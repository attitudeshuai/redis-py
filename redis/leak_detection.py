"""Stale checkout (connection leak) detection for connection pools.

The pool implementations (sync/async, regular/blocking) embed a single
:class:`_LeakTracker` each. The tracker is deliberately transport agnostic and
synchronous, so the same code backs both stacks and the async side never awaits
inside its borrow/return hot path.

The feature is opt-in via :class:`LeakDetectionConfig`. When it is not enabled
the tracker short-circuits every hook, leaving the borrow/return path with its
original behaviour and cost.

A tracked checkout records:

* the monotonic checkout time,
* a short summary of the call site that borrowed the connection (a few stack
  frames that live outside the ``redis`` package - never a full stack trace),
* a connection identifier captured eagerly, so the source of a checkout stays
  visible even after the connection object has been garbage collected,
* optional exemption state for legitimate long checkouts (pub/sub, watched
  transactions, blocking commands, single-connection clients).

Checkout records never keep their connection alive: the tracker holds a
``weakref`` and a ``weakref.finalize`` reclassifies the record when the
connection is collected without ever being returned.
"""

import logging
import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Optional

logger = logging.getLogger(__name__)

#: Directory of the ``redis`` package itself; stack frames captured from files
#: inside it are treated as library internals and skipped.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Never walk an unbounded number of frames while capturing a call site.
_MAX_FRAME_WALK = 64

ERROR_DETAIL_MINIMAL = "minimal"
ERROR_DETAIL_SUMMARY = "summary"
ERROR_DETAIL_FULL = "full"
ERROR_DETAIL_LEVELS = (
    ERROR_DETAIL_MINIMAL,
    ERROR_DETAIL_SUMMARY,
    ERROR_DETAIL_FULL,
)

RESET_POLICY_CLEAR = "clear"
RESET_POLICY_ARCHIVE = "archive"
RESET_POLICIES = (RESET_POLICY_CLEAR, RESET_POLICY_ARCHIVE)


@dataclass(frozen=True)
class StaleCheckoutInfo:
    """Read-only view of a checkout that has outlived the configured threshold.

    Returned by ``ConnectionPool.stale_checkouts()`` (sync and async) and passed
    to the configured alert callback. Instances are snapshots - they never
    mutate after they are handed out.

    Attributes:
        checkout_time: the ``time.monotonic()`` value captured at checkout.
        age: seconds elapsed since the checkout at snapshot time.
        connection: connection identifier captured at checkout. Survives the
            connection object being garbage collected.
        context: call-site summary, outer (non-redis) frames first, formatted
            as ``"filename:lineno:function"``. Empty when context capture is
            disabled.
        exempt: True for checkouts explicitly marked as legitimate long
            checkouts (pub/sub, watched transactions, configured commands, ...).
        exempt_reason: optional reason given when the checkout was exempted.
        gc_collected: True when the connection object was garbage collected
            before being returned to the pool.
        command_name: command name recorded at checkout, if one was supplied.
        archived: True for records retained from a previous pool reset/close
            (``on_reset="archive"``) instead of live checkouts.
    """

    checkout_time: float
    age: float
    connection: str
    context: tuple
    exempt: bool
    exempt_reason: Optional[str]
    gc_collected: bool
    command_name: Optional[str]
    archived: bool


@dataclass
class LeakDetectionConfig:
    """Configuration for stale checkout detection on a connection pool.

    Args:
        enabled: master switch. When False (the default) every pool hook
            short-circuits and the borrow/return path behaves and costs the
            same as without the feature.
        threshold: checkout age in seconds after which a still-outstanding
            checkout is considered stale, becomes visible through
            ``stale_checkouts()`` and triggers ``alert_callback`` once. A value
            of ``0`` makes every active checkout immediately visible (useful in
            tests).
        alert_callback: optional callable invoked with a
            :class:`StaleCheckoutInfo` when a checkout crosses ``threshold``.
            Each stale checkout is reported at most once. Exceptions raised by
            the callback are logged and never affect the pool. On async pools
            the callback is a regular synchronous callable scheduled with
            ``loop.call_soon`` - it must not block.
        capture_context: whether to capture the call-site summary at checkout.
        context_depth: number of stack frames outside the ``redis`` package to
            keep in the call-site summary.
        error_detail: how much diagnostics pool-exhaustion errors carry:

            * ``"minimal"`` - the original error message, unchanged;
            * ``"summary"`` - pool capacity, in-use/idle counts, wait time and
              the oldest active checkout age;
            * ``"full"`` - summary plus the oldest stale checkouts with their
              call sites.
        on_reset: what happens to checkout records when the pool is reset
            (e.g. after a fork) or closed: ``"clear"`` drops them,
            ``"archive"`` retains them, retrievable through
            ``stale_checkouts(include_archived=True)``. Sentinel failover
            always clears records regardless of this setting.
        exempt_commands: command names (case-insensitive) whose checkouts are
            automatically marked exempt, e.g. blocking commands such as
            ``BLPOP`` or ``XREAD``.
        max_error_reports: maximum number of stale checkouts listed in a
            ``"full"`` detail exhaustion error.
    """

    enabled: bool = False
    threshold: float = 60.0
    alert_callback: Optional[Callable[[StaleCheckoutInfo], Any]] = None
    capture_context: bool = True
    context_depth: int = 3
    error_detail: str = ERROR_DETAIL_SUMMARY
    on_reset: str = RESET_POLICY_CLEAR
    exempt_commands: Collection[str] = field(default_factory=frozenset)
    max_error_reports: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.threshold, (int, float)) or self.threshold < 0:
            raise ValueError('"threshold" must be a non-negative number')
        if not isinstance(self.context_depth, int) or self.context_depth < 0:
            raise ValueError('"context_depth" must be a non-negative integer')
        if self.error_detail not in ERROR_DETAIL_LEVELS:
            raise ValueError(
                '"error_detail" must be one of: {}'.format(
                    ", ".join(ERROR_DETAIL_LEVELS)
                )
            )
        if self.on_reset not in RESET_POLICIES:
            raise ValueError(
                '"on_reset" must be one of: {}'.format(", ".join(RESET_POLICIES))
            )
        if not isinstance(self.max_error_reports, int) or self.max_error_reports < 1:
            raise ValueError('"max_error_reports" must be a positive integer')
        self.exempt_commands = frozenset(
            _normalize_command_name(name) for name in self.exempt_commands
        )

    def is_exempt_command(self, command_name: Any) -> bool:
        normalized = _normalize_command_name(command_name)
        return normalized is not None and normalized in self.exempt_commands


def _normalize_command_name(command_name: Any) -> Optional[str]:
    if command_name is None:
        return None
    if isinstance(command_name, bytes):
        command_name = command_name.decode("utf-8", "replace")
    return str(command_name).strip().upper() or None


def pool_mark_checkout_exempt(
    pool: Any, connection: Any, reason: Optional[str] = None
) -> bool:
    """Mark ``connection`` exempt on a pool supporting leak detection.

    Safe to call against arbitrary pool implementations: pools without the
    marker method (third-party pools) are simply ignored.
    """
    marker = getattr(pool, "mark_checkout_exempt", None)
    if marker is None:
        return False
    return marker(connection, reason)


def pool_mark_command_exemption(pool: Any, connection: Any, command_name: Any) -> bool:
    """Exempt a checkout when the pool config lists ``command_name`` as exempt.

    Safe to call against arbitrary pool implementations.
    """
    marker = getattr(pool, "mark_command_checkout_exempt", None)
    if marker is None:
        return False
    return marker(connection, command_name)


def _capture_context(depth: int) -> tuple:
    """Return up to ``depth`` non-redis stack frames, oldest callers first."""
    if depth <= 0:
        return ()
    try:
        frame = sys._getframe(1)
    except ValueError:
        return ()
    frames = []
    walked = 0
    while frame is not None and walked < _MAX_FRAME_WALK and len(frames) < depth:
        filename = frame.f_code.co_filename
        try:
            internal = os.path.abspath(filename).startswith(_PACKAGE_DIR)
        except (OSError, ValueError):
            internal = False
        if not internal:
            frames.append(
                "{}:{}:{}".format(
                    filename, frame.f_lineno, frame.f_code.co_name
                )
            )
        frame = frame.f_back
        walked += 1
    return tuple(frames)


def _connection_identity(connection: Any) -> str:
    """Build a stable identifier for ``connection`` at checkout time."""
    host = getattr(connection, "host", None)
    port = getattr(connection, "port", None)
    db = getattr(connection, "db", None)
    location = None
    if host is not None and port is not None:
        location = f"{host}:{port}"
    elif host is not None:
        location = str(host)
    details = type(connection).__name__
    if location is not None:
        details += f"<{location}"
        if db is not None:
            details += f",db={db}"
        details += ">"
    details += f" id=0x{id(connection):x}"
    return details


class _CheckoutRecord:
    __slots__ = (
        "seq",
        "conn_id",
        "conn_ref",
        "finalizer",
        "connection_name",
        "checkout_time",
        "context",
        "command_name",
        "exempt",
        "exempt_reason",
        "alerted",
        "gc_collected",
        "generation",
    )

    def __init__(
        self,
        seq: int,
        tracker: "_LeakTracker",
        connection: Any,
        checkout_time: float,
        context: tuple,
        command_name: Optional[str],
        generation: int,
    ):
        self.seq = seq
        self.conn_id = id(connection)
        self.connection_name = _connection_identity(connection)
        self.checkout_time = checkout_time
        self.context = context
        self.command_name = command_name
        self.exempt = False
        self.exempt_reason = None
        self.alerted = False
        self.gc_collected = False
        self.generation = generation
        # ``weakref.ref`` lets the connection be garbage collected while the
        # record (and therefore the leak attribution) survives. Objects that
        # cannot be weakly referenced fall back to no reference at all; the
        # finalizer is not installed in that case.
        try:
            self.conn_ref = weakref.ref(connection)
            self.finalizer = weakref.finalize(
                connection, _finalize_checkout_record, tracker, self
            )
        except TypeError:
            self.conn_ref = None
            self.finalizer = None


def _finalize_checkout_record(
    tracker: "_LeakTracker", record: "_CheckoutRecord"
) -> None:
    """Reclassify a record whose connection was collected without release.

    Runs from the garbage collector, possibly at interpreter shutdown, so it
    must never raise.
    """
    try:
        tracker._note_gc(record)
    except Exception:
        logger.debug("failed to reclassify a garbage-collected checkout", exc_info=True)


class _LeakTracker:
    """Per-pool bookkeeping for stale checkouts. Shared by sync and async pools."""

    def __init__(self, config: Optional[LeakDetectionConfig]):
        self.config = config if config is not None else LeakDetectionConfig()
        self._lock = threading.RLock()
        self._active: dict = {}
        self._leaked: list = []
        self._archived: list = []
        self._seq = 0
        self._generation = 0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    # ------------------------------------------------------------------ hooks

    def checkout(self, connection: Any, command_name: Any = None) -> None:
        """Register a connection being checked out of the pool."""
        if not self.enabled:
            return
        command = _normalize_command_name(command_name)
        context = (
            _capture_context(self.config.context_depth)
            if self.config.capture_context
            else ()
        )
        now = time.monotonic()
        with self._lock:
            self._seq += 1
            record = _CheckoutRecord(
                self._seq,
                self,
                connection,
                now,
                context,
                command,
                self._generation,
            )
            if command is not None and self.config.is_exempt_command(command):
                record.exempt = True
                record.exempt_reason = command
            self._active[id(connection)] = record

    def release(self, connection: Any) -> None:
        """Forget a checkout once its connection is returned to the pool."""
        if not self.enabled:
            return
        with self._lock:
            record = self._active.pop(id(connection), None)
        if record is not None and record.finalizer is not None:
            record.finalizer.detach()

    def set_exempt(
        self,
        connection: Any,
        exempt: bool = True,
        reason: Optional[str] = None,
    ) -> bool:
        """Mark/unmark the active checkout of ``connection`` as exempt."""
        if not self.enabled:
            return False
        with self._lock:
            record = self._active.get(id(connection))
            if record is None:
                return False
            record.exempt = exempt
            record.exempt_reason = reason if exempt else None
            return True

    def apply_command_exemption(self, connection: Any, command_name: Any) -> bool:
        """Mark the checkout exempt when ``command_name`` is configured exempt."""
        if not self.enabled or not self.config.is_exempt_command(command_name):
            return False
        command = _normalize_command_name(command_name)
        return self.set_exempt(connection, True, command)

    def _note_gc(self, record: "_CheckoutRecord") -> None:
        with self._lock:
            if self._active.get(record.conn_id) is record:
                del self._active[record.conn_id]
                record.gc_collected = True
                self._leaked.append(record)

    # ------------------------------------------------------------------ alerts

    def poll_alerts(self, now: Optional[float] = None) -> list:
        """Return (once) views of checkouts that newly crossed the threshold."""
        if not self.enabled:
            return []
        threshold = self.config.threshold
        if now is None:
            now = time.monotonic()
        alerted_views = []
        with self._lock:
            for record in self._iter_live_records():
                if (
                    not record.alerted
                    and not record.exempt
                    and now - record.checkout_time >= threshold
                ):
                    record.alerted = True
                    alerted_views.append(self._view(record, now))
        return alerted_views

    # ----------------------------------------------------------------- queries

    def stale_checkouts(
        self,
        threshold: Optional[float] = None,
        include_exempt: bool = False,
        include_archived: bool = False,
        now: Optional[float] = None,
    ) -> tuple:
        """Return stale checkout views, oldest first.

        ``threshold`` defaults to the configured threshold. Live checkouts
        whose connection was garbage collected before release are always
        included once they are stale.
        """
        if not self.enabled:
            return ()
        if threshold is None:
            threshold = self.config.threshold
        if now is None:
            now = time.monotonic()
        candidates = []
        with self._lock:
            for record in self._iter_live_records():
                if not include_exempt and record.exempt:
                    continue
                if now - record.checkout_time >= threshold:
                    candidates.append(self._view(record, now))
            if include_archived:
                for record in tuple(self._archived):
                    if not include_exempt and record.exempt:
                        continue
                    candidates.append(self._view(record, now, archived=True))
        candidates.sort(key=lambda item: item.checkout_time)
        return tuple(candidates)

    def oldest_active_age(self, now: Optional[float] = None) -> Optional[float]:
        """Return the age of the oldest live checkout, exempt or not."""
        if not self.enabled:
            return None
        if now is None:
            now = time.monotonic()
        with self._lock:
            oldest = min(
                (r.checkout_time for r in self._iter_live_records()),
                default=None,
            )
        return None if oldest is None else now - oldest

    def clear_archive(self) -> None:
        with self._lock:
            self._archived = []

    # ------------------------------------------------------------ lifecycle

    def reset(self, archive: Optional[bool] = None) -> None:
        """Reset tracking state on pool reset (e.g. fork) or close.

        ``archive=None`` defers to ``config.on_reset``.
        """
        if not self.enabled:
            return
        if archive is None:
            archive = self.config.on_reset == RESET_POLICY_ARCHIVE
        with self._lock:
            records = list(self._active.values()) + list(self._leaked)
            for record in records:
                if record.finalizer is not None:
                    record.finalizer.detach()
                if record.conn_ref is not None and record.conn_ref() is None:
                    record.gc_collected = True
            if archive:
                self._archived.extend(records)
            self._active = {}
            self._leaked = []
            self._generation += 1

    def clear(self) -> None:
        """Unconditionally drop every live record (sentinel failover)."""
        if not self.enabled:
            return
        with self._lock:
            records = list(self._active.values()) + list(self._leaked)
            for record in records:
                if record.finalizer is not None:
                    record.finalizer.detach()
            self._active = {}
            self._leaked = []
            self._generation += 1

    # ------------------------------------------------------- error reporting

    def format_exhaustion_message(
        self,
        base_message: str,
        *,
        capacity: int,
        in_use: int,
        idle: Optional[int] = None,
        waited: Optional[float] = None,
        now: Optional[float] = None,
    ) -> str:
        """Enrich a pool-exhaustion error message per ``error_detail``.

        Returns ``base_message`` unchanged when detection is disabled or the
        detail level is ``"minimal"``.
        """
        if not self.enabled or self.config.error_detail == ERROR_DETAIL_MINIMAL:
            return base_message
        details = [
            f"pool capacity: {capacity}",
            f"connections in use: {in_use}",
        ]
        if idle is not None:
            details.append(f"idle connections: {idle}")
        if waited is not None:
            details.append(f"wait time: {waited:.3f}s")
        oldest = self.oldest_active_age(now)
        if oldest is not None:
            details.append(f"oldest active checkout: {oldest:.3f}s")
        message = f"{base_message} ({'; '.join(details)})"
        if self.config.error_detail == ERROR_DETAIL_FULL:
            stale = self.stale_checkouts(now=now)[: self.config.max_error_reports]
            if stale:
                lines = [
                    "\nstale checkouts exceeding "
                    f"{self.config.threshold:.3f}s, oldest first:"
                ]
                for item in stale:
                    origin = item.context[0] if item.context else "<unknown call site>"
                    suffix = " [gc collected]" if item.gc_collected else ""
                    if item.exempt_reason:
                        suffix += f" [exempt: {item.exempt_reason}]"
                    lines.append(
                        f"  - age {item.age:.3f}s, {item.connection}{suffix}: {origin}"
                    )
                message += "\n".join(lines)
        return message

    # ---------------------------------------------------------------- helpers

    def _iter_live_records(self):
        # Active records plus checkouts whose connections were collected
        # without being returned. ``_active`` is iterated from a snapshot to
        # stay safe if a finalizer re-enters the tracker mid-iteration.
        yield from list(self._active.values())
        yield from self._leaked

    def _view(
        self,
        record: "_CheckoutRecord",
        now: float,
        archived: bool = False,
    ) -> StaleCheckoutInfo:
        gc_collected = record.gc_collected
        if not gc_collected and record.conn_ref is not None:
            gc_collected = record.conn_ref() is None
        return StaleCheckoutInfo(
            checkout_time=record.checkout_time,
            age=max(0.0, now - record.checkout_time),
            connection=record.connection_name,
            context=record.context,
            exempt=record.exempt,
            exempt_reason=record.exempt_reason,
            gc_collected=gc_collected,
            command_name=record.command_name,
            archived=archived,
        )
