"""Tests for stale checkout (connection leak) detection on sync pools.

These tests are self-contained (no Redis server): they drive the pools with a
dummy connection class, exactly like ``tests/test_connection_pool.py``.
"""

import gc
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from redis import (
    BlockingConnectionPool,
    ConnectionPool,
    LeakDetectionConfig,
    MaxConnectionsError,
    StaleCheckoutInfo,
)
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.leak_detection import (
    _LeakTracker,
    pool_mark_checkout_exempt,
    pool_mark_command_exemption,
)
from redis.sentinel import SentinelConnectionPool

THIS_FILE = os.path.abspath(__file__)


class DummyConnection:
    description_format = "DummyConnection<>"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pid = os.getpid()
        self._sock = None
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 6379)
        self.db = kwargs.get("db", 0)

    def connect(self):
        self._sock = mock.Mock()

    def disconnect(self):
        self._sock = None

    def can_read(self, timeout: float = 0) -> bool:
        return False

    def should_reconnect(self):
        return False

    def re_auth(self):
        pass


def config(**overrides):
    overrides.setdefault("enabled", True)
    overrides.setdefault("threshold", 0)
    return LeakDetectionConfig(**overrides)


def make_pool(cls=ConnectionPool, **kwargs):
    kwargs.setdefault("connection_class", DummyConnection)
    kwargs.setdefault("max_connections", 10)
    return cls(**kwargs)


class TestLeakDetectionConfigValidation:
    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            LeakDetectionConfig(enabled=True, threshold=-1)

    def test_invalid_context_depth(self):
        with pytest.raises(ValueError):
            LeakDetectionConfig(enabled=True, context_depth=-1)

    def test_invalid_error_detail(self):
        with pytest.raises(ValueError):
            LeakDetectionConfig(enabled=True, error_detail="nope")

    def test_invalid_on_reset(self):
        with pytest.raises(ValueError):
            LeakDetectionConfig(enabled=True, on_reset="nope")

    def test_exempt_commands_normalized(self):
        cfg = LeakDetectionConfig(
            enabled=True, exempt_commands=["blpop", b"XREAD", " Wait "]
        )
        assert cfg.exempt_commands == frozenset({"BLPOP", "XREAD", "WAIT"})
        assert cfg.is_exempt_command(b"blpop")
        assert not cfg.is_exempt_command("GET")


class TestDisabledByDefault:
    def test_borrow_return_path_unchanged(self):
        pool = make_pool()
        conn = pool.get_connection()
        assert pool.stale_checkouts() == ()
        pool.release(conn)
        assert pool.stale_checkouts() == ()

    def test_capacity_error_message_unchanged(self):
        pool = make_pool(max_connections=1)
        conn = pool.get_connection()
        with pytest.raises(MaxConnectionsError) as exc_info:
            pool.get_connection()
        assert str(exc_info.value) == "Too many connections"
        pool.release(conn)

    def test_blocking_timeout_message_unchanged(self):
        pool = make_pool(BlockingConnectionPool, max_connections=1, timeout=0.05)
        conn = pool.get_connection()
        with pytest.raises(RedisConnectionError) as exc_info:
            pool.get_connection()
        assert str(exc_info.value) == "No connection available."
        pool.release(conn)

    def test_marker_methods_return_false(self):
        pool = make_pool()
        conn = pool.get_connection()
        assert pool.mark_checkout_exempt(conn, "x") is False
        assert pool.unmark_checkout_exempt(conn) is False
        assert pool.mark_command_checkout_exempt(conn, "BLPOP") is False
        pool.release(conn)


class TestStaleCheckoutQuery:
    def test_record_contents(self):
        pool = make_pool(leak_detection=config(context_depth=3))
        conn = pool.get_connection()
        stale = pool.stale_checkouts()
        assert len(stale) == 1
        info = stale[0]
        assert isinstance(info, StaleCheckoutInfo)
        assert info.age >= 0
        assert info.gc_collected is False
        assert info.exempt is False
        assert info.archived is False
        assert info.command_name is None
        assert "DummyConnection" in info.connection
        assert "id=0x" in info.connection
        # The call-site summary skips redis internals and lands in this file.
        assert info.context
        assert info.context[0].startswith(THIS_FILE)
        pool.release(conn)

    def test_return_removes_record(self):
        pool = make_pool(leak_detection=config())
        conn = pool.get_connection()
        assert len(pool.stale_checkouts()) == 1
        pool.release(conn)
        assert pool.stale_checkouts() == ()

    def test_healthy_short_checkouts_not_stale(self):
        pool = make_pool(leak_detection=config(threshold=100))
        conn = pool.get_connection()
        assert pool.stale_checkouts() == ()
        pool.release(conn)

    def test_custom_threshold_argument(self):
        pool = make_pool(leak_detection=config(threshold=100))
        conn = pool.get_connection()
        assert pool.stale_checkouts() == ()
        # Querying with a lower threshold surfaces the same record.
        assert len(pool.stale_checkouts(threshold=0)) == 1
        pool.release(conn)

    def test_ordered_oldest_first(self):
        pool = make_pool(max_connections=10, leak_detection=config())
        conns = [pool.get_connection() for _ in range(3)]
        stale = pool.stale_checkouts()
        times = [item.checkout_time for item in stale]
        assert times == sorted(times)
        for conn in conns:
            pool.release(conn)

    def test_context_capture_can_be_disabled(self):
        pool = make_pool(leak_detection=config(capture_context=False))
        conn = pool.get_connection()
        assert pool.stale_checkouts()[0].context == ()
        pool.release(conn)


class TestAlerts:
    def test_alert_fires_once_per_checkout(self):
        seen = []
        pool = make_pool(leak_detection=config(alert_callback=seen.append))
        conn = pool.get_connection()
        assert len(seen) == 1
        # Further pool activity must not re-alert for the same checkout.
        pool._dispatch_leak_alerts()
        pool.stale_checkouts()
        pool._dispatch_leak_alerts()
        assert len(seen) == 1
        assert seen[0].connection == pool.stale_checkouts()[0].connection
        pool.release(conn)

    def test_no_alert_for_healthy_checkout(self):
        seen = []
        pool = make_pool(
            leak_detection=config(threshold=100, alert_callback=seen.append)
        )
        conn = pool.get_connection()
        pool.release(conn)
        assert seen == []

    def test_callback_exception_does_not_break_pool(self):
        def bad_callback(_info):
            raise RuntimeError("boom")

        pool = make_pool(leak_detection=config(alert_callback=bad_callback))
        conn = pool.get_connection()  # must not raise
        stale = pool.stale_checkouts()
        assert len(stale) == 1
        pool.release(conn)
        assert pool.stale_checkouts() == ()

    def test_alert_per_checkout_then_clean(self):
        seen = []
        pool = make_pool(
            max_connections=10,
            leak_detection=config(threshold=0, alert_callback=seen.append),
        )
        long_held = pool.get_connection()
        second = pool.get_connection()
        assert len(seen) == 2
        pool.release(second)
        pool.release(long_held)
        assert pool.stale_checkouts() == ()


class TestExemptions:
    def test_manual_exempt_and_unmark(self):
        pool = make_pool(leak_detection=config())
        conn = pool.get_connection()
        assert pool.mark_checkout_exempt(conn, "pubsub") is True
        assert pool.stale_checkouts() == ()
        exempt = pool.stale_checkouts(include_exempt=True)
        assert len(exempt) == 1
        assert exempt[0].exempt is True
        assert exempt[0].exempt_reason == "pubsub"
        assert pool.unmark_checkout_exempt(conn) is True
        assert len(pool.stale_checkouts()) == 1
        pool.release(conn)

    def test_configured_command_exemption(self):
        pool = make_pool(leak_detection=config(exempt_commands=["BLPOP"]))
        conn = pool.get_connection()
        assert pool.mark_command_checkout_exempt(conn, b"blpop") is True
        assert pool.stale_checkouts() == ()
        assert pool.stale_checkouts(include_exempt=True)[0].exempt_reason == "BLPOP"
        pool.release(conn)

    def test_command_not_in_exempt_list(self):
        pool = make_pool(leak_detection=config(exempt_commands=["BLPOP"]))
        conn = pool.get_connection()
        assert pool.mark_command_checkout_exempt(conn, "GET") is False
        assert len(pool.stale_checkouts()) == 1
        pool.release(conn)

    def test_mark_unknown_connection_returns_false(self):
        pool = make_pool(leak_detection=config())
        assert pool.mark_checkout_exempt(DummyConnection(), "x") is False


class TestClientMarkerHelpers:
    """The helpers the client layer uses must tolerate arbitrary pools."""

    def test_third_party_pool_without_marker_is_ignored(self):
        conn = DummyConnection()
        assert pool_mark_checkout_exempt(object(), conn, "pubsub") is False
        assert pool_mark_command_exemption(object(), conn, "BLPOP") is False

    def test_markers_delegate_to_supported_pool(self):
        pool = make_pool(leak_detection=config(exempt_commands=["BLPOP"]))
        conn = pool.get_connection()
        assert pool_mark_checkout_exempt(pool, conn, "pubsub") is True
        assert pool.stale_checkouts() == ()
        pool.unmark_checkout_exempt(conn)
        assert pool_mark_command_exemption(pool, conn, "blpop") is True
        assert pool.stale_checkouts() == ()
        pool.release(conn)


class TestExhaustionErrors:
    def test_minimal_detail_keeps_legacy_message(self):
        pool = make_pool(
            max_connections=1,
            leak_detection=config(error_detail="minimal"),
        )
        conn = pool.get_connection()
        with pytest.raises(MaxConnectionsError, match="^Too many connections$"):
            pool.get_connection()
        pool.release(conn)

    def test_summary_detail_for_regular_pool(self):
        pool = make_pool(
            max_connections=1,
            leak_detection=config(error_detail="summary"),
        )
        conn = pool.get_connection()
        with pytest.raises(MaxConnectionsError) as exc_info:
            pool.get_connection()
        message = str(exc_info.value)
        assert message.startswith("Too many connections (")
        assert "pool capacity: 1" in message
        assert "connections in use: 1" in message
        assert "oldest active checkout:" in message
        pool.release(conn)

    def test_full_detail_includes_call_site(self):
        pool = make_pool(
            max_connections=1,
            leak_detection=config(error_detail="full"),
        )
        conn = pool.get_connection()
        with pytest.raises(MaxConnectionsError) as exc_info:
            pool.get_connection()
        message = str(exc_info.value)
        assert "stale checkouts exceeding" in message
        assert "DummyConnection" in message
        assert THIS_FILE in message
        pool.release(conn)

    def test_blocking_pool_timeout_detail(self):
        pool = make_pool(
            BlockingConnectionPool,
            max_connections=1,
            timeout=0.05,
            leak_detection=config(error_detail="summary"),
        )
        conn = pool.get_connection()
        with pytest.raises(RedisConnectionError) as exc_info:
            pool.get_connection()
        message = str(exc_info.value)
        assert message.startswith("No connection available. (")
        assert "pool capacity: 1" in message
        assert "connections in use: 1" in message
        assert "idle connections: 0" in message
        assert "wait time: 0.050s" in message
        assert "oldest active checkout:" in message
        pool.release(conn)

    def test_blocking_pool_full_detail(self):
        pool = make_pool(
            BlockingConnectionPool,
            max_connections=1,
            timeout=0.05,
            leak_detection=config(error_detail="full"),
        )
        conn = pool.get_connection()
        with pytest.raises(RedisConnectionError) as exc_info:
            pool.get_connection()
        assert THIS_FILE in str(exc_info.value)
        pool.release(conn)


class TestGarbageCollectedConnections:
    def test_record_survives_connection_gc(self):
        tracker = _LeakTracker(config())
        conn = DummyConnection()
        tracker.checkout(conn)
        identity = tracker.stale_checkouts()[0].connection
        del conn
        gc.collect()
        stale = tracker.stale_checkouts()
        assert len(stale) == 1
        assert stale[0].gc_collected is True
        assert stale[0].connection == identity

    def test_gc_record_carries_call_site(self):
        tracker = _LeakTracker(config())
        conn = DummyConnection()
        tracker.checkout(conn)
        del conn
        gc.collect()
        info = tracker.stale_checkouts()[0]
        assert info.context[0].startswith(THIS_FILE)

    def test_release_detaches_gc_finalizer(self):
        tracker = _LeakTracker(config())
        conn = DummyConnection()
        tracker.checkout(conn)
        tracker.release(conn)
        del conn
        gc.collect()
        assert tracker.stale_checkouts() == ()


class TestResetAndArchive:
    def test_reset_clears_by_default(self):
        pool = make_pool(leak_detection=config())
        conn = pool.get_connection()
        assert len(pool.stale_checkouts()) == 1
        pool.reset()
        assert pool.stale_checkouts() == ()
        assert pool.stale_checkouts(include_archived=True) == ()
        # The borrowed connection slot is gone after reset, nothing to release.
        del conn

    def test_reset_archives_records(self):
        pool = make_pool(leak_detection=config(on_reset="archive"))
        conn = pool.get_connection()
        pool.reset()
        assert pool.stale_checkouts() == ()
        archived = pool.stale_checkouts(include_archived=True)
        assert len(archived) == 1
        assert archived[0].archived is True
        pool.clear_stale_checkout_archive()
        assert pool.stale_checkouts(include_archived=True) == ()
        del conn

    def test_close_archives_records_when_configured(self):
        pool = make_pool(leak_detection=config(on_reset="archive"))
        conn = pool.get_connection()
        pool.close()
        assert len(pool.stale_checkouts(include_archived=True)) == 1
        del conn

    def test_gc_records_included_in_archive(self):
        tracker = _LeakTracker(config(on_reset="archive"))
        conn = DummyConnection()
        tracker.checkout(conn)
        del conn
        gc.collect()
        tracker.reset()
        archived = tracker.stale_checkouts(include_archived=True)
        assert len(archived) == 1
        assert archived[0].archived is True
        assert archived[0].gc_collected is True


class TestSentinelFailover:
    def _make_sentinel_pool(self):
        manager = MagicMock()
        manager.discover_master.return_value = ("10.0.0.1", 6379)
        pool = SentinelConnectionPool("mymaster", manager, leak_detection=config())
        return pool, manager

    def test_failover_clears_checkout_records(self):
        pool, manager = self._make_sentinel_pool()
        # Initial discovery registers the first master address.
        assert pool.get_master_address() == ("10.0.0.1", 6379)

        forgotten = DummyConnection()
        pool._leak_tracker.checkout(forgotten)
        assert len(pool.stale_checkouts()) == 1

        manager.discover_master.return_value = ("10.0.0.2", 6379)
        assert pool.get_master_address() == ("10.0.0.2", 6379)
        assert pool.stale_checkouts() == ()
        assert pool.stale_checkouts(include_archived=True) == ()
        del forgotten

    def test_failover_ignores_archive_policy(self):
        manager = MagicMock()
        manager.discover_master.return_value = ("10.0.0.1", 6379)
        pool = SentinelConnectionPool(
            "mymaster",
            manager,
            leak_detection=config(on_reset="archive"),
        )
        pool.get_master_address()
        forgotten = DummyConnection()
        pool._leak_tracker.checkout(forgotten)
        manager.discover_master.return_value = ("10.0.0.2", 6379)
        pool.get_master_address()
        assert pool.stale_checkouts(include_archived=True) == ()
        del forgotten
