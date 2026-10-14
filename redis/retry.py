import abc
import logging
import math
import socket
from time import monotonic, sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from redis.exceptions import ConnectionError, RetryTimeoutError, TimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")
E = TypeVar("E", bound=Exception, covariant=True)

#: When the ``on_failure`` callback raises, log it (never silently) and continue
#: the retry process without changing the retry decision.
ON_FAILURE_LOG = "log"
#: When the ``on_failure`` callback raises, propagate the exception and abort
#: the retry session.
ON_FAILURE_RAISE = "raise"
_ON_FAILURE_ERROR_POLICIES = (ON_FAILURE_LOG, ON_FAILURE_RAISE)

#: Per-failure observer callback, invoked as
#: ``on_failure(attempt, elapsed, backoff, error)`` after every failed attempt:
#: ``attempt`` is the 1-based attempt number, ``elapsed`` the seconds elapsed
#: since the first attempt of the session, ``backoff`` the seconds the session
#: is about to sleep before the next attempt (``0`` when no further attempt
#: follows) and ``error`` the exception raised by the attempt.
OnFailureCallback = Callable[[int, float, float, Exception], Any]

if TYPE_CHECKING:
    from redis.backoff import AbstractBackoff


def _validate_total_timeout(total_timeout: Any) -> Optional[float]:
    """Validate the total retry timeout and normalize it to a float."""
    if total_timeout is None:
        return None
    # ``bool`` is a subclass of ``int``; reject it explicitly.
    if isinstance(total_timeout, bool) or not isinstance(total_timeout, (int, float)):
        raise TypeError("total_timeout must be a positive number or None")
    if not math.isfinite(total_timeout) or total_timeout <= 0:
        raise ValueError("total_timeout must be greater than 0")
    return float(total_timeout)


class AbstractRetry(Generic[E], abc.ABC):
    """Retry a specific number of times after a failure"""

    _supported_errors: Tuple[Type[E], ...]

    def __init__(
        self,
        backoff: "AbstractBackoff",
        retries: int,
        supported_errors: Tuple[Type[E], ...],
        total_timeout: Optional[Union[int, float]] = None,
        on_failure: Optional[OnFailureCallback] = None,
        on_failure_error: str = ON_FAILURE_LOG,
    ):
        """
        Initialize a `Retry` object with a `Backoff` object
        that retries a maximum of `retries` times.
        `retries` can be negative to retry forever.
        You can specify the types of supported errors which trigger
        a retry with the `supported_errors` parameter.
        `total_timeout` optionally bounds the total time, in seconds, of a
        retry session (attempts plus backoff sleeps), measured from the first
        attempt. When it elapses, no further attempt is made and the last
        failure is raised wrapped in `RetryTimeoutError`.
        `on_failure` is an optional callback invoked after every failed
        attempt with `(attempt, elapsed, backoff, error)`.
        `on_failure_error` selects what happens when `on_failure` itself
        raises: `"log"` (the default) logs the error and keeps retrying,
        `"raise"` propagates it.
        """
        if on_failure is not None and not callable(on_failure):
            raise TypeError("on_failure must be callable or None")
        if on_failure_error not in _ON_FAILURE_ERROR_POLICIES:
            raise ValueError(
                "on_failure_error must be one of "
                f"{_ON_FAILURE_ERROR_POLICIES!r}, got {on_failure_error!r}"
            )
        self._backoff = backoff
        self._retries = retries
        self._supported_errors = supported_errors
        self._total_timeout = _validate_total_timeout(total_timeout)
        self._on_failure = on_failure
        self._on_failure_error = on_failure_error

    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool:
        return NotImplemented

    def __hash__(self) -> int:
        return hash(
            (
                self._backoff,
                self._retries,
                frozenset(self._supported_errors),
                self._total_timeout,
                self._on_failure_error,
            )
        )

    def update_supported_errors(self, specified_errors: Iterable[Type[E]]) -> None:
        """
        Updates the supported errors with the specified error types
        """
        self._supported_errors = tuple(
            set(self._supported_errors + tuple(specified_errors))
        )

    def get_retries(self) -> int:
        """
        Get the number of retries.
        """
        return self._retries

    def update_retries(self, value: int) -> None:
        """
        Set the number of retries.
        """
        self._retries = value

    def _timeout_error(
        self, failures: int, elapsed: float, error: Exception
    ) -> RetryTimeoutError:
        return RetryTimeoutError(
            failures,
            elapsed,
            error,
            f"Total retry timeout of {self._total_timeout:g}s exceeded after "
            f"{failures} attempt(s)",
        )


class Retry(AbstractRetry[Exception]):
    __hash__ = AbstractRetry.__hash__

    def __init__(
        self,
        backoff: "AbstractBackoff",
        retries: int,
        supported_errors: Tuple[Type[Exception], ...] = (
            ConnectionError,
            TimeoutError,
            socket.timeout,
        ),
        total_timeout: Optional[Union[int, float]] = None,
        on_failure: Optional[OnFailureCallback] = None,
        on_failure_error: str = ON_FAILURE_LOG,
    ):
        super().__init__(
            backoff,
            retries,
            supported_errors,
            total_timeout=total_timeout,
            on_failure=on_failure,
            on_failure_error=on_failure_error,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Retry):
            return NotImplemented

        return (
            self._backoff == other._backoff
            and self._retries == other._retries
            and set(self._supported_errors) == set(other._supported_errors)
            and self._total_timeout == other._total_timeout
            and self._on_failure_error == other._on_failure_error
            # callbacks are compared by identity
            and self._on_failure is other._on_failure
        )

    def _notify_failure(
        self, failures: int, elapsed: float, backoff: float, error: Exception
    ) -> None:
        """Invoke the optional per-failure observer callback."""
        if self._on_failure is None:
            return
        try:
            self._on_failure(failures, elapsed, backoff, error)
        except Exception:
            if self._on_failure_error == ON_FAILURE_RAISE:
                raise
            logger.warning(
                "redis retry on_failure callback raised an exception after "
                "attempt %d; continuing the retry process",
                failures,
                exc_info=True,
            )

    def call_with_retry(
        self,
        do: Callable[[], T],
        fail: Union[Callable[[Exception], Any], Callable[[Exception, int], Any]],
        is_retryable: Optional[Callable[[Exception], bool]] = None,
        with_failure_count: bool = False,
    ) -> T:
        """
        Execute an operation that might fail and returns its result, or
        raise the exception that was thrown depending on the `Backoff` object.
        `do`: the operation to call. Expects no argument.
        `fail`: the failure handler, expects the last error that was thrown
        ``is_retryable``: optional function to determine if an error is retryable
        ``with_failure_count``: if True, the failure count is passed to the failure handler

        When `total_timeout` is configured, timing starts at the first attempt.
        An in-flight attempt is never interrupted; once it fails, no new
        attempt is started if the deadline elapsed (including when the deadline
        is reached during the backoff sleep, which is cut short), and the last
        failure is raised wrapped in `RetryTimeoutError`. Retry count and total
        timeout whichever is reached first terminates the session.
        """
        self._backoff.reset()
        failures = 0
        start_time = monotonic()
        deadline = (
            None if self._total_timeout is None else start_time + self._total_timeout
        )
        while True:
            try:
                return do()
            except self._supported_errors as error:
                if is_retryable and not is_retryable(error):
                    raise
                failures += 1

                if with_failure_count:
                    fail(error, failures)
                else:
                    fail(error)

                elapsed = monotonic() - start_time
                count_exhausted = self._retries >= 0 and failures > self._retries
                deadline_exceeded = (
                    deadline is not None and elapsed >= self._total_timeout
                )
                # Only consult the backoff when another attempt may follow,
                # preserving its reset()/compute() call pattern exactly.
                if not count_exhausted and not deadline_exceeded:
                    backoff = self._backoff.compute(failures)
                else:
                    backoff = 0.0

                self._notify_failure(failures, elapsed, backoff, error)

                if count_exhausted:
                    raise error
                if deadline_exceeded:
                    raise self._timeout_error(failures, elapsed, error) from error

                if backoff > 0:
                    if deadline is None:
                        sleep(backoff)
                    else:
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            elapsed = monotonic() - start_time
                            raise self._timeout_error(
                                failures, elapsed, error
                            ) from error
                        if backoff < remaining:
                            # the full backoff fits within the budget
                            sleep(backoff)
                        else:
                            # the deadline falls inside the sleep: drain the
                            # remaining time (a sleep may return early on some
                            # platforms) and terminate without a new attempt
                            while remaining > 0:
                                sleep(remaining)
                                remaining = deadline - monotonic()
                            elapsed = monotonic() - start_time
                            raise self._timeout_error(
                                failures, elapsed, error
                            ) from error
