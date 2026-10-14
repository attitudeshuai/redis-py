import logging
import warnings
from asyncio import sleep
from inspect import isawaitable, iscoroutinefunction
from time import monotonic
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from redis.exceptions import ConnectionError, TimeoutError
from redis.retry import (
    ON_FAILURE_LOG,
    AbstractRetry,
    OnFailureCallback,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

if TYPE_CHECKING:
    from redis.backoff import AbstractBackoff


class Retry(AbstractRetry[Exception]):
    __hash__ = AbstractRetry.__hash__

    def __init__(
        self,
        backoff: "AbstractBackoff",
        retries: int,
        supported_errors: Tuple[Type[Exception], ...] = (
            ConnectionError,
            TimeoutError,
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

    async def _notify_failure(
        self, failures: int, elapsed: float, backoff: float, error: Exception
    ) -> None:
        """Invoke the optional per-failure observer callback.

        Both regular callables and coroutine functions are supported; waiting
        never blocks the event loop.
        """
        if self._on_failure is None:
            return
        try:
            result = self._on_failure(failures, elapsed, backoff, error)
            if isawaitable(result):
                await result
        except Exception:
            if self._on_failure_error == ON_FAILURE_LOG:
                logger.warning(
                    "redis retry on_failure callback raised an exception after "
                    "attempt %d; continuing the retry process",
                    failures,
                    exc_info=True,
                )
            else:
                raise

    async def call_with_retry(
        self,
        do: Callable[[], Awaitable[T]],
        fail: Union[
            Callable[[Exception], Any],
            Callable[[Exception, int], Any],
        ],
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
        timeout whichever is reached first terminates the session. Timing and
        sleeping do not block the event loop.
        """
        self._backoff.reset()
        failures = 0
        start_time = monotonic()
        deadline = (
            None if self._total_timeout is None else start_time + self._total_timeout
        )
        while True:
            try:
                return await do()
            except self._supported_errors as error:
                if is_retryable and not is_retryable(error):
                    raise
                failures += 1

                if with_failure_count:
                    await fail(error, failures)
                else:
                    await fail(error)

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

                await self._notify_failure(failures, elapsed, backoff, error)

                if count_exhausted:
                    raise error
                if deadline_exceeded:
                    raise self._timeout_error(failures, elapsed, error) from error

                if backoff > 0:
                    if deadline is None:
                        await sleep(backoff)
                    else:
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            elapsed = monotonic() - start_time
                            raise self._timeout_error(
                                failures, elapsed, error
                            ) from error
                        if backoff < remaining:
                            # the full backoff fits within the budget
                            await sleep(backoff)
                        else:
                            # the deadline falls inside the sleep: drain the
                            # remaining time (a sleep may return early on some
                            # platforms) and terminate without a new attempt
                            while remaining > 0:
                                await sleep(remaining)
                                remaining = deadline - monotonic()
                            elapsed = monotonic() - start_time
                            raise self._timeout_error(
                                failures, elapsed, error
                            ) from error


def _to_async_retry(retry: Any) -> Any:
    """Convert a synchronous retry policy while preserving async-shaped ones."""
    if iscoroutinefunction(getattr(type(retry), "call_with_retry", None)):
        return retry
    if not isinstance(retry, AbstractRetry):
        return retry

    warnings.warn(
        "A synchronous redis.retry.Retry was passed to an asyncio client and has "
        "been converted to redis.asyncio.retry.Retry. A custom call_with_retry "
        "implementation is not preserved - please use redis.asyncio.retry.Retry.",
        UserWarning,
        stacklevel=2,
    )

    return Retry(
        backoff=retry._backoff,
        retries=retry._retries,
        supported_errors=retry._supported_errors,
        total_timeout=retry._total_timeout,
        on_failure=retry._on_failure,
        on_failure_error=retry._on_failure_error,
    )
