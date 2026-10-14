import asyncio
import copy
import time
import warnings
from threading import Lock
from unittest.mock import AsyncMock, patch

import pytest
from redis.asyncio import Redis, RedisCluster
from redis.asyncio.cluster import TransactionStrategy
from redis.asyncio.connection import (
    Connection,
    ConnectionPool,
    UnixDomainSocketConnection,
)
from redis.asyncio.retry import Retry, _to_async_retry
from redis.backoff import (
    AbstractBackoff,
    ConstantBackoff,
    ExponentialBackoff,
    NoBackoff,
)
from redis.exceptions import ConnectionError, RetryTimeoutError, TimeoutError
from redis.retry import ON_FAILURE_RAISE, AbstractRetry
from redis.retry import Retry as SyncRetry


class BackoffMock(AbstractBackoff):
    def __init__(self):
        self.reset_calls = 0
        self.calls = 0

    def reset(self):
        self.reset_calls += 1

    def compute(self, failures):
        self.calls += 1
        return 0


class UncopyableBackoff(AbstractBackoff):
    def __init__(self):
        self._lock = Lock()

    def compute(self, failures):
        with self._lock:
            return 0


class CustomAsyncRetry(AbstractRetry):
    def __init__(self):
        super().__init__(NoBackoff(), 1, (ConnectionError,))

    def __eq__(self, other):
        return self is other

    async def call_with_retry(self, do, fail, **kwargs):
        return await do()


class DuckTypedAsyncRetry:
    async def call_with_retry(self, do, fail, **kwargs):
        return await do()

    def get_retries(self):
        return 1

    def update_supported_errors(self, specified_errors):
        pass


class DuckTypedAwaitableRetry:
    def call_with_retry(self, do, fail, **kwargs):
        return do()

    def get_retries(self):
        return 1

    def update_supported_errors(self, specified_errors):
        pass


@pytest.mark.fixed_client
class TestConnectionConstructorWithRetry:
    "Test that the Connection constructors properly handles Retry objects"

    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_error_set(self, Class):
        class CustomError(Exception):
            pass

        retry_on_error = [ConnectionError, TimeoutError, CustomError]
        c = Class(retry_on_error=retry_on_error)
        assert c.retry_on_error == retry_on_error
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == 1
        assert set(c.retry._supported_errors) == set(retry_on_error)

    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_error_not_set(self, Class):
        c = Class()
        assert c.retry_on_error == []
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == 0

    @pytest.mark.parametrize("retry_on_timeout", [False, True])
    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_timeout(self, Class, retry_on_timeout):
        c = Class(retry_on_timeout=retry_on_timeout)
        assert c.retry_on_timeout == retry_on_timeout
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == (1 if retry_on_timeout else 0)

    @pytest.mark.parametrize("retries", range(10))
    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_with_retry_on_timeout(self, Class, retries: int):
        retry_on_timeout = retries > 0
        c = Class(retry_on_timeout=retry_on_timeout, retry=Retry(NoBackoff(), retries))
        assert c.retry_on_timeout == retry_on_timeout
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == retries

    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_error_does_not_mutate_caller_list(self, Class):
        retry_on_error = [ValueError]
        snapshot = list(retry_on_error)

        # Two constructions must not grow or otherwise mutate the caller's list.
        c1 = Class(retry_on_error=retry_on_error, retry_on_timeout=True)
        c2 = Class(retry_on_error=retry_on_error, retry_on_timeout=True)

        assert retry_on_error == snapshot
        assert retry_on_error is not c1.retry_on_error
        assert ValueError in c1.retry_on_error
        assert TimeoutError in c1.retry_on_error
        assert c1.retry_on_error == c2.retry_on_error

    @pytest.mark.parametrize("retries", range(10))
    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_with_retry_on_error(self, Class, retries: int):
        class CustomError(Exception):
            pass

        retry_on_error = [ConnectionError, TimeoutError, CustomError]
        c = Class(retry_on_error=retry_on_error, retry=Retry(NoBackoff(), retries))
        assert c.retry_on_error == retry_on_error
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == retries
        assert set(c.retry._supported_errors) == set(retry_on_error)

    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    @pytest.mark.asyncio
    async def test_sync_retry_is_used_by_connect(self, Class, monkeypatch):
        retry = SyncRetry(NoBackoff(), 2)
        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            connection = Class(retry=retry)
        attempts = 0
        failures = 0

        async def connect_check_health(
            _self, check_health=True, retry_socket_connect=True
        ):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError

        async def disconnect(_self, *args, **kwargs):
            nonlocal failures
            failures += 1

        monkeypatch.setattr(Class, "connect_check_health", connect_check_health)
        monkeypatch.setattr(Class, "disconnect", disconnect)

        assert isinstance(connection.retry, Retry)
        await connection.connect()

        assert attempts == 3
        assert failures == 2

    @pytest.mark.asyncio
    async def test_pool_from_url_converts_sync_retry(self, monkeypatch):
        retry = SyncRetry(NoBackoff(), 2)
        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            pool = ConnectionPool.from_url(
                "redis://localhost:6379",
                retry=retry,
                retry_on_error=[ConnectionError],
            )

        async def ensure_connection(_self, _connection):
            pass

        monkeypatch.setattr(ConnectionPool, "ensure_connection", ensure_connection)
        connection = await pool.get_connection()

        assert isinstance(connection.retry, Retry)
        assert connection.retry.get_retries() == 2
        assert ConnectionError in connection.retry._supported_errors

        await pool.release(connection)
        await pool.aclose()

    @pytest.mark.parametrize(
        "retry",
        [CustomAsyncRetry(), DuckTypedAsyncRetry(), DuckTypedAwaitableRetry()],
    )
    def test_async_shaped_retry_is_preserved(self, retry):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pool = ConnectionPool(retry=retry)

        assert pool.connection_kwargs["retry"] is retry

    @pytest.mark.asyncio
    async def test_pool_set_retry_applies_to_new_connection(self, monkeypatch):
        pool = ConnectionPool()
        retry = Retry(NoBackoff(), 2)
        pool.set_retry(retry)

        async def ensure_connection(_self, _connection):
            pass

        monkeypatch.setattr(ConnectionPool, "ensure_connection", ensure_connection)
        connection = await pool.get_connection()

        assert connection.retry.get_retries() == retry.get_retries()
        assert isinstance(connection.retry._backoff, NoBackoff)

        await pool.release(connection)
        await pool.aclose()

    def test_pool_converts_sync_retry(self):
        retry = SyncRetry(NoBackoff(), 2)
        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            client = Redis(retry=retry)

        assert isinstance(client.get_retry(), Retry)
        assert client.get_retry().get_retries() == retry.get_retries()

        new_retry = SyncRetry(ExponentialBackoff(), 3)
        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            client.set_retry(new_retry)

        assert isinstance(client.get_retry(), Retry)
        assert client.get_retry().get_retries() == new_retry.get_retries()

    def test_redis_set_retry_converts_without_pool_write(self, monkeypatch):
        client = Redis()
        monkeypatch.setattr(client.connection_pool, "set_retry", lambda retry: None)
        sync_retry = SyncRetry(NoBackoff(), 2)
        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            client.set_retry(sync_retry)

        assert isinstance(client.get_retry(), Retry)
        assert client.get_retry().get_retries() == 2
        assert type(sync_retry) is SyncRetry

    def test_cluster_converts_sync_retry(self):
        retry = SyncRetry(NoBackoff(), 2)
        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            client = RedisCluster(host="127.0.0.1", port=6379, retry=retry)

        assert isinstance(client.retry, Retry)
        assert client.retry.get_retries() == retry.get_retries()

        new_retry = SyncRetry(ExponentialBackoff(), 3)
        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            client.set_retry(new_retry)

        assert isinstance(client.retry, Retry)
        assert client.retry.get_retries() == new_retry.get_retries()

    def test_cluster_preserves_uncopyable_sync_backoff(self):
        backoff = UncopyableBackoff()
        retry = SyncRetry(backoff, 2)

        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            client = RedisCluster(host="127.0.0.1", port=6379, retry=retry)

        assert isinstance(client.retry, Retry)
        assert client.retry._backoff is backoff


@pytest.mark.fixed_client
class TestRetry:
    "Test that Retry calls backoff and retries the expected number of times"

    def setup_method(self, test_method):
        self.actual_attempts = 0
        self.actual_failures = 0

    async def _do(self):
        self.actual_attempts += 1
        raise ConnectionError()

    async def _fail(self, error):
        self.actual_failures += 1

    async def _fail_inf(self, error):
        self.actual_failures += 1
        if self.actual_failures == 5:
            raise ConnectionError()

    @pytest.mark.parametrize("retries", range(10))
    @pytest.mark.asyncio
    async def test_retry(self, retries: int):
        backoff = BackoffMock()
        retry = Retry(backoff, retries)
        with pytest.raises(ConnectionError):
            await retry.call_with_retry(self._do, self._fail)

        assert self.actual_attempts == 1 + retries
        assert self.actual_failures == 1 + retries
        assert backoff.reset_calls == 1
        assert backoff.calls == retries

    @pytest.mark.asyncio
    async def test_infinite_retry(self):
        backoff = BackoffMock()
        # specify infinite retries, but give up after 5
        retry = Retry(backoff, -1)
        with pytest.raises(ConnectionError):
            await retry.call_with_retry(self._do, self._fail_inf)

        assert self.actual_attempts == 5
        assert self.actual_failures == 5


@pytest.mark.fixed_client
class TestAsyncRetryTotalTimeout:
    "Total timeout semantics for the asynchronous Retry"

    def setup_method(self, test_method):
        self.attempts = 0

    async def _do_fail_fast(self):
        self.attempts += 1
        raise ConnectionError()

    def _do_fail_slow(self, delay):
        async def do():
            self.attempts += 1
            await asyncio.sleep(delay)
            raise ConnectionError()

        return do

    async def _fail(self, error):
        pass

    @pytest.mark.asyncio
    async def test_deadline_in_sleep_wakes_up_early(self):
        # the backoff alone would sleep for 10 seconds per failure
        retry = Retry(ConstantBackoff(10), 100, total_timeout=0.1)
        start = time.monotonic()
        with pytest.raises(RetryTimeoutError) as exc_info:
            await retry.call_with_retry(self._do_fail_fast, self._fail)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0
        error = exc_info.value
        assert error.attempts == 1
        assert error.elapsed >= 0.1 - 0.05
        assert isinstance(error.last_error, ConnectionError)
        assert error.__cause__ is error.last_error
        assert self.attempts == 1

    @pytest.mark.asyncio
    async def test_deadline_stops_further_attempts(self):
        retry = Retry(NoBackoff(), 10_000, total_timeout=0.15)
        start = time.monotonic()
        with pytest.raises(RetryTimeoutError) as exc_info:
            await retry.call_with_retry(self._do_fail_slow(0.02), self._fail)
        elapsed = time.monotonic() - start

        error = exc_info.value
        assert 2 <= error.attempts <= 9
        assert error.elapsed >= 0.15 - 0.05
        assert elapsed < 1.0
        assert self.attempts == error.attempts

    @pytest.mark.asyncio
    async def test_count_exhaustion_is_distinguishable_from_timeout(self):
        retry = Retry(NoBackoff(), 2, total_timeout=10)
        with pytest.raises(ConnectionError):
            await retry.call_with_retry(self._do_fail_fast, self._fail)
        assert self.attempts == 3

    @pytest.mark.asyncio
    async def test_infinite_retries_are_bounded_by_the_deadline(self):
        retry = Retry(NoBackoff(), -1, total_timeout=0.05)
        with pytest.raises(RetryTimeoutError) as exc_info:
            await retry.call_with_retry(self._do_fail_fast, self._fail)
        assert exc_info.value.attempts >= 1

    @pytest.mark.asyncio
    async def test_timing_and_sleep_do_not_block_the_event_loop(self):
        ticks = []

        async def ticker():
            while True:
                await asyncio.sleep(0.02)
                ticks.append(time.monotonic())

        ticker_task = asyncio.create_task(ticker())
        try:
            retry = Retry(ConstantBackoff(10), 100, total_timeout=0.1)
            with pytest.raises(RetryTimeoutError):
                await retry.call_with_retry(self._do_fail_fast, self._fail)
        finally:
            ticker_task.cancel()

        # the 0.1s deadline sleep yielded control to the event loop repeatedly
        assert len(ticks) >= 3

    @pytest.mark.asyncio
    async def test_backoff_is_not_consulted_when_deadline_terminates(self):
        backoff = BackoffMock()
        retry = Retry(backoff, 100, total_timeout=0.01)
        with pytest.raises(RetryTimeoutError):
            await retry.call_with_retry(self._do_fail_slow(0.05), self._fail)
        assert backoff.calls == 0
        assert backoff.reset_calls == 1


@pytest.mark.fixed_client
class TestAsyncRetryOnFailureCallback:
    "Per-failure callback semantics for the asynchronous Retry"

    def setup_method(self, test_method):
        self.attempts = 0

    async def _do(self):
        self.attempts += 1
        raise ConnectionError(f"failure {self.attempts}")

    async def _fail(self, error):
        pass

    @pytest.mark.asyncio
    async def test_sync_callback_is_supported(self):
        events = []

        retry = Retry(
            ConstantBackoff(0.001),
            2,
            on_failure=lambda attempt, elapsed, backoff, error: events.append(
                (attempt, backoff)
            ),
        )
        with pytest.raises(ConnectionError):
            await retry.call_with_retry(self._do, self._fail)

        assert [event[0] for event in events] == [1, 2, 3]
        assert events[0][1] == pytest.approx(0.001)
        assert events[-1][1] == 0.0
        assert isinstance(events[-1][1], float)

    @pytest.mark.asyncio
    async def test_async_callback_is_supported(self):
        events = []

        async def on_failure(attempt, elapsed, backoff, error):
            await asyncio.sleep(0)
            events.append((attempt, type(error)))

        retry = Retry(NoBackoff(), 2, on_failure=on_failure)
        with pytest.raises(ConnectionError):
            await retry.call_with_retry(self._do, self._fail)

        assert [event[0] for event in events] == [1, 2, 3]
        assert all(event[1] is ConnectionError for event in events)

    @pytest.mark.asyncio
    async def test_callback_error_is_logged_and_retry_continues(self, caplog):
        def on_failure(attempt, elapsed, backoff, error):
            raise RuntimeError("boom in callback")

        retry = Retry(NoBackoff(), 2, on_failure=on_failure)
        with caplog.at_level("WARNING", logger="redis.asyncio.retry"):
            with pytest.raises(ConnectionError):
                await retry.call_with_retry(self._do, self._fail)

        assert self.attempts == 3
        assert any(
            "on_failure callback raised" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_callback_error_can_abort_the_session(self):
        async def on_failure(attempt, elapsed, backoff, error):
            raise RuntimeError("boom in callback")

        retry = Retry(
            NoBackoff(),
            2,
            on_failure=on_failure,
            on_failure_error=ON_FAILURE_RAISE,
        )
        with pytest.raises(RuntimeError, match="boom in callback"):
            await retry.call_with_retry(self._do, self._fail)
        assert self.attempts == 1

    @pytest.mark.asyncio
    async def test_nested_session_inside_callback_has_no_state_crosstalk(self):
        outer_attempt_numbers = []
        nested_outcomes = []

        async def outer_do():
            raise ConnectionError("outer")

        async def on_failure(attempt, elapsed, backoff, error):
            outer_attempt_numbers.append(attempt)
            # simulate the callback issuing another command through the very
            # same client, i.e. re-entering the same Retry policy object
            nested_outcomes.append(
                await retry.call_with_retry(
                    lambda: _return_nested_ok(), lambda nested: None
                )
            )

        async def _return_nested_ok():
            return "nested-ok"

        retry = Retry(NoBackoff(), 3, total_timeout=10, on_failure=on_failure)
        with pytest.raises(ConnectionError):
            await retry.call_with_retry(outer_do, self._fail)

        assert outer_attempt_numbers == [1, 2, 3, 4]
        assert nested_outcomes == ["nested-ok"] * 4


@pytest.mark.fixed_client
class TestAsyncRetryConfigurationPropagation:
    "New options survive sync -> async conversion and cluster construction"

    def test_sync_to_async_conversion_preserves_new_options(self):
        def callback(*args):
            return None

        sync_retry = SyncRetry(
            NoBackoff(),
            2,
            total_timeout=2.5,
            on_failure=callback,
            on_failure_error=ON_FAILURE_RAISE,
        )
        with pytest.warns(UserWarning, match="synchronous redis.retry.Retry"):
            converted = _to_async_retry(sync_retry)

        assert isinstance(converted, Retry)
        assert converted._total_timeout == 2.5
        assert converted._on_failure is callback
        assert converted._on_failure_error == ON_FAILURE_RAISE

    def test_cluster_keeps_total_timeout_configuration(self):
        retry = Retry(NoBackoff(), 2, total_timeout=3.0)
        client = RedisCluster(host="127.0.0.1", port=6379, retry=retry)
        assert client.retry._total_timeout == 3.0

    @pytest.mark.asyncio
    async def test_cluster_transaction_entry_honors_total_timeout(self):
        # exercise the async cluster transaction retry entry (no server needed)
        strategy = TransactionStrategy.__new__(TransactionStrategy)
        strategy._retry = Retry(ConstantBackoff(10), 100, total_timeout=0.1)

        async def fake_reinitialize(error, failure_count):
            return None

        strategy._reinitialize_on_error = fake_reinitialize
        attempts = []

        async def fake_execute_transaction(stack, raise_on_error):
            attempts.append(1)
            raise ConnectionError()

        strategy._execute_transaction = fake_execute_transaction
        start = time.monotonic()
        with pytest.raises(RetryTimeoutError) as exc_info:
            await strategy._execute_transaction_with_retries([], True)

        assert time.monotonic() - start < 1.0
        assert sum(attempts) == 1
        assert exc_info.value.attempts == 1

    def test_cluster_pipeline_retry_copy_preserves_new_options(self):
        def callback(*args):
            return None

        retry = Retry(
            NoBackoff(),
            2,
            total_timeout=2.5,
            on_failure=callback,
            on_failure_error=ON_FAILURE_RAISE,
        )
        copied = copy.copy(retry)
        assert copied._total_timeout == 2.5
        assert copied._on_failure is callback
        assert copied._on_failure_error == ON_FAILURE_RAISE


class TestRedisClientRetry:
    "Test the Redis client behavior with retries"

    async def test_get_set_retry_object(self, request):
        retry = Retry(NoBackoff(), 2)
        url = request.config.getoption("--redis-url")
        r = await Redis.from_url(url, retry_on_timeout=True, retry=retry)
        assert r.get_retry()._retries == retry._retries
        assert isinstance(r.get_retry()._backoff, NoBackoff)
        new_retry_policy = Retry(ExponentialBackoff(), 3)
        exiting_conn = await r.connection_pool.get_connection()
        r.set_retry(new_retry_policy)
        assert r.get_retry()._retries == new_retry_policy._retries
        assert isinstance(r.get_retry()._backoff, ExponentialBackoff)
        assert exiting_conn.retry._retries == new_retry_policy._retries
        await r.connection_pool.release(exiting_conn)
        new_conn = await r.connection_pool.get_connection()
        assert new_conn.retry._retries == new_retry_policy._retries
        await r.connection_pool.release(new_conn)
        await r.aclose()

    @pytest.mark.asyncio
    async def test_client_retry_total_timeout(self, request):
        url = request.config.getoption("--redis-url")
        r = await Redis.from_url(
            url, retry=Retry(ConstantBackoff(5), 100, total_timeout=0.2)
        )
        start = time.monotonic()
        try:
            with patch.object(
                Redis, "parse_response", new=AsyncMock(side_effect=TimeoutError())
            ):
                with pytest.raises(RetryTimeoutError):
                    await r.get("foo")
        finally:
            elapsed = time.monotonic() - start
            await r.aclose()
        assert elapsed < 2

    @pytest.mark.asyncio
    async def test_client_retry_on_failure_callback(self, request):
        url = request.config.getoption("--redis-url")
        events = []
        r = await Redis.from_url(
            url,
            retry=Retry(
                NoBackoff(),
                2,
                on_failure=lambda attempt, elapsed, backoff, error: events.append(
                    (attempt, type(error))
                ),
            ),
        )
        with patch.object(
            Redis, "parse_response", new=AsyncMock(side_effect=TimeoutError())
        ):
            with pytest.raises(TimeoutError):
                await r.get("foo")
        await r.aclose()
        assert [event[0] for event in events] == [1, 2, 3]
        assert all(event[1] is TimeoutError for event in events)

    @pytest.mark.asyncio
    async def test_pipeline_retry_total_timeout(self, request):
        url = request.config.getoption("--redis-url")
        r = await Redis.from_url(
            url, retry=Retry(ConstantBackoff(5), 100, total_timeout=0.2)
        )
        start = time.monotonic()
        try:
            with patch.object(
                Redis, "parse_response", new=AsyncMock(side_effect=TimeoutError())
            ):
                with pytest.raises(RetryTimeoutError):
                    pipe = r.pipeline(transaction=False)
                    pipe.get("foo")
                    await pipe.execute()
        finally:
            elapsed = time.monotonic() - start
            await r.aclose()
        assert elapsed < 2
