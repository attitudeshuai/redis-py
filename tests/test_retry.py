import copy
import time
from unittest.mock import patch

import pytest
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import (
    AbstractBackoff,
    ConstantBackoff,
    DecorrelatedJitterBackoff,
    EqualJitterBackoff,
    ExponentialBackoff,
    ExponentialWithJitterBackoff,
    FullJitterBackoff,
    NoBackoff,
)
from redis.cluster import TransactionStrategy
from redis.client import Redis
from redis.connection import Connection, UnixDomainSocketConnection
from redis.exceptions import (
    BusyLoadingError,
    ConnectionError,
    ReadOnlyError,
    RetryTimeoutError,
    TimeoutError,
)
from redis.retry import ON_FAILURE_RAISE, Retry

from .conftest import _get_client


class BackoffMock(AbstractBackoff):
    def __init__(self):
        self.reset_calls = 0
        self.calls = 0

    def reset(self):
        self.reset_calls += 1

    def compute(self, failures):
        self.calls += 1
        return 0


@pytest.mark.fixed_client
class TestConnectionConstructorWithRetry:
    "Test that the Connection constructors properly handles Retry objects"

    @pytest.mark.parametrize("retry_on_timeout", [False, True])
    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_timeout_boolean(self, Class, retry_on_timeout):
        c = Class(retry_on_timeout=retry_on_timeout)
        assert c.retry_on_timeout == retry_on_timeout
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == (1 if retry_on_timeout else 0)

    @pytest.mark.parametrize("retries", range(10))
    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_timeout_retry(self, Class, retries):
        retry_on_timeout = retries > 0
        c = Class(retry_on_timeout=retry_on_timeout, retry=Retry(NoBackoff(), retries))
        assert c.retry_on_timeout == retry_on_timeout
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == retries

    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_error(self, Class):
        c = Class(retry_on_error=[ReadOnlyError])
        assert c.retry_on_error == [ReadOnlyError]
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == 1

    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_error_empty_value(self, Class):
        c = Class(retry_on_error=[])
        assert c.retry_on_error == []
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == 0

    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_error_and_timeout(self, Class):
        c = Class(
            retry_on_error=[ReadOnlyError, BusyLoadingError], retry_on_timeout=True
        )
        assert c.retry_on_error == [ReadOnlyError, BusyLoadingError, TimeoutError]
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == 1

    @pytest.mark.parametrize("retries", range(10))
    @pytest.mark.parametrize("Class", [Connection, UnixDomainSocketConnection])
    def test_retry_on_error_retry(self, Class, retries):
        c = Class(retry_on_error=[ReadOnlyError], retry=Retry(NoBackoff(), retries))
        assert c.retry_on_error == [ReadOnlyError]
        assert isinstance(c.retry, Retry)
        assert c.retry._retries == retries


@pytest.mark.fixed_client
@pytest.mark.parametrize("retry_class", [Retry, AsyncRetry])
@pytest.mark.parametrize(
    "args",
    [
        (ConstantBackoff(0), 0),
        (ConstantBackoff(10), 5),
        (NoBackoff(), 0),
    ]
    + [
        backoff
        for Backoff in (
            DecorrelatedJitterBackoff,
            EqualJitterBackoff,
            ExponentialBackoff,
            ExponentialWithJitterBackoff,
            FullJitterBackoff,
        )
        for backoff in ((Backoff(), 2), (Backoff(25), 5), (Backoff(25, 5), 5))
    ],
)
def test_retry_eq_and_hashable(retry_class, args):
    assert retry_class(*args) == retry_class(*args)

    # create another retry object with different parameters
    copy = list(args)
    if isinstance(copy[0], ConstantBackoff):
        copy[1] = 9000
    else:
        copy[0] = ConstantBackoff(9000)

    assert retry_class(*args) != retry_class(*copy)
    assert retry_class(*copy) != retry_class(*args)
    assert (
        len(
            {
                retry_class(*args),
                retry_class(*args),
                retry_class(*copy),
                retry_class(*copy),
            }
        )
        == 2
    )


@pytest.mark.fixed_client
class TestRetry:
    "Test that Retry calls backoff and retries the expected number of times"

    def setup_method(self, test_method):
        self.actual_attempts = 0
        self.actual_failures = 0

    def _do(self):
        self.actual_attempts += 1
        raise ConnectionError()

    def _fail(self, error):
        self.actual_failures += 1

    def _fail_inf(self, error):
        self.actual_failures += 1
        if self.actual_failures == 5:
            raise ConnectionError()

    @pytest.mark.parametrize("retries", range(10))
    def test_retry(self, retries):
        backoff = BackoffMock()
        retry = Retry(backoff, retries)
        with pytest.raises(ConnectionError):
            retry.call_with_retry(self._do, self._fail)

        assert self.actual_attempts == 1 + retries
        assert self.actual_failures == 1 + retries
        assert backoff.reset_calls == 1
        assert backoff.calls == retries

    def test_infinite_retry(self):
        backoff = BackoffMock()
        # specify infinite retries, but give up after 5
        retry = Retry(backoff, -1)
        with pytest.raises(ConnectionError):
            retry.call_with_retry(self._do, self._fail_inf)

        assert self.actual_attempts == 5
        assert self.actual_failures == 5


@pytest.mark.fixed_client
class TestRetryConfiguration:
    "Validation of the total timeout and failure callback configuration"

    @pytest.mark.parametrize("retry_class", [Retry, AsyncRetry], ids=["sync", "async"])
    @pytest.mark.parametrize("value", [0, -1, -0.01])
    def test_non_positive_total_timeout_rejected(self, retry_class, value):
        with pytest.raises(ValueError):
            retry_class(NoBackoff(), 1, total_timeout=value)

    @pytest.mark.parametrize("retry_class", [Retry, AsyncRetry], ids=["sync", "async"])
    @pytest.mark.parametrize("value", ["1", [1], False, True])
    def test_invalid_total_timeout_type_rejected(self, retry_class, value):
        with pytest.raises(TypeError):
            retry_class(NoBackoff(), 1, total_timeout=value)

    @pytest.mark.parametrize("retry_class", [Retry, AsyncRetry], ids=["sync", "async"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_total_timeout_rejected(self, retry_class, value):
        with pytest.raises(ValueError):
            retry_class(NoBackoff(), 1, total_timeout=value)

    @pytest.mark.parametrize("retry_class", [Retry, AsyncRetry], ids=["sync", "async"])
    def test_invalid_on_failure_rejected(self, retry_class):
        with pytest.raises(TypeError):
            retry_class(NoBackoff(), 1, on_failure=42)

    @pytest.mark.parametrize("retry_class", [Retry, AsyncRetry], ids=["sync", "async"])
    def test_invalid_on_failure_error_rejected(self, retry_class):
        with pytest.raises(ValueError):
            retry_class(
                NoBackoff(),
                1,
                on_failure=lambda *args: None,
                on_failure_error="bogus",
            )

    @pytest.mark.parametrize("retry_class", [Retry, AsyncRetry], ids=["sync", "async"])
    def test_defaults_leave_configuration_disabled(self, retry_class):
        retry = retry_class(NoBackoff(), 3)
        assert retry._total_timeout is None
        assert retry._on_failure is None
        assert retry._on_failure_error == "log"

    def test_eq_and_hash_include_new_configuration(self):
        def cb(*args):
            return None

        assert Retry(NoBackoff(), 1, total_timeout=5) == Retry(
            NoBackoff(), 1, total_timeout=5
        )
        assert Retry(NoBackoff(), 1, total_timeout=5) != Retry(
            NoBackoff(), 1, total_timeout=6
        )
        assert Retry(NoBackoff(), 1, on_failure=cb) != Retry(NoBackoff(), 1)
        assert Retry(
            NoBackoff(), 1, on_failure=cb, on_failure_error=ON_FAILURE_RAISE
        ) != Retry(NoBackoff(), 1, on_failure=cb)
        assert hash(Retry(NoBackoff(), 1, total_timeout=5)) == hash(
            Retry(NoBackoff(), 1, total_timeout=5)
        )
        assert (
            len(
                {
                    Retry(NoBackoff(), 1, total_timeout=5),
                    Retry(NoBackoff(), 1, total_timeout=5),
                    Retry(NoBackoff(), 1, total_timeout=6),
                }
            )
            == 2
        )


@pytest.mark.fixed_client
class TestRetryTotalTimeout:
    "Total timeout semantics for the synchronous Retry"

    def setup_method(self, test_method):
        self.attempts = 0

    def _do_fail_fast(self):
        self.attempts += 1
        raise ConnectionError()

    def _do_fail_slow(self, delay):
        def do():
            self.attempts += 1
            time.sleep(delay)
            raise ConnectionError()

        return do

    def _fail(self, error):
        pass

    def test_deadline_in_sleep_wakes_up_early(self):
        # the backoff alone would sleep for 10 seconds per failure
        retry = Retry(ConstantBackoff(10), 100, total_timeout=0.1)
        start = time.monotonic()
        with pytest.raises(RetryTimeoutError) as exc_info:
            retry.call_with_retry(self._do_fail_fast, self._fail)
        elapsed = time.monotonic() - start

        # the sleep must be cut short by the deadline, not run for 10 seconds
        assert elapsed < 1.0
        error = exc_info.value
        assert error.attempts == 1
        assert error.elapsed >= 0.1 - 0.05
        assert isinstance(error.last_error, ConnectionError)
        assert error.__cause__ is error.last_error
        assert self.attempts == 1

    def test_deadline_stops_further_attempts(self):
        # every attempt costs ~20ms; the 0.15s budget allows a handful only
        retry = Retry(NoBackoff(), 10_000, total_timeout=0.15)
        start = time.monotonic()
        with pytest.raises(RetryTimeoutError) as exc_info:
            retry.call_with_retry(self._do_fail_slow(0.02), self._fail)
        elapsed = time.monotonic() - start

        error = exc_info.value
        assert 2 <= error.attempts <= 9
        assert error.elapsed >= 0.15 - 0.05
        assert elapsed < 1.0
        assert self.attempts == error.attempts

    def test_first_attempt_always_runs_even_when_over_budget(self):
        # the attempt itself is never interrupted; total cost is the budget
        # plus the cost of the final attempt
        retry = Retry(NoBackoff(), 100, total_timeout=0.01)
        start = time.monotonic()
        with pytest.raises(RetryTimeoutError) as exc_info:
            retry.call_with_retry(self._do_fail_slow(0.1), self._fail)
        elapsed = time.monotonic() - start

        # the ~0.1s attempt ran to completion instead of being cut at 0.01s
        assert elapsed >= 0.08
        assert exc_info.value.attempts == 1

    def test_count_exhaustion_is_distinguishable_from_timeout(self):
        retry = Retry(NoBackoff(), 2, total_timeout=10)
        with pytest.raises(ConnectionError):
            retry.call_with_retry(self._do_fail_fast, self._fail)
        assert self.attempts == 3

    def test_infinite_retries_are_bounded_by_the_deadline(self):
        retry = Retry(NoBackoff(), -1, total_timeout=0.05)
        with pytest.raises(RetryTimeoutError) as exc_info:
            retry.call_with_retry(self._do_fail_fast, self._fail)
        assert exc_info.value.attempts >= 1

    def test_success_within_budget_returns_result(self):
        retry = Retry(NoBackoff(), 3, total_timeout=10)

        def do():
            self.attempts += 1
            if self.attempts < 2:
                raise ConnectionError()
            return "ok"

        assert retry.call_with_retry(do, self._fail) == "ok"
        assert self.attempts == 2

    def test_backoff_is_not_consulted_when_deadline_terminates(self):
        backoff = BackoffMock()
        retry = Retry(backoff, 100, total_timeout=0.01)
        with pytest.raises(RetryTimeoutError):
            retry.call_with_retry(self._do_fail_slow(0.05), self._fail)
        assert backoff.calls == 0
        assert backoff.reset_calls == 1


@pytest.mark.fixed_client
class TestRetryOnFailureCallback:
    "Per-failure callback semantics for the synchronous Retry"

    def setup_method(self, test_method):
        self.attempts = 0

    def _do(self):
        self.attempts += 1
        raise ConnectionError(f"failure {self.attempts}")

    def _fail(self, error):
        pass

    def test_callback_reports_attempts_elapsed_and_backoff_in_order(self):
        events = []

        def on_failure(attempt, elapsed, backoff, error):
            events.append((attempt, elapsed, backoff, error))

        retry = Retry(ConstantBackoff(0.001), 3, on_failure=on_failure)
        with pytest.raises(ConnectionError):
            retry.call_with_retry(self._do, self._fail)

        assert [event[0] for event in events] == [1, 2, 3, 4]
        assert events[0][2] == pytest.approx(0.001)
        # no further attempt follows the terminal failure
        assert events[-1][2] == 0.0
        assert isinstance(events[-1][2], float)
        assert [event[3] for event in events]
        elapsed_values = [event[1] for event in events]
        assert elapsed_values == sorted(elapsed_values)
        assert all(isinstance(event[3], ConnectionError) for event in events)

    def test_callback_not_called_on_success(self):
        calls = []
        retry = Retry(NoBackoff(), 2, on_failure=lambda *args: calls.append(args))

        def do():
            return "ok"

        assert retry.call_with_retry(do, self._fail) == "ok"
        assert calls == []

    def test_callback_not_called_for_non_retryable_error(self):
        calls = []
        retry = Retry(
            NoBackoff(),
            2,
            on_failure=lambda *args: calls.append(args),
        )
        with pytest.raises(ConnectionError):
            retry.call_with_retry(
                self._do,
                self._fail,
                is_retryable=lambda error: False,
            )
        assert calls == []
        assert self.attempts == 1

    def test_callback_error_is_logged_and_retry_continues(self, caplog):
        def on_failure(attempt, elapsed, backoff, error):
            raise RuntimeError("boom in callback")

        retry = Retry(NoBackoff(), 2, on_failure=on_failure)
        with caplog.at_level("WARNING", logger="redis.retry"):
            with pytest.raises(ConnectionError):
                retry.call_with_retry(self._do, self._fail)

        assert self.attempts == 3
        assert any(
            "on_failure callback raised" in record.message
            and "RuntimeError" in record.exc_info[0].__name__
            for record in caplog.records
        )

    def test_callback_error_can_abort_the_session(self):
        def on_failure(attempt, elapsed, backoff, error):
            raise RuntimeError("boom in callback")

        retry = Retry(
            NoBackoff(),
            2,
            on_failure=on_failure,
            on_failure_error=ON_FAILURE_RAISE,
        )
        with pytest.raises(RuntimeError, match="boom in callback"):
            retry.call_with_retry(self._do, self._fail)
        assert self.attempts == 1

    def test_nested_session_inside_callback_has_no_state_crosstalk(self):
        outer_attempt_numbers = []
        nested_outcomes = []

        def outer_do():
            raise ConnectionError("outer")

        def on_failure(attempt, elapsed, backoff, error):
            outer_attempt_numbers.append(attempt)
            # simulate the callback issuing another command through the very
            # same client, i.e. re-entering the same Retry policy object
            nested_outcomes.append(
                retry.call_with_retry(lambda: "nested-ok", lambda nested: None)
            )

        retry = Retry(NoBackoff(), 3, total_timeout=10, on_failure=on_failure)
        with pytest.raises(ConnectionError):
            retry.call_with_retry(outer_do, self._fail)

        # the outer session bookkeeping is local and survives nested sessions
        assert outer_attempt_numbers == [1, 2, 3, 4]
        assert nested_outcomes == ["nested-ok"] * 4


@pytest.mark.fixed_client
class TestClusterRetryEntry:
    "The cluster pipeline builds its retry from the node retry via copy.copy"

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

    def test_cluster_transaction_entry_honors_total_timeout(self):
        # exercise TransactionStrategy._execute_transaction_with_retries
        # directly, the cluster transaction retry entry (no server required)
        strategy = TransactionStrategy.__new__(TransactionStrategy)
        strategy._retry = Retry(ConstantBackoff(10), 100, total_timeout=0.1)
        strategy._reinitialize_on_error = lambda error, failure_count: None
        attempts = []

        def fake_execute_transaction(stack, raise_on_error):
            attempts.append(1)
            raise ConnectionError()

        strategy._execute_transaction = fake_execute_transaction
        start = time.monotonic()
        with pytest.raises(RetryTimeoutError) as exc_info:
            strategy._execute_transaction_with_retries([], True)

        assert time.monotonic() - start < 1.0
        assert sum(attempts) == 1
        assert exc_info.value.attempts == 1


@pytest.mark.onlynoncluster
class TestRedisClientRetry:
    "Test the standalone Redis client behavior with retries"

    def test_client_retry_on_error_with_success(self, request):
        with patch.object(Redis, "parse_response") as parse_response:

            def mock_parse_response(connection, *args, **options):
                def ok_response(connection, *args, **options):
                    return "MOCK_OK"

                parse_response.side_effect = ok_response
                raise ReadOnlyError()

            parse_response.side_effect = mock_parse_response
            r = _get_client(Redis, request, retry_on_error=[ReadOnlyError])
            assert r.get("foo") == "MOCK_OK"
            assert parse_response.call_count == 2

    def test_client_retry_on_error_raise(self, request):
        with patch.object(Redis, "parse_response") as parse_response:
            parse_response.side_effect = BusyLoadingError()
            retries = 3
            r = _get_client(
                Redis,
                request,
                retry_on_error=[ReadOnlyError, BusyLoadingError],
                retry=Retry(NoBackoff(), retries),
            )
            with pytest.raises(BusyLoadingError):
                try:
                    r.get("foo")
                finally:
                    assert parse_response.call_count == retries + 1

    def test_client_retry_on_error_different_error_raised(self, request):
        with patch.object(Redis, "parse_response") as parse_response:
            parse_response.side_effect = OSError()
            retries = 3
            r = _get_client(
                Redis,
                request,
                retry_on_error=[ReadOnlyError],
                retry=Retry(NoBackoff(), retries),
            )
            with pytest.raises(OSError):
                try:
                    r.get("foo")
                finally:
                    assert parse_response.call_count == 1

    def test_client_retry_on_error_and_timeout(self, request):
        with patch.object(Redis, "parse_response") as parse_response:
            parse_response.side_effect = TimeoutError()
            retries = 3
            r = _get_client(
                Redis,
                request,
                retry_on_error=[ReadOnlyError],
                retry_on_timeout=True,
                retry=Retry(NoBackoff(), retries),
            )
            with pytest.raises(TimeoutError):
                try:
                    r.get("foo")
                finally:
                    assert parse_response.call_count == retries + 1

    def test_client_retry_on_timeout(self, request):
        with patch.object(Redis, "parse_response") as parse_response:
            parse_response.side_effect = TimeoutError()
            retries = 3
            r = _get_client(
                Redis, request, retry_on_timeout=True, retry=Retry(NoBackoff(), retries)
            )
            with pytest.raises(TimeoutError):
                try:
                    r.get("foo")
                finally:
                    assert parse_response.call_count == retries + 1

    def test_client_retry_total_timeout(self, request):
        with patch.object(Redis, "parse_response") as parse_response:
            parse_response.side_effect = TimeoutError()
            # without the total timeout, a 5s backoff and 100 retries would
            # keep the call running for minutes
            r = _get_client(
                Redis,
                request,
                retry=Retry(ConstantBackoff(5), 100, total_timeout=0.2),
            )
            start = time.monotonic()
            try:
                with pytest.raises(RetryTimeoutError) as exc_info:
                    r.get("foo")
            finally:
                elapsed = time.monotonic() - start
            assert elapsed < 2
            assert exc_info.value.attempts == parse_response.call_count
            assert isinstance(exc_info.value.last_error, TimeoutError)

    def test_client_retry_on_failure_callback(self, request):
        events = []
        with patch.object(Redis, "parse_response") as parse_response:
            parse_response.side_effect = TimeoutError()
            r = _get_client(
                Redis,
                request,
                retry=Retry(
                    NoBackoff(),
                    2,
                    on_failure=lambda attempt, elapsed, backoff, error: events.append(
                        (attempt, type(error))
                    ),
                ),
            )
            with pytest.raises(TimeoutError):
                r.get("foo")
        assert [event[0] for event in events] == [1, 2, 3]
        assert all(event[1] is TimeoutError for event in events)

    def test_pipeline_retry_total_timeout(self, request):
        with patch.object(Redis, "parse_response") as parse_response:
            parse_response.side_effect = TimeoutError()
            r = _get_client(
                Redis,
                request,
                retry=Retry(ConstantBackoff(5), 100, total_timeout=0.2),
            )
            start = time.monotonic()
            with pytest.raises(RetryTimeoutError):
                r.pipeline(transaction=False).get("foo").execute()
            assert time.monotonic() - start < 2

    @pytest.mark.onlycluster
    def test_get_set_retry_object_for_cluster_client(self, request):
        retry = Retry(NoBackoff(), 2)
        r = _get_client(Redis, request, retry_on_timeout=True, retry=retry)
        exist_conn = r.connection_pool.get_connection()
        assert r.retry._retries == retry._retries
        assert isinstance(r.retry._backoff, NoBackoff)
        new_retry_policy = Retry(ExponentialBackoff(), 3)
        r.set_retry(new_retry_policy)
        assert r.retry._retries == new_retry_policy._retries
        assert isinstance(r.retry._backoff, ExponentialBackoff)
        assert exist_conn.retry._retries == new_retry_policy._retries
        new_conn = r.connection_pool.get_connection()
        assert new_conn.retry._retries == new_retry_policy._retries

    @pytest.mark.onlynoncluster
    def test_get_set_retry_object(self, request):
        retry = Retry(NoBackoff(), 2)
        r = _get_client(Redis, request, retry_on_timeout=True, retry=retry)
        exist_conn = r.connection_pool.get_connection()
        assert r.get_retry()._retries == retry._retries
        assert isinstance(r.get_retry()._backoff, NoBackoff)
        new_retry_policy = Retry(ExponentialBackoff(), 3)
        r.set_retry(new_retry_policy)
        assert r.get_retry()._retries == new_retry_policy._retries
        assert isinstance(r.get_retry()._backoff, ExponentialBackoff)
        assert exist_conn.retry._retries == new_retry_policy._retries
        new_conn = r.connection_pool.get_connection()
        assert new_conn.retry._retries == new_retry_policy._retries
