import asyncio
import logging
from unittest.mock import Mock, AsyncMock

import pytest

from redis.event import (
    AfterConnectionReleasedEvent,
    AsyncAfterConnectionReleasedEvent,
    AsyncEventListenerInterface,
    EventDispatcher,
    EventListenerInterface,
    ListenerFailure,
    ListenerFailureError,
    ListenerFailureStrategy,
)


@pytest.mark.fixed_client
class TestEventDispatcher:
    def test_register_listeners(self):
        mock_event = Mock(spec=object)
        mock_event_listener = Mock(spec=EventListenerInterface)
        listener_called = 0

        def callback(event):
            nonlocal listener_called
            listener_called += 1

        mock_event_listener.listen = callback

        # Register via constructor
        dispatcher = EventDispatcher(
            event_listeners={type(mock_event): [mock_event_listener]}
        )
        dispatcher.dispatch(mock_event)

        assert listener_called == 1

        # Register additional listener for the same event
        mock_another_event_listener = Mock(spec=EventListenerInterface)
        mock_another_event_listener.listen = callback
        dispatcher.register_listeners(
            mappings={type(mock_event): [mock_another_event_listener]}
        )
        dispatcher.dispatch(mock_event)

        assert listener_called == 3

    async def test_register_listeners_async(self):
        mock_event = Mock(spec=object)
        mock_event_listener = AsyncMock(spec=AsyncEventListenerInterface)
        listener_called = 0

        async def callback(event):
            nonlocal listener_called
            listener_called += 1

        mock_event_listener.listen = callback

        # Register via constructor
        dispatcher = EventDispatcher(
            event_listeners={type(mock_event): [mock_event_listener]}
        )
        await dispatcher.dispatch_async(mock_event)

        assert listener_called == 1

        # Register additional listener for the same event
        mock_another_event_listener = Mock(spec=AsyncEventListenerInterface)
        mock_another_event_listener.listen = callback
        dispatcher.register_listeners(
            mappings={type(mock_event): [mock_another_event_listener]}
        )
        await dispatcher.dispatch_async(mock_event)

        assert listener_called == 3

    def test_listener_can_unregister_itself_during_dispatch(self):
        """Listener calling unregister_listeners from inside listen() must
        not deadlock (dispatch must not hold the lock across invocation)."""

        class _Evt:
            pass

        dispatcher = EventDispatcher()

        class SelfUnregisteringListener(EventListenerInterface):
            def __init__(self) -> None:
                self.calls = 0

            def listen(self, event: object) -> None:
                self.calls += 1
                dispatcher.unregister_listeners({type(event): [self]})

        listener = SelfUnregisteringListener()
        dispatcher.register_listeners({_Evt: [listener]})

        dispatcher.dispatch(_Evt())
        dispatcher.dispatch(_Evt())

        assert listener.calls == 1

    def test_listener_can_register_during_dispatch(self):
        """Listener calling register_listeners from inside listen() must
        not deadlock, and the newly registered listener must not receive
        the in-flight event (snapshot semantics)."""

        class _Evt:
            pass

        dispatcher = EventDispatcher()
        later_calls = 0

        class LateListener(EventListenerInterface):
            def listen(self, event: object) -> None:
                nonlocal later_calls
                later_calls += 1

        late = LateListener()

        class RegisteringListener(EventListenerInterface):
            def listen(self, event: object) -> None:
                dispatcher.register_listeners({type(event): [late]})

        dispatcher.register_listeners({_Evt: [RegisteringListener()]})

        dispatcher.dispatch(_Evt())
        assert later_calls == 0
        dispatcher.dispatch(_Evt())
        assert later_calls == 1

    async def test_listener_can_unregister_itself_during_dispatch_async(self):
        class _Evt:
            pass

        dispatcher = EventDispatcher()

        class SelfUnregisteringListener(AsyncEventListenerInterface):
            def __init__(self) -> None:
                self.calls = 0

            async def listen(self, event: object) -> None:
                self.calls += 1
                dispatcher.unregister_listeners({type(event): [self]})

        listener = SelfUnregisteringListener()
        dispatcher.register_listeners({_Evt: [listener]})

        await dispatcher.dispatch_async(_Evt())
        await dispatcher.dispatch_async(_Evt())

        assert listener.calls == 1

    async def test_listener_can_register_during_dispatch_async(self):
        class _Evt:
            pass

        dispatcher = EventDispatcher()
        later_calls = 0

        class LateListener(AsyncEventListenerInterface):
            async def listen(self, event: object) -> None:
                nonlocal later_calls
                later_calls += 1

        late = LateListener()

        class RegisteringListener(AsyncEventListenerInterface):
            async def listen(self, event: object) -> None:
                dispatcher.register_listeners({type(event): [late]})

        dispatcher.register_listeners({_Evt: [RegisteringListener()]})

        await dispatcher.dispatch_async(_Evt())
        assert later_calls == 0
        await dispatcher.dispatch_async(_Evt())
        assert later_calls == 1


class _Evt:
    pass


class _SyncListener(EventListenerInterface):
    def __init__(self, name, call_log, failure=None):
        self.name = name
        self._call_log = call_log
        self._failure = failure

    def listen(self, event: object) -> None:
        self._call_log.append(self.name)
        if self._failure is not None:
            raise self._failure


class _AsyncListener(AsyncEventListenerInterface):
    def __init__(self, name, call_log, failure=None):
        self.name = name
        self._call_log = call_log
        self._failure = failure

    async def listen(self, event: object) -> None:
        self._call_log.append(self.name)
        if self._failure is not None:
            raise self._failure


@pytest.mark.fixed_client
class TestListenerFailureStrategy:
    def test_propagate_is_the_default_strategy(self):
        dispatcher = EventDispatcher()
        assert dispatcher.failure_strategy is ListenerFailureStrategy.PROPAGATE

    def test_string_strategy_value_is_accepted(self):
        dispatcher = EventDispatcher(failure_strategy="isolate")
        assert dispatcher.failure_strategy is ListenerFailureStrategy.ISOLATE

    def test_invalid_strategy_value_is_rejected(self):
        with pytest.raises(ValueError):
            EventDispatcher(failure_strategy="no-such-strategy")
        with pytest.raises(TypeError):
            EventDispatcher(failure_strategy=1)

    def test_propagate_stops_at_first_failure_sync(self):
        calls = []
        first = _SyncListener("first", calls, RuntimeError("boom"))
        second = _SyncListener("second", calls)

        dispatcher = EventDispatcher({_Evt: [first, second]})

        with pytest.raises(RuntimeError, match="boom"):
            dispatcher.dispatch(_Evt())
        assert calls == ["first"]

    async def test_propagate_stops_at_first_failure_async(self):
        calls = []
        first = _AsyncListener("first", calls, RuntimeError("boom"))
        second = _AsyncListener("second", calls)

        dispatcher = EventDispatcher({_Evt: [first, second]})

        with pytest.raises(RuntimeError, match="boom"):
            await dispatcher.dispatch_async(_Evt())
        assert calls == ["first"]

    def test_isolate_invokes_every_listener_sync(self, caplog):
        calls = []
        first = _SyncListener("first", calls, ValueError("bad"))
        second = _SyncListener("second", calls, RuntimeError("worse"))
        third = _SyncListener("third", calls)

        dispatcher = EventDispatcher(
            {_Evt: [first, second, third]},
            failure_strategy=ListenerFailureStrategy.ISOLATE,
        )

        with caplog.at_level(logging.WARNING, logger="redis.event"):
            result = dispatcher.dispatch(_Evt())

        assert result is None
        assert calls == ["first", "second", "third"]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert all("_SyncListener" in r.getMessage() for r in warnings)

    async def test_isolate_invokes_every_listener_async(self):
        calls = []
        first = _AsyncListener("first", calls, ValueError("bad"))
        second = _AsyncListener("second", calls, RuntimeError("worse"))
        third = _AsyncListener("third", calls)

        dispatcher = EventDispatcher(
            {_Evt: [first, second, third]},
            failure_strategy=ListenerFailureStrategy.ISOLATE,
        )

        result = await dispatcher.dispatch_async(_Evt())

        assert result is None
        assert calls == ["first", "second", "third"]

    def test_aggregate_raises_all_failures_in_registration_order_sync(self):
        calls = []
        first_failure = ValueError("bad")
        second_failure = RuntimeError("worse")
        first = _SyncListener("first", calls, first_failure)
        second = _SyncListener("second", calls, second_failure)
        third = _SyncListener("third", calls)

        dispatcher = EventDispatcher(
            {_Evt: [first, second, third]},
            failure_strategy=ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
        )

        with pytest.raises(ListenerFailureError) as exc_info:
            dispatcher.dispatch(_Evt())

        assert calls == ["first", "second", "third"]
        failures = exc_info.value.failures
        assert len(failures) == 2
        assert [f.listener for f in failures] == [first, second]
        assert all(f.event_type is _Evt for f in failures)
        assert failures[0].exception is first_failure
        assert failures[1].exception is second_failure
        assert isinstance(failures[0], ListenerFailure)
        assert "_SyncListener" in failures[0].listener_identifier
        assert "bad" in str(exc_info.value) and "worse" in str(exc_info.value)

    async def test_aggregate_raises_all_failures_in_registration_order_async(self):
        calls = []
        first = _AsyncListener("first", calls, ValueError("bad"))
        second = _AsyncListener("second", calls, RuntimeError("worse"))
        third = _AsyncListener("third", calls)

        dispatcher = EventDispatcher(
            {_Evt: [first, second, third]},
            failure_strategy=ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
        )

        with pytest.raises(ListenerFailureError) as exc_info:
            await dispatcher.dispatch_async(_Evt())

        assert calls == ["first", "second", "third"]
        assert [f.listener for f in exc_info.value.failures] == [first, second]

    def test_aggregate_invokes_failure_callback_in_order_sync(self):
        calls = []
        received = []
        first = _SyncListener("first", calls, ValueError("bad"))
        second = _SyncListener("second", calls, RuntimeError("worse"))

        def failure_callback(failure):
            received.append(failure)

        dispatcher = EventDispatcher(
            {_Evt: [first, second]},
            failure_strategy=ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
            failure_callback=failure_callback,
        )

        result = dispatcher.dispatch(_Evt())

        assert result is None
        assert [f.listener for f in received] == [first, second]

    async def test_aggregate_invokes_async_failure_callback_in_order(self):
        calls = []
        received = []
        first = _AsyncListener("first", calls, ValueError("bad"))
        second = _AsyncListener("second", calls, RuntimeError("worse"))

        async def failure_callback(failure):
            received.append(failure)
            await asyncio.sleep(0)

        dispatcher = EventDispatcher(
            {_Evt: [first, second]},
            failure_strategy=ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
            failure_callback=failure_callback,
        )

        await dispatcher.dispatch_async(_Evt())

        assert [f.listener for f in received] == [first, second]

    def test_failure_callback_exception_is_suppressed_sync(self, caplog):
        calls = []
        received = []
        first = _SyncListener("first", calls, ValueError("bad"))
        second = _SyncListener("second", calls, RuntimeError("worse"))

        def failure_callback(failure):
            received.append(failure)
            raise RuntimeError("broken callback")

        dispatcher = EventDispatcher(
            {_Evt: [first, second]},
            failure_strategy=ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
            failure_callback=failure_callback,
        )

        with caplog.at_level(logging.ERROR, logger="redis.event"):
            dispatcher.dispatch(_Evt())

        # The callback is still invoked for every failure and its own errors
        # never reach the event source.
        assert [f.listener for f in received] == [first, second]
        callback_errors = sum(
            "failure callback raised" in record.getMessage()
            for record in caplog.records
        )
        assert callback_errors == 2

    async def test_failure_callback_exception_is_suppressed_async(self):
        received = []
        first = _AsyncListener("first", [], ValueError("bad"))
        second = _AsyncListener("second", [], RuntimeError("worse"))

        async def failure_callback(failure):
            received.append(failure)
            raise RuntimeError("broken callback")

        dispatcher = EventDispatcher(
            {_Evt: [first, second]},
            failure_strategy=ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
            failure_callback=failure_callback,
        )

        await dispatcher.dispatch_async(_Evt())

        assert [f.listener for f in received] == [first, second]

    @pytest.mark.parametrize(
        "strategy",
        [
            ListenerFailureStrategy.ISOLATE,
            ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
        ],
    )
    @pytest.mark.parametrize(
        "interrupt", [KeyboardInterrupt("stop"), SystemExit("exit")]
    )
    def test_process_exit_exceptions_always_propagate_sync(self, strategy, interrupt):
        calls = []
        first = _SyncListener("first", calls, interrupt)
        second = _SyncListener("second", calls)

        dispatcher = EventDispatcher({_Evt: [first, second]}, failure_strategy=strategy)

        with pytest.raises(BaseException) as exc_info:
            dispatcher.dispatch(_Evt())
        assert exc_info.value is interrupt
        # The remaining listeners must not run once the interpreter is being
        # interrupted.
        assert calls == ["first"]

    @pytest.mark.parametrize(
        "strategy",
        [
            ListenerFailureStrategy.ISOLATE,
            ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
        ],
    )
    async def test_process_exit_exceptions_always_propagate_async(self, strategy):
        calls = []
        interrupt = KeyboardInterrupt("stop")
        first = _AsyncListener("first", calls, interrupt)
        second = _AsyncListener("second", calls)

        dispatcher = EventDispatcher({_Evt: [first, second]}, failure_strategy=strategy)

        with pytest.raises(KeyboardInterrupt) as exc_info:
            await dispatcher.dispatch_async(_Evt())
        assert exc_info.value is interrupt
        assert calls == ["first"]

    def test_isolation_strategy_is_scoped_to_dispatcher_instance(self):
        calls_a = []
        calls_b = []
        listener_a = _SyncListener("a", calls_a, RuntimeError("boom"))
        listener_b = _SyncListener("b", calls_b, RuntimeError("boom"))

        isolating = EventDispatcher(
            {_Evt: [listener_a]},
            failure_strategy=ListenerFailureStrategy.ISOLATE,
        )
        propagating = EventDispatcher({_Evt: [listener_b]})

        isolating.dispatch(_Evt())  # must not raise
        with pytest.raises(RuntimeError):
            propagating.dispatch(_Evt())
        assert calls_a == ["a"]
        assert calls_b == ["b"]

    def test_registration_order_is_deterministic_for_aggregation(self):
        failures_to_raise = [RuntimeError(f"err-{i}") for i in range(5)]
        listeners = [
            _SyncListener(f"listener-{i}", [], failures_to_raise[i]) for i in range(5)
        ]

        dispatcher = EventDispatcher(
            failure_strategy=ListenerFailureStrategy.ISOLATE_AND_AGGREGATE
        )
        # Re-registering the same instance must not duplicate or reorder it.
        dispatcher.register_listeners({_Evt: listeners})
        dispatcher.register_listeners({_Evt: [listeners[0], listeners[2]]})

        with pytest.raises(ListenerFailureError) as exc_info:
            dispatcher.dispatch(_Evt())

        assert [f.listener for f in exc_info.value.failures] == listeners

    def test_snapshot_is_used_when_failing_listener_mutates_registration(self):
        calls = []
        surviving = _SyncListener("surviving", calls)
        late = _SyncListener("late", calls)
        dispatcher = EventDispatcher(failure_strategy=ListenerFailureStrategy.ISOLATE)

        class MutatingAndFailingListener(EventListenerInterface):
            def listen(self, event: object) -> None:
                calls.append("mutating")
                dispatcher.unregister_listeners({_Evt: [surviving]})
                dispatcher.register_listeners({_Evt: [late]})
                raise RuntimeError("failing")

        mutating = MutatingAndFailingListener()
        dispatcher.register_listeners({_Evt: [mutating, surviving]})

        # Current dispatch follows the pre-dispatch snapshot: the unregistered
        # listener still gets it, the newly registered one does not.
        dispatcher.dispatch(_Evt())
        assert calls == ["mutating", "surviving"]

        # The next dispatch follows the mutated registration.
        dispatcher.dispatch(_Evt())
        assert calls == ["mutating", "surviving", "mutating", "late"]

    def test_builtin_listener_failure_is_logged_and_does_not_starve_users(self, caplog):
        class FailingReAuthConnection:
            def re_auth(self):
                raise RuntimeError("re-auth failed")

        user_calls = []
        user_listener = _SyncListener("user", user_calls)

        dispatcher = EventDispatcher(failure_strategy=ListenerFailureStrategy.ISOLATE)
        dispatcher.register_listeners({AfterConnectionReleasedEvent: [user_listener]})

        with caplog.at_level(logging.ERROR, logger="redis.event"):
            dispatcher.dispatch(AfterConnectionReleasedEvent(FailingReAuthConnection()))

        # The user listener still receives the event even though the built-in
        # re-auth listener failed before it.
        assert user_calls == ["user"]
        error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_logs) == 1
        message = error_logs[0].getMessage()
        assert "Built-in event listener" in message
        assert "ReAuthConnectionListener" in message
        assert error_logs[0].exc_info is not None

    async def test_builtin_async_listener_failure_is_logged_and_isolated(self, caplog):
        class FailingReAuthConnection:
            async def re_auth(self):
                raise RuntimeError("re-auth failed")

        dispatcher = EventDispatcher(failure_strategy=ListenerFailureStrategy.ISOLATE)

        with caplog.at_level(logging.ERROR, logger="redis.event"):
            await dispatcher.dispatch_async(
                AsyncAfterConnectionReleasedEvent(FailingReAuthConnection())
            )

        error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_logs) == 1
        assert "AsyncReAuthConnectionListener" in error_logs[0].getMessage()

    async def test_async_dispatch_supports_plain_sync_listener(self):
        calls = []

        class PlainSyncListener(EventListenerInterface):
            def listen(self, event: object) -> None:
                calls.append(event)

        dispatcher = EventDispatcher({_Evt: [PlainSyncListener()]})

        # Previously this raised "object NoneType can't be used in 'await'".
        await dispatcher.dispatch_async(_Evt())
        assert len(calls) == 1
        assert isinstance(calls[0], _Evt)

    async def test_async_dispatch_awaits_coroutine_returned_by_sync_listener(self):
        ran = False

        class SyncListenerReturningCoroutine(EventListenerInterface):
            def listen(self, event: object):
                async def work():
                    nonlocal ran
                    await asyncio.sleep(0)
                    ran = True

                return work()

        dispatcher = EventDispatcher({_Evt: [SyncListenerReturningCoroutine()]})

        await dispatcher.dispatch_async(_Evt())
        assert ran is True

    async def test_async_aggregate_handles_mixed_listener_kinds(self):
        calls = []

        class SyncFailingListener(EventListenerInterface):
            def listen(self, event: object) -> None:
                calls.append("sync")
                raise ValueError("sync failure")

        class SyncReturningFailingCoroutine(EventListenerInterface):
            def listen(self, event: object):
                async def work():
                    calls.append("unawaited")
                    raise RuntimeError("unawaited failure")

                return work()

        async_listener = _AsyncListener("async", calls, TypeError("async failure"))

        dispatcher = EventDispatcher(
            {
                _Evt: [
                    SyncFailingListener(),
                    SyncReturningFailingCoroutine(),
                    async_listener,
                ]
            },
            failure_strategy=ListenerFailureStrategy.ISOLATE_AND_AGGREGATE,
        )

        with pytest.raises(ListenerFailureError) as exc_info:
            await dispatcher.dispatch_async(_Evt())

        assert calls == ["sync", "unawaited", "async"]
        failures = exc_info.value.failures
        assert [type(f.exception) for f in failures] == [
            ValueError,
            RuntimeError,
            TypeError,
        ]
