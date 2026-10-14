import asyncio
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

from redis.auth.token import TokenInterface
from redis.credentials import CredentialProvider, StreamingCredentialProvider
from redis.observability.recorder import (
    init_connection_count,
    register_pools_connection_count,
)
from redis.utils import check_protocol_version, deprecated_function

if TYPE_CHECKING:
    from redis.maint_notifications import (
        MaintenanceNotification,
        MaintenanceState,
        MaintNotificationsConfig,
    )

logger = logging.getLogger(__name__)


class EventListenerInterface(ABC):
    """
    Represents a listener for given event object.
    """

    @abstractmethod
    def listen(self, event: object):
        pass


class AsyncEventListenerInterface(ABC):
    """
    Represents an async listener for given event object.
    """

    @abstractmethod
    async def listen(self, event: object):
        pass


class EventDispatcherInterface(ABC):
    """
    Represents a dispatcher that dispatches events to listeners
    associated with given event.
    """

    @abstractmethod
    def dispatch(self, event: object):
        pass

    @abstractmethod
    async def dispatch_async(self, event: object):
        pass

    @abstractmethod
    def register_listeners(
        self,
        mappings: Dict[
            Type[object],
            List[Union[EventListenerInterface, AsyncEventListenerInterface]],
        ],
    ):
        """Register additional listeners."""
        pass

    @abstractmethod
    def unregister_listeners(
        self,
        mappings: Dict[
            Type[object],
            List[Union[EventListenerInterface, AsyncEventListenerInterface]],
        ],
    ):
        """Remove previously registered listeners by identity."""
        pass


class EventException(Exception):
    """
    Exception wrapper that adds an event object into exception context.
    """

    def __init__(self, exception: Exception, event: object):
        self.exception = exception
        self.event = event
        super().__init__(exception)


class ListenerFailureStrategy(Enum):
    """
    How an :class:`EventDispatcher` treats a listener that raises while
    handling an event. Configured per dispatcher instance.

    ``PROPAGATE``
        The raised exception propagates to the event source immediately and
        the remaining listeners for the event are not called. This is the
        default and preserves the historical behavior.

    ``ISOLATE``
        The failing listener is skipped, the failure is logged, and every
        remaining listener still receives the event. The event source never
        sees the exception.

    ``ISOLATE_AND_AGGREGATE``
        Every listener is invoked as with ``ISOLATE``, but the failures are
        collected in listener registration order. Once all listeners have run,
        they are handed one by one to the configured failure callback, or -
        when no callback is configured - a single
        :class:`ListenerFailureError` holding every failure is raised to the
        event source.

    ``KeyboardInterrupt``, ``SystemExit`` and every other ``BaseException``
    that is not an ``Exception`` are never isolated under any strategy: they
    propagate immediately exactly as the listener raised them.
    """

    PROPAGATE = "propagate"
    ISOLATE = "isolate"
    ISOLATE_AND_AGGREGATE = "isolate_and_aggregate"


@dataclass(frozen=True)
class ListenerFailure:
    """
    A single listener failure recorded while dispatching an event.

    :param event_type: The type of the event being dispatched.
    :param listener: The listener instance that raised.
    :param exception: The original exception raised by the listener.
    """

    event_type: Type[object]
    listener: Union[EventListenerInterface, AsyncEventListenerInterface]
    exception: Exception

    @property
    def listener_identifier(self) -> str:
        """A stable, human-readable identifier of the failing listener."""
        listener_type = type(self.listener)
        return f"{listener_type.__module__}.{listener_type.__name__}@0x{id(self.listener):x}"


class ListenerFailureError(Exception):
    """
    Aggregate raised to the event source when every listener has been invoked
    under :attr:`ListenerFailureStrategy.ISOLATE_AND_AGGREGATE`, one or more
    listeners failed, and no failure callback is configured.

    :param failures: The :class:`ListenerFailure` records in the same order
        as the corresponding listeners were registered.
    """

    def __init__(self, failures: List[ListenerFailure]):
        self.failures = list(failures)
        details = "\n".join(
            f"  {index}. event '{failure.event_type.__name__}', "
            f"listener '{failure.listener_identifier}', "
            f"{type(failure.exception).__name__}: {failure.exception}"
            for index, failure in enumerate(self.failures, start=1)
        )
        super().__init__(
            f"{len(self.failures)} event listener(s) failed while dispatching "
            f"the event:\n{details}"
        )


#: Receives each recorded failure, in registration order, when the
#: ``ISOLATE_AND_AGGREGATE`` strategy is used with a callback. May be a regular
#: function or a coroutine function; ``dispatch_async`` also awaits an
#: awaitable returned by a regular function. Anything it raises is logged and
#: never reaches the event source.
ListenerFailureCallback = Callable[[ListenerFailure], Any]


class EventDispatcher(EventDispatcherInterface):
    # TODO: Make dispatcher to accept external mappings.
    def __init__(
        self,
        event_listeners: Optional[
            Dict[Type[object], List[EventListenerInterface]]
        ] = None,
        failure_strategy: Union[
            ListenerFailureStrategy, str
        ] = ListenerFailureStrategy.PROPAGATE,
        failure_callback: Optional[ListenerFailureCallback] = None,
    ):
        """
        Dispatcher that dispatches events to listeners associated with given event.

        :param event_listeners: Additional listeners registered on construction.
        :param failure_strategy: How a listener raising an ``Exception`` is
            treated while dispatching a single event. Defaults to
            :attr:`ListenerFailureStrategy.PROPAGATE`, which preserves the
            historical behavior. The strategy is scoped to this dispatcher
            instance and never affects other dispatchers.
        :param failure_callback: Used only with
            :attr:`ListenerFailureStrategy.ISOLATE_AND_AGGREGATE`: invoked once
            per recorded failure, in listener registration order, after every
            listener has run. It may be a regular or a coroutine function.
            When it raises, the error is logged but never reaches the event
            source. When no callback is configured, the aggregated failures are
            raised as a :class:`ListenerFailureError` instead.
        """
        self._event_listeners_mapping: Dict[
            Type[object], List[EventListenerInterface]
        ] = {
            AfterConnectionReleasedEvent: [
                ReAuthConnectionListener(),
            ],
            AfterPooledConnectionsInstantiationEvent: [
                RegisterReAuthForPooledConnections(),
            ],
            AfterSingleConnectionInstantiationEvent: [
                RegisterReAuthForSingleConnection()
            ],
            AfterPubSubConnectionInstantiationEvent: [RegisterReAuthForPubSub()],
            AfterAsyncClusterInstantiationEvent: [RegisterReAuthForAsyncClusterNodes()],
            AsyncAfterConnectionReleasedEvent: [
                AsyncReAuthConnectionListener(),
            ],
        }

        # Reentrant so a finalizer/listener that runs on the same thread
        # while the lock is held (e.g. a weakref.finalize callback fired
        # from cyclic GC during an allocation inside register_listeners /
        # unregister_listeners) can re-enter without deadlocking.
        self._lock = threading.RLock()
        self._async_lock = None

        self._failure_strategy = self._coerce_failure_strategy(failure_strategy)
        self._failure_callback = failure_callback

        if event_listeners:
            self.register_listeners(event_listeners)

    @property
    def failure_strategy(self) -> ListenerFailureStrategy:
        """The strategy applied when a listener raises during dispatch."""
        return self._failure_strategy

    @staticmethod
    def _coerce_failure_strategy(
        failure_strategy: Union[ListenerFailureStrategy, str],
    ) -> ListenerFailureStrategy:
        if isinstance(failure_strategy, ListenerFailureStrategy):
            return failure_strategy
        if isinstance(failure_strategy, str):
            return ListenerFailureStrategy(failure_strategy)
        raise TypeError(
            "failure_strategy must be a ListenerFailureStrategy or one of its "
            f"string values, got {type(failure_strategy).__name__}"
        )

    def dispatch(self, event: object):
        # Snapshot listeners under the lock, then release it before invoking
        # them. Holding the lock across listener execution would turn any
        # listener that calls register_listeners / unregister_listeners /
        # dispatch back into the dispatcher into a deadlock. The snapshot also
        # makes an in-flight dispatch immune to listeners mutating the
        # registration of the event being dispatched.
        with self._lock:
            listeners = list(self._event_listeners_mapping.get(type(event), []))
        # Allocated lazily: connection release is a hot path and dispatch
        # succeeds without any failure in the common case.
        failures: Optional[List[ListenerFailure]] = None
        for listener in listeners:
            try:
                listener.listen(event)
            except Exception as exception:
                if self._failure_strategy is ListenerFailureStrategy.PROPAGATE:
                    raise
                if failures is None:
                    failures = []
                failures.append(self._record_failure(event, listener, exception))
        if failures is not None:
            self._report_failures(failures)

    async def dispatch_async(self, event: object):
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        # Snapshot listeners under the lock, then release it before awaiting
        # them. See the note in dispatch(); the same rationale applies here
        # for dispatch_async re-entry from within a listener.
        async with self._async_lock:
            listeners = list(self._event_listeners_mapping.get(type(event), []))
        # Allocated lazily, see the note in dispatch().
        failures: Optional[List[ListenerFailure]] = None
        for listener in listeners:
            try:
                # Listeners may freely be either coroutine functions or plain
                # functions. A plain function returning an awaitable (an
                # un-awaited coroutine/future) is awaited as well, so both
                # kinds of mixing work; a plain ``None`` return needs no await.
                outcome = listener.listen(event)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception as exception:
                if self._failure_strategy is ListenerFailureStrategy.PROPAGATE:
                    raise
                if failures is None:
                    failures = []
                failures.append(self._record_failure(event, listener, exception))
        if failures is not None:
            await self._report_failures_async(failures)

    def _record_failure(
        self,
        event: object,
        listener: Union[EventListenerInterface, AsyncEventListenerInterface],
        exception: Exception,
    ) -> ListenerFailure:
        failure = ListenerFailure(type(event), listener, exception)
        self._log_isolated_failure(failure)
        return failure

    def _log_isolated_failure(self, failure: ListenerFailure) -> None:
        # A failure that does not reach the event source must never vanish
        # silently. Failures of the listeners the library registers itself
        # are logged at error level with their traceback.
        if isinstance(failure.listener, _BUILTIN_LISTENER_CLASSES):
            logger.error(
                "Built-in event listener '%s' failed handling event '%s'; the "
                "failure is isolated by the '%s' strategy and will not be "
                "raised to the event source.",
                failure.listener_identifier,
                failure.event_type.__name__,
                self._failure_strategy.value,
                exc_info=failure.exception,
            )
        else:
            logger.warning(
                "Event listener '%s' failed handling event '%s'; the failure "
                "is isolated by the '%s' strategy and will not be raised to "
                "the event source.",
                failure.listener_identifier,
                failure.event_type.__name__,
                self._failure_strategy.value,
                exc_info=failure.exception,
            )

    def _report_failures(self, failures: List[ListenerFailure]) -> None:
        if not failures:
            return
        if self._failure_strategy is not ListenerFailureStrategy.ISOLATE_AND_AGGREGATE:
            return
        if self._failure_callback is None:
            raise ListenerFailureError(failures)
        for failure in failures:
            try:
                outcome = self._failure_callback(failure)
            except Exception:
                logger.exception(
                    "Event listener failure callback raised while handling a "
                    "failure of listener '%s'; the error is suppressed and "
                    "cannot affect the event source.",
                    failure.listener_identifier,
                )
                continue
            if inspect.isawaitable(outcome):
                # A synchronous dispatch has no event loop to run the
                # callback on; close a bare coroutine so it is not reported
                # as never awaited instead of failing silently.
                if inspect.iscoroutine(outcome):
                    outcome.close()
                logger.warning(
                    "Event listener failure callback returned an awaitable "
                    "from the synchronous dispatch path for listener '%s'; "
                    "it cannot be awaited and is ignored.",
                    failure.listener_identifier,
                )

    async def _report_failures_async(self, failures: List[ListenerFailure]) -> None:
        if not failures:
            return
        if self._failure_strategy is not ListenerFailureStrategy.ISOLATE_AND_AGGREGATE:
            return
        if self._failure_callback is None:
            raise ListenerFailureError(failures)
        for failure in failures:
            try:
                outcome = self._failure_callback(failure)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                logger.exception(
                    "Event listener failure callback raised while handling a "
                    "failure of listener '%s'; the error is suppressed and "
                    "cannot affect the event source.",
                    failure.listener_identifier,
                )

    def register_listeners(
        self,
        mappings: Dict[
            Type[object],
            List[Union[EventListenerInterface, AsyncEventListenerInterface]],
        ],
    ):
        with self._lock:
            for event_type in mappings:
                current = self._event_listeners_mapping.get(event_type, [])
                # De-duplicate by identity (matching unregister_listeners)
                # while preserving registration order, so aggregated failures
                # surface deterministically in registration order.
                merged = list(current)
                for listener in mappings[event_type]:
                    if all(listener is not existing for existing in merged):
                        merged.append(listener)
                self._event_listeners_mapping[event_type] = merged

    def unregister_listeners(
        self,
        mappings: Dict[
            Type[object],
            List[Union[EventListenerInterface, AsyncEventListenerInterface]],
        ],
    ):
        with self._lock:
            for event_type, to_remove in mappings.items():
                current = self._event_listeners_mapping.get(event_type)
                if not current:
                    continue
                # Remove by identity to match register semantics and to avoid
                # reliance on listener __eq__ implementations.
                self._event_listeners_mapping[event_type] = [
                    listener
                    for listener in current
                    if all(listener is not target for target in to_remove)
                ]


class AfterConnectionReleasedEvent:
    """
    Event that will be fired before each command execution.
    """

    def __init__(self, connection):
        self._connection = connection

    @property
    def connection(self):
        return self._connection


class AsyncAfterConnectionReleasedEvent(AfterConnectionReleasedEvent):
    pass


class AfterSlotsCacheRefreshEvent:
    """
    Event fired after NodesManager's slots cache is refreshed, either via a
    full re-initialization or a MOVED-driven slot re-mapping. Signal-only;
    carries no payload. Listeners typically reconcile per-node bookkeeping
    (e.g. ClusterPubSub shard subscriptions).
    """

    pass


class AsyncAfterSlotsCacheRefreshEvent(AfterSlotsCacheRefreshEvent):
    pass


class MaintenanceEvent:
    """
    Base class of the events a maintenance notification pushed by the server
    produces while it relaxes timeouts: on a connection pool for a node handoff
    (``MaintenanceState.MOVING``), or on a single connection for a shard
    migration or failover (``MaintenanceState.MAINTENANCE``).

    Dispatched synchronously, by both the sync and the async client, to the
    ``event_dispatcher`` of the ``MaintNotificationsConfig`` the handler acted
    with; listeners implement ``EventListenerInterface``.

    :param connection_pool: The pool whose connections are affected. ``None``
        when the notification was handled by a connection without a pool
        handler, such as an OSS cluster node connection.
    :param connection: The connection that received the notification. ``None``
        when a pool-level handler had no connection to attribute it to.
    :param state: The ``MaintenanceState`` the notification puts its source in.
    :param notification: The ``MaintenanceNotification`` that was handled.
    :param config: The ``MaintNotificationsConfig`` the handler acted with.
    """

    def __init__(
        self,
        connection_pool: Optional[object],
        connection: Optional[object],
        state: "MaintenanceState",
        notification: Optional["MaintenanceNotification"],
        config: "MaintNotificationsConfig",
    ):
        self._connection_pool = connection_pool
        self._connection = connection
        self._state = state
        self._notification = notification
        self._config = config

    @property
    def connection_pool(self) -> Optional[object]:
        return self._connection_pool

    @property
    def connection(self) -> Optional[object]:
        return self._connection

    @property
    def state(self) -> "MaintenanceState":
        return self._state

    @property
    def notification(self) -> Optional["MaintenanceNotification"]:
        return self._notification

    @property
    def config(self) -> "MaintNotificationsConfig":
        return self._config


class MaintenanceStartedEvent(MaintenanceEvent):
    """
    Event fired when a maintenance notification starts relaxing the timeouts of
    its source. Paired with a ``MaintenanceCompletedEvent`` carrying the same
    ``state`` and source.
    """

    pass


class MaintenanceCompletedEvent(MaintenanceEvent):
    """
    Event fired when the timeout relaxation announced by a
    ``MaintenanceStartedEvent`` is reverted: the completion notification
    arrived, the handoff TTL expired, or the connection was closed while
    still under maintenance. Also fired, with ``MaintenanceState.MAINTENANCE``,
    when a handoff takes over a connection under maintenance: the connection
    stays relaxed, but under the handoff's ``MOVING`` pair from then on. In
    the last two cases ``notification`` is ``None``.

    Not fired when a newer handoff superseded the one that completed; the
    source stays relaxed until the newer one completes.
    """

    pass


class ClientType(Enum):
    SYNC = ("sync",)
    ASYNC = ("async",)


class AfterPooledConnectionsInstantiationEvent:
    """
    Event that will be fired after pooled connection instances was created.
    """

    def __init__(
        self,
        connection_pools: List,
        client_type: ClientType,
        credential_provider: Optional[CredentialProvider] = None,
    ):
        self._connection_pools = connection_pools
        self._client_type = client_type
        self._credential_provider = credential_provider

    @property
    def connection_pools(self):
        return self._connection_pools

    @property
    def client_type(self) -> ClientType:
        return self._client_type

    @property
    def credential_provider(self) -> Union[CredentialProvider, None]:
        return self._credential_provider


class AfterSingleConnectionInstantiationEvent:
    """
    Event that will be fired after single connection instances was created.

    :param connection_lock: For sync client thread-lock should be provided,
    for async asyncio.Lock
    """

    def __init__(
        self,
        connection,
        client_type: ClientType,
        connection_lock: Union[threading.RLock, asyncio.Lock],
    ):
        self._connection = connection
        self._client_type = client_type
        self._connection_lock = connection_lock

    @property
    def connection(self):
        return self._connection

    @property
    def client_type(self) -> ClientType:
        return self._client_type

    @property
    def connection_lock(self) -> Union[threading.RLock, asyncio.Lock]:
        return self._connection_lock


class AfterPubSubConnectionInstantiationEvent:
    def __init__(
        self,
        pubsub_connection,
        connection_pool,
        client_type: ClientType,
        connection_lock: Union[threading.RLock, asyncio.Lock],
    ):
        self._pubsub_connection = pubsub_connection
        self._connection_pool = connection_pool
        self._client_type = client_type
        self._connection_lock = connection_lock

    @property
    def pubsub_connection(self):
        return self._pubsub_connection

    @property
    def connection_pool(self):
        return self._connection_pool

    @property
    def client_type(self) -> ClientType:
        return self._client_type

    @property
    def connection_lock(self) -> Union[threading.RLock, asyncio.Lock]:
        return self._connection_lock


class AfterAsyncClusterInstantiationEvent:
    """
    Event that will be fired after async cluster instance was created.

    Async cluster doesn't use connection pools,
    instead ClusterNode object manages connections.
    """

    def __init__(
        self,
        nodes: dict,
        credential_provider: Optional[CredentialProvider] = None,
    ):
        self._nodes = nodes
        self._credential_provider = credential_provider

    @property
    def nodes(self) -> dict:
        return self._nodes

    @property
    def credential_provider(self) -> Union[CredentialProvider, None]:
        return self._credential_provider


class OnCommandsFailEvent:
    """
    Event fired whenever a command fails during the execution.
    """

    def __init__(
        self,
        commands: tuple,
        exception: Exception,
    ):
        self._commands = commands
        self._exception = exception

    @property
    def commands(self) -> tuple:
        return self._commands

    @property
    def exception(self) -> Exception:
        return self._exception


class AsyncOnCommandsFailEvent(OnCommandsFailEvent):
    pass


class ReAuthConnectionListener(EventListenerInterface):
    """
    Listener that performs re-authentication of given connection.
    """

    def listen(self, event: AfterConnectionReleasedEvent):
        event.connection.re_auth()


class AsyncReAuthConnectionListener(AsyncEventListenerInterface):
    """
    Async listener that performs re-authentication of given connection.
    """

    async def listen(self, event: AsyncAfterConnectionReleasedEvent):
        await event.connection.re_auth()


class RegisterReAuthForPooledConnections(EventListenerInterface):
    """
    Listener that registers a re-authentication callback for pooled connections.
    Required by :class:`StreamingCredentialProvider`.
    """

    def __init__(self):
        self._event = None

    def listen(self, event: AfterPooledConnectionsInstantiationEvent):
        if isinstance(event.credential_provider, StreamingCredentialProvider):
            self._event = event

            if event.client_type == ClientType.SYNC:
                event.credential_provider.on_next(self._re_auth)
                event.credential_provider.on_error(self._raise_on_error)
            else:
                event.credential_provider.on_next(self._re_auth_async)
                event.credential_provider.on_error(self._raise_on_error_async)

    def _re_auth(self, token):
        for pool in self._event.connection_pools:
            pool.re_auth_callback(token)

    async def _re_auth_async(self, token):
        for pool in self._event.connection_pools:
            await pool.re_auth_callback(token)

    def _raise_on_error(self, error: Exception):
        raise EventException(error, self._event)

    async def _raise_on_error_async(self, error: Exception):
        raise EventException(error, self._event)


class RegisterReAuthForSingleConnection(EventListenerInterface):
    """
    Listener that registers a re-authentication callback for single connection.
    Required by :class:`StreamingCredentialProvider`.
    """

    def __init__(self):
        self._event = None

    def listen(self, event: AfterSingleConnectionInstantiationEvent):
        if isinstance(
            event.connection.credential_provider, StreamingCredentialProvider
        ):
            self._event = event

            if event.client_type == ClientType.SYNC:
                event.connection.credential_provider.on_next(self._re_auth)
                event.connection.credential_provider.on_error(self._raise_on_error)
            else:
                event.connection.credential_provider.on_next(self._re_auth_async)
                event.connection.credential_provider.on_error(
                    self._raise_on_error_async
                )

    def _re_auth(self, token):
        with self._event.connection_lock:
            self._event.connection.send_command(
                "AUTH", token.try_get("oid"), token.get_value()
            )
            self._event.connection.read_response()

    async def _re_auth_async(self, token):
        async with self._event.connection_lock:
            await self._event.connection.send_command(
                "AUTH", token.try_get("oid"), token.get_value()
            )
            await self._event.connection.read_response()

    def _raise_on_error(self, error: Exception):
        raise EventException(error, self._event)

    async def _raise_on_error_async(self, error: Exception):
        raise EventException(error, self._event)


class RegisterReAuthForAsyncClusterNodes(EventListenerInterface):
    def __init__(self):
        self._event = None

    def listen(self, event: AfterAsyncClusterInstantiationEvent):
        if isinstance(event.credential_provider, StreamingCredentialProvider):
            self._event = event
            event.credential_provider.on_next(self._re_auth)
            event.credential_provider.on_error(self._raise_on_error)

    async def _re_auth(self, token: TokenInterface):
        for key in self._event.nodes:
            await self._event.nodes[key].re_auth_callback(token)

    async def _raise_on_error(self, error: Exception):
        raise EventException(error, self._event)


class RegisterReAuthForPubSub(EventListenerInterface):
    def __init__(self):
        self._connection = None
        self._connection_pool = None
        self._client_type = None
        self._connection_lock = None
        self._event = None

    def listen(self, event: AfterPubSubConnectionInstantiationEvent):
        if isinstance(
            event.pubsub_connection.credential_provider, StreamingCredentialProvider
        ) and check_protocol_version(event.pubsub_connection.get_protocol(), 3):
            self._event = event
            self._connection = event.pubsub_connection
            self._connection_pool = event.connection_pool
            self._client_type = event.client_type
            self._connection_lock = event.connection_lock

            if self._client_type == ClientType.SYNC:
                self._connection.credential_provider.on_next(self._re_auth)
                self._connection.credential_provider.on_error(self._raise_on_error)
            else:
                self._connection.credential_provider.on_next(self._re_auth_async)
                self._connection.credential_provider.on_error(
                    self._raise_on_error_async
                )

    def _re_auth(self, token: TokenInterface):
        with self._connection_lock:
            self._connection.send_command(
                "AUTH", token.try_get("oid"), token.get_value()
            )
            self._connection.read_response()

        self._connection_pool.re_auth_callback(token)

    async def _re_auth_async(self, token: TokenInterface):
        async with self._connection_lock:
            await self._connection.send_command(
                "AUTH", token.try_get("oid"), token.get_value()
            )
            await self._connection.read_response()

        await self._connection_pool.re_auth_callback(token)

    def _raise_on_error(self, error: Exception):
        raise EventException(error, self._event)

    async def _raise_on_error_async(self, error: Exception):
        raise EventException(error, self._event)


class InitializeConnectionCountObservability(EventListenerInterface):
    """
    Listener that initializes connection count observability.
    """

    @deprecated_function(
        reason="Connection count is now tracked via record_connection_count(). "
        "This functionality will be removed in the next major version",
        version="7.4.0",
    )
    def listen(self, event: AfterPooledConnectionsInstantiationEvent):
        # Initialize gauge only once, subsequent calls won't have an affect.
        # Note: init_connection_count() and register_pools_connection_count()
        # are deprecated and will emit their own warnings.
        init_connection_count()

        # Register pools for connection count observability.
        register_pools_connection_count(event.connection_pools)


#: Listener classes the library itself registers in the default mapping (and
#: other built-in listeners shipped with the library). Their isolated failures
#: are logged explicitly so a misconfigured credential provider or a failing
#: re-authentication stays visible even when the event source is shielded.
_BUILTIN_LISTENER_CLASSES = (
    ReAuthConnectionListener,
    AsyncReAuthConnectionListener,
    RegisterReAuthForPooledConnections,
    RegisterReAuthForSingleConnection,
    RegisterReAuthForPubSub,
    RegisterReAuthForAsyncClusterNodes,
    InitializeConnectionCountObservability,
)
