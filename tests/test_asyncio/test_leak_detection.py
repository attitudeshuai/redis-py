"""Tests for stale checkout (connection leak) detection on async pools.

Self-contained (no Redis server): pools are driven with an async dummy
connection class.
"""

import asyncio
import gc
import os
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from redis.asyncio import BlockingConnectionPool, ConnectionPool, LeakDetectionConfig
from redis.asyncio.sentinel import SentinelConnectionPool
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    MaxConnectionsError,
)

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

    async def connect(self):
        self._sock = mock.Mock()

    async def disconnect(self):
        self._sock = None

    async def can_read(self, timeout: float = 0):
        return False

    def should_reconnect(self):
        return False

    async def re_auth(self):
        pass


def config(**overrides):
    overrides.setdefault("enabled", True)
    overrides.setdefault("threshold", 0)
    return LeakDetectionConfig(**overrides)


def make_pool(cls=ConnectionPool, **kwargs):
    kwargs.setdefault("connection_class", DummyConnection)
    kwargs.setdefault("max_connections", 10)
    return cls(**kwargs)


class TestDisabledByDefault:
    async def test_borrow_return_path_unchanged(self):
        pool = make_pool()
        conn = await pool.get_connection()
        assert pool.stale_checkouts() == ()
        await pool.release(conn)
        assert pool.stale_checkouts() == ()
        await pool.aclose()

    async def test_capacity_error_message_unchanged(self):
        pool = make_pool(max_connections=1)
        conn = await pool.get_connection()
        with pytest.raises(MaxConnectionsError, match="^Too many connections$"):
            await pool.get_connection()
        await pool.release(conn)
        await pool.aclose()

    async def test_blocking_timeout_message_unchanged(self):
        pool = make_pool(
            BlockingConnectionPool, max_connections=1, timeout=0.05
        )
        conn = await pool.get_connection()
        with pytest.raises(
            RedisConnectionError, match="^No connection available\\.$"
        ):
            await pool.get_connection()
        await pool.release(conn)
        await pool.aclose()


class TestStaleCheckoutQuery:
    async def test_record_contents(self):
        pool = make_pool(leak_detection=config())
        conn = await pool.get_connection()
        stale = pool.stale_checkouts()
        assert len(stale) == 1
        info = stale[0]
        assert info.age >= 0
        assert info.gc_collected is False
        assert info.exempt is False
        assert "DummyConnection" in info.connection
        assert "id=0x" in info.connection
        assert info.context[0].startswith(THIS_FILE)
        await pool.release(conn)
        await pool.aclose()

    async def test_return_removes_record(self):
        pool = make_pool(leak_detection=config())
        conn = await pool.get_connection()
        assert len(pool.stale_checkouts()) == 1
        await pool.release(conn)
        assert pool.stale_checkouts() == ()
        await pool.aclose()

    async def test_healthy_short_checkouts_not_stale(self):
        pool = make_pool(leak_detection=config(threshold=100))
        conn = await pool.get_connection()
        assert pool.stale_checkouts() == ()
        await pool.release(conn)
        await pool.aclose()


class TestAlerts:
    async def test_alert_delivered_off_borrow_path_and_only_once(self):
        seen = []
        pool = make_pool(leak_detection=config(alert_callback=seen.append))
        conn = await pool.get_connection()
        # Alert delivery is scheduled with loop.call_soon, so it must not be
        # visible before the event loop gets control back.
        assert seen == []
        await asyncio.sleep(0)
        assert len(seen) == 1
        pool._dispatch_leak_alerts()
        assert len(seen) == 1
        await pool.release(conn)
        await pool.aclose()

    async def test_callback_exception_does_not_break_pool(self):
        def bad_callback(_info):
            raise RuntimeError("boom")

        pool = make_pool(leak_detection=config(alert_callback=bad_callback))
        conn = await pool.get_connection()
        await asyncio.sleep(0)
        assert len(pool.stale_checkouts()) == 1
        await pool.release(conn)
        assert pool.stale_checkouts() == ()
        # The pool is still fully operational after a failing callback.
        conn2 = await pool.get_connection()
        await pool.release(conn2)
        await pool.aclose()

    async def test_no_alert_for_healthy_checkout(self):
        seen = []
        pool = make_pool(
            leak_detection=config(threshold=100, alert_callback=seen.append)
        )
        conn = await pool.get_connection()
        await pool.release(conn)
        await asyncio.sleep(0)
        assert seen == []
        await pool.aclose()


class TestExemptions:
    async def test_manual_exempt_and_unmark(self):
        pool = make_pool(leak_detection=config())
        conn = await pool.get_connection()
        assert pool.mark_checkout_exempt(conn, "pubsub") is True
        assert pool.stale_checkouts() == ()
        exempt = pool.stale_checkouts(include_exempt=True)
        assert exempt[0].exempt_reason == "pubsub"
        assert pool.unmark_checkout_exempt(conn) is True
        assert len(pool.stale_checkouts()) == 1
        await pool.release(conn)
        await pool.aclose()

    async def test_configured_command_exemption(self):
        pool = make_pool(leak_detection=config(exempt_commands=["XREAD"]))
        conn = await pool.get_connection()
        assert pool.mark_command_checkout_exempt(conn, "xread") is True
        assert pool.stale_checkouts() == ()
        await pool.release(conn)
        await pool.aclose()


class TestExhaustionErrors:
    async def test_summary_detail_regular_pool(self):
        pool = make_pool(
            max_connections=1, leak_detection=config(error_detail="summary")
        )
        conn = await pool.get_connection()
        with pytest.raises(MaxConnectionsError) as exc_info:
            await pool.get_connection()
        message = str(exc_info.value)
        assert "pool capacity: 1" in message
        assert "connections in use: 1" in message
        assert "oldest active checkout:" in message
        await pool.release(conn)
        await pool.aclose()

    async def test_full_detail_regular_pool(self):
        pool = make_pool(
            max_connections=1, leak_detection=config(error_detail="full")
        )
        conn = await pool.get_connection()
        with pytest.raises(MaxConnectionsError) as exc_info:
            await pool.get_connection()
        assert THIS_FILE in str(exc_info.value)
        await pool.release(conn)
        await pool.aclose()

    async def test_blocking_pool_timeout_detail(self):
        pool = make_pool(
            BlockingConnectionPool,
            max_connections=1,
            timeout=0.05,
            leak_detection=config(error_detail="summary"),
        )
        conn = await pool.get_connection()
        with pytest.raises(RedisConnectionError) as exc_info:
            await pool.get_connection()
        message = str(exc_info.value)
        assert message.startswith("No connection available. (")
        assert "pool capacity: 1" in message
        assert "connections in use: 1" in message
        assert "idle connections: 0" in message
        assert "wait time:" in message
        assert "oldest active checkout:" in message
        await pool.release(conn)
        await pool.aclose()


class TestGarbageCollectedConnections:
    async def test_record_survives_connection_gc(self):
        pool = make_pool(leak_detection=config())
        # ``reset`` installs the weak in-use set the async pool uses after a
        # fork, allowing checked-out connections to be garbage collected.
        pool.reset()
        conn = await pool.get_connection()
        identity = pool.stale_checkouts()[0].connection
        del conn
        gc.collect()
        stale = pool.stale_checkouts()
        assert len(stale) == 1
        assert stale[0].gc_collected is True
        assert stale[0].connection == identity
        assert stale[0].context[0].startswith(THIS_FILE)
        await pool.aclose()


class TestResetAndArchive:
    async def test_aclose_clears_by_default(self):
        pool = make_pool(leak_detection=config())
        conn = await pool.get_connection()
        await pool.aclose()
        assert pool.stale_checkouts(include_archived=True) == ()
        del conn

    async def test_aclose_archives_records(self):
        pool = make_pool(leak_detection=config(on_reset="archive"))
        conn = await pool.get_connection()
        await pool.aclose()
        assert pool.stale_checkouts() == ()
        archived = pool.stale_checkouts(include_archived=True)
        assert len(archived) == 1
        assert archived[0].archived is True
        pool.clear_stale_checkout_archive()
        del conn


class TestSentinelFailover:
    async def test_failover_clears_checkout_records(self):
        manager = AsyncMock()
        manager.discover_master.return_value = ("10.0.0.1", 6379)
        pool = SentinelConnectionPool(
            "mymaster", manager, leak_detection=config()
        )
        assert await pool.get_master_address() == ("10.0.0.1", 6379)

        forgotten = DummyConnection()
        pool._leak_tracker.checkout(forgotten)
        assert len(pool.stale_checkouts()) == 1

        manager.discover_master.return_value = ("10.0.0.2", 6379)
        assert await pool.get_master_address() == ("10.0.0.2", 6379)
        assert pool.stale_checkouts() == ()
        assert pool.stale_checkouts(include_archived=True) == ()
        del forgotten

    async def test_failover_ignores_archive_policy(self):
        manager = AsyncMock()
        manager.discover_master.return_value = ("10.0.0.1", 6379)
        pool = SentinelConnectionPool(
            "mymaster",
            manager,
            leak_detection=config(on_reset="archive"),
        )
        await pool.get_master_address()
        forgotten = DummyConnection()
        pool._leak_tracker.checkout(forgotten)
        manager.discover_master.return_value = ("10.0.0.2", 6379)
        await pool.get_master_address()
        assert pool.stale_checkouts(include_archived=True) == ()
        del forgotten
