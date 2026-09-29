"""MCP connection pool with circuit breaker (Task 6).

The pool keeps one lazy client session per configured MCP server
(``mcp.servers`` in ``config/app.yaml``) and exposes a single
``call_tool`` entry point used by
:mod:`inference.app.backends.mcp_backend`.

Semantics (per SPEC.md):

- **lazy connect:** no server is contacted until the first tool call.
- **reconnect on failure:** a failed call drops the cached session so the
  next call reconnects from scratch.
- **per-call timeout:** default 10 s (``asyncio.wait_for``).
- **circuit breaker:** opens after 3 consecutive failures for 60 s. While
  open, calls are rejected immediately with :class:`CircuitOpenError`
  (a :class:`BackendError`, mapped to HTTP 502 by the gateway) so the
  strategy falls back without waiting on a dead server.

The pool never imports the official ``mcp`` SDK at module load: transports
import it lazily inside ``_ensure_connected`` and raise a clear
:class:`BackendError` when the ``mcp`` extra is not installed. Tests inject
fake callers via ``caller_factory`` so no SDK is needed to exercise the
pool, breaker, or backend logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from common.contracts import McpServerSpec

from .backends.base import BackendError

logger = logging.getLogger(__name__)

DEFAULT_CALL_TIMEOUT_S = 10.0
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_RECOVERY_TIMEOUT_S = 60.0


class CircuitOpenError(BackendError):
    """Circuit breaker is open; the call was rejected without server contact."""


class CircuitBreaker:
    """Consecutive-failure circuit breaker with half-open trial.

    Opens after ``failure_threshold`` consecutive failures and stays open
    for ``recovery_timeout`` seconds. Afterwards a single trial call is let
    through (half-open): success closes the breaker, failure re-opens it.
    While half-open, concurrent extra calls are rejected.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"

    def __init__(
        self,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT_S,
        time_fn: Callable[[], float] | None = None,
        name: str = "",
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._time = time_fn or time.monotonic
        self.name = name or "breaker"
        self._consecutive_failures = 0
        self._state = self.CLOSED
        self._opened_at = 0.0
        self._trial_in_flight = False

    @property
    def state(self) -> str:
        if self._state == self.OPEN and self._time() - self._opened_at >= self.recovery_timeout:
            return self.HALF_OPEN
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def can_execute(self) -> bool:
        """True when a call may proceed (closed, or a half-open trial slot)."""
        current = self.state
        if current == self.CLOSED:
            return True
        if current == self.HALF_OPEN:
            if self._state == self.OPEN:
                # Timeout elapsed: promote to half-open and take the trial slot.
                self._state = self.HALF_OPEN
                self._trial_in_flight = True
                return True
            return not self._trial_in_flight
        return False

    def before_call(self) -> None:
        """Raise :class:`CircuitOpenError` when the call must not proceed."""
        if not self.can_execute():
            raise CircuitOpenError(
                f"circuit breaker {self.name!r} is {self.state}; "
                f"server unreachable (retry in {self.retry_in():.1f}s)"
            )

    def retry_in(self) -> float:
        """Seconds until a trial call is allowed (0 when already allowed)."""
        if self._state != self.OPEN:
            return 0.0
        return max(0.0, self.recovery_timeout - (self._time() - self._opened_at))

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = self.CLOSED
        self._trial_in_flight = False

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._trial_in_flight = False
        if self._consecutive_failures >= self.failure_threshold:
            if self._state != self.OPEN:
                logger.warning(
                    "circuit breaker %r opened after %d consecutive failures",
                    self.name,
                    self._consecutive_failures,
                )
            self._state = self.OPEN
            self._opened_at = self._time()


# A tool caller invokes ``tool`` on a connected server and returns the raw
# (untrusted) result. ``timeout`` is in seconds.
ToolCaller = Callable[[str, dict[str, Any], float], Awaitable[Any]]
CallerFactory = Callable[[str, McpServerSpec], ToolCaller]


def _require_mcp_sdk() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise BackendError(
            "the 'mcp' package is not installed; install the 'mcp' extra "
            "(pip install -e .[mcp]) to use MCP-backed models"
        ) from exc


class _McpSession:
    """Persistent SDK session with lazy connect and explicit invalidation.

    Holds the SDK context managers open across calls; on any failure the
    caller drops the session (:meth:`ainvalidate`) so the next call
    reconnects. Subclasses only implement :meth:`_open`.
    """

    def __init__(self, server_name: str, spec: McpServerSpec) -> None:
        self.server_name = server_name
        self.spec = spec
        self._lock = asyncio.Lock()
        self._session: Any = None
        self._exits: list[Any] = []

    async def _open(self) -> Any:
        raise NotImplementedError

    async def _ensure(self) -> Any:
        async with self._lock:
            if self._session is None:
                self._session = await self._open()
            return self._session

    async def ainvalidate(self) -> None:
        async with self._lock:
            session, self._session = self._session, None
            exits, self._exits = self._exits, []
        for ctx in reversed(exits):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug("error closing MCP session for %r: %s", self.server_name, exc)
        _ = session

    async def call_tool(self, tool: str, arguments: dict[str, Any], timeout: float) -> Any:
        session = await self._ensure()

        async def _invoke() -> Any:
            return await session.call_tool(tool, arguments)

        return await asyncio.wait_for(_invoke(), timeout=timeout)


class _StdioSession(_McpSession):
    """``stdio`` transport: spawns ``spec.command`` via the MCP SDK."""

    async def _open(self) -> Any:
        _require_mcp_sdk()
        if not self.spec.command:
            raise BackendError(f"mcp server {self.server_name!r}: stdio needs 'command'")
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.spec.command[0], args=list(self.spec.command[1:])
        )
        logger.info("connecting to MCP stdio server %r: %s", self.server_name, self.spec.command)
        stdio_ctx = stdio_client(params)
        read, write = await stdio_ctx.__aenter__()
        self._exits.append(stdio_ctx)
        session_ctx = ClientSession(read, write)
        session = await session_ctx.__aenter__()
        self._exits.append(session_ctx)
        await session.initialize()
        logger.info("MCP stdio server %r ready", self.server_name)
        return session


class _HttpSession(_McpSession):
    """``http`` transport: streamable HTTP via the MCP SDK."""

    async def _open(self) -> Any:
        _require_mcp_sdk()
        if not self.spec.url:
            raise BackendError(f"mcp server {self.server_name!r}: http needs 'url'")
        from mcp import ClientSession

        try:
            from mcp.client.streamable_http import streamablehttp_client as http_client
        except ImportError:  # older SDKs exposed plain `streamable_http`
            from mcp.client.streamable_http import (  # type: ignore[no-redef]
                streamable_http_client as http_client,
            )
        logger.info("connecting to MCP http server %r: %s", self.server_name, self.spec.url)
        http_ctx = http_client(self.spec.url)
        streams = await http_ctx.__aenter__()
        self._exits.append(http_ctx)
        read, write = streams[0], streams[1]
        session_ctx = ClientSession(read, write)
        session = await session_ctx.__aenter__()
        self._exits.append(session_ctx)
        await session.initialize()
        logger.info("MCP http server %r ready", self.server_name)
        return session


def default_caller_factory(server_name: str, spec: McpServerSpec) -> ToolCaller:
    """Build an SDK-backed caller for ``spec.transport`` (lazy connect)."""
    if spec.transport == "stdio":
        session = _StdioSession(server_name, spec)
    elif spec.transport == "http":
        session = _HttpSession(server_name, spec)
    else:  # pragma: no cover - pydantic constrains the literal
        raise BackendError(f"mcp server {server_name!r}: unknown transport {spec.transport!r}")
    return session.call_tool


class McpConnectionPool:
    """Per-server lazy sessions + per-server circuit breakers."""

    def __init__(
        self,
        servers: Mapping[str, McpServerSpec] | None = None,
        default_timeout: float = DEFAULT_CALL_TIMEOUT_S,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT_S,
        caller_factory: CallerFactory | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._servers: dict[str, McpServerSpec] = dict(servers or {})
        self.default_timeout = default_timeout
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._caller_factory = caller_factory or default_caller_factory
        self._time_fn = time_fn
        self._callers: dict[str, ToolCaller] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def configure(self, servers: Mapping[str, McpServerSpec]) -> None:
        """(Re)configure the known servers (only source: ``app.yaml``)."""
        self._servers = dict(servers)

    def breaker_for(self, server: str) -> CircuitBreaker:
        breaker = self._breakers.get(server)
        if breaker is None:
            breaker = CircuitBreaker(
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                time_fn=self._time_fn,
                name=server,
            )
            self._breakers[server] = breaker
        return breaker

    def breaker_state(self, server: str) -> str:
        return self.breaker_for(server).state

    async def _get_caller(self, server: str) -> ToolCaller:
        async with self._lock:
            caller = self._callers.get(server)
            if caller is None:
                try:
                    spec = self._servers[server]
                except KeyError:
                    raise BackendError(f"unknown mcp server {server!r}") from None
                caller = self._caller_factory(server, spec)
                self._callers[server] = caller
            return caller

    async def _drop_caller(self, server: str) -> None:
        async with self._lock:
            caller = self._callers.pop(server, None)
        invalidate = getattr(caller, "__self__", None)
        if invalidate is not None:
            ainvalidate = getattr(invalidate, "ainvalidate", None)
            if callable(ainvalidate):
                try:
                    await ainvalidate()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("error invalidating MCP caller for %r: %s", server, exc)

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Call ``tool`` on ``server`` with breaker + timeout protection."""
        breaker = self.breaker_for(server)
        breaker.before_call()  # raises CircuitOpenError while open
        effective_timeout = timeout if timeout is not None else self.default_timeout
        try:
            caller = await self._get_caller(server)
        except BackendError:
            breaker.record_failure()
            raise
        try:
            result = await caller(tool, arguments, effective_timeout)
        except TimeoutError as exc:
            logger.warning(
                "mcp server %r tool %r timed out after %.1fs", server, tool, effective_timeout
            )
            breaker.record_failure()
            await self._drop_caller(server)  # reconnect on next call
            raise BackendError(
                f"mcp server {server!r} tool {tool!r} timed out after {effective_timeout}s"
            ) from exc
        except CircuitOpenError:
            raise
        except BackendError:
            breaker.record_failure()
            await self._drop_caller(server)
            raise
        except Exception as exc:
            logger.warning("mcp server %r tool %r failed: %s", server, tool, exc)
            breaker.record_failure()
            await self._drop_caller(server)  # reconnect on next call
            raise BackendError(f"mcp server {server!r} tool {tool!r} failed: {exc}") from exc
        breaker.record_success()
        return result

    async def aclose(self) -> None:
        """Drop all cached sessions (called on gateway shutdown)."""
        for server in list(self._callers):
            await self._drop_caller(server)


__all__ = [
    "DEFAULT_CALL_TIMEOUT_S",
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_RECOVERY_TIMEOUT_S",
    "CircuitBreaker",
    "CircuitOpenError",
    "McpConnectionPool",
    "ToolCaller",
    "CallerFactory",
    "default_caller_factory",
]
