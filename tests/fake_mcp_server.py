"""Fake MCP server harness for Task 6 tests (no ``mcp`` SDK required).

:class:`FakeMcpServer` mimics an MCP server's ``call_tool`` surface with
switchable behaviours (valid JSON, free text, bad schema, errors, hangs),
so the pool, breaker, and backend can be exercised end to end without any
external process.
"""

from __future__ import annotations

import asyncio
from typing import Any

from common.contracts import McpServerSpec
from inference.app.mcp_pool import McpConnectionPool


class FakeMcpServer:
    """In-memory stand-in for one MCP server."""

    def __init__(
        self,
        mode: str = "ok",
        signal: float = 0.42,
        confidence: float = 0.77,
        delay: float = 0.0,
    ) -> None:
        self.mode = mode
        self.signal = signal
        self.confidence = confidence
        self.delay = delay
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool: str, arguments: dict[str, Any], timeout: float) -> Any:
        self.calls.append((tool, arguments))

        async def _respond() -> Any:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.mode == "ok":
                return {"signal": self.signal, "confidence": self.confidence}
            if self.mode == "out_of_range":
                return {"signal": 5.0, "confidence": -3.0}
            if self.mode == "free_text":
                return "The market looks bullish today, consider buying soon!"
            if self.mode == "missing_fields":
                return {"signal": 0.5}
            if self.mode == "bad_types":
                return {"signal": "bullish", "confidence": [0.9]}
            if self.mode == "non_finite":
                return {"signal": float("nan"), "confidence": float("inf")}
            if self.mode == "error":
                raise RuntimeError("fake server exploded")
            if self.mode == "hang":
                await asyncio.sleep(timeout + 5.0)
                return {"signal": 0.0, "confidence": 0.0}  # unreachable
            raise AssertionError(f"unknown fake mode {self.mode!r}")

        return await asyncio.wait_for(_respond(), timeout=timeout)

    def caller(self):  # type: ignore[no-untyped-def]
        async def _call(tool: str, arguments: dict[str, Any], timeout: float) -> Any:
            return await self.call_tool(tool, arguments, timeout)

        return _call


def make_pool(
    fake: FakeMcpServer,
    server_name: str = "llm-tools",
    transport: str = "stdio",
    **pool_kwargs: Any,
) -> McpConnectionPool:
    """Pool wired to ``fake`` for ``server_name`` (spec still config-driven)."""
    spec = (
        McpServerSpec(transport="stdio", command=["python", "-m", "fake_mcp_server"])
        if transport == "stdio"
        else McpServerSpec(transport="http", url="http://fake-mcp:8000/mcp")
    )
    return McpConnectionPool(
        servers={server_name: spec},
        caller_factory=lambda _name, _spec: fake.caller(),
        **pool_kwargs,
    )


__all__ = ["FakeMcpServer", "make_pool"]
