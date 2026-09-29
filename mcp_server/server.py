"""Bot monitoring/control MCP server (Task 7).

Read-mostly toolset over the Freqtrade REST API so MCP-capable assistants
can inspect the bot:

- ``get_status`` — bot status (open trades)
- ``get_open_trades`` — open trades
- ``get_profit`` — profit/loss summary
- ``get_balance`` — account balance per currency
- ``get_performance`` — per-pair performance

Write tools are registered **only** when ``MCP_ALLOW_WRITE=yes``:

- ``pause_trading`` — stop opening new trades (open trades keep running)
- ``force_exit(trade_id)`` — instantly exit one open trade

No tool opens new positions.

Run over stdio (Docker ``mcp-server`` service, or any MCP client)::

    MCP_ALLOW_WRITE=yes python -m mcp_server.server
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from .ft_client import FreqtradeApiError, FreqtradeClient

logger = logging.getLogger(__name__)

SERVER_NAME = "freqtrade-bot"

READ_TOOL_NAMES = (
    "get_status",
    "get_open_trades",
    "get_profit",
    "get_balance",
    "get_performance",
)

WRITE_TOOL_NAMES = (
    "pause_trading",
    "force_exit",
)

_WRITE_TRUTHY = {"1", "yes", "true", "on"}

__all__ = [
    "create_server",
    "is_write_enabled",
    "main",
    "READ_TOOL_NAMES",
    "WRITE_TOOL_NAMES",
    "SERVER_NAME",
]


def is_write_enabled() -> bool:
    """Write tools exist only when ``MCP_ALLOW_WRITE=yes`` (or 1/true/on)."""
    return os.environ.get("MCP_ALLOW_WRITE", "").strip().lower() in _WRITE_TRUTHY


def _error_result(exc: Exception) -> dict[str, Any]:
    logger.warning("bot tool failed: %s", exc)
    return {"ok": False, "error": str(exc)}


def create_server(client: FreqtradeClient | None = None) -> FastMCP:
    """Build the MCP server; write tools are added only if enabled."""
    mcp = FastMCP(
        SERVER_NAME,
        instructions=(
            "Read-mostly Freqtrade bot inspector. "
            "Use the get_* tools to report state; "
            "pause_trading/force_exit only exist when writes are enabled, "
            "and no tool can open new positions."
        ),
    )
    ft = client if client is not None else FreqtradeClient.from_env()

    @mcp.tool()
    def get_status() -> dict[str, Any]:
        """Bot status: currently open trades with profit and entry info."""
        try:
            # Wrapped in a dict so the MCP result is a single JSON object
            # (FastMCP turns each element of a bare list into its own block).
            return {"status": ft.get_status()}
        except FreqtradeApiError as exc:
            return _error_result(exc)

    @mcp.tool()
    def get_open_trades() -> dict[str, Any]:
        """List all currently open trades."""
        try:
            trades = ft.get_open_trades()
            if isinstance(trades, list):
                return {"open_trades": trades, "count": len(trades)}
            return {"open_trades": trades, "count": 0}
        except FreqtradeApiError as exc:
            return _error_result(exc)

    @mcp.tool()
    def get_profit() -> Any:
        """Profit/loss summary over closed trades plus performance stats."""
        try:
            return ft.get_profit()
        except FreqtradeApiError as exc:
            return _error_result(exc)

    @mcp.tool()
    def get_balance() -> Any:
        """Account balance per currency."""
        try:
            return ft.get_balance()
        except FreqtradeApiError as exc:
            return _error_result(exc)

    @mcp.tool()
    def get_performance() -> dict[str, Any]:
        """Per-pair performance of finished trades."""
        try:
            return {"performance": ft.get_performance()}
        except FreqtradeApiError as exc:
            return _error_result(exc)

    if is_write_enabled():

        @mcp.tool()
        def pause_trading() -> Any:
            """Stop opening new trades; open trades keep running their rules."""
            try:
                return ft.pause_trading()
            except FreqtradeApiError as exc:
                return _error_result(exc)

        @mcp.tool()
        def force_exit(trade_id: str) -> Any:
            """Instantly exit one open trade, ignoring minimum_roi.

            Args:
                trade_id: id of the open trade to exit (as shown by
                    get_open_trades).
            """
            try:
                return ft.force_exit(trade_id)
            except FreqtradeApiError as exc:
                return _error_result(exc)

    return mcp


def main() -> None:
    """Run the server over stdio (default transport)."""
    create_server().run()


if __name__ == "__main__":
    main()
