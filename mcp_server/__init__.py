"""Bot monitoring/control MCP server (Task 7)."""

from mcp_server.ft_client import (
    AuthError,
    BotUnreachableError,
    FreqtradeApiError,
    FreqtradeClient,
    InvalidTradeIdError,
)
from mcp_server.server import (
    READ_TOOL_NAMES,
    SERVER_NAME,
    WRITE_TOOL_NAMES,
    create_server,
    is_write_enabled,
)

__all__ = [
    "FreqtradeClient",
    "FreqtradeApiError",
    "BotUnreachableError",
    "AuthError",
    "InvalidTradeIdError",
    "create_server",
    "is_write_enabled",
    "READ_TOOL_NAMES",
    "WRITE_TOOL_NAMES",
    "SERVER_NAME",
]
