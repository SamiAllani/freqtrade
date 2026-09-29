"""Re-export of :mod:`inference.app.mcp_pool` for Task 6 callers.

The canonical pool/breaker implementation lives in
``inference/app/mcp_pool.py`` (per SPEC.md); this module re-exports it so
``inference.app.backends.mcp_pool`` works too.
"""

from ..mcp_pool import (
    DEFAULT_CALL_TIMEOUT_S,
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_RECOVERY_TIMEOUT_S,
    CallerFactory,
    CircuitBreaker,
    CircuitOpenError,
    McpConnectionPool,
    ToolCaller,
    default_caller_factory,
)

__all__ = [
    "DEFAULT_CALL_TIMEOUT_S",
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_RECOVERY_TIMEOUT_S",
    "CallerFactory",
    "CircuitBreaker",
    "CircuitOpenError",
    "McpConnectionPool",
    "ToolCaller",
    "default_caller_factory",
]
