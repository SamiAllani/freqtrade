"""MCP client backend for external models (Task 6).

Delegates predictions to an external model exposed through an MCP server.
Only the numeric ``signal``/``confidence`` fields of the tool result may
influence trading decisions; everything else is treated as untrusted input
and validated strictly (free text or non-conforming JSON is rejected and
logged, never turned into a signal).

The connection pool and circuit breaker live in
:mod:`inference.app.mcp_pool` and are re-exported here so the whole Task 6
surface (backend + pool + breaker) is available from this module.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from common.contracts import Candle, McpServerSpec, ModelSpec, PredictRequest, PredictResponse

from ..mcp_pool import (
    DEFAULT_CALL_TIMEOUT_S,
    CallerFactory,
    CircuitBreaker,
    CircuitOpenError,
    McpConnectionPool,
)
from .base import BackendError, WindowTooSmallError

logger = logging.getLogger(__name__)

#: Minimum candles needed to compute return statistics (one return minimum).
MIN_CANDLES = 2

__all__ = [
    "McpBackend",
    "ToolOutputError",
    "build_tool_payload",
    "summarize_candles",
    "parse_tool_result",
    "clamp_signal",
    "clamp_confidence",
    "McpConnectionPool",
    "CircuitBreaker",
    "CircuitOpenError",
    "DEFAULT_CALL_TIMEOUT_S",
    "MIN_CANDLES",
]


class ToolOutputError(BackendError):
    """An MCP tool returned non-conforming output. Never becomes a signal."""


def clamp_signal(value: float) -> float:
    """Clamp a signal into the contract range ``[-1.0, 1.0]``."""
    return max(-1.0, min(1.0, float(value)))


def clamp_confidence(value: float) -> float:
    """Clamp a confidence into the contract range ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(value)))


def _returns(closes: Sequence[float]) -> list[float]:
    """Simple returns r[i] = c[i]/c[i-1] - 1, guarded against zero prices."""
    out: list[float] = []
    for prev, cur in zip(closes[:-1], closes[1:], strict=True):
        out.append(0.0 if prev == 0 else cur / prev - 1.0)
    return out


def summarize_candles(candles: Sequence[Candle], window: int | None = None) -> dict[str, Any]:
    """Compact statistics over the most recent ``window`` candles.

    Only aggregates are returned — never raw OHLCV series — so the tool
    payload stays small and model-agnostic.
    """
    selected = list(candles[-window:] if window is not None else candles)
    closes = [c.c for c in selected]
    highs = [c.h for c in selected]
    lows = [c.l for c in selected]
    volumes = [c.v for c in selected]
    rets = _returns(closes)
    n = len(rets)
    mean = sum(rets) / n if n else 0.0
    var = sum((r - mean) ** 2 for r in rets) / n if n else 0.0
    std = math.sqrt(var)
    first, last = closes[0], closes[-1]
    return {
        "n_candles": len(selected),
        "start_t": selected[0].t,
        "end_t": selected[-1].t,
        "first_close": first,
        "last_close": last,
        "high": max(highs),
        "low": min(lows),
        "total_volume": sum(volumes),
        "mean_volume": sum(volumes) / len(volumes),
        "momentum": (last / first - 1.0) if first != 0 else 0.0,
        "mean_return": mean,
        "std_return": std,
        "min_return": min(rets) if rets else 0.0,
        "max_return": max(rets) if rets else 0.0,
        "last_return": rets[-1] if rets else 0.0,
    }


def build_tool_payload(
    req: PredictRequest,
    include_raw_candles: bool = False,
    summary_window: int | None = None,
) -> dict[str, Any]:
    """Build the compact tool payload for ``req``.

    Raw candles are excluded by default; pass ``include_raw_candles=True``
    only for tools that explicitly need the full series.
    """
    payload: dict[str, Any] = {
        "pair": req.pair,
        "timeframe": req.timeframe,
        "model": req.model,
        "summary": summarize_candles(req.candles, window=summary_window),
    }
    if include_raw_candles:
        payload["candles"] = [c.model_dump() for c in req.candles]
    return payload


def _coerce_number(value: Any, field: str) -> float:
    """Coerce ``value`` to a finite float; reject bools/non-numerics."""
    if isinstance(value, bool):
        raise ToolOutputError(f"tool field {field!r} must be numeric, got bool")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise ToolOutputError(f"tool field {field!r} is not numeric: {value!r}") from None
    else:
        raise ToolOutputError(f"tool field {field!r} must be numeric, got {type(value).__name__}")
    if not math.isfinite(number):
        raise ToolOutputError(f"tool field {field!r} must be finite, got {value!r}")
    return number


def _extract_json_dict(raw: Any) -> dict[str, Any]:
    """Unwrap SDK result objects / JSON text into a plain dict.

    Accepts real MCP ``CallToolResult`` objects (``structuredContent`` or a
    ``content`` block list), JSON strings, and plain dicts. Anything else —
    free text, empty content, binary blocks — raises :class:`ToolOutputError`.
    """
    # Official SDK CallToolResult: prefer structured content when present.
    structured = getattr(raw, "structuredContent", None)
    if isinstance(structured, dict) and structured:
        return structured
    content = getattr(raw, "content", None)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        texts = [getattr(block, "text", None) for block in content]
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            logger.warning("mcp tool returned no text content; rejecting")
            raise ToolOutputError("mcp tool returned no text content")
        for text in texts:
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        logger.warning("mcp tool returned free text, not JSON; rejecting")
        raise ToolOutputError("mcp tool returned free text, not JSON")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("mcp tool returned free text, not JSON; rejecting")
            raise ToolOutputError("mcp tool returned free text, not JSON") from None
        if not isinstance(parsed, dict):
            raise ToolOutputError("mcp tool result JSON must be an object")
        return parsed
    if isinstance(raw, Mapping):
        return dict(raw)
    logger.warning("mcp tool returned type %s, not JSON; rejecting", type(raw).__name__)
    raise ToolOutputError(f"mcp tool returned non-conforming output ({type(raw).__name__})")


def parse_tool_result(raw: Any) -> tuple[float, float]:
    """Extract and clamp ``(signal, confidence)`` from a raw tool result.

    Raises :class:`ToolOutputError` for anything that is not an object with
    numeric ``signal`` and ``confidence`` fields. Out-of-range values are
    clamped, never rejected.
    """
    data = _extract_json_dict(raw)
    missing = [k for k in ("signal", "confidence") if k not in data]
    if missing:
        logger.warning("mcp tool result missing fields %s; rejecting", missing)
        raise ToolOutputError(f"mcp tool result missing fields: {missing}")
    try:
        signal = _coerce_number(data["signal"], "signal")
        confidence = _coerce_number(data["confidence"], "confidence")
    except ToolOutputError:
        logger.warning("mcp tool result has non-numeric signal/confidence; rejecting")
        raise
    return clamp_signal(signal), clamp_confidence(confidence)


def _load_server_specs() -> dict[str, McpServerSpec]:
    """Lazily load ``mcp.servers`` from the central ``app.yaml``.

    Used only when the backend was built without explicit server specs
    (e.g. direct construction); the registry always passes them explicitly.
    Failures yield an empty mapping so the backend reports ``unknown mcp
    server`` at predict time (HTTP 502) instead of crashing at import.
    """
    import os
    from pathlib import Path

    candidate = Path(os.environ.get("APP_CONFIG_PATH", "config/app.yaml"))
    if not candidate.is_absolute():
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            if (parent / "config" / "app.yaml.example").exists():
                candidate = parent / candidate
                break
    if not candidate.exists():
        return {}
    try:
        import yaml

        from common.contracts import AppConfig

        data = yaml.safe_load(candidate.read_text()) or {}
        return dict(AppConfig.model_validate(data).mcp.servers)
    except Exception as exc:
        logger.warning("could not load mcp servers from %s: %s", candidate, exc)
        return {}


class McpBackend:
    """``mcp`` backend: predict via a configured MCP tool.

    Servers come only from ``app.yaml`` (``mcp.servers``), passed either via
    ``servers=`` or loaded lazily from the central config file.
    """

    backend_name = "mcp"

    def __init__(
        self,
        name: str | None = None,
        spec: ModelSpec | None = None,
        model_name: str | None = None,
        server: str | None = None,
        tool: str | None = None,
        servers: Mapping[str, McpServerSpec] | None = None,
        pool: McpConnectionPool | None = None,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,
        include_raw_candles: bool = False,
        summary_window: int | None = None,
        caller_factory: CallerFactory | None = None,
    ) -> None:
        """Resolve server/tool settings and build the connection pool."""
        resolved_name = name or model_name or (spec.tool if spec else None) or "mcp"
        self.model_name = resolved_name
        self.server = server or (spec.server if spec else None)
        self.tool = tool or (spec.tool if spec else None)
        self.timeout = timeout
        self.include_raw_candles = include_raw_candles
        self.summary_window = summary_window
        self._servers: dict[str, McpServerSpec] | None = (
            dict(servers) if servers is not None else None
        )
        if pool is not None:
            self._pool = pool
        else:
            self._pool = McpConnectionPool(
                servers=self._servers or {},
                default_timeout=timeout,
                caller_factory=caller_factory,
            )
            if self._servers is None:
                self._pool.configure({})  # real specs resolved lazily per call

    @property
    def server_specs(self) -> dict[str, McpServerSpec]:
        """Server specs in effect (explicit, else central ``app.yaml``)."""
        if self._servers is not None:
            return self._servers
        loaded = _load_server_specs()
        self._servers = loaded
        self._pool.configure(loaded)
        return loaded

    def _ensure_specs(self) -> None:
        """Trigger the lazy ``app.yaml`` server-spec load if not provided."""
        if self._servers is None:
            _ = self.server_specs  # triggers lazy load + pool.configure

    async def predict(self, req: PredictRequest) -> PredictResponse:
        """Call the configured MCP tool and map its result to a prediction."""
        start = time.perf_counter()
        if not self.server or not self.tool:
            raise BackendError(
                f"mcp backend for model {self.model_name!r} is misconfigured: "
                "ModelSpec needs 'server' and 'tool'"
            )
        if len(req.candles) < MIN_CANDLES:
            raise WindowTooSmallError(
                f"mcp model {self.model_name!r} needs at least {MIN_CANDLES} candles, "
                f"got {len(req.candles)}"
            )
        self._ensure_specs()
        payload = build_tool_payload(
            req,
            include_raw_candles=self.include_raw_candles,
            summary_window=self.summary_window,
        )
        raw = await self._pool.call_tool(self.server, self.tool, payload, timeout=self.timeout)
        signal, confidence = parse_tool_result(raw)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return PredictResponse(
            signal=signal,
            confidence=confidence,
            model=req.model or self.model_name,
            backend="mcp",
            latency_ms=latency_ms,
        )
