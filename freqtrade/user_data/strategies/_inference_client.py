"""HTTP client for the inference gateway (``POST /v1/predict``).

Shared by :mod:`AiSignalStrategy` (Task 3) and :mod:`HybridStrategy`
(Task 11). The client never raises into the bot loop: every failure mode
(gateway down, timeout, malformed response, empty candles) is logged and
reported as ``None`` so the strategy can fall back to its pure-indicator
rule.

Contract (see SPEC.md — Shared Contracts)::

    POST /v1/predict
      Request:  {"pair", "timeframe", "model", "candles": [{t,o,h,l,c,v}]}
      Response: {"signal" (-1..1), "confidence" (0..1),
                 "model", "backend", "latency_ms"}

Policy: 2 s timeout, 1 retry on transport errors/timeouts and on
502/503/504. No retry on 404/422 (the request itself is wrong).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:  # httpx is installed in the freqtrade image (see freqtrade/Dockerfile)
    import httpx
except ImportError:  # pragma: no cover - unit-test env without httpx
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PREDICT_PATH = "/v1/predict"
REQUEST_TIMEOUT = 2.0
MAX_RETRIES = 1  # one retry => at most two attempts
DEFAULT_N_CANDLES = 50
RETRYABLE_STATUS = {502, 503, 504}


@dataclass(frozen=True)
class InferenceResult:
    """A validated ``/v1/predict`` response."""

    signal: float
    confidence: float
    model: str
    backend: str
    latency_ms: float


def dataframe_to_candles(dataframe: Any, limit: int = DEFAULT_N_CANDLES) -> list[dict[str, Any]]:
    """Convert the last ``limit`` rows of a Freqtrade OHLCV dataframe.

    Expected columns are ``date, open, high, low, close, volume`` (the
    standard Freqtrade names). Returns ``[]`` when the frame is empty so
    callers can skip the gateway call (edge case: empty candles).
    """
    if dataframe is None or len(dataframe) == 0:
        return []
    frame = dataframe.tail(limit)
    candles: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        date = row.get("date") if hasattr(row, "get") else row["date"]
        try:
            ts = int(date.timestamp()) if hasattr(date, "timestamp") else int(date)  # type: ignore[arg-type]
            candle = {
                "t": ts,
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "v": float(row["volume"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed candle row: %s", exc)
            continue
        candles.append(candle)
    return candles


def build_predict_payload(
    pair: str,
    candles: list[dict[str, Any]],
    timeframe: str = "5m",
    model: str = "local-gru",
) -> dict[str, Any]:
    """Build a ``POST /v1/predict`` JSON body."""
    return {"pair": pair, "timeframe": timeframe, "model": model, "candles": candles}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_predict_response(data: Any) -> InferenceResult | None:
    """Validate a decoded ``/v1/predict`` body.

    Returns ``None`` for malformed payloads (missing/non-numeric fields).
    Slightly out-of-range ``signal``/``confidence`` floats are clamped
    into range instead of rejected.
    """
    if not isinstance(data, dict):
        logger.warning("Malformed predict response: not a JSON object: %r", data)
        return None
    try:
        signal = float(data["signal"])
        confidence = float(data["confidence"])
        model = str(data["model"])
        backend = str(data["backend"])
        latency_ms = float(data.get("latency_ms", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Malformed predict response (%s): %r", exc, data)
        return None
    if backend not in ("local", "mcp"):
        logger.warning("Unknown backend in predict response: %r", data)
        return None
    if signal != _clamp(signal, -1.0, 1.0):
        logger.debug("Clamping out-of-range signal %r", signal)
    if confidence != _clamp(confidence, 0.0, 1.0):
        logger.debug("Clamping out-of-range confidence %r", confidence)
    return InferenceResult(
        signal=_clamp(signal, -1.0, 1.0),
        confidence=_clamp(confidence, 0.0, 1.0),
        model=model,
        backend=backend,
        latency_ms=latency_ms,
    )


def query_inference_gateway(
    pair: str,
    candles: list[dict[str, Any]] | Any,
    timeframe: str = "5m",
    model: str = "local-gru",
    url: str = "http://inference:8000",
    timeout: float = REQUEST_TIMEOUT,
) -> InferenceResult | None:
    """Ask the gateway for a signal, returning ``None`` on any failure.

    Never raises: transport errors, timeouts, HTTP error statuses and
    malformed bodies are logged (warning) and mapped to ``None`` so the
    caller can use its fallback rule. ``candles`` may be a ready-made
    list of ``{t,o,h,l,c,v}`` dicts or a dataframe (converted with the
    default window).
    """
    if candles is None or (hasattr(candles, "__len__") and len(candles) == 0):
        logger.warning("No candles for %s — skipping gateway call", pair)
        return None
    if not isinstance(candles, list):
        candles = dataframe_to_candles(candles)
        if not candles:
            logger.warning("No usable candles for %s — skipping gateway call", pair)
            return None
    if httpx is None:  # pragma: no cover - env without httpx
        logger.warning("httpx is not installed — cannot query gateway for %s", pair)
        return None

    payload = build_predict_payload(pair, candles, timeframe=timeframe, model=model)
    endpoint = url.rstrip("/") + PREDICT_PATH

    attempts = 1 + MAX_RETRIES
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.post(endpoint, json=payload, timeout=timeout)
        except Exception as exc:  # timeout, DNS, connection refused, ...
            if attempt < attempts:
                logger.warning(
                    "Gateway call failed (attempt %d/%d) for %s: %s — retrying",
                    attempt,
                    attempts,
                    pair,
                    exc,
                )
                continue
            logger.warning("Inference gateway unreachable for %s: %s", pair, exc)
            return None
        if response.status_code in RETRYABLE_STATUS and attempt < attempts:
            logger.warning(
                "Gateway returned %d for %s (attempt %d/%d) — retrying",
                response.status_code,
                pair,
                attempt,
                attempts,
            )
            continue
        if response.status_code != 200:
            logger.warning(
                "Gateway returned HTTP %d for %s: %s",
                response.status_code,
                pair,
                response.text[:200],
            )
            return None
        try:
            data = response.json()
        except ValueError as exc:
            logger.warning("Gateway returned invalid JSON for %s: %s", pair, exc)
            return None
        result = parse_predict_response(data)
        if result is None:
            logger.warning("Gateway response for %s was unusable — using fallback", pair)
        return result
    return None  # pragma: no cover - loop always returns
