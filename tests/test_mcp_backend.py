"""Task 6 tests: MCP client backend, pool, and circuit breaker."""

from __future__ import annotations

import asyncio
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from common.contracts import (
    AppConfig,
    Candle,
    InferenceConfig,
    McpServerSpec,
    ModelSpec,
    PredictRequest,
)
from inference.app.backends.base import BackendError, WindowTooSmallError
from inference.app.backends.mcp_backend import (
    McpBackend,
    ToolOutputError,
    build_tool_payload,
    parse_tool_result,
    summarize_candles,
)
from inference.app.main import create_app
from inference.app.mcp_pool import (
    CircuitBreaker,
    CircuitOpenError,
    McpConnectionPool,
)
from inference.app.registry import BackendRegistry
from tests.fake_mcp_server import FakeMcpServer, make_pool


def make_candles(n: int = 40, start: float = 100.0) -> list[Candle]:
    out = []
    price = start
    for i in range(n):
        price *= 1.001
        out.append(
            Candle(t=1710000000 + i * 300, o=price, h=price + 1, l=price - 1, c=price, v=10.0)
        )
    return out


def make_req(model: str = "remote-llm", n: int = 40) -> PredictRequest:
    return PredictRequest(pair="BTC/USDT", timeframe="5m", model=model, candles=make_candles(n))


def make_backend(fake: FakeMcpServer, **kwargs) -> McpBackend:  # type: ignore[no-untyped-def]
    spec = ModelSpec(backend="mcp", server="llm-tools", tool="predict")
    pool = make_pool(fake, **{k: v for k, v in kwargs.items() if k in ("recovery_timeout",)})
    backend_kwargs = {k: v for k, v in kwargs.items() if k not in ("recovery_timeout",)}
    return McpBackend(
        name="remote-llm",
        spec=spec,
        servers={"llm-tools": McpServerSpec(transport="stdio", command=["python", "-m", "x"])},
        pool=pool,
        **backend_kwargs,
    )


# -- payload: compact summary, no raw data by default ----------------------


def test_payload_is_compact_summary_without_raw_candles() -> None:
    payload = build_tool_payload(make_req())
    assert payload["pair"] == "BTC/USDT"
    assert payload["timeframe"] == "5m"
    assert "candles" not in payload
    summary = payload["summary"]
    assert summary["n_candles"] == 40
    assert summary["last_close"] > summary["first_close"]  # rising prices
    assert summary["momentum"] > 0


def test_payload_opt_in_raw_candles() -> None:
    payload = build_tool_payload(make_req(), include_raw_candles=True)
    assert len(payload["candles"]) == 40


def test_summarize_stats_sane() -> None:
    summary = summarize_candles(make_candles(10))
    assert summary["n_candles"] == 10
    assert summary["std_return"] >= 0
    assert summary["min_return"] <= summary["max_return"]
    assert summary["total_volume"] == pytest.approx(100.0)


# -- round-trip via the fake server ----------------------------------------


def test_fake_server_round_trips_prediction() -> None:
    fake = FakeMcpServer(mode="ok", signal=0.42, confidence=0.77)
    backend = make_backend(fake)
    resp = asyncio.run(backend.predict(make_req()))
    assert resp.backend == "mcp"
    assert resp.model == "remote-llm"
    assert resp.signal == pytest.approx(0.42)
    assert resp.confidence == pytest.approx(0.77)
    assert resp.latency_ms >= 0.0
    tool, arguments = fake.calls[0]
    assert tool == "predict"
    assert arguments["pair"] == "BTC/USDT"
    assert "candles" not in arguments  # compact by default


def test_out_of_range_values_are_clamped() -> None:
    fake = FakeMcpServer(mode="out_of_range")
    backend = make_backend(fake)
    resp = asyncio.run(backend.predict(make_req()))
    assert resp.signal == pytest.approx(1.0)
    assert resp.confidence == pytest.approx(0.0)


def test_parse_accepts_json_string_and_numeric_strings() -> None:
    signal, conf = parse_tool_result('{"signal": "0.5", "confidence": "0.25"}')
    assert (signal, conf) == pytest.approx((0.5, 0.25))


# -- non-conforming output never produces a signal --------------------------


@pytest.mark.parametrize("mode", ["free_text", "missing_fields", "bad_types", "non_finite"])
def test_non_conforming_output_raises_never_signals(mode: str) -> None:
    fake = FakeMcpServer(mode=mode)
    backend = make_backend(fake)
    with pytest.raises(BackendError):
        asyncio.run(backend.predict(make_req()))


def test_free_text_rejected_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="inference.app.backends.mcp_backend"):
        with pytest.raises(ToolOutputError):
            parse_tool_result("buy now, trust me")
    assert any("free text" in rec.message for rec in caplog.records)


def test_missing_tool_or_server_is_misconfigured() -> None:
    backend = McpBackend(name="bad", spec=ModelSpec(backend="mcp"))
    with pytest.raises(BackendError, match="misconfigured"):
        asyncio.run(backend.predict(make_req()))


def test_too_few_candles_422() -> None:
    fake = FakeMcpServer()
    backend = make_backend(fake)
    with pytest.raises(WindowTooSmallError):
        asyncio.run(backend.predict(make_req(n=1)))
    assert fake.calls == []  # rejected before any server contact


# -- circuit breaker: opens after 3 failures, recovers after 60 s -----------


def test_breaker_unit_opens_and_recovers() -> None:
    now = [1000.0]
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0, time_fn=lambda: now[0])
    assert breaker.state == "closed"
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        breaker.before_call()
    now[0] += 59.0
    assert breaker.state == "open"
    now[0] += 1.0  # 60 s elapsed -> half-open trial allowed
    breaker.before_call()
    breaker.record_success()
    assert breaker.state == "closed"


def test_pool_breaker_opens_after_3_consecutive_failures() -> None:
    fake = FakeMcpServer(mode="error")
    pool = make_pool(fake)
    req_args = {"pair": "BTC/USDT"}
    for _ in range(3):
        with pytest.raises(BackendError):
            asyncio.run(pool.call_tool("llm-tools", "predict", req_args))
    assert pool.breaker_state("llm-tools") == "open"
    calls_before = len(fake.calls)
    with pytest.raises(CircuitOpenError):
        asyncio.run(pool.call_tool("llm-tools", "predict", req_args))
    assert len(fake.calls) == calls_before  # rejected without server contact
    assert fake.calls  # earlier failures did reach the server


def test_pool_breaker_recovers_after_timeout() -> None:
    fake = FakeMcpServer(mode="error")
    pool = make_pool(fake, recovery_timeout=0.05)
    for _ in range(3):
        with pytest.raises(BackendError):
            asyncio.run(pool.call_tool("llm-tools", "predict", {"pair": "BTC/USDT"}))
    assert pool.breaker_state("llm-tools") == "open"
    time.sleep(0.06)
    fake.mode = "ok"  # server healthy again -> half-open trial succeeds
    result = asyncio.run(pool.call_tool("llm-tools", "predict", {"pair": "BTC/USDT"}))
    assert result == {"signal": 0.42, "confidence": 0.77}
    assert pool.breaker_state("llm-tools") == "closed"


def test_pool_success_resets_consecutive_failures() -> None:
    fake = FakeMcpServer(mode="error")
    pool = make_pool(fake)
    for _ in range(2):
        with pytest.raises(BackendError):
            asyncio.run(pool.call_tool("llm-tools", "predict", {}))
    fake.mode = "ok"
    asyncio.run(pool.call_tool("llm-tools", "predict", {}))
    fake.mode = "error"
    for _ in range(2):  # counter was reset: still closed after 2 more failures
        with pytest.raises(BackendError):
            asyncio.run(pool.call_tool("llm-tools", "predict", {}))
    assert pool.breaker_state("llm-tools") == "closed"


def test_pool_reconnects_on_failure() -> None:
    """A failed transport is dropped; the next call builds a fresh one."""
    created = []
    attempts = {"n": 0}

    def factory(name: str, spec: McpServerSpec):  # type: ignore[no-untyped-def]
        created.append(name)

        async def _call(tool: str, args: dict, timeout: float):  # type: ignore[no-untyped-def]
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("connection lost")
            return {"signal": 0.1, "confidence": 0.9}

        return _call

    pool = McpConnectionPool(
        servers={"llm-tools": McpServerSpec(transport="stdio", command=["x"])},
        caller_factory=factory,
    )
    with pytest.raises(BackendError):
        asyncio.run(pool.call_tool("llm-tools", "predict", {}))
    result = asyncio.run(pool.call_tool("llm-tools", "predict", {}))
    assert result == {"signal": 0.1, "confidence": 0.9}
    assert len(created) == 2  # fresh transport after the failure


# -- timeouts ---------------------------------------------------------------


def test_hung_call_hits_per_call_timeout() -> None:
    fake = FakeMcpServer(mode="hang")
    pool = make_pool(fake, default_timeout=0.05)
    start = time.perf_counter()
    with pytest.raises(BackendError, match="[Tt]imed out"):
        asyncio.run(pool.call_tool("llm-tools", "predict", {}))
    assert (time.perf_counter() - start) < 5.0


def test_backend_timeout_surfaces_as_backend_error() -> None:
    fake = FakeMcpServer(mode="hang")
    backend = make_backend(fake, timeout=0.05)
    with pytest.raises(BackendError):
        asyncio.run(backend.predict(make_req()))


# -- config-driven servers only ----------------------------------------------


def test_servers_come_only_from_app_yaml() -> None:
    data = yaml.safe_load(
        """
mcp:
  servers:
    llm-tools:
      transport: http
      url: https://example.com/mcp
inference:
  url: http://inference:8000
  default_model: remote-llm
  models:
    remote-llm: {backend: mcp, server: llm-tools, tool: predict}
"""
    )
    cfg = AppConfig.model_validate(data)
    assert cfg.mcp.servers["llm-tools"].transport == "http"
    reg = BackendRegistry.from_config(cfg, models_dir="/tmp")
    name, backend = reg.resolve("remote-llm")
    assert name == "remote-llm"
    assert isinstance(backend, McpBackend)
    assert backend.server_specs["llm-tools"].url == "https://example.com/mcp"


def test_unknown_server_is_502() -> None:
    spec = ModelSpec(backend="mcp", server="nope", tool="predict")
    pool = McpConnectionPool(servers={}, caller_factory=lambda n, s: FakeMcpServer().caller())
    backend = McpBackend(name="remote-llm", spec=spec, servers={}, pool=pool)
    with pytest.raises(BackendError, match="unknown mcp server"):
        asyncio.run(backend.predict(make_req()))


def test_missing_sdk_is_backend_error_not_import_error() -> None:
    """Without the ``mcp`` extra, predict fails cleanly (gateway -> 502)."""
    import inference.app.mcp_pool as pool_mod

    if pool_mod.default_caller_factory is None:  # pragma: no cover - sanity
        pytest.skip("no factory")
    try:
        import mcp  # noqa: F401
    except ImportError:
        backend = McpBackend(
            name="remote-llm",
            spec=ModelSpec(backend="mcp", server="llm-tools", tool="predict"),
            servers={"llm-tools": McpServerSpec(transport="stdio", command=["python", "-m", "x"])},
        )
        with pytest.raises(BackendError):
            asyncio.run(backend.predict(make_req()))
    else:
        pytest.skip("mcp SDK installed; SDK-missing path not exercisable")


# -- gateway wiring ----------------------------------------------------------


def _gateway_app_with_fake(fake: FakeMcpServer) -> TestClient:
    cfg = AppConfig(
        inference=InferenceConfig(
            default_model="remote-llm",
            models={"remote-llm": ModelSpec(backend="mcp", server="llm-tools", tool="predict")},
        )
    )
    app = create_app(cfg)
    backend = make_backend(fake)
    app.state.registry.register("remote-llm", backend)  # type: ignore[arg-type]
    return TestClient(app)


def test_gateway_round_trip_labels_backend_mcp() -> None:
    client = _gateway_app_with_fake(FakeMcpServer(mode="ok", signal=0.3, confidence=0.9))
    candles = [
        {
            "t": 1710000000 + i * 300,
            "o": 100.0,
            "h": 101.0,
            "l": 99.0,
            "c": 100.0 + i,
            "v": 10.0,
        }
        for i in range(40)
    ]
    r = client.post(
        "/v1/predict",
        json={"pair": "BTC/USDT", "timeframe": "5m", "model": "remote-llm", "candles": candles},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == "mcp"
    assert body["signal"] == pytest.approx(0.3)


def test_gateway_backend_failure_is_502() -> None:
    client = _gateway_app_with_fake(FakeMcpServer(mode="error"))
    candles = [
        {
            "t": 1710000000 + i * 300,
            "o": 100.0,
            "h": 101.0,
            "l": 99.0,
            "c": 100.0,
            "v": 10.0,
        }
        for i in range(40)
    ]
    r = client.post(
        "/v1/predict",
        json={"pair": "BTC/USDT", "timeframe": "5m", "model": "remote-llm", "candles": candles},
    )
    assert r.status_code == 502


def test_gateway_too_few_candles_is_422() -> None:
    client = _gateway_app_with_fake(FakeMcpServer())
    r = client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": "5m",
            "model": "remote-llm",
            "candles": [
                {"t": 1710000000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0},
            ],
        },
    )
    assert r.status_code == 422
