"""Task 5 tests: routing and error paths of the inference gateway."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from common.contracts import (
    AppConfig,
    Candle,
    InferenceConfig,
    ModelSpec,
    PredictRequest,
    PredictResponse,
)
from inference.app.backends.base import BackendError, WindowTooSmallError
from inference.app.backends.local_torch import LocalTorchBackend
from inference.app.main import create_app, default_config
from inference.app.registry import BackendRegistry, UnknownModelError


def make_candles(n: int, start: float = 100.0) -> list[dict]:
    out = []
    price = start
    for i in range(n):
        price *= 1.001
        out.append(
            {
                "t": 1710000000 + i * 300,
                "o": price,
                "h": price + 1,
                "l": price - 1,
                "c": price,
                "v": 10.0,
            }
        )
    return out


def make_req(model: str = "local-gru", n: int = 40) -> PredictRequest:
    return PredictRequest(
        pair="BTC/USDT",
        timeframe="5m",
        model=model,
        candles=[Candle(**c) for c in make_candles(n)],
    )


class FakeBackend:
    def __init__(self, signal: float = 0.5, confidence: float = 0.8) -> None:
        self.signal = signal
        self.confidence = confidence

    async def predict(self, req: PredictRequest) -> PredictResponse:
        return PredictResponse(
            signal=self.signal,
            confidence=self.confidence,
            model=req.model,
            backend="local",
            latency_ms=1.0,
        )


class FailingBackend:
    async def predict(self, req: PredictRequest) -> PredictResponse:
        raise BackendError("boom")


def make_app() -> TestClient:
    return TestClient(create_app(default_config()))


# -- registry -----------------------------------------------------------


def test_registry_routes_default_fallback() -> None:
    reg = BackendRegistry.from_config(default_config())
    name, _ = reg.resolve("")
    assert name == "local-gru"


def test_registry_unknown_model_raises() -> None:
    reg = BackendRegistry.from_config(default_config())
    with pytest.raises(UnknownModelError):
        reg.resolve("nope")


def test_registry_mcp_model_without_sdk_is_502() -> None:
    cfg = AppConfig(
        inference=InferenceConfig(
            default_model="local-gru",
            models={
                "local-gru": ModelSpec(backend="local", device="cpu"),
                "remote-llm": ModelSpec(backend="mcp", server="llm-tools", tool="predict"),
            },
        )
    )
    reg = BackendRegistry.from_config(cfg, models_dir="/tmp")
    name, backend = reg.resolve("remote-llm")
    assert name == "remote-llm"
    with pytest.raises(BackendError):
        asyncio.run(backend.predict(make_req(model="remote-llm")))


# -- local backend (no torch on CPU CI) ----------------------------------


def test_local_backend_heuristic_contract() -> None:
    backend = LocalTorchBackend(model_name="test-gru", device="cpu", models_dir="/tmp")
    resp = asyncio.run(backend.predict(make_req(model="test-gru")))
    assert resp.backend == "local"
    assert -1.0 <= resp.signal <= 1.0
    assert 0.0 <= resp.confidence <= 1.0
    assert resp.latency_ms >= 0.0


def test_local_backend_too_few_candles() -> None:
    backend = LocalTorchBackend(model_name="test-gru", device="cpu", models_dir="/tmp")
    with pytest.raises(WindowTooSmallError):
        asyncio.run(backend.predict(make_req(n=5)))


# -- HTTP contract -------------------------------------------------------


def test_predict_matches_contract() -> None:
    client = make_app()
    r = client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": "5m",
            "model": "local-gru",
            "candles": make_candles(40),
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"signal", "confidence", "model", "backend", "latency_ms"}
    assert -1.0 <= body["signal"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model"] == "local-gru"
    assert body["backend"] == "local"


def test_predict_unknown_model_404() -> None:
    client = make_app()
    r = client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": "5m",
            "model": "nope",
            "candles": make_candles(40),
        },
    )
    assert r.status_code == 404


def test_predict_too_few_candles_422() -> None:
    client = make_app()
    r = client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": "5m",
            "model": "local-gru",
            "candles": make_candles(2),
        },
    )
    assert r.status_code == 422


def test_predict_invalid_payload_422() -> None:
    client = make_app()
    r = client.post("/v1/predict", json={"pair": "BTC/USDT", "candles": []})
    assert r.status_code == 422


def test_predict_backend_failure_502() -> None:
    app = create_app(default_config())
    app.state.registry.register("bad", FailingBackend())  # type: ignore[arg-type]
    client = TestClient(app)
    r = client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": "5m",
            "model": "bad",
            "candles": make_candles(40),
        },
    )
    assert r.status_code == 502


def test_predict_routing_to_fake_backend() -> None:
    app = create_app(default_config())
    app.state.registry.register("fake", FakeBackend(signal=0.9, confidence=0.95))  # type: ignore[arg-type]
    client = TestClient(app)
    r = client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": "5m",
            "model": "fake",
            "candles": make_candles(40),
        },
    )
    assert r.status_code == 200
    assert r.json()["signal"] == pytest.approx(0.9)


def test_healthz_reports_gpu_and_models() -> None:
    client = make_app()
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["gpu"], bool)
    assert "local-gru" in body["models"]


def test_metrics_prometheus_format() -> None:
    client = make_app()
    client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": "5m",
            "model": "local-gru",
            "candles": make_candles(40),
        },
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "gateway_predict_requests_total" in r.text
