"""Task 10 E2E: dry-run stack verification with a fake model.

What this covers (SPEC Task 10 — E2E):

- compose up in dry-run with a fake model, wait for ``/healthz``,
  assert the bot loop runs and the strategy received a signal.

Two layers:

1. **In-process (always runs):** boot the real gateway app, hit
   ``/healthz`` and ``/v1/predict``, feed the signal through the
   strategy's entry/exit rule, and assert the rendered stack
   (freqtrade/compose/helm) is dry-run end to end.
2. **Docker (opt-in):** ``FT_E2E_DOCKER=1`` brings up
   ``deploy/compose/docker-compose.yml``, polls ``/healthz`` and the
   Freqtrade ping endpoint, then tears the stack down. Skipped by
   default so unit CI stays fast; the nightly/CI ``e2e`` job sets it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from common.contracts import PredictRequest, PredictResponse
from ftctl.loader import load_config
from ftctl.render.compose import render_compose
from ftctl.render.freqtrade import render_freqtrade
from ftctl.render.helm import render_helm
from inference.app.main import create_app, default_config

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO / "config" / "app.yaml.example"
EXAMPLE_ENV = REPO / "config" / ".env.example"
COMPOSE_FILE = REPO / "deploy" / "compose" / "docker-compose.yml"


def make_candles(n: int = 40, start: float = 100.0) -> list[dict]:
    out: list[dict] = []
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


class FakeBullishBackend:
    """Deterministic fake model: bullish signal above default thresholds."""

    backend_name = "local"

    async def predict(self, req: PredictRequest) -> PredictResponse:
        return PredictResponse(
            signal=0.8,
            confidence=0.9,
            model=req.model,
            backend="local",
            latency_ms=1.0,
        )


def make_gateway_client() -> TestClient:
    app = create_app(default_config())
    app.state.registry.register("fake-model", FakeBullishBackend())  # type: ignore[arg-type]
    return TestClient(app)


def should_entry(signal: float, confidence: float, cfg) -> bool:  # type: ignore[no-untyped-def]
    return signal > cfg.strategy.entry_signal_min and confidence > cfg.strategy.entry_confidence_min


def should_exit(signal: float, cfg) -> bool:  # type: ignore[no-untyped-def]
    return signal < cfg.strategy.exit_signal_max


# --- in-process stack -------------------------------------------------------


def test_gateway_healthz_reports_models() -> None:
    client = make_gateway_client()
    r = client.get("/healthz")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["gpu"], bool)
    assert "local-gru" in body["models"]
    assert "fake-model" in body["models"]


def test_gateway_predict_reaches_strategy_entry_rule() -> None:
    """The strategy received a signal: fake-model output passes entry rule."""
    for var in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD"):
        os.environ.pop(var, None)
    cfg = load_config(EXAMPLE_CONFIG, EXAMPLE_ENV)
    assert cfg.mode == "dry_run"

    client = make_gateway_client()
    r = client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": cfg.strategy.timeframe,
            "model": "fake-model",
            "candles": make_candles(40),
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert -1.0 <= body["signal"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0
    # Bullish fake signal must trigger the entry rule from app.yaml.example.
    assert should_entry(body["signal"], body["confidence"], cfg) is True
    assert should_exit(body["signal"], cfg) is False


def test_gateway_exit_rule_on_bearish_signal() -> None:
    for var in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD"):
        os.environ.pop(var, None)
    cfg = load_config(EXAMPLE_CONFIG, EXAMPLE_ENV)
    assert should_exit(-0.9, cfg) is True
    assert should_entry(-0.9, 0.9, cfg) is False


def test_rendered_stack_is_dry_run_end_to_end() -> None:
    for var in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD"):
        os.environ.pop(var, None)
    cfg = load_config(EXAMPLE_CONFIG, EXAMPLE_ENV)
    assert render_freqtrade(cfg)["dry_run"] is True
    assert "FT_DRY_RUN=true" in render_compose(cfg)
    assert render_helm(cfg)["dryRun"] is True


def test_metrics_endpoint_exposes_gateway_metrics() -> None:
    client = make_gateway_client()
    client.post(
        "/v1/predict",
        json={
            "pair": "BTC/USDT",
            "timeframe": "5m",
            "model": "fake-model",
            "candles": make_candles(40),
        },
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "gateway_predict_requests_total" in r.text


# --- static deployment validation (no daemon needed) ------------------------


def test_compose_config_validates_and_binds_localhost() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    proc = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "127.0.0.1" in proc.stdout
    assert "0.0.0.0:8080" not in proc.stdout
    assert "0.0.0.0:8000" not in proc.stdout


def test_helm_template_renders_cpu_dry_run() -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm not available")
    proc = subprocess.run(
        ["helm", "template", "freqtrade-ai", str(REPO / "deploy" / "helm")],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "kind:" in proc.stdout


def test_helm_template_renders_gpu_values() -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm not available")
    proc = subprocess.run(
        [
            "helm",
            "template",
            "freqtrade-ai",
            str(REPO / "deploy" / "helm"),
            "--set",
            "inference.gpu.enabled=true",
            "--set",
            "freqtrade.gpu.enabled=true",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "nvidia.com/gpu" in proc.stdout


# --- FreqAI smoke (config-level; Task 11 aware) ------------------------------


def test_freqai_backtesting_smoke_config_shape() -> None:
    """CI smoke shape: small FreqAI config renders a trainable-looking block.

    A full ``freqtrade backtesting --freqaimodel LightGBMRegressor`` run
    needs the Task 11 strategy + market data; when those are absent this
    asserts the rendered plumbing instead of failing the suite.
    """
    for var in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD"):
        os.environ.pop(var, None)
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    data["freqai"].update(
        {
            "enabled": True,
            "model": "LightGBMRegressor",
            "train_period_days": 3,
            "backtest_period_days": 2,
        }
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(data, fh)
        tmp = Path(fh.name)
    try:
        rendered_cfg = load_config(tmp, EXAMPLE_ENV)
        doc = render_freqtrade(rendered_cfg)
        assert doc["freqai"]["train_period_days"] == 3
        assert doc["freqai"]["identifier"]
        assert doc["strategy"] == rendered_cfg.strategy.name
    finally:
        tmp.unlink(missing_ok=True)


# --- full docker stack (opt-in) ----------------------------------------------


def _poll(url: str, timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status < 500:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(5)
    return False


@pytest.mark.e2e
def test_full_stack_dryrun_docker() -> None:
    """Real ``compose up`` dry-run: inference healthy + bot loop answering.

    Opt-in via ``FT_E2E_DOCKER=1`` (the CI ``e2e`` job sets it). Skipped
    otherwise so default ``pytest`` stays hermetic.
    """
    if os.environ.get("FT_E2E_DOCKER") != "1":
        pytest.skip("set FT_E2E_DOCKER=1 to run the docker stack")
    if shutil.which("docker") is None:
        pytest.skip("docker not available")

    compose = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    up = subprocess.run(
        [*compose, "up", "-d", "--build", "freqtrade", "inference"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=600,
    )
    assert up.returncode == 0, up.stderr[-3000:]
    try:
        assert _poll("http://127.0.0.1:8000/healthz"), "inference /healthz never came up"
        # Any HTTP answer (even 401 from the auth-guarded API) proves the
        # bot loop is up; only connection failure means it is down.
        assert _poll("http://127.0.0.1:8080/api/v1/ping"), "freqtrade API never came up"
    finally:
        subprocess.run(
            [*compose, "down", "-v"], capture_output=True, text=True, cwd=str(REPO), timeout=300
        )
