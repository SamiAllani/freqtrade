"""Task 9 tests: observability wiring (exporter, prometheus, dashboards)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from inference.app.main import create_app, default_config
from mcp_server.ft_client import FreqtradeApiError
from monitoring.freqtrade_exporter import (
    FreqtradeExporter,
    parse_balance,
    parse_performance,
    parse_profit,
    parse_status,
)

REPO = Path(__file__).resolve().parents[1]


# -- exporter pure parsers --------------------------------------------------


def test_parse_status_counts_and_sums() -> None:
    trades = [
        {"trade_id": "1", "pair": "BTC/USDT", "profit_abs": 5.0},
        {"trade_id": "2", "pair": "ETH/USDT", "profit_abs": -2.0},
    ]
    count, profit = parse_status(trades)
    assert count == 2
    assert profit == pytest.approx(3.0)


def test_parse_status_non_list_is_empty() -> None:
    assert parse_status(None) == (0, 0.0)
    assert parse_status({"oops": True}) == (0, 0.0)


def test_parse_profit_summary_and_win_rate() -> None:
    body = {
        "profit_closed_coin": 12.5,
        "profit_closed_percent": 3.2,
        "closed_trade_count": 10,
        "winning_trades": 6,
        "losing_trades": 4,
    }
    out = parse_profit(body)
    assert out["closed_profit"] == pytest.approx(12.5)
    assert out["closed_pct"] == pytest.approx(3.2)
    assert out["closed_count"] == pytest.approx(10.0)
    assert out["win_rate"] == pytest.approx(0.6)


def test_parse_profit_empty_is_zero() -> None:
    out = parse_profit({})
    assert out["win_rate"] == 0.0
    assert out["closed_profit"] == 0.0


def test_parse_performance_rows() -> None:
    rows = [
        {"pair": "BTC/USDT", "profit_abs": 7.0, "count": 3},
        {"pair": "ETH/USDT", "profit": -1.5, "count": 2},
    ]
    assert parse_performance(rows) == [
        ("BTC/USDT", 7.0, 3.0),
        ("ETH/USDT", -1.5, 2.0),
    ]
    assert parse_performance(None) == []


def test_parse_balance_variants() -> None:
    assert parse_balance({"currencies": [{"currency": "USDT", "balance": 100.0}]}) == [
        ("USDT", 100.0)
    ]
    assert parse_balance({"USDT": {"balance": 50.0}}) == [("USDT", 50.0)]
    assert parse_balance(None) == []


# -- exporter poll loop with a fake client -----------------------------------


class _FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def get_status(self):  # type: ignore[no-untyped-def]
        if self.fail:
            raise FreqtradeApiError("down")
        return [{"trade_id": "1", "profit_abs": 4.0}]

    def get_profit(self):  # type: ignore[no-untyped-def]
        if self.fail:
            raise FreqtradeApiError("down")
        return {
            "profit_closed_coin": 9.0,
            "profit_closed_percent": 2.0,
            "closed_trade_count": 4,
            "winning_trades": 3,
            "losing_trades": 1,
        }

    def get_performance(self):  # type: ignore[no-untyped-def]
        if self.fail:
            raise FreqtradeApiError("down")
        return [{"pair": "BTC/USDT", "profit_abs": 9.0, "count": 4}]

    def get_balance(self):  # type: ignore[no-untyped-def]
        if self.fail:
            raise FreqtradeApiError("down")
        return {"currencies": [{"currency": "USDT", "balance": 1000.0}]}


def test_exporter_poll_success_sets_up() -> None:
    exp = FreqtradeExporter(client=_FakeClient())  # type: ignore[arg-type]
    assert exp.poll_once() is True
    from monitoring.freqtrade_exporter import OPEN_TRADES, UP, WIN_RATE

    assert UP._value.get() == pytest.approx(1.0)
    assert OPEN_TRADES._value.get() == pytest.approx(1.0)
    assert WIN_RATE._value.get() == pytest.approx(0.75)


def test_exporter_poll_failure_sets_down_not_raise() -> None:
    exp = FreqtradeExporter(client=_FakeClient(fail=True))  # type: ignore[arg-type]
    assert exp.poll_once() is False
    from monitoring.freqtrade_exporter import UP

    assert UP._value.get() == pytest.approx(0.0)


# -- inference /metrics exposes the gateway series ---------------------------


def test_metrics_exposes_gateway_series() -> None:
    client = TestClient(create_app(default_config()))
    r = client.get("/metrics")
    assert r.status_code == 200
    for series in (
        "gateway_predict_requests_total",
        "gateway_predict_latency_seconds",
        "gateway_backend_errors_total",
        "gateway_gpu_available",
    ):
        assert series in r.text


# -- prometheus + grafana provisioning files ---------------------------------


def test_prometheus_scrapes_inference_and_exporter() -> None:
    doc = yaml.safe_load((REPO / "deploy/compose/prometheus/prometheus.yml").read_text())
    jobs = {j["job_name"]: j for j in doc["scrape_configs"]}
    assert "inference" in jobs
    assert "freqtrade-exporter" in jobs
    assert jobs["inference"]["metrics_path"] == "/metrics"
    assert jobs["freqtrade-exporter"]["metrics_path"] == "/metrics"
    targets = " ".join(str(j) for j in jobs.values())
    assert "inference:8000" in targets
    assert "freqtrade-exporter:9108" in targets


def test_grafana_datasource_provisioned() -> None:
    doc = yaml.safe_load(
        (REPO / "deploy/compose/grafana/provisioning/datasources/prometheus.yml").read_text()
    )
    ds = doc["datasources"][0]
    assert ds["type"] == "prometheus"
    assert ds["url"] == "http://prometheus:9090"
    assert ds["isDefault"] is True


def test_grafana_dashboard_provider_points_at_mount() -> None:
    doc = yaml.safe_load(
        (REPO / "deploy/compose/grafana/provisioning/dashboards/dashboards.yml").read_text()
    )
    assert doc["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"


def _dashboard_queries(path: Path) -> str:
    dash = json.loads(path.read_text())
    assert dash["title"]
    assert dash["panels"], f"{path.name} has no panels"
    blob = json.dumps(dash)
    assert "DS_PROMETHEUS" in blob
    return blob


def test_trading_dashboard_panels() -> None:
    blob = _dashboard_queries(REPO / "dashboards/trading.json")
    for q in (
        "freqtrade_open_trades",
        "freqtrade_open_profit_usdt",
        "freqtrade_closed_profit_usdt",
        "freqtrade_win_rate",
        "freqtrade_pair_profit_usdt",
    ):
        assert q in blob, f"trading dashboard missing {q}"


def test_inference_dashboard_uses_metrics() -> None:
    blob = _dashboard_queries(REPO / "dashboards/inference.json")
    for q in (
        "gateway_predict_requests_total",
        "gateway_predict_latency_seconds_bucket",
        "gateway_backend_errors_total",
        "gateway_gpu_available",
        "histogram_quantile",
    ):
        assert q in blob, f"inference dashboard missing {q}"


def test_compose_wires_exporter_and_dashboards() -> None:
    text = (REPO / "deploy/compose/docker-compose.yml").read_text()
    assert "freqtrade-exporter" in text
    assert "monitoring/Dockerfile" in text
    assert "../../dashboards:/var/lib/grafana/dashboards:ro" in text
