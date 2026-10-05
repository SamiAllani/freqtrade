"""Task 1 scaffold tests: shared contracts load the example config."""

from pathlib import Path

import yaml

from common.contracts import AppConfig

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "app.yaml.example"


def test_app_config_loads_example() -> None:
    data = yaml.safe_load(EXAMPLE.read_text())
    cfg = AppConfig.model_validate(data)
    assert cfg.mode == "dry_run"
    assert cfg.exchange.pairs == ["BTC/USDT", "ETH/USDT"]
