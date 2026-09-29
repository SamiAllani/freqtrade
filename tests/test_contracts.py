"""Task 1 scaffold tests: shared contracts load the example config."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from common.contracts import AppConfig

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "app.yaml.example"


def test_app_config_loads_example() -> None:
    data = yaml.safe_load(EXAMPLE.read_text())
    cfg = AppConfig.model_validate(data)
    assert cfg.mode == "dry_run"
    assert cfg.strategy.signal_source == "gateway"
    assert cfg.exchange.pairs == ["BTC/USDT", "ETH/USDT"]


def test_freqai_signal_source_requires_enabled() -> None:
    data = yaml.safe_load(EXAMPLE.read_text())
    data["strategy"]["signal_source"] = "freqai"
    data["freqai"]["enabled"] = False
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)

    data["freqai"]["enabled"] = True
    cfg = AppConfig.model_validate(data)
    assert cfg.freqai.enabled is True
