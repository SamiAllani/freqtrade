"""Task 2 tests: ftctl loader, interpolation, guards and target renderers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from common.contracts import AppConfig
from ftctl.cli import app
from ftctl.guards import GuardError, check_guards, is_placeholder, missing_secrets
from ftctl.loader import (
    DRY_RUN_SECRET_PLACEHOLDER,
    ConfigError,
    interpolate_obj,
    load_config,
    load_dotenv,
)
from ftctl.render.compose import render_compose
from ftctl.render.freqtrade import freqtrade_args, freqtrade_tag, render_freqtrade
from ftctl.render.helm import render_helm

SECRET_VARS = ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD")

BASE_YAML: dict[str, Any] = {
    "mode": "dry_run",
    "exchange": {
        "name": "binance",
        "pairs": ["BTC/USDT", "ETH/USDT"],
        "stake_currency": "USDT",
        "stake_amount": 50,
        "max_stake": 100,
    },
    "strategy": {
        "signal_source": "gateway",
        "name": "AiSignalStrategy",
        "timeframe": "5m",
        "entry_signal_min": 0.4,
        "entry_confidence_min": 0.6,
        "exit_signal_max": -0.2,
    },
    "freqai": {"enabled": False},
    "inference": {"url": "http://inference:8000", "default_model": "local-gru"},
    "secrets": {
        "binance_key": "${BINANCE_API_KEY}",
        "binance_secret": "${BINANCE_API_SECRET}",
        "ft_api_username": "${FT_API_USERNAME}",
        "ft_api_password": "${FT_API_PASSWORD}",
    },
}

REAL_SECRETS = {
    "BINANCE_API_KEY": "livekeyABC123xyz",
    "BINANCE_API_SECRET": "livesecretABC123xyz",
    "FT_API_USERNAME": "botoperator",
    "FT_API_PASSWORD": "s3cure-pw-99",
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (*SECRET_VARS, "FT_ALLOW_LIVE"):
        monkeypatch.delenv(var, raising=False)


def write_yaml(tmp_path: Path, data: dict[str, Any], name: str = "app.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def write_env(tmp_path: Path, values: dict[str, str], name: str = ".env") -> Path:
    path = tmp_path / name
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    return path


# --- dotenv parsing -------------------------------------------------------


def test_load_dotenv_parses_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nPLAIN=abc\nQUOTED=\"hello world\"\nSINGLE='x#y'\nexport EXPORTED=1\n\nEMPTY=\n"
    )
    assert load_dotenv(env_file) == {
        "PLAIN": "abc",
        "QUOTED": "hello world",
        "SINGLE": "x#y",
        "EXPORTED": "1",
        "EMPTY": "",
    }


def test_load_dotenv_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}
    assert load_dotenv(None) == {}


# --- interpolation --------------------------------------------------------


def test_interpolate_replaces_vars_and_defaults() -> None:
    node = {"a": "x-${FOO}-y", "b": ["${BAR:-fallback}"], "c": 3}
    out = interpolate_obj(node, {"FOO": "1"}, set(), "dry_run")
    assert out == {"a": "x-1-y", "b": ["fallback"], "c": 3}


def test_interpolate_unresolved_var_fails() -> None:
    with pytest.raises(ConfigError, match="MISSING_VAR"):
        interpolate_obj({"a": "${MISSING_VAR}"}, {}, set(), "dry_run")


def test_process_env_wins_over_dotenv(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_yaml(tmp_path, BASE_YAML)
    env_file = write_env(
        tmp_path,
        {
            "BINANCE_API_KEY": "from-dotenv",
            "BINANCE_API_SECRET": "s",
            "FT_API_USERNAME": "u",
            "FT_API_PASSWORD": "p",
        },
    )
    monkeypatch.setenv("BINANCE_API_KEY", "from-process")
    cfg = load_config(config, env_file)
    assert cfg.secrets.binance_key == "from-process"
    assert cfg.secrets.binance_secret == "s"


def test_dry_run_missing_secrets_use_placeholders(tmp_path: Path, clean_env: None) -> None:
    cfg = load_config(write_yaml(tmp_path, BASE_YAML))
    assert cfg.secrets.binance_key == DRY_RUN_SECRET_PLACEHOLDER
    assert cfg.secrets.ft_api_password == DRY_RUN_SECRET_PLACEHOLDER


def test_dry_run_unrelated_missing_var_still_fails(tmp_path: Path, clean_env: None) -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["inference"]["url"] = "http://${GATEWAY_HOST}:8000"
    with pytest.raises(ConfigError, match="GATEWAY_HOST"):
        load_config(write_yaml(tmp_path, data))


def test_live_missing_secrets_fail_at_load(tmp_path: Path, clean_env: None) -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["mode"] = "live"
    with pytest.raises(ConfigError, match="BINANCE_API_KEY"):
        load_config(write_yaml(tmp_path, data))


def test_live_loads_with_real_secrets(tmp_path: Path, clean_env: None) -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["mode"] = "live"
    cfg = load_config(write_yaml(tmp_path, data), extra_env=REAL_SECRETS)
    assert cfg.mode == "live"
    assert cfg.secrets.binance_key == REAL_SECRETS["BINANCE_API_KEY"]


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "app.yaml"
    bad.write_text("mode: [unclosed\n")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(bad)


def test_example_config_loads_with_example_env(clean_env: None) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "config" / "app.yaml.example", repo / "config" / ".env.example")
    assert cfg.mode == "dry_run"
    assert cfg.secrets.binance_key == "test_key_placeholder"


# --- guards ---------------------------------------------------------------


def test_guards_dry_run_passes_without_live_flag(clean_env: None) -> None:
    cfg = AppConfig.model_validate(BASE_YAML | {"secrets": {}})
    check_guards(cfg, env={})  # must not raise


def test_guards_stake_above_max_fails() -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["exchange"]["stake_amount"] = 150
    cfg = AppConfig.model_validate(data)
    with pytest.raises(GuardError, match="stake_amount"):
        check_guards(cfg, env={})


def test_guards_live_without_flag_fails(clean_env: None) -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["mode"] = "live"
    cfg = AppConfig.model_validate({**data, "secrets": {"binance_key": "k", "binance_secret": "s"}})
    with pytest.raises(GuardError, match="FT_ALLOW_LIVE"):
        check_guards(cfg, env={})
    # Wrong value also fails.
    with pytest.raises(GuardError, match="FT_ALLOW_LIVE"):
        check_guards(cfg, env={"FT_ALLOW_LIVE": "no"})


def test_guards_live_with_flag_and_real_secrets_passes() -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["mode"] = "live"
    data["secrets"] = {
        "binance_key": "k",
        "binance_secret": "s",
        "ft_api_username": "u",
        "ft_api_password": "p",
    }
    cfg = AppConfig.model_validate(data)
    check_guards(cfg, env={"FT_ALLOW_LIVE": "yes"})  # must not raise


def test_guards_live_rejects_placeholder_secrets() -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["mode"] = "live"
    cfg = AppConfig.model_validate(data)  # empty-string secrets
    with pytest.raises(GuardError, match="requires real secrets"):
        check_guards(cfg, env={"FT_ALLOW_LIVE": "yes"})


def test_is_placeholder_and_missing_secrets() -> None:
    assert is_placeholder("")
    assert is_placeholder("  ")
    assert is_placeholder(DRY_RUN_SECRET_PLACEHOLDER)
    assert is_placeholder("test_key_placeholder")
    assert is_placeholder("changeme")
    assert not is_placeholder("livekeyABC123xyz")
    cfg = AppConfig.model_validate(BASE_YAML | {"secrets": {}})
    assert set(missing_secrets(cfg)) == {
        "binance_key",
        "binance_secret",
        "ft_api_username",
        "ft_api_password",
    }


# --- freqtrade renderer ---------------------------------------------------


def test_render_freqtrade_dry_run(tmp_path: Path, clean_env: None) -> None:
    cfg = load_config(write_yaml(tmp_path, BASE_YAML), extra_env=REAL_SECRETS)
    doc = render_freqtrade(cfg)
    assert doc["dry_run"] is True
    assert doc["stake_currency"] == "USDT"
    assert doc["stake_amount"] == 50
    assert doc["strategy"] == "AiSignalStrategy"
    assert doc["timeframe"] == "5m"
    assert doc["exchange"]["name"] == "binance"
    assert doc["exchange"]["pair_whitelist"] == ["BTC/USDT", "ETH/USDT"]
    assert doc["pairlists"] == [{"method": "StaticPairList"}]
    assert doc["api_server"]["listen_ip_address"] == "0.0.0.0"
    assert doc["api_server"]["username"] == REAL_SECRETS["FT_API_USERNAME"]
    assert doc["exchange"]["key"] == REAL_SECRETS["BINANCE_API_KEY"]
    assert doc["strategy_params"] == {
        "entry_signal_min": 0.4,
        "entry_confidence_min": 0.6,
        "exit_signal_max": -0.2,
    }
    assert "freqai" not in doc
    assert "LightGBMRegressor" not in json.dumps(doc)  # --freqaimodel only


def test_render_freqtrade_live_and_freqai(tmp_path: Path, clean_env: None) -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["mode"] = "live"
    data["strategy"]["signal_source"] = "freqai"
    data["freqai"] = {
        "enabled": True,
        "identifier": "ft-freqai-v1",
        "model": "LightGBMRegressor",
        "train_period_days": 30,
        "backtest_period_days": 7,
        "live_retrain_hours": 1,
        "feature_parameters": {
            "include_timeframes": ["5m", "1h"],
            "include_corr_pairlist": ["ETH/USDT"],
            "label_period_candles": 24,
        },
        "data_split_parameters": {"test_size": 0.33, "shuffle": False},
        "model_training_parameters": {},
    }
    cfg = load_config(write_yaml(tmp_path, data), extra_env=REAL_SECRETS)
    doc = render_freqtrade(cfg)
    assert doc["dry_run"] is False
    freqai = doc["freqai"]
    assert freqai["identifier"] == "ft-freqai-v1"
    assert freqai["train_period_days"] == 30
    assert freqai["backtest_period_days"] == 7
    assert freqai["live_retrain_hours"] == 1
    assert freqai["feature_parameters"]["label_period_candles"] == 24
    assert freqai["data_split_parameters"] == {"test_size": 0.33, "shuffle": False}
    assert "model" not in freqai  # passed as --freqaimodel instead


def test_freqtrade_args_and_tags() -> None:
    cfg = AppConfig.model_validate(BASE_YAML | {"secrets": {}})
    assert freqtrade_args(cfg) == ["--strategy", "AiSignalStrategy"]
    assert freqtrade_tag(cfg) == "stable"

    freqai_cfg = AppConfig.model_validate(
        {
            **BASE_YAML,
            "strategy": {**BASE_YAML["strategy"], "signal_source": "freqai"},
            "freqai": {"enabled": True, "model": "LightGBMRegressor"},
            "secrets": {},
        }
    )
    assert freqtrade_args(freqai_cfg) == [
        "--strategy",
        "AiSignalStrategy",
        "--freqaimodel",
        "LightGBMRegressor",
    ]
    assert freqtrade_tag(freqai_cfg) == "stable_freqai"

    torch_cfg = AppConfig.model_validate(
        {**freqai_cfg.model_dump(), "freqai": {"enabled": True, "model": "PyTorchMLPRegressor"}}
    )
    assert freqtrade_tag(torch_cfg) == "stable_freqaitorch"


# --- compose renderer -----------------------------------------------------


def test_render_compose_gateway(tmp_path: Path, clean_env: None) -> None:
    cfg = load_config(write_yaml(tmp_path, BASE_YAML), extra_env=REAL_SECRETS)
    text = render_compose(cfg)
    assert "FT_STRATEGY=AiSignalStrategy" in text
    assert 'FREQTRADE_ARGS="--strategy AiSignalStrategy"' in text
    assert "--freqaimodel" not in text
    assert "FREQTRADE_TAG=stable" in text
    assert "FT_DRY_RUN=true" in text
    assert f"BINANCE_API_KEY={REAL_SECRETS['BINANCE_API_KEY']}" in text


def test_render_compose_freqai_args() -> None:
    cfg = AppConfig.model_validate(
        {
            **BASE_YAML,
            "strategy": {**BASE_YAML["strategy"], "signal_source": "freqai"},
            "freqai": {"enabled": True, "model": "LightGBMRegressor"},
            "secrets": {},
        }
    )
    text = render_compose(cfg)
    assert "--freqaimodel LightGBMRegressor" in text
    assert "FREQTRADE_TAG=stable_freqai" in text


# --- helm renderer --------------------------------------------------------


def test_render_helm_uses_secret_refs_not_values(tmp_path: Path, clean_env: None) -> None:
    cfg = load_config(write_yaml(tmp_path, BASE_YAML), extra_env=REAL_SECRETS)
    values = render_helm(cfg)
    assert values["dryRun"] is True
    assert values["strategy"]["args"] == ["--strategy", "AiSignalStrategy"]
    assert values["freqtrade"]["tag"] == "stable"
    assert values["existingSecret"]["name"] == "freqtrade-secrets"
    dumped = yaml.safe_dump(values)
    for secret in REAL_SECRETS.values():
        assert secret not in dumped
    for key in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD"):
        assert key in dumped  # referenced by name only


def test_render_helm_freqai() -> None:
    cfg = AppConfig.model_validate(
        {
            **BASE_YAML,
            "strategy": {**BASE_YAML["strategy"], "signal_source": "hybrid"},
            "freqai": {"enabled": True, "model": "LightGBMRegressor"},
            "secrets": {},
        }
    )
    values = render_helm(cfg)
    assert values["freqtrade"]["freqaimodel"] == "LightGBMRegressor"
    assert values["strategy"]["args"] == [
        "--strategy",
        "AiSignalStrategy",
        "--freqaimodel",
        "LightGBMRegressor",
    ]
    assert values["freqai"]["enabled"] is True


# --- CLI ------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, clean_env: None) -> tuple[Path, Path]:
    config = write_yaml(tmp_path, BASE_YAML)
    env_file = write_env(tmp_path, REAL_SECRETS)
    return config, env_file


def test_cli_validate_ok(cli_env: tuple[Path, Path]) -> None:
    config, env_file = cli_env
    result = CliRunner().invoke(
        app, ["validate", "--config", str(config), "--env-file", str(env_file)]
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_cli_validate_missing_config_fails(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["validate", "--config", str(tmp_path / "no.yaml")])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_cli_validate_live_without_guard_fails(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FT_ALLOW_LIVE", raising=False)
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["mode"] = "live"
    config = write_yaml(tmp_path, data)
    env_file = write_env(tmp_path, REAL_SECRETS)
    result = CliRunner().invoke(
        app, ["validate", "--config", str(config), "--env-file", str(env_file)]
    )
    assert result.exit_code != 0
    assert "FT_ALLOW_LIVE" in result.output


def test_cli_render_all_targets(cli_env: tuple[Path, Path], tmp_path: Path) -> None:
    config, env_file = cli_env
    runner = CliRunner()
    out_ft = tmp_path / "config.json"
    result = runner.invoke(
        app,
        [
            "render",
            "freqtrade",
            "--out",
            str(out_ft),
            "--config",
            str(config),
            "--env-file",
            str(env_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out_ft.read_text())["dry_run"] is True

    out_env = tmp_path / ".env.generated"
    result = runner.invoke(
        app,
        [
            "render",
            "compose",
            "--out",
            str(out_env),
            "--config",
            str(config),
            "--env-file",
            str(env_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "FT_STRATEGY" in out_env.read_text()

    out_helm = tmp_path / "values.generated.yaml"
    result = runner.invoke(
        app,
        [
            "render",
            "helm",
            "--out",
            str(out_helm),
            "--config",
            str(config),
            "--env-file",
            str(env_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "existingSecret" in out_helm.read_text()


def test_cli_render_bad_target_rejected(cli_env: tuple[Path, Path], tmp_path: Path) -> None:
    config, env_file = cli_env
    result = CliRunner().invoke(
        app,
        [
            "render",
            "terraform",
            "--out",
            str(tmp_path / "x"),
            "--config",
            str(config),
            "--env-file",
            str(env_file),
        ],
    )
    assert result.exit_code != 0


def test_cli_show_redacts_secrets(cli_env: tuple[Path, Path]) -> None:
    config, env_file = cli_env
    result = CliRunner().invoke(app, ["show", "--config", str(config), "--env-file", str(env_file)])
    assert result.exit_code == 0, result.output
    assert "***REDACTED***" in result.output
    for secret in REAL_SECRETS.values():
        assert secret not in result.output


def test_cli_show_json_redacts_secrets(cli_env: tuple[Path, Path]) -> None:
    config, env_file = cli_env
    result = CliRunner().invoke(
        app,
        ["show", "--config", str(config), "--env-file", str(env_file), "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data["secrets"].values()) == {"***REDACTED***"}


def test_cli_validate_stake_guard(tmp_path: Path, clean_env: None) -> None:
    data = yaml.safe_load(yaml.safe_dump(BASE_YAML))
    data["exchange"]["stake_amount"] = 500
    config = write_yaml(tmp_path, data)
    result = CliRunner().invoke(app, ["validate", "--config", str(config)])
    assert result.exit_code != 0
    assert "stake_amount" in result.output


def test_os_environ_not_leaked_by_show(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Secrets come from the process env here; show must still redact them.
    for key, value in REAL_SECRETS.items():
        monkeypatch.setenv(key, value)
    config = write_yaml(tmp_path, BASE_YAML)
    result = CliRunner().invoke(app, ["show", "--config", str(config)])
    assert result.exit_code == 0, result.output
    for value in REAL_SECRETS.values():
        assert value not in result.output
    assert os.environ["BINANCE_API_KEY"] == REAL_SECRETS["BINANCE_API_KEY"]
