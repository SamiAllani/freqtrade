"""Task 10 safety tests: dry-run defaults, live guards, secret hygiene, FreqAI.

These are the guard rails that keep the bot from accidentally trading live:

- the example ``app.yaml`` renders ``dry_run: true`` on every target;
- flipping the default to live fails without ``FT_ALLOW_LIVE=yes``;
- secret *values* never appear in Helm values, ``ftctl show`` output, or logs;
- generated/secret-bearing local files are git-ignored.
"""

from __future__ import annotations

import copy
import fnmatch
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from common.contracts import AppConfig
from ftctl.cli import app
from ftctl.guards import GuardError, check_guards
from ftctl.loader import load_config
from ftctl.render.compose import render_compose
from ftctl.render.freqtrade import render_freqtrade
from ftctl.render.helm import render_helm

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO / "config" / "app.yaml.example"
EXAMPLE_ENV = REPO / "config" / ".env.example"
SECRET_VARS = ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD")

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


def load_example(clean_env: None) -> AppConfig:  # noqa: ANN001 - fixture name
    return load_config(EXAMPLE_CONFIG, EXAMPLE_ENV)


# --- dry-run defaults -------------------------------------------------------


def test_example_config_is_dry_run(clean_env: None) -> None:
    cfg = load_example(clean_env)
    assert cfg.mode == "dry_run"


def test_rendered_freqtrade_from_example_is_dry_run(clean_env: None) -> None:
    doc = render_freqtrade(load_example(clean_env))
    assert doc["dry_run"] is True
    assert doc["dry_run_wallet"] == 1000


def test_rendered_compose_from_example_is_dry_run(clean_env: None) -> None:
    text = render_compose(load_example(clean_env))
    assert "FT_MODE=dry_run" in text
    assert "FT_DRY_RUN=true" in text
    assert "FT_DRY_RUN=false" not in text


def test_rendered_helm_from_example_is_dry_run(clean_env: None) -> None:
    values = render_helm(load_example(clean_env))
    assert values["mode"] == "dry_run"
    assert values["dryRun"] is True


def test_safety_tests_fail_if_default_flipped_to_live(clean_env: None) -> None:
    """Mutating the example to ``mode: live`` must trip the guards.

    This is the regression test for the acceptance criterion
    "safety tests fail if a default is flipped to live": with the live
    guard variable unset, validation must refuse to continue.
    """
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    data["mode"] = "live"
    cfg = AppConfig.model_validate(data)
    with pytest.raises(GuardError, match="FT_ALLOW_LIVE"):
        check_guards(cfg, env={})


# --- live guards ------------------------------------------------------------


def test_live_without_guard_fails(clean_env: None) -> None:
    cfg = load_example(clean_env).model_copy(update={"mode": "live"})
    with pytest.raises(GuardError, match="FT_ALLOW_LIVE"):
        check_guards(cfg, env={})
    with pytest.raises(GuardError, match="FT_ALLOW_LIVE"):
        check_guards(cfg, env={"FT_ALLOW_LIVE": "no"})


def test_live_with_guard_but_placeholder_secrets_fails(clean_env: None) -> None:
    cfg = load_example(clean_env).model_copy(update={"mode": "live"})
    with pytest.raises(GuardError, match="requires real secrets"):
        check_guards(cfg, env={"FT_ALLOW_LIVE": "yes"})


def test_live_with_guard_and_real_secrets_passes(clean_env: None) -> None:
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    data["mode"] = "live"
    data["secrets"] = {
        "binance_key": REAL_SECRETS["BINANCE_API_KEY"],
        "binance_secret": REAL_SECRETS["BINANCE_API_SECRET"],
        "ft_api_username": REAL_SECRETS["FT_API_USERNAME"],
        "ft_api_password": REAL_SECRETS["FT_API_PASSWORD"],
    }
    cfg = AppConfig.model_validate(data)
    check_guards(cfg, env={"FT_ALLOW_LIVE": "yes"})  # must not raise


def test_stake_guard_blocks_oversized_stake(clean_env: None) -> None:
    cfg = load_example(clean_env)
    oversized = cfg.model_copy(
        update={"exchange": cfg.exchange.model_copy(update={"stake_amount": 5000.0})}
    )
    with pytest.raises(GuardError, match="stake_amount"):
        check_guards(oversized, env={})


def test_cli_validate_rejects_live_without_guard(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FT_ALLOW_LIVE", raising=False)
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    data["mode"] = "live"
    config = tmp_path / "app.yaml"
    config.write_text(yaml.safe_dump(data))
    # Real secrets so interpolation succeeds and the *guard* is what refuses.
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in REAL_SECRETS.items()))
    result = CliRunner().invoke(
        app, ["validate", "--config", str(config), "--env-file", str(env_file)]
    )
    assert result.exit_code != 0
    assert "FT_ALLOW_LIVE" in result.output


# --- secret hygiene ---------------------------------------------------------


def test_helm_values_contain_no_secret_values(clean_env: None) -> None:
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    # Build the config with real secrets mapped onto the secret fields.
    cfg = AppConfig.model_validate(
        {
            **data,
            "secrets": {
                "binance_key": REAL_SECRETS["BINANCE_API_KEY"],
                "binance_secret": REAL_SECRETS["BINANCE_API_SECRET"],
                "ft_api_username": REAL_SECRETS["FT_API_USERNAME"],
                "ft_api_password": REAL_SECRETS["FT_API_PASSWORD"],
            },
        }
    )
    dumped = yaml.safe_dump(render_helm(cfg))
    for secret in REAL_SECRETS.values():
        assert secret not in dumped
    # ... but the key *names* are referenced so the chart can mount them.
    for key in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD"):
        assert key in dumped


def test_ftctl_show_never_prints_secret_values(clean_env: None) -> None:
    result = CliRunner().invoke(
        app, ["show", "--config", str(EXAMPLE_CONFIG), "--env-file", str(EXAMPLE_ENV)]
    )
    assert result.exit_code == 0, result.output
    assert "***REDACTED***" in result.output
    for line in EXAMPLE_ENV.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            _, _, value = line.partition("=")
            value = value.strip()
            if value and value != "no":  # FT_ALLOW_LIVE is not a secret
                assert value not in result.output


def test_secret_bearing_renders_are_gitignored() -> None:
    """Local renders that embed secrets must never be committable.

    ``config.json`` and ``.env.generated`` DO carry secret values by design
    (the bot needs them); the safety property is that they are git-ignored.
    Helm values must not carry them at all (tested above).
    """
    gitignore = (REPO / ".gitignore").read_text().splitlines()
    patterns = [line.strip() for line in gitignore if line.strip() and not line.startswith("#")]
    must_match = [
        "freqtrade/user_data/config.json",
        "deploy/compose/.env.generated",
        ".env",
    ]
    for target in must_match:
        assert any(fnmatch.fnmatch(target, pat) for pat in patterns), (
            f"{target} is not covered by .gitignore"
        )


def test_env_example_has_no_real_credentials() -> None:
    values = {}
    for line in EXAMPLE_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    assert values.get("FT_ALLOW_LIVE") != "yes"
    for var in SECRET_VARS:
        assert var in values, f"{var} missing from .env.example"
        lowered = values[var].lower()
        # FT_API_USERNAME=freqtrader is Freqtrade's own default login name,
        # not a credential — allow it alongside placeholder-looking values.
        allowed = ("placeholder", "changeme", "test_", "example", "freqtrader")
        assert any(m in lowered for m in allowed), (
            f"{var} in .env.example does not look like a placeholder: {values[var]!r}"
        )


# --- FreqAI config ----------------------------------------------------------


def test_freqai_example_keeps_time_ordered_split(clean_env: None) -> None:
    cfg = load_example(clean_env)
    assert cfg.freqai.data_split_parameters.shuffle is False


def test_no_hardcoded_live_secrets_in_examples() -> None:
    """Example configs must not contain anything resembling a live credential."""
    blob = EXAMPLE_CONFIG.read_text() + "\n" + EXAMPLE_ENV.read_text()
    lowered = blob.lower()
    for marker in ("livekey", "livesecret", "sk-live", "prod-secret"):
        assert marker not in lowered
    # Secrets in app.yaml.example must be ${VAR} references, not literals.
    secrets = yaml.safe_load(EXAMPLE_CONFIG.read_text())["secrets"]
    for key, value in secrets.items():
        assert "${" in str(value), f"secrets.{key} should be a ${{VAR}} reference, got {value!r}"


def test_rendered_freqtrade_with_short_train_period_for_ci(tmp_path: Path) -> None:
    """CI smoke-test shape: a small FreqAI config renders a valid freqai block."""
    data: dict = copy.deepcopy(yaml.safe_load(EXAMPLE_CONFIG.read_text()))
    data["freqai"].update({"enabled": True, "model": "LightGBMRegressor", "train_period_days": 3})
    cfg = AppConfig.model_validate(data)
    doc = render_freqtrade(cfg)
    assert doc["freqai"]["train_period_days"] == 3
    assert doc["freqai"]["identifier"] == cfg.freqai.identifier
    assert "model" not in doc["freqai"]  # passed as --freqaimodel instead
