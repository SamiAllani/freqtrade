"""ftctl — validate the central config and render deployment targets.

Commands:

- ``ftctl validate`` — load, interpolate and guard-check the config.
- ``ftctl render freqtrade|compose|helm --out <file>`` — write the target file.
- ``ftctl show`` — print the effective config with secrets redacted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
import yaml

from ftctl.guards import GuardError, check_guards
from ftctl.loader import ConfigError, load_config
from ftctl.render.compose import render_compose
from ftctl.render.freqtrade import render_freqtrade
from ftctl.render.helm import render_helm

app = typer.Typer(name="ftctl", help="Central config CLI.", no_args_is_help=True)

RenderTarget = Literal["freqtrade", "compose", "helm"]

DEFAULT_CONFIG_CANDIDATES = (Path("config/app.yaml"), Path("config/app.yaml.example"))
DEFAULT_ENV_CANDIDATES = (Path("config/.env"), Path(".env"))

REDACTED = "***REDACTED***"


def resolve_config_path(config: Path | None) -> Path:
    """Return the config file to use (explicit path or first default found)."""
    if config is not None:
        return config
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_CONFIG_CANDIDATES[0]


def resolve_env_file(env_file: Path | None) -> Path | None:
    """Return the .env file to use, or None when no candidate exists."""
    if env_file is not None:
        return env_file
    for candidate in DEFAULT_ENV_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _load(config: Path | None, env_file: Path | None):  # type: ignore[no-untyped-def]
    """Resolve config/env paths and load the validated :class:`AppConfig`."""
    config_path = resolve_config_path(config)
    return config_path, load_config(config_path, resolve_env_file(env_file))


@app.command()
def validate(
    config: Path | None = typer.Option(None, "--config", "-c", help="Central app.yaml."),
    env_file: Path | None = typer.Option(None, "--env-file", "-e", help=".env file."),
) -> None:
    """Validate the central config (interpolation + guards)."""
    try:
        config_path, cfg = _load(config, env_file)
        check_guards(cfg)
    except (ConfigError, GuardError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"OK: {config_path} is valid "
        f"(mode={cfg.mode}, signal_source={cfg.strategy.signal_source}).",
        fg=typer.colors.GREEN,
    )


@app.command()
def render(
    target: RenderTarget = typer.Argument(..., help="Render target."),
    out: Path = typer.Option(..., "--out", "-o", help="Output file to write."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Central app.yaml."),
    env_file: Path | None = typer.Option(None, "--env-file", "-e", help=".env file."),
) -> None:
    """Render TARGET (freqtrade|compose|helm) into --out."""
    try:
        config_path, cfg = _load(config, env_file)
        check_guards(cfg)
    except (ConfigError, GuardError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if target == "freqtrade":
        payload = json.dumps(render_freqtrade(cfg), indent=2) + "\n"
    elif target == "compose":
        payload = render_compose(cfg, source=str(config_path))
    else:
        payload = yaml.safe_dump(render_helm(cfg), sort_keys=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload)
    typer.secho(f"Wrote {target} config to {out}.", fg=typer.colors.GREEN)


@app.command()
def show(
    config: Path | None = typer.Option(None, "--config", "-c", help="Central app.yaml."),
    env_file: Path | None = typer.Option(None, "--env-file", "-e", help=".env file."),
    output: Literal["yaml", "json"] = typer.Option("yaml", "--format", "-f"),
) -> None:
    """Print the effective config with secret values redacted."""
    try:
        _, cfg = _load(config, env_file)
    except ConfigError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    data = cfg.model_dump()
    data["secrets"] = {key: REDACTED for key in data.get("secrets", {})}
    if output == "json":
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(yaml.safe_dump(data, sort_keys=False).rstrip("\n"))


if __name__ == "__main__":
    app()
