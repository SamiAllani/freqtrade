"""Load the central ``app.yaml`` config with ``${ENV_VAR}`` interpolation.

Resolution order for variables (highest precedence first):

1. caller-provided ``extra_env`` mapping,
2. the process environment (``os.environ``),
3. values parsed from the ``.env`` file (if given).

Any ``${VAR}`` (or ``${VAR:-default}``) left unresolved is a hard error,
except for variables referenced from the ``secrets:`` block while the
config is in ``dry_run`` mode: those fall back to
:data:`DRY_RUN_SECRET_PLACEHOLDER` so a secrets-less local render works.
Live mode never tolerates missing secrets.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from common.contracts import AppConfig

DRY_RUN_SECRET_PLACEHOLDER = "DRY_RUN_PLACEHOLDER"

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Raised when the central config cannot be loaded or validated."""


def load_dotenv(env_file: str | Path | None) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` dotenv file. Missing file -> empty mapping."""
    if env_file is None:
        return {}
    path = Path(env_file)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _secret_var_names(raw: Mapping[str, Any]) -> set[str]:
    """Collect ``${VAR}`` names referenced inside the ``secrets:`` block."""
    secrets = raw.get("secrets", {})
    names: set[str] = set()
    if isinstance(secrets, Mapping):
        for value in secrets.values():
            if isinstance(value, str):
                names.update(m.group(1) for m in _VAR_PATTERN.finditer(value))
    return names


def _interpolate_string(
    value: str, env: Mapping[str, str], secret_vars: set[str], mode: str
) -> str:
    """Substitute ``${VAR}``/``${VAR:-default}`` references in one string."""

    def _replace(match: re.Match[str]) -> str:
        """Resolve a single ``${VAR}`` match from env, default or placeholder."""
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        if mode == "dry_run" and name in secret_vars:
            return DRY_RUN_SECRET_PLACEHOLDER
        raise ConfigError(
            f"Unresolved variable '${{{name}}}': set it in the environment or in the .env file."
        )

    return _VAR_PATTERN.sub(_replace, value)


def interpolate_obj(node: Any, env: Mapping[str, str], secret_vars: set[str], mode: str) -> Any:
    """Recursively interpolate ``${VAR}`` in all strings of a YAML tree."""
    if isinstance(node, str):
        return _interpolate_string(node, env, secret_vars, mode)
    if isinstance(node, Mapping):
        return {key: interpolate_obj(value, env, secret_vars, mode) for key, value in node.items()}
    if isinstance(node, list):
        return [interpolate_obj(item, env, secret_vars, mode) for item in node]
    return node


def load_config(
    config_path: str | Path,
    env_file: str | Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load, interpolate and validate the central config file."""
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config file {path} must contain a YAML mapping.")
    raw_dict = dict(raw)
    mode = str(raw_dict.get("mode", "dry_run"))

    env: dict[str, str] = {}
    env.update(load_dotenv(env_file))
    env.update(os.environ)
    if extra_env:
        env.update(extra_env)

    try:
        resolved = interpolate_obj(raw_dict, env, _secret_var_names(raw_dict), mode)
    except ConfigError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Failed to interpolate {path}: {exc}") from exc

    try:
        return AppConfig.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config in {path}:\n{exc}") from exc
