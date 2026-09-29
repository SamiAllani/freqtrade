"""Environment guards for ftctl.

Live trading is dangerous by default, so two independent checks apply:

1. ``mode: live`` requires ``FT_ALLOW_LIVE=yes`` in the process environment.
2. ``stake_amount <= max_stake`` always holds (checked in every mode).

Live mode additionally rejects missing or placeholder-looking secrets.
On violation a :class:`GuardError` is raised; the CLI turns it into a
clear message on stderr and a non-zero exit code.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from common.contracts import AppConfig
from ftctl.loader import DRY_RUN_SECRET_PLACEHOLDER

LIVE_GUARD_VAR = "FT_ALLOW_LIVE"
LIVE_GUARD_VALUE = "yes"

#: Substrings (case-insensitive) that mark a secret value as an example
#: placeholder rather than a real credential.
PLACEHOLDER_MARKERS = ("placeholder", "changeme", "example")

SECRET_FIELDS = ("binance_key", "binance_secret", "ft_api_username", "ft_api_password")


class GuardError(Exception):
    """Raised when a safety guard blocks the requested operation."""


def is_placeholder(value: str) -> bool:
    """Return True for empty or example-looking secret values."""
    if not value or not value.strip():
        return True
    lowered = value.strip().lower()
    if lowered == DRY_RUN_SECRET_PLACEHOLDER.lower():
        return True
    if "${" in value:
        # Uninterpolated variable reference: no real credential was provided.
        return True
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def missing_secrets(cfg: AppConfig) -> list[str]:
    """Return the names of secret fields that are missing or placeholders."""
    return [name for name in SECRET_FIELDS if is_placeholder(getattr(cfg.secrets, name))]


def check_guards(cfg: AppConfig, env: Mapping[str, str] | None = None) -> None:
    """Enforce live-trading and stake guards. Raises :class:`GuardError`."""
    if env is None:
        env = os.environ

    if cfg.exchange.stake_amount > cfg.exchange.max_stake:
        raise GuardError(
            f"stake_amount ({cfg.exchange.stake_amount}) exceeds "
            f"max_stake ({cfg.exchange.max_stake}): "
            "refusing to continue. Lower stake_amount or raise max_stake."
        )

    if cfg.mode != "live":
        return

    if env.get(LIVE_GUARD_VAR) != LIVE_GUARD_VALUE:
        raise GuardError(
            f"Live trading requested (mode: live) but {LIVE_GUARD_VAR}={LIVE_GUARD_VALUE} "
            f"is not set in the environment (got {LIVE_GUARD_VAR}={env.get(LIVE_GUARD_VAR)!r}). "
            "Refusing to continue. Keep mode: dry_run, or explicitly opt in."
        )

    missing = missing_secrets(cfg)
    if missing:
        raise GuardError(
            "Live mode requires real secrets, but these are missing or "
            f"still example placeholders: {', '.join(missing)}. "
            "Set them in the environment or the .env file."
        )
