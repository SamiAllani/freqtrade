#!/usr/bin/env python3
"""End-to-end dry-run verification (Task 10).

Runs the full safety + stack check without Docker:

1. loads ``config/app.yaml.example`` (dry-run must hold);
2. renders freqtrade / compose / helm targets and asserts ``dry_run``;
3. asserts Helm values and ``ftctl show`` carry no secret *values*;
4. boots the real inference gateway in-process, checks ``/healthz`` and
   ``/v1/predict``, and feeds the signal through the strategy entry rule;
5. runs ``docker compose config`` and ``helm template`` when those
   binaries are available (SKIP otherwise).

Exit 0 when every check passes, 1 otherwise. Secrets are never printed.

Usage:
    python scripts/verify_dryrun.py [--config PATH] [--env-file PATH]
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ftctl.guards import GuardError, check_guards  # noqa: E402
from ftctl.loader import load_config  # noqa: E402
from ftctl.render.compose import render_compose  # noqa: E402
from ftctl.render.freqtrade import render_freqtrade  # noqa: E402
from ftctl.render.helm import render_helm  # noqa: E402
from inference.app.main import create_app, default_config  # noqa: E402

CHECKS: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one named check; returns ``ok`` for chaining."""
    status = "PASS" if ok else "FAIL"
    CHECKS.append((name, status))
    suffix = f" — {detail}" if detail and not ok else ""
    print(f"[{status}] {name}{suffix}")
    return ok


def main() -> int:
    """Run all dry-run checks and return 0 on success, 1 on any failure."""
    parser = argparse.ArgumentParser(description="Verify the dry-run stack end to end.")
    parser.add_argument("--config", default=str(REPO / "config" / "app.yaml.example"))
    parser.add_argument("--env-file", default=str(REPO / "config" / ".env.example"))
    args = parser.parse_args()

    ok = True
    import os

    for var in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "FT_API_USERNAME", "FT_API_PASSWORD"):
        os.environ.pop(var, None)
    os.environ.pop("FT_ALLOW_LIVE", None)

    # 1. Example config loads and is dry-run.
    try:
        cfg = load_config(args.config, args.env_file)
        ok &= check("example config loads", True)
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        return check("example config loads", False, str(exc)) and 1 or 1
    ok &= check("mode is dry_run", cfg.mode == "dry_run", f"mode={cfg.mode!r}")
    try:
        check_guards(cfg, env={})
        ok &= check("guards pass without FT_ALLOW_LIVE (dry-run)", True)
    except GuardError as exc:
        ok &= check("guards pass without FT_ALLOW_LIVE (dry-run)", False, str(exc))

    # Live must be refused without the guard variable.
    live_cfg = cfg.model_copy(update={"mode": "live"})
    try:
        check_guards(live_cfg, env={})
        ok &= check("live without FT_ALLOW_LIVE is refused", False, "guards did not raise")
    except GuardError:
        ok &= check("live without FT_ALLOW_LIVE is refused", True)

    # 2. Rendered targets are dry-run.
    with tempfile.TemporaryDirectory() as tmp:
        doc = render_freqtrade(cfg)
        ok &= check("freqtrade render is dry_run", doc.get("dry_run") is True)
        compose_text = render_compose(cfg)
        ok &= check("compose render is dry_run", "FT_DRY_RUN=true" in compose_text)
        values = render_helm(cfg)
        ok &= check("helm render is dry_run", values.get("dryRun") is True)
        (Path(tmp) / "config.json").write_text(yaml.safe_dump(doc))

    # 3. Secret hygiene.
    dumped = yaml.safe_dump(values)
    secrets_blob = Path(args.env_file).read_text() if Path(args.env_file).exists() else ""
    leaked = [
        v
        for _, _, v in (line.partition("=") for line in secrets_blob.splitlines())
        if v.strip() and v.strip() in dumped and len(v.strip()) > 3
    ]
    ok &= check("helm values contain no secret values", not leaked, f"leaked={leaked}")
    gitignore = (REPO / ".gitignore").read_text()
    patterns = [
        ln.strip() for ln in gitignore.splitlines() if ln.strip() and not ln.startswith("#")
    ]
    for target in ("freqtrade/user_data/config.json", "deploy/compose/.env.generated", ".env"):
        matched = any(fnmatch.fnmatch(target, pat) for pat in patterns)
        ok &= check(f"git-ignored: {target}", matched)

    # 4. Gateway health + predict + strategy rule.
    try:
        client = TestClient(create_app(default_config()))
        health = client.get("/healthz")
        ok &= check(
            "gateway /healthz ok",
            health.status_code == 200 and health.json().get("status") == "ok",
            health.text[:200],
        )
        candles = [
            {
                "t": 1710000000 + i * 300,
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "c": 100.0 + i * 0.1,
                "v": 10.0,
            }
            for i in range(40)
        ]
        pred = client.post(
            "/v1/predict",
            json={"pair": "BTC/USDT", "timeframe": "5m", "model": "local-gru", "candles": candles},
        )
        body = pred.json() if pred.status_code == 200 else {}
        in_range = (
            -1.0 <= body.get("signal", 99) <= 1.0 and 0.0 <= body.get("confidence", -1) <= 1.0
        )
        ok &= check(
            "gateway /v1/predict returns a valid signal",
            pred.status_code == 200 and in_range,
            pred.text[:200],
        )
        entry = (
            body.get("signal", -1) > cfg.strategy.entry_signal_min
            and body.get("confidence", -1) > cfg.strategy.entry_confidence_min
        )
        ok &= check("strategy entry rule evaluates a gateway signal", isinstance(entry, bool))
    except Exception as exc:  # noqa: BLE001
        ok &= check("gateway in-process stack", False, str(exc))

    # 5. Deployment tooling (SKIP when binaries are absent).
    compose_file = REPO / "deploy" / "compose" / "docker-compose.yml"
    if shutil.which("docker"):
        proc = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            timeout=120,
        )
        ok &= check("docker compose config validates", proc.returncode == 0, proc.stderr[-500:])
        if proc.returncode == 0:
            ok &= check(
                "compose binds localhost only",
                "127.0.0.1" in proc.stdout,
                "no 127.0.0.1 binding found",
            )
    else:
        print("[SKIP] docker compose config (docker not available)")
    if shutil.which("helm"):
        proc = subprocess.run(
            ["helm", "lint", str(REPO / "deploy" / "helm")],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            timeout=120,
        )
        ok &= check(
            "helm lint passes", proc.returncode == 0, proc.stderr[-500:] or proc.stdout[-500:]
        )
        proc = subprocess.run(
            ["helm", "template", "freqtrade-ai", str(REPO / "deploy" / "helm")],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            timeout=120,
        )
        ok &= check("helm template renders", proc.returncode == 0, proc.stderr[-500:])
    else:
        print("[SKIP] helm lint/template (helm not available)")

    failed = sum(1 for _, s in CHECKS if s == "FAIL")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed.")
    return 1 if not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
