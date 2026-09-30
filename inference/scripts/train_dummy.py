"""Train a dummy GRU model so the inference pipeline works end to end.

Generates a synthetic random-walk price series, labels each window with the
future return, and:

- with ``torch`` installed (``pip install -e .[gpu]``): trains the tiny
  :class:`SignalGRU` for a few epochs and saves a ``.pt`` state dict;
- without ``torch``: writes a ``.json`` sidecar with a ``gain`` scale param
  that the NumPy fallback in ``local_torch`` picks up.

Usage:
    python -m inference.scripts.train_dummy --out /models/local-gru.pt
    python -m inference.scripts.train_dummy --out inference/models/local-gru.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def gen_series(n: int, seed: int = 7) -> list[float]:
    """Generate a synthetic random-walk price series of ``n`` points."""
    rng = random.Random(seed)
    price = 100.0
    out = [price]
    for _ in range(n - 1):
        price *= 1.0 + rng.gauss(0.0, 0.002)
        out.append(price)
    return out


def windows(closes: list[float], window: int) -> tuple[list[list[float]], list[float]]:
    """Return (normalized-return windows, tanh-scaled future-return labels)."""
    xs, ys = [], []
    for i in range(len(closes) - window - 1):
        w = closes[i : i + window + 1]
        rets = [w[j] / w[j - 1] - 1.0 if w[j - 1] else 0.0 for j in range(1, len(w))]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var) + 1e-9
        xs.append([(r - mean) / std for r in rets])
        nxt = closes[i + window + 1]
        fut = nxt / closes[i + window] - 1.0 if closes[i + window] else 0.0
        ys.append(math.tanh(fut * 500.0))
    return xs, ys


def train_torch(xs: list[list[float]], ys: list[float], epochs: int) -> dict:
    """Train :class:`SignalGRU` on ``xs``/``ys`` and return its state dict."""
    import torch

    from inference.app.backends.local_torch import SignalGRU

    model = SignalGRU()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    xb = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
    yb = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point: train the dummy model and return the output path."""
    p = argparse.ArgumentParser(description="Train a dummy GRU model.")
    p.add_argument("--out", default="/models/local-gru.pt")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    closes = gen_series(args.samples, seed=args.seed)
    xs, ys = windows(closes, args.window)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch  # noqa: F401

        state = train_torch(xs, ys, args.epochs)
        torch.save({"state_dict": state, "window": args.window}, out)
        print(f"saved torch weights for {len(xs)} windows -> {out}")
    except ImportError:
        mean_abs = sum(abs(y) for y in ys) / max(len(ys), 1)
        sidecar = out.with_suffix(".json")
        sidecar.write_text(json.dumps({"gain": 1.0, "mean_abs_label": mean_abs}))
        print(f"torch not installed; wrote dummy params -> {sidecar}")
    return out


if __name__ == "__main__":
    main()
