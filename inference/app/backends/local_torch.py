"""Local PyTorch GRU backend (Task 5).

Runs a small GRU over normalized returns on GPU when available, CPU
otherwise. ``torch`` is an optional dependency (``pip install -e .[gpu]``):
when it is not installed the backend falls back to a deterministic NumPy
heuristic with the same API contract, so the CPU image stays small and the
pipeline works end to end without CUDA libraries.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from common.contracts import PredictRequest, PredictResponse

from .base import BackendError, WindowTooSmallError

logger = logging.getLogger(__name__)

try:  # torch is optional; CPU-only images do not ship it.
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on CPU-only hosts
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

DEFAULT_WINDOW = 30
DEFAULT_MODELS_DIR = "/models"


def gpu_available() -> bool:
    """True when a CUDA device can be used by torch."""
    if not TORCH_AVAILABLE:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_device(preference: str | None) -> str:
    """Map a ``device`` pref (``auto`` | ``cuda`` | ``cpu``) to ``cuda``/``cpu``."""
    pref = (preference or "auto").lower()
    if pref == "cuda":
        return "cuda" if gpu_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    return "cuda" if gpu_available() else "cpu"  # auto


if TORCH_AVAILABLE:

    class SignalGRU(nn.Module):  # type: ignore[no-redef]
        """Tiny baseline: one-layer GRU over normalized returns -> signal."""

        def __init__(self, input_size: int = 1, hidden_size: int = 16, num_layers: int = 1) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: Any) -> Any:
            _, h = self.gru(x)
            return self.head(h[-1])

else:  # placeholder so attribute access fails loudly, not silently

    class SignalGRU:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise BackendError("torch is not installed; install the 'gpu' extra")


def _closes(req: PredictRequest) -> list[float]:
    return [c.c for c in req.candles]


def _returns(closes: list[float]) -> list[float]:
    """Simple returns r[i] = c[i]/c[i-1] - 1, guarded against zero prices."""
    out: list[float] = []
    for prev, cur in zip(closes[:-1], closes[1:], strict=True):
        if prev == 0:
            out.append(0.0)
        else:
            out.append(cur / prev - 1.0)
    return out


def heuristic_signal_confidence(returns: list[float]) -> tuple[float, float]:
    """Deterministic NumPy-free fallback used with and without torch.

    Signal is a Sharpe-like ratio squashed through tanh; confidence blends
    signal strength with directional agreement of recent returns.
    """
    n = len(returns)
    if n == 0:
        return 0.0, 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    std = math.sqrt(var)
    sharpe_like = mean / (std / math.sqrt(n) + 1e-9)
    signal = math.tanh(sharpe_like)
    direction = 1.0 if mean >= 0 else -1.0
    agree = sum(1 for r in returns if (r >= 0) == (direction >= 0)) / n
    confidence = 0.5 * abs(signal) + 0.5 * agree
    return max(-1.0, min(1.0, signal)), max(0.0, min(1.0, confidence))


class LocalTorchBackend:
    """``local`` backend: GRU on torch when available, heuristic otherwise."""

    def __init__(
        self,
        model_name: str = "local-gru",
        device: str | None = "auto",
        models_dir: str | Path = DEFAULT_MODELS_DIR,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        self.model_name = model_name
        self.device_pref = (device or "auto").lower()
        self.device = resolve_device(self.device_pref)
        self.models_dir = Path(models_dir)
        self.window = window
        self._model: Any = None
        self._cpu_fallback = False
        self._load_model()

    # -- model loading --------------------------------------------------
    def _weights_path(self) -> Path:
        return self.models_dir / f"{self.model_name}.pt"

    def _load_model(self) -> None:
        if not TORCH_AVAILABLE:
            logger.info("torch not installed; %s uses heuristic inference", self.model_name)
            return
        try:
            model = SignalGRU()
            path = self._weights_path()
            if path.exists():
                state = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                logger.info("loaded weights for %s from %s", self.model_name, path)
            else:
                logger.info("no weights at %s; using untrained baseline", path)
            if self.device == "cuda":
                model = model.to("cuda")
            model.eval()
            self._model = model
        except Exception as exc:
            logger.warning("could not init torch model %s: %s", self.model_name, exc)
            self._model = None

    def _dummy_params(self) -> dict[str, float]:
        """Optional ``<model>.json`` scale params written by train_dummy.py."""
        path = self.models_dir / f"{self.model_name}.json"
        try:
            if path.exists():
                data = json.loads(path.read_text())
                return {"gain": float(data.get("gain", 1.0))}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ignoring bad dummy params at %s: %s", path, exc)
        return {"gain": 1.0}

    # -- inference ------------------------------------------------------
    def _torch_signal(self, returns: list[float]) -> float | None:
        """Run the GRU; return None when torch inference is unavailable."""
        if not TORCH_AVAILABLE or self._model is None:
            return None
        try:
            import math as _math

            mean = sum(returns) / len(returns)
            var = sum((r - mean) ** 2 for r in returns) / len(returns)
            std = _math.sqrt(var) + 1e-9
            norm = [(r - mean) / std for r in returns]
            x = torch.tensor(norm, dtype=torch.float32).view(1, len(norm), 1)
            device = "cuda" if (self.device == "cuda" and not self._cpu_fallback) else "cpu"
            with torch.no_grad():
                out = self._model.to(device)(x.to(device))
                return float(torch.tanh(out).view(-1)[0].item())
        except Exception as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                logger.warning("CUDA inference failed, falling back to CPU: %s", exc)
                self._cpu_fallback = True
                try:
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                return None
            raise BackendError(f"torch inference failed: {exc}") from exc

    async def predict(self, req: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        if len(req.candles) < self.window + 1:
            raise WindowTooSmallError(
                f"model {self.model_name!r} needs at least {self.window + 1} candles, "
                f"got {len(req.candles)}"
            )
        closes = _closes(req)
        returns = _returns(closes)[-self.window :]

        signal: float | None = self._torch_signal(returns)
        fallback_signal, fallback_conf = heuristic_signal_confidence(returns)
        if signal is None:
            gain = self._dummy_params()["gain"]
            signal = max(-1.0, min(1.0, math.tanh(gain * fallback_signal * 2.0)))
            # Blend toward the heuristic signal when no torch model ran.
            signal = max(-1.0, min(1.0, 0.5 * signal + 0.5 * fallback_signal))
            confidence = fallback_conf
        else:
            confidence = fallback_conf

        latency_ms = (time.perf_counter() - start) * 1000.0
        return PredictResponse(
            signal=max(-1.0, min(1.0, float(signal))),
            confidence=max(0.0, min(1.0, float(confidence))),
            model=req.model or self.model_name,
            backend="local",
            latency_ms=latency_ms,
        )
