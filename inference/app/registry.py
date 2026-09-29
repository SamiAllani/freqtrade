"""Backend registry (Task 5).

Builds backends from ``AppConfig.inference.models`` and routes predict
requests by their ``model`` field, falling back to ``default_model``.
The ``mcp`` backend (Task 6) is resolved lazily so a gateway without the
MCP extras still starts; unknown/misconfigured models surface as errors
at request time, not at import time.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from common.contracts import AppConfig, PredictRequest, PredictResponse

from .backends.base import BackendError, WindowTooSmallError
from .backends.local_torch import LocalTorchBackend

if TYPE_CHECKING:  # avoids importing Task 6 code at module load
    from .backends.base import Backend

logger = logging.getLogger(__name__)


class UnknownModelError(KeyError):
    """Requested model is not in the registry. Mapped to HTTP 404."""


class MisconfiguredBackendError(BackendError):
    """Model entry references a backend that cannot be built (HTTP 502)."""


def _unavailable_mcp_backend(model_name: str, detail: str) -> Backend:
    class _UnavailableMcp:
        async def predict(self, req: PredictRequest) -> PredictResponse:
            raise BackendError(f"mcp backend for model {model_name!r} unavailable: {detail}")

    return _UnavailableMcp()  # type: ignore[return-value]


class BackendRegistry:
    def __init__(self, default_model: str = "local-gru") -> None:
        self.default_model = default_model
        self._backends: dict[str, Backend] = {}

    def register(self, name: str, backend: Backend) -> None:
        self._backends[name] = backend

    @classmethod
    def from_config(cls, cfg: AppConfig, models_dir: str | None = None) -> BackendRegistry:
        reg = cls(default_model=cfg.inference.default_model)
        base_dir = models_dir or os.environ.get("MODELS_DIR", "/models")
        for name, spec in cfg.inference.models.items():
            if spec.backend == "local":
                reg.register(
                    name,
                    LocalTorchBackend(
                        model_name=name, device=spec.device or "auto", models_dir=base_dir
                    ),
                )
            elif spec.backend == "mcp":
                try:
                    from .backends import mcp_backend as _mcp  # type: ignore[attr-defined]

                    reg.register(
                        name, _mcp.McpBackend(name=name, spec=spec, servers=cfg.mcp.servers)
                    )
                except ImportError as exc:
                    logger.warning("mcp backend for %r not available: %s", name, exc)
                    reg.register(name, _unavailable_mcp_backend(name, str(exc)))
            else:  # pragma: no cover - pydantic constrains the literal
                raise MisconfiguredBackendError(f"unknown backend for model {name!r}")
        # Guarantee the default model always resolves (local baseline).
        if reg.default_model not in reg._backends:
            logger.info("default model %r not configured; adding local baseline", reg.default_model)
            reg.register(
                reg.default_model,
                LocalTorchBackend(model_name=reg.default_model, models_dir=base_dir),
            )
        return reg

    def resolve(self, requested: str | None) -> tuple[str, Backend]:
        """Return ``(name, backend)`` for ``requested`` or the default model.

        An empty ``requested`` name falls back to the default model. An
        explicit but unknown name raises :class:`UnknownModelError` (404).
        """
        name = (requested or "").strip() or self.default_model
        try:
            return name, self._backends[name]
        except KeyError:
            raise UnknownModelError(f"unknown model {requested!r}") from None

    def model_names(self) -> list[str]:
        return sorted(self._backends)


__all__ = [
    "BackendRegistry",
    "MisconfiguredBackendError",
    "UnknownModelError",
    "BackendError",
    "WindowTooSmallError",
]
