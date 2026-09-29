"""FastAPI inference gateway (Task 5).

Endpoints (see SPEC.md — Shared Contracts / Inference API):

- ``POST /v1/predict`` -> ``PredictResponse``
- ``GET /healthz`` -> ``{"status": "ok", "gpu": bool, "models": [...]}``
- ``GET /metrics`` -> Prometheus text format
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from common.contracts import AppConfig, InferenceConfig, ModelSpec, PredictRequest

from .backends.base import BackendError, WindowTooSmallError
from .backends.local_torch import gpu_available
from .registry import BackendRegistry, UnknownModelError

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "gateway_predict_requests_total",
    "Total /v1/predict requests.",
    ["model", "backend", "status"],
)
REQUEST_LATENCY = Histogram(
    "gateway_predict_latency_seconds",
    "Latency of /v1/predict requests in seconds.",
    ["model", "backend"],
)
BACKEND_ERRORS = Counter(
    "gateway_backend_errors_total",
    "Per-backend predict failures.",
    ["model", "backend"],
)
GPU_AVAILABLE = Gauge(
    "gateway_gpu_available",
    "1 when a CUDA device is available to the gateway, 0 otherwise.",
)


def default_config() -> AppConfig:
    return AppConfig(
        inference=InferenceConfig(
            default_model="local-gru",
            models={"local-gru": ModelSpec(backend="local", device="auto")},
        )
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load the central ``app.yaml``; fall back to a local-only default.

    The gateway only needs the ``inference`` section, but it reuses the
    shared ``AppConfig`` contract so validation stays in one place.
    """
    candidate = Path(path or os.environ.get("APP_CONFIG_PATH", "config/app.yaml"))
    if not candidate.is_absolute():
        # Resolve relative to the repo root (two levels above this file's package).
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            if (parent / "config" / "app.yaml.example").exists():
                repo_root = parent
                break
        else:
            repo_root = Path.cwd()
        candidate = repo_root / candidate
    if candidate.exists():
        import yaml

        try:
            data = yaml.safe_load(candidate.read_text()) or {}
            # Secrets use ${ENV_VAR} interpolation in ftctl; tolerate
            # unresolved placeholders here (they live in other sections).
            cfg = AppConfig.model_validate(data)
            logger.info("loaded inference config from %s", candidate)
            return cfg
        except Exception as exc:
            logger.warning("could not parse %s (%s); using default config", candidate, exc)
    else:
        logger.info("no config at %s; using default local-only config", candidate)
    return default_config()


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    config = cfg or load_config()
    registry = BackendRegistry.from_config(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        app.state.registry = registry
        app.state.config = config
        yield

    app = FastAPI(title="Inference Gateway", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry
    app.state.config = config

    @app.post("/v1/predict")
    async def predict(req: PredictRequest):  # type: ignore[no-untyped-def]
        try:
            name, backend = registry.resolve(req.model)
        except UnknownModelError as exc:
            REQUEST_COUNT.labels(model=req.model, backend="unknown", status="404").inc()
            raise HTTPException(status_code=404, detail=str(exc)) from None
        backend_label = getattr(backend, "backend_name", "local")
        try:
            resp = await backend.predict(req)
        except WindowTooSmallError as exc:
            REQUEST_COUNT.labels(model=name, backend=backend_label, status="422").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except BackendError as exc:
            BACKEND_ERRORS.labels(model=name, backend=backend_label).inc()
            REQUEST_COUNT.labels(model=name, backend=backend_label, status="502").inc()
            raise HTTPException(status_code=502, detail=str(exc)) from None
        REQUEST_COUNT.labels(model=name, backend=resp.backend, status="200").inc()
        REQUEST_LATENCY.labels(model=name, backend=resp.backend).observe(
            max(resp.latency_ms, 0.0) / 1000.0
        )
        return resp

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "gpu": gpu_available(),
            "models": registry.model_names(),
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        # Refresh the GPU gauge on every scrape so restarts / device changes
        # are visible without a process restart. Cheap: single cuda check.
        try:
            GPU_AVAILABLE.set(1.0 if gpu_available() else 0.0)
        except Exception:  # pragma: no cover - defensive
            GPU_AVAILABLE.set(0.0)
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


def main() -> None:  # pragma: no cover - exercised via uvicorn/Docker
    import uvicorn

    uvicorn.run(
        "inference.app.main:app",
        host="0.0.0.0",  # noqa: S104 - container serves on all interfaces
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
