# Freqtrade AI Trading App — Agent Task Breakdown

This document decomposes the project into 11 tasks. Each task can be handed to a separate coding agent. Read the **Assumptions**, **Architecture** and **Shared Contracts** sections first: every agent must follow them.

## Goals

A trading app built on Freqtrade that:

- automates trading on Binance
- runs locally via Docker Compose
- deploys to Kubernetes
- can use a GPU to run models locally
- can use external models through MCP
- can train and run models in-process with FreqAI (Freqtrade's built-in ML module), as an alternative or complement to the inference gateway
- centralizes all configuration in one place

## Assumptions

- **Language:** Python 3.11 for all code.
- **Trading engine:** the official Freqtrade Docker image, unmodified. Extras are added as strategy code and sidecar services. The base image is `freqtradeorg/freqtrade:stable`. When FreqAI is enabled, use a FreqAI variant instead (`stable_freqai`, or `stable_freqaitorch` for PyTorch models). Verify the exact tags in the Freqtrade docs when building.
- **Two model paths, one switch:** `strategy.signal_source` selects `gateway` (Inference Gateway, Tasks 3, 5, 6), `freqai` (FreqAI, trained and run inside the Freqtrade process, Task 11) or `hybrid` (both must agree to enter).
- **Safety default:** `dry_run: true` everywhere. Live trading requires an explicit flag in the central config **and** an environment guard (`FT_ALLOW_LIVE=yes`).
- **MCP for external models:** the inference service contains an MCP client that calls model-providing MCP servers. An optional separate MCP server exposes the bot for inspection and control.
- **UI:** no custom frontend. Use FreqUI (bundled with Freqtrade) and Grafana.
- **Repo:** monorepo.

## Architecture

```
                 ┌──────────────────────────────┐
 app.yaml + .env │  ftctl (config renderer)     │
 ───────────────►│  → freqtrade config.json     │
                 │  → compose .env / overrides  │
                 │  → Helm values.yaml          │
                 └──────────────┬───────────────┘
                                ▼
 ┌──────────────┐  HTTP   ┌───────────────────┐  local   ┌─────────────┐
 │  Freqtrade   │────────►│ Inference Gateway │─────────►│ GPU backend │
 │  + Strategy  │ /v1/... │ (FastAPI)         │          │ (PyTorch)   │
 │  (Binance)   │         │                   │─── MCP ─►│             │
 └──────┬───────┘         └───────────────────┘  client  └─────────────┘
        │ REST API                                  └────► external MCP servers
        ▼
 ┌──────────────┐    ┌─────────────────────┐
 │ FreqUI       │    │ MCP server          │
 │ Prometheus/  │    │ (bot status/control)│
 │ Grafana      │    └─────────────────────┘
 └──────────────┘
```

### FreqAI path

FreqAI is not a client of the gateway. It runs **inside** the Freqtrade process: it builds features from candles, trains a model (LightGBM, XGBoost, CatBoost or PyTorch), retrains it periodically on a sliding window in a background thread, and hands predictions to the strategy. Trained models live in `user_data/models/<identifier>` and must be on a persistent volume. With `signal_source: freqai` the gateway is not needed at all; with `hybrid` both are used.

### Repo layout

```
repo/
├── common/            contracts.py (shared pydantic models)
├── config/            app.yaml.example, .env.example
├── ftctl/             central config CLI (Python package)
├── freqtrade/         Dockerfile(s), user_data/strategies, user_data/freqaimodels
├── inference/         FastAPI gateway + backends
├── mcp_server/        bot control MCP server
├── deploy/compose/    docker-compose.yml + overrides
├── deploy/helm/       Helm chart
└── tests/
```

## Shared Contracts

### Central config (`config/app.yaml`)

```yaml
mode: dry_run            # dry_run | live
exchange:
  name: binance
  pairs: ["BTC/USDT", "ETH/USDT"]
  stake_currency: USDT
  stake_amount: 50
  max_stake: 100
strategy:
  signal_source: gateway   # gateway | freqai | hybrid
  name: AiSignalStrategy   # gateway -> AiSignalStrategy, freqai -> FreqAiStrategy, hybrid -> HybridStrategy
  timeframe: 5m
  entry_signal_min: 0.4
  entry_confidence_min: 0.6
  exit_signal_max: -0.2
freqai:
  enabled: false
  identifier: ft-freqai-v1       # change it whenever the feature set changes
  model: LightGBMRegressor       # passed to Freqtrade as --freqaimodel
  train_period_days: 30
  backtest_period_days: 7
  live_retrain_hours: 1
  device: auto                   # auto | cuda | cpu (PyTorch models only)
  feature_parameters:
    include_timeframes: ["5m", "1h"]
    include_corr_pairlist: ["ETH/USDT"]
    label_period_candles: 24
  data_split_parameters: {test_size: 0.33, shuffle: false}
  model_training_parameters: {}
inference:
  url: http://inference:8000
  default_model: local-gru
  models:
    local-gru: {backend: local, device: auto}        # auto | cuda | cpu
    remote-llm: {backend: mcp, server: llm-tools, tool: predict}
mcp:
  servers:
    llm-tools:
      transport: stdio                                # stdio | http
      command: ["python", "-m", "some_mcp_server"]    # for stdio
      # url: https://example.com/mcp                  # for http
secrets:
  binance_key: ${BINANCE_API_KEY}
  binance_secret: ${BINANCE_API_SECRET}
  ft_api_username: ${FT_API_USERNAME}
  ft_api_password: ${FT_API_PASSWORD}
```

### Inference API

```
POST /v1/predict
  Request:
  { "pair": "BTC/USDT", "timeframe": "5m", "model": "local-gru",
    "candles": [{"t": 1710000000, "o": 0.0, "h": 0.0, "l": 0.0, "c": 0.0, "v": 0.0}] }
  Response:
  { "signal": -1.0..1.0, "confidence": 0..1, "model": "local-gru",
    "backend": "local|mcp", "latency_ms": 12 }

GET /healthz  -> { "status": "ok", "gpu": true|false, "models": ["local-gru", ...] }
GET /metrics  -> Prometheus text format
```

Error codes: unknown model → 404, too few candles or invalid payload → 422, backend failure → 502.

### Conventions

- All shared models live in `common/contracts.py` (pydantic v2). Do not rename fields without updating that file.
- Rendered artifacts (`config.json`, `.env.generated`, `values.generated.yaml`) are generated, never committed.
- Secrets never appear in rendered files in plain text, logs, or Helm values. Use env vars / K8s Secrets.
- Lint: ruff + mypy. Tests: pytest.

---

## Task 1: Repo scaffold and shared contracts

**Category:** `infra` | **Agent:** 1 | **Depends on:** None

### Context
Foundation for every other task. Defines the workspace, tooling, and shared models.

### Objective
Create the monorepo skeleton, the shared pydantic schema for config and the inference API, and the tooling baseline.

### Technical Spec
- **Files:** `pyproject.toml` (uv or poetry workspace), `Makefile`, `.pre-commit-config.yaml` (ruff, mypy), `.gitignore` (excludes `.env`, `user_data/`, `*.generated.*`), `common/contracts.py`, `config/app.yaml.example`, `config/.env.example`, `README.md` stub.
- **`contracts.py`:** pydantic v2 models `AppConfig`, `ExchangeConfig`, `StrategyConfig`, `ModelSpec`, `McpServerSpec`, `FreqAiConfig`, `PredictRequest`, `PredictResponse`, matching the shared contracts above exactly. Validation rule: `signal_source` of `freqai` or `hybrid` requires `freqai.enabled: true`.
- **Makefile targets:** `up`, `down`, `lint`, `test`, `render-config`.

### Acceptance Criteria
- [ ] `make lint` and `make test` run (empty tests pass)
- [ ] `AppConfig.model_validate` loads `app.yaml.example`
- [ ] `.env` is git-ignored and `.env.example` contains no real values

### Notes for the Agent
Every other task imports models from `common/contracts.py`. Keep it small and stable.

---

## Task 2: Centralized configuration (`ftctl`)

**Category:** `backend` | **Agent:** 2 | **Depends on:** Task 1 (contracts only)

### Context
One YAML file plus one `.env` must drive every deployment target (Freqtrade, Compose, Helm).

### Objective
Build the `ftctl` CLI that validates the central config and renders target-specific files.

### Technical Spec
- **Files:** `ftctl/cli.py` (typer), `ftctl/loader.py`, `ftctl/render/freqtrade.py`, `ftctl/render/compose.py`, `ftctl/render/helm.py`, `ftctl/guards.py`, `tests/test_ftctl.py`.
- **Commands:**
  - `ftctl validate`
  - `ftctl render freqtrade --out freqtrade/user_data/config.json`
  - `ftctl render compose --out deploy/compose/.env.generated`
  - `ftctl render helm --out deploy/helm/values.generated.yaml`
  - `ftctl show` (secrets redacted)
- **Loader:** parse YAML, then `${ENV_VAR}` interpolation from `.env` and the process environment. Fail on unresolved variables.
- **Freqtrade render:** map `mode` to `dry_run`; fill the exchange block, `pairlists`, `api_server` (listen `0.0.0.0` inside the container, credentials from env), and `stake_*`. When `freqai.enabled`, also render the `freqai` block (`identifier`, `train_period_days`, `backtest_period_days`, `live_retrain_hours`, `feature_parameters`, `data_split_parameters`, `model_training_parameters`). `freqai.model` is not part of `config.json`: Freqtrade takes it as `--freqaimodel`, so the compose and Helm renderers must pass it as a command argument next to `--strategy` (derived from `strategy.name`).
- **Helm render:** emit `existingSecret` references, not secret values.
- **Guards:** `mode: live` requires `FT_ALLOW_LIVE=yes` in the environment, and `stake_amount <= max_stake`. On violation, print a clear message and exit non-zero.
- **Edge cases:** missing secrets are allowed in dry-run (use placeholders) but rejected in live mode.

### Acceptance Criteria
- [ ] The same `app.yaml` renders valid Freqtrade JSON, compose env, and Helm values
- [ ] `ftctl show` never prints secret values
- [ ] Live mode without the guard variable fails
- [ ] Unit tests cover interpolation, guards, and each renderer

### Notes for the Agent
Rendered files are generated artifacts. Never write plain-text secrets into any rendered file that could be committed.

---

## Task 3: Freqtrade image and AI-signal strategy

**Category:** `integration` | **Agent:** 3 | **Depends on:** None (uses the inference API contract only)

### Context
The trading engine. The strategy asks the inference gateway for signals and must degrade gracefully when it is unavailable.

### Objective
Provide a Freqtrade image and a strategy that consumes `/v1/predict`, with a pure-indicator fallback.

### Technical Spec
- **Files:** `freqtrade/Dockerfile` (`FROM freqtradeorg/freqtrade:stable`, adds `httpx`), `freqtrade/user_data/strategies/AiSignalStrategy.py`, `freqtrade/user_data/strategies/_inference_client.py`.
- **Strategy:** `INTERFACE_VERSION = 3`, long-only spot. `populate_indicators` adds RSI/EMA (used as fallback features). `populate_entry_trend` calls `POST /v1/predict` per pair with the last N candles. Enter long if `signal > entry_signal_min` and `confidence > entry_confidence_min`. Exit if `signal < exit_signal_max`. Thresholds come from the config `strategy` block.
- **Client:** 2 s timeout, 1 retry. On failure, fall back to the indicator rule and log a warning. Never crash the bot loop.
- **Edge cases:** gateway down, malformed response, empty candles, pair missing from the response.

### Acceptance Criteria
- [ ] `freqtrade backtesting` runs with the gateway mocked
- [ ] Gateway failure triggers the fallback and does not raise
- [ ] Thresholds are configurable without editing code

### Notes for the Agent
Do not modify Freqtrade itself. Keep all logic in the strategy and client modules. This task covers `signal_source: gateway` only. FreqAI is Task 11, and its hybrid strategy reuses `_inference_client.py` from this task.

---

## Task 4: Docker Compose stack (CPU and GPU)

**Category:** `infra` | **Agent:** 4 | **Depends on:** Tasks 3, 5 (build contexts only)

### Context
Local deployment path.

### Objective
`docker compose up` gives a working local stack in dry-run, with an opt-in GPU override.

### Technical Spec
- **Files:** `deploy/compose/docker-compose.yml`, `deploy/compose/docker-compose.gpu.yml`, `deploy/compose/.env.example`, `deploy/compose/README.md`.
- **Services:** `freqtrade` (API port bound to `127.0.0.1:8080`), `inference`, `mcp-server` (optional profile), `prometheus`, `grafana`.
- **GPU override** adds to `inference`:
  ```yaml
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  ```
  Usage: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`. The override also switches to the GPU Dockerfile.
- **Config:** mount the rendered `config.json` read-only; secrets via `env_file`.
- **FreqAI variant:** add a build arg `FREQTRADE_TAG` (default `stable`; `stable_freqai` or `stable_freqaitorch` when FreqAI is used). Mount a persistent volume on `user_data/models` so trained models survive restarts. The `freqtrade` service command takes `--strategy` and `--freqaimodel` from the rendered env.
- **GPU for FreqAI:** for PyTorch FreqAI models the GPU reservation must also apply to the `freqtrade` service (a separate override file or a profile). The stock FreqAI images may not ship CUDA-enabled PyTorch; check the Freqtrade docs, and if needed build a custom image on a CUDA base.
- **Healthchecks:** `inference` checks `/healthz`; `freqtrade` uses `depends_on` with `condition: service_healthy`.

### Acceptance Criteria
- [ ] The CPU stack starts in dry-run without any GPU
- [ ] The GPU override reports `gpu: true` on `/healthz` on an NVIDIA host
- [ ] Ports are bound to localhost, not `0.0.0.0`
- [ ] Volumes persist `user_data` and the trades DB

### Notes for the Agent
Document the NVIDIA Container Toolkit prerequisite in the README.

---

## Task 5: Inference gateway with local GPU backend

**Category:** `backend` | **Agent:** 5 | **Depends on:** Task 1

### Context
Central model-serving component. Local models run here; the MCP backend (Task 6) plugs into the same interface.

### Objective
A FastAPI service that serves `/v1/predict` from a local model on GPU when available, CPU otherwise.

### Technical Spec
- **Files:** `inference/app/main.py`, `inference/app/registry.py`, `inference/app/backends/base.py`, `inference/app/backends/local_torch.py`, `inference/Dockerfile`, `inference/Dockerfile.gpu` (CUDA runtime base), `inference/models/README.md`, `inference/scripts/train_dummy.py`.
- **Backend interface:** `class Backend(Protocol): async def predict(self, req: PredictRequest) -> PredictResponse`.
- **`local_torch`:** load a model file from `/models`. `device=auto` selects `cuda` if `torch.cuda.is_available()`. Include a tiny baseline model (small GRU over normalized returns) and a script that trains a dummy one so the pipeline works end to end.
- **Registry:** builds backends from `AppConfig.inference.models`; routes by the request's `model` field; falls back to `default_model`.
- **Endpoints:** `/v1/predict`, `/healthz` (reports GPU availability and loaded models), `/metrics` (request count, latency histogram, per-backend errors).
- **Edge cases:** unknown model → 404; CUDA OOM → fall back to CPU with a warning; fewer candles than the window → 422.

### Acceptance Criteria
- [ ] Works on CPU with no CUDA libraries installed
- [ ] Uses the GPU when present (verified by `/healthz`)
- [ ] Matches the API contract exactly
- [ ] Tests cover routing and error paths using a fake backend

### Notes for the Agent
Keep torch behind an extras group (e.g. `inference[gpu]`) so the CPU image stays small.

---

## Task 6: MCP client backend for external models

**Category:** `integration` | **Agent:** 6 | **Depends on:** Task 5 (backend interface only)

### Context
Lets the gateway delegate predictions to external models exposed through MCP servers.

### Objective
Implement a backend that calls a configured MCP tool and converts its result into a `PredictResponse`.

### Technical Spec
- **Files:** `inference/app/backends/mcp_backend.py`, `inference/app/mcp_pool.py`, `tests/test_mcp_backend.py`, `tests/fake_mcp_server.py`.
- **Library:** the official `mcp` Python SDK. Support `stdio` and streamable HTTP transports as set in `mcp.servers`.
- **Flow:** on `predict`, build a compact payload (pair, timeframe, summary statistics of the recent candles, not raw data by default), call the configured `tool`, and parse the result. The tool must return JSON matching `{signal, confidence}`. Clamp values to valid ranges.
- **Pool:** lazy connect, reconnect on failure, per-call timeout (default 10 s), circuit breaker (opens after 3 consecutive failures for 60 s).
- **Edge cases:** tool returns free text (reject and log), server unreachable (breaker opens, error surfaces so the strategy falls back), hung call (timeout).

### Acceptance Criteria
- [ ] The fake MCP server round-trips a prediction
- [ ] The breaker opens and recovers as specified
- [ ] Non-conforming output never produces a signal
- [ ] Servers are configurable only via `app.yaml`

### Notes for the Agent
Treat MCP tool output as untrusted input. Only the numeric `signal` and `confidence` fields may influence trading decisions.

---

## Task 7: MCP server for bot monitoring and control

**Category:** `integration` | **Agent:** 7 | **Depends on:** None (uses the Freqtrade REST API)

### Context
Optional service so MCP-capable assistants can inspect the bot.

### Objective
Expose a read-mostly toolset over the Freqtrade REST API.

### Technical Spec
- **Files:** `mcp_server/server.py`, `mcp_server/ft_client.py`, `mcp_server/Dockerfile`.
- **Read tools:** `get_status`, `get_open_trades`, `get_profit`, `get_balance`, `get_performance`.
- **Write tools** (registered only when `MCP_ALLOW_WRITE=yes`): `pause_trading` (stopentry), `force_exit(trade_id)`. There is **no** `force_enter` tool.
- **Client:** Freqtrade REST with JWT login; credentials from env; re-login on token expiry.
- **Edge cases:** bot unreachable, invalid trade id.

### Acceptance Criteria
- [ ] Read tools work against a dry-run bot
- [ ] Write tools are absent from the tool list when the flag is off
- [ ] No tool can open new positions

---

## Task 8: Kubernetes Helm chart

**Category:** `infra` | **Agent:** 8 | **Depends on:** Tasks 3, 5 (images and ports)

### Context
Cluster deployment path, with optional GPU scheduling for inference.

### Objective
A Helm chart that deploys the stack to Kubernetes.

### Technical Spec
- **Files:** `deploy/helm/Chart.yaml`, `values.yaml`, `templates/freqtrade-deployment.yaml`, `templates/inference-deployment.yaml`, `templates/mcp-server-deployment.yaml`, `templates/services.yaml`, `templates/configmap.yaml`, `templates/secret.yaml` (optional), `templates/pvc.yaml`, `templates/servicemonitor.yaml` (optional), `templates/NOTES.txt`.
- **Freqtrade:** `replicas: 1` with `strategy: Recreate` (two traders must never run at once). PVC for `user_data`, which must include `user_data/models` when FreqAI is enabled. Config from a ConfigMap; credentials from `existingSecret`. `freqtrade.image.tag` selects the FreqAI variant, and `freqtrade.gpu.enabled` mirrors the inference GPU toggle for PyTorch models. FreqAI retrains in a background thread, so give this pod higher CPU and memory requests than in gateway mode.
- **Inference GPU toggle:** `inference.gpu.enabled=true` sets `resources.limits["nvidia.com/gpu"]: 1`, plus configurable `nodeSelector`, `tolerations`, and optional `runtimeClassName: nvidia`.
- **Probes:** liveness and readiness on `/healthz`. The Freqtrade API is ClusterIP only, no Ingress by default.
- **Edge cases:** GPU node unavailable (pod stays Pending; document this), missing secret (clear failure).

### Acceptance Criteria
- [ ] `helm lint` and `helm template` pass with default and GPU values
- [ ] Freqtrade can never run more than 1 replica
- [ ] Deploys to kind/minikube in dry-run with CPU inference

### Notes for the Agent
The GPU path requires the NVIDIA device plugin on the cluster. Document it; do not install it from the chart.

---

## Task 9: Observability

**Category:** `infra` | **Agent:** 9 | **Depends on:** None

### Context
Metrics and dashboards for the bot and the gateway.

### Objective
Provision Prometheus and Grafana with dashboards for trading and inference.

### Technical Spec
- **Files:** `deploy/compose/prometheus/prometheus.yml`, `deploy/compose/grafana/provisioning/datasources/*.yml`, `deploy/compose/grafana/provisioning/dashboards/*.yml`, `dashboards/trading.json`, `dashboards/inference.json`.
- **Sources:** Freqtrade `/api/v1` via a small exporter, and the inference `/metrics` endpoint.
- **Panels:** open trades, profit, win rate; gateway latency (p50/p95), error rate, GPU vs CPU backend usage.

### Acceptance Criteria
- [ ] Grafana starts with the datasource and dashboards pre-provisioned
- [ ] The inference dashboard shows data from `/metrics`

---

## Task 10: Testing, safety checks and CI

**Category:** `testing` | **Agent:** 10 | **Depends on:** Tasks 1–3, 11 (run last, or against stubs)

### Context
Guard rails that keep the bot from accidentally trading live.

### Objective
Automated checks for correctness, packaging, and safety defaults.

### Technical Spec
- **Files:** `.github/workflows/ci.yml`, `tests/e2e/test_stack_dryrun.py`, `tests/test_safety.py`.
- **CI jobs:** lint (ruff, mypy), unit tests, `helm lint`, `docker compose config`, image builds, secret scan.
- **Safety tests:** the rendered config from the example `app.yaml` always has `dry_run: true`; live mode without `FT_ALLOW_LIVE` fails; no secrets appear in rendered artifacts or logs.
- **E2E:** compose up in dry-run with a fake model, wait for `/healthz`, assert the bot loop runs and the strategy received a signal.
- **FreqAI:** a CI smoke test that runs `freqtrade backtesting` with `--freqaimodel LightGBMRegressor` on a small dataset (short `train_period_days`), and a check that `signal_source: freqai` without `freqai.enabled` is rejected.

### Acceptance Criteria
- [ ] CI is green on a fresh clone
- [ ] Safety tests fail if a default is flipped to live
- [ ] A secret-scan step is included

---

## Task 11: FreqAI strategies and configuration

**Category:** `integration` | **Agent:** 11 | **Depends on:** Task 1 (contracts); Task 3 only for the hybrid strategy's inference client

### Context
FreqAI is Freqtrade's built-in ML module. It engineers features from candles, trains a model, retrains it periodically on a live sliding window, and exposes predictions to the strategy. It runs inside the Freqtrade process, so it is an alternative to the Inference Gateway, not a client of it.

### Objective
Add a FreqAI strategy and a hybrid strategy, plus the config plumbing so `strategy.signal_source` switches between `gateway`, `freqai` and `hybrid` without code changes.

### Technical Spec
- **Files:** `freqtrade/user_data/strategies/FreqAiStrategy.py`, `freqtrade/user_data/strategies/HybridStrategy.py`, `freqtrade/user_data/freqaimodels/` (optional custom model classes), `freqtrade/Dockerfile.freqai`, short `docs/freqai.md`.
- **FreqAiStrategy:** implement `feature_engineering_expand_all` (RSI, EMA, ATR, returns over `indicator_periods_candles`), `feature_engineering_expand_basic`, `feature_engineering_standard`, and `set_freqai_targets` (label: future return over `label_period_candles`, e.g. `&-s_close`). `populate_indicators` calls `self.freqai.start(dataframe, metadata, self)`. Enter long when `do_predict == 1` and the prediction is above the configured threshold; exit when it drops below the exit threshold. Thresholds come from the config `strategy` block.
- **HybridStrategy:** enter long only if the FreqAI prediction is bullish **and** the gateway signal passes its thresholds; exit if either says exit. If the gateway is down, degrade to the FreqAI-only decision and log a warning. If `do_predict == 0` (outlier), never enter.
- **Models:** default `LightGBMRegressor` (CPU, fast, a good baseline). Optional PyTorch model (for example `PyTorchMLPRegressor`) using `freqai.device`. Reinforcement learning (`stable_freqairl` image) is out of scope for the first version.
- **Persistence:** models, training data and metadata are stored in `user_data/models/<identifier>` on a persistent volume. Changing the feature set requires a new `identifier`.
- **Edge cases:** cold start (`startup_candle_count` must cover the longest indicator period, the first training takes time, and no entries are allowed until a model exists), outlier detection (`do_predict == 0`), retraining during live trading (runs in the background while the bot keeps using the previous model), pairlist changes (new training needed).

### Acceptance Criteria
- [ ] `freqtrade backtesting` with FreqAI runs on a small dataset and produces a trades result
- [ ] `signal_source` switches between the strategies without editing code
- [ ] No entry is opened while `do_predict == 0` or before the first model is trained
- [ ] Hybrid mode falls back to FreqAI-only when the gateway is unreachable
- [ ] Trained models persist across a container restart

### Notes for the Agent
Keep `shuffle: false` in `data_split_parameters` so the train/test split respects time order. FreqAI handles the sliding-window training itself, so do not add a second training loop. Training is CPU-heavy: keep `train_period_days` small in CI and dev configs.

---

## Summary

| # | Task | Category | Depends On | Complexity |
|---|------|----------|------------|------------|
| 1 | Scaffold and contracts | infra | None | Low |
| 2 | Central config (`ftctl`) | backend | 1 (contracts) | Med |
| 3 | Freqtrade image and strategy | integration | None | Med |
| 4 | Docker Compose (CPU/GPU) | infra | 3, 5 | Med |
| 5 | Inference gateway and GPU | backend | 1 | High |
| 6 | MCP client backend | integration | 5 (interface) | High |
| 7 | MCP control server | integration | None | Med |
| 8 | Helm chart | infra | 3, 5 | Med |
| 9 | Observability | infra | None | Low |
| 10 | Tests, safety and CI | testing | 1–3, 11 | Med |
| 11 | FreqAI strategies and config | integration | 1 (3 for hybrid) | High |

## Parallelization Plan

- **Wave 0 (first, small):** Task 1, so the shared contracts exist.
- **Wave 1 (fully parallel):** Tasks 2, 3, 5, 7, 9, 11. Task 11's hybrid strategy imports the client from Task 3, so it can stub it until Task 3 lands.
- **Wave 2 (parallel):** Task 6 (needs the backend interface from Task 5), Tasks 4 and 8 (need image contexts from Tasks 3 and 5).
- **Wave 3:** Task 10, which ties everything together.

## Safety Note

Keep the bot in dry-run until strategies have been backtested and paper-traded. This document is engineering scaffolding, not trading advice.
