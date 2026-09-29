# Freqtrade AI Trading App

Automated crypto trading on **Binance** built on the official **Freqtrade** engine,
with a **FastAPI inference gateway** (local PyTorch + external models via MCP),
a **bot-control MCP server**, **Docker Compose** for local runs, **Helm** for
Kubernetes, and **Prometheus + Grafana** observability.

**Safety default: `dry_run: true` everywhere.** Live trading requires an explicit
`mode: live` in the central config **and** `FT_ALLOW_LIVE=yes` in the environment.

> Not trading advice. Backtest and paper-trade before risking any capital.
> See [SPEC.md](SPEC.md) for the full task breakdown, architecture, and contracts.

## Features

- **One config drives everything** — `config/app.yaml` + `config/.env`, rendered by the `ftctl` CLI into Freqtrade JSON, Compose env, and Helm values.
- **AI-signal strategy** (`AiSignalStrategy`) — queries `POST /v1/predict` per pair, falls back to a pure RSI rule when the gateway is down. Never crashes the bot loop.
- **Inference gateway** — FastAPI service with a pluggable backend registry (`local` PyTorch GRU baseline, `mcp` external-model backend), Prometheus metrics, GPU auto-detect.
- **MCP on both sides** — gateway *calls* external model MCP servers (client backend with pool + circuit breaker); a separate MCP server *exposes* the bot (`get_status`, `get_profit`, …) to assistants.
- **FreqAI-ready plumbing** — `ftctl`/Compose/Helm already render the `freqai` block, image tags (`stable` / `stable_freqai` / `stable_freqaitorch`), and `--freqaimodel` args. (FreqAI/Hybrid strategies themselves are not yet in `freqtrade/user_data/strategies/` — see Roadmap.)
- **Local + cluster deploys** — Compose CPU stack with opt-in GPU overrides; Helm chart with GPU toggle, single-replica Freqtrade, PVC, ClusterIP-only API.
- **Observability included** — `freqtrade-exporter` sidecar, Prometheus scrape config, two pre-provisioned Grafana dashboards.
- **Safety tests + CI hygiene** — guards enforced in code and tested; rendered artifacts and secret values are never committed.

## Architecture

```
              ┌──────────────────────────────┐
app.yaml+.env │  ftctl (config renderer)     │
──────────────►│  → freqtrade config.json     │
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

### Signal sources (`strategy.signal_source`)

| Value     | Strategy             | Gateway needed? | Status in this repo |
|-----------|----------------------|-----------------|---------------------|
| `gateway` | `AiSignalStrategy`   | Yes             | Implemented |
| `freqai`  | `FreqAiStrategy`     | No (in-process) | Plumbing only (config renders, image tags, volumes); strategy file not yet added |
| `hybrid`  | `HybridStrategy`     | Yes + FreqAI    | Same as above — planned, reuses `_inference_client.py` |

`signal_source: freqai|hybrid` without `freqai.enabled: true` is rejected by
`common/contracts.py` validation.

## Repo layout

```
├── common/            contracts.py — shared Pydantic v2 models (single source of truth)
├── config/            app.yaml.example, .env.example (examples only; rendered files are generated)
├── ftctl/             central config CLI — loader, guards, render/{freqtrade,compose,helm}.py
├── freqtrade/         Dockerfile (+ build-arg FREQTRADE_TAG), user_data/strategies/
├── inference/         FastAPI gateway (app/main.py, registry.py, backends/), Dockerfiles, models/, scripts/train_dummy.py
├── mcp_server/        bot-control MCP server (server.py, ft_client.py, Dockerfile)
├── monitoring/        freqtrade_exporter.py (bot REST → Prometheus metrics), Dockerfile
├── deploy/compose/    docker-compose.yml + .gpu.yml + .freqai-gpu.yml, prometheus/, grafana/, README.md
├── deploy/helm/       Chart.yaml, values.yaml, templates/ (freqtrade/inference/mcp-server, services, configmap, pvc, …)
├── dashboards/        trading.json, inference.json (canonical Grafana dashboards)
├── scripts/           verify_dryrun.py — 20-check end-to-end dry-run verification
└── tests/             unit, safety, gateway, MCP, observability, e2e/ (opt-in via FT_E2E_DOCKER=1)
```

## Prerequisites

- Python **3.11+**, Docker Engine + Compose v2.
- For GPU overrides: NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`nvidia-smi` works on the host).
- For cluster deploys: Helm, a cluster (kind/minikube is fine), and the NVIDIA device plugin if you schedule GPUs.
- Install extras as needed (see `pyproject.toml`): `.[dev,ftctl,gateway]` for CLI+gateway, `.[gpu]` for torch, `.[mcp]` for MCP (`mcp>=1.9,<2`, the bot server uses the 1.x FastMCP API).

## Quickstart (local CPU, dry-run)

```bash
cp config/.env.example config/.env   # fill in secrets, never commit .env
pip install -e ".[dev,ftctl,gateway]"

# Validate the central config (interpolation + safety guards)
python3 -m ftctl.cli validate

# Render the deployment artifacts (git-ignored, never commit these)
python3 -m ftctl.cli render freqtrade --out freqtrade/user_data/config.json
python3 -m ftctl.cli render compose  --out deploy/compose/.env.generated

# Lint + tests
make lint
make test                              # 142 passed, 2 skipped
python3 scripts/verify_dryrun.py       # 20/20 checks (compose/helm steps SKIP without binaries)

# Boot the stack
docker compose -f deploy/compose/docker-compose.yml up --build
```

Then open: FreqUI/API `http://127.0.0.1:8080` · gateway `http://127.0.0.1:8000/healthz` ·
Prometheus `http://127.0.0.1:9090` · Grafana `http://127.0.0.1:3000` (admin/admin locally).
All host ports bind to `127.0.0.1`, never `0.0.0.0`.

Optional bot-control MCP server (no host port):

```bash
docker compose -f deploy/compose/docker-compose.yml --profile mcp up --build
```

GPU gateway (CUDA image + device reservation; `/healthz` reports `"gpu": true`):

```bash
docker compose -f deploy/compose/docker-compose.yml \
               -f deploy/compose/docker-compose.gpu.yml up --build
```

PyTorch FreqAI models additionally need a GPU on `freqtrade`:

```bash
docker compose -f deploy/compose/docker-compose.yml \
               -f deploy/compose/docker-compose.gpu.yml \
               -f deploy/compose/docker-compose.freqai-gpu.yml up --build
```

## Configuration (`ftctl`)

One YAML plus one `.env` drives every target. Edit `config/app.yaml` (start from
`config/app.yaml.example`); secrets stay in `config/.env`/environment as `${VAR}`
references that `ftctl` interpolates (unresolved variables are a hard error).

```bash
python3 -m ftctl.cli validate                                   # check config + guards
python3 -m ftctl.cli render freqtrade --out freqtrade/user_data/config.json
python3 -m ftctl.cli render compose  --out deploy/compose/.env.generated
python3 -m ftctl.cli render helm     --out deploy/helm/values.generated.yaml
python3 -m ftctl.cli show                                       # effective config, secrets redacted
python3 -m ftctl.cli show --format json
# Both accept --config PATH (-c) and --env-file PATH (-e); defaults are
# config/app.yaml (fallback: config/app.yaml.example) and config/.env (fallback: .env).
```

Key `app.yaml` sections (see `config/app.yaml.example` for the full schema):

- `mode: dry_run|live`, `exchange:` (name, pairs, stake_currency/amount, max_stake).
- `strategy:` (`signal_source`, `name`, `timeframe`, `entry_signal_min`, `entry_confidence_min`, `exit_signal_max`).
- `freqai:` (enabled, identifier, model → `--freqaimodel`, train/backtest/retrain windows, feature/data-split/training params).
- `inference:` (url, default_model, per-model `{backend: local|mcp, device, server, tool}`).
- `mcp.servers:` (`transport: stdio|http`, `command` or `url`).
- `secrets:` (always `${ENV_VAR}` references, never literal values).

Rendered mapping highlights: `mode` → Freqtrade `dry_run`; thresholds → a custom
`strategy_params` block the strategies read without code changes; `freqai.model`
is **not** in `config.json` (Freqtrade takes it as `--freqaimodel`, so Compose/Helm
carry it in `FREQTRADE_ARGS` / `strategy.args`); image variant comes from
`freqtrade_tag()` (`stable` → `stable_freqai` → `stable_freqaitorch` for torch models);
Helm output uses `existingSecret` references, never secret values.

## Safety model

- `mode: live` requires `FT_ALLOW_LIVE=yes` in the environment — otherwise `ftctl` exits non-zero with a clear message.
- `stake_amount <= max_stake` is enforced in every mode.
- Live mode rejects missing/placeholder secrets (empty, `${...}`, or containing `placeholder`/`changeme`/`example`).
- Dry-run tolerates missing secrets (placeholders are used so `docker compose config` works on a fresh clone).
- `tests/test_safety.py` + `scripts/verify_dryrun.py` assert the example config always renders `dry_run: true`, live-without-guard fails, and no secret values leak into rendered artifacts or `ftctl show` output.
- The MCP control server **cannot open positions** (there is no `force_enter` tool), and its write tools don't even exist unless `MCP_ALLOW_WRITE=yes`.

## Inference gateway

FastAPI service in `inference/app/`. Config reuses the shared `AppConfig` contract
(only the `inference` section matters) loaded from `APP_CONFIG_PATH`/`config/app.yaml`,
falling back to a local-only default.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/predict` | `{pair, timeframe, model, candles:[{t,o,h,l,c,v}]}` → `{signal −1..1, confidence 0..1, model, backend, latency_ms}` |
| `GET /healthz` | `{"status":"ok","gpu":bool,"models":[...]}` |
| `GET /metrics` | Prometheus text (`gateway_predict_requests_total`, `gateway_predict_latency_seconds_*`, `gateway_backend_errors_total`, `gateway_gpu_available`) |

Error codes: unknown model → **404**, too few candles/invalid payload → **422**,
backend failure → **502**. `device: auto` selects CUDA when `torch.cuda.is_available()`;
CUDA OOM falls back to CPU with a warning. The CPU image (`inference/Dockerfile`)
ships no torch/CUDA; `inference/Dockerfile.gpu` is the CUDA runtime variant.
`inference/scripts/train_dummy.py` trains a tiny baseline GRU so the pipeline works
end to end; models load from `/models` (`MODELS_DIR`).

MCP client backend (`backends/mcp_backend.py` + `mcp_pool.py`): sends a **compact
summary payload** (pair, timeframe, return/volume statistics — not raw candles unless
explicitly enabled), requires the tool to return JSON `{signal, confidence}`
(free text / non-conforming output is rejected and logged, never traded on;
out-of-range numbers are clamped), with lazy connect, per-call timeout (default
10 s), and a circuit breaker (opens after 3 consecutive failures for 60 s).
Servers are configurable only via `app.yaml`.

## Strategies

`freqtrade/user_data/strategies/`:

- **`_inference_client.py`** — shared gateway client (2 s timeout, 1 retry on transport errors and 502/503/504, no retry on 404/422). Returns `None` on any failure so strategies fall back instead of raising.
- **`AiSignalStrategy.py`** — `INTERFACE_VERSION = 3`, long-only spot. `populate_indicators` adds RSI(14)/EMA(12,26). Entry: `signal > entry_signal_min and confidence > entry_confidence_min` (one `/v1/predict` call per pair, last 50 candles, shared entry/exit cache per bot loop). Exit: `signal < exit_signal_max`. Fallback when the gateway gives no signal: enter on RSI < 30, exit on RSI > 70. Thresholds, URL, and model come from `strategy_params`/`inference` config — no code edits needed. Defaults: stoploss −10 %, `startup_candle_count = 30`, market orders.

Freqtrade image (`freqtrade/Dockerfile`, `FROM freqtradeorg/freqtrade:${FREQTRADE_TAG}`)
adds only `httpx`; all logic lives in the strategy files.

## Deploy

### Docker Compose (`deploy/compose/` — details in its [README](deploy/compose/README.md))

| Service | Endpoint | Notes |
|---------|----------|-------|
| `freqtrade` | `127.0.0.1:8080` | `trade` with read-only `config.json`; DB on `freqtrade-user-data` volume (`--db-url sqlite:////freqtrade/user_data/tradesv3.sqlite`); waits for `inference` healthy |
| `inference` | `127.0.0.1:8000` | `/healthz`, `/metrics`, `/v1/predict`; `/models` mount |
| `freqtrade-exporter` | in-network `freqtrade-exporter:9108` | polls bot REST, exposes `/metrics` |
| `prometheus` | `127.0.0.1:9090` | scrapes gateway + exporter |
| `grafana` | `127.0.0.1:3000` | datasource + dashboards pre-provisioned |
| `mcp-server` | no host port | only with `--profile mcp` |

`FREQTRADE_TAG`/`FREQTRADE_ARGS` come from the rendered `.env.generated` (export it
before building non-default tags: `set -a; source deploy/compose/.env.generated; set +a`).
Trained FreqAI models persist in `user_data/models/<identifier>` on the named volume —
bump `freqai.identifier` whenever the feature set changes.

### Helm (`deploy/helm/`)

```bash
python3 -m ftctl.cli render helm --out deploy/helm/values.generated.yaml
helm lint ./deploy/helm
helm template freqtrade-ai ./deploy/helm -f deploy/helm/values.generated.yaml
helm install freqtrade-ai ./deploy/helm -f deploy/helm/values.generated.yaml
```

- Freqtrade is pinned to `replicas: 1`, `strategy: Recreate` — two traders must never run at once.
- Freqtrade API is ClusterIP-only; no Ingress by default. Liveness/readiness probe the gateway `/healthz`.
- Credentials come from a pre-created `existingSecret` (`freqtrade-secrets`); the in-chart `secret.create` is dev/kind-only and off by default.
- GPU: `inference.gpu.enabled=true` requests `nvidia.com/gpu: 1` (+ nodeSelector/tolerations/runtimeClassName); `freqtrade.gpu.enabled` mirrors it for PyTorch FreqAI (needs a CUDA-capable FreqAI image — stock ones may not ship CUDA PyTorch). Requires the NVIDIA device plugin on the cluster (not installed by the chart).
- PVC (default 5 Gi) holds `user_data` (trades DB + FreqAI models); FreqAI mode raises CPU/memory requests for the background retraining thread. Optional `ServiceMonitor` for Prometheus Operator.

## Observability

- `monitoring/freqtrade_exporter.py` polls bot `/status`, `/profit`, `/performance`, `/balance` every 15 s and exposes `freqtrade_up`, `freqtrade_open_trades`, `freqtrade_open_profit_usdt`, `freqtrade_closed_profit_usdt[/_percent]`, `freqtrade_closed_trades_total`, `freqtrade_winning/losing_trades_total`, `freqtrade_win_rate`, `freqtrade_pair_profit_usdt{pair}`, `freqtrade_pair_trades_total{pair}`, `freqtrade_balance{currency}`. Serves `freqtrade_up 0` until the first successful poll; never crashes on API errors.
- Canonical dashboards in `dashboards/`: **trading** (open trades, profit, win rate, per-pair) and **inference** (rate by model/backend, p50/p95 latency, error rate, GPU availability). Mounted read-only into Grafana via `deploy/compose/grafana/provisioning/`.

## Testing & CI

```bash
make up            # docker compose up (base CPU stack)
make down          # docker compose down
make lint          # ruff check + ruff format --check + mypy common
make test          # pytest -q
make render-config # render freqtrade config.json
```

- `tests/`: contracts, `ftctl` (interpolation/guards/renderers), gateway routing + error paths (fake backend), MCP backend (fake MCP server, breaker open/recovery, non-conforming output rejected), MCP control server (write tools absent without the flag), observability (exporter parsing, dashboard JSON validity), safety (dry-run default, live guard, no leaked secrets), `e2e/test_stack_dryrun.py` (opt-in container test, skipped unless `FT_E2E_DOCKER=1`).
- `scripts/verify_dryrun.py` runs the full dry-run proof without Docker (config → guards → renders → secret hygiene → in-process gateway `/healthz`+`/v1/predict` → strategy rule → `compose config`/`helm lint+template` when binaries exist).
- `.pre-commit-config.yaml` runs ruff + mypy; `.gitignore` excludes `.env`, `user_data/config.json`, and all `*.generated.*` files.

## Secrets hygiene

Secrets live in `config/.env` (or the process environment) and are referenced from
`app.yaml` as `${BINANCE_API_KEY}` etc. — never as literals, never in rendered files,
logs, or Helm values (`existingSecret` keys only). `ftctl show` prints `***REDACTED***`
for every secret field. `config/.env.example` contains placeholder values only.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|-------------------|
| `ftctl` fails on live mode | Set `FT_ALLOW_LIVE=yes` **and** real (non-placeholder) secrets; check `stake_amount <= max_stake` |
| Strategy only emits `fallback_rsi` tags | Gateway unreachable/misconfigured — check `INFERENCE_URL`, `docker compose logs inference`, `curl 127.0.0.1:8000/healthz` |
| `/v1/predict` → 404 / 422 / 502 | 404: wrong `model` name (see `/healthz` models list); 422: fewer candles than the window; 502: backend down / MCP breaker open — check gateway logs + `/metrics` |
| `gpu: false` on `/healthz` despite a GPU host | CPU image in use or toolkit missing — use the `.gpu.yml` override, verify `nvidia-smi`, rebuild with `inference/Dockerfile.gpu` |
| Grafana shows no data | `curl 127.0.0.1:8000/metrics \| grep gateway_` and `:9090/api/v1/targets`; exporter needs bot API credentials from the rendered env |
| Helm pod Pending on GPU values | No GPU node / device plugin — `kubectl describe pod`, install the NVIDIA device plugin or set `gpu.enabled=false` |
| Compose build ignores `FREQTRADE_TAG` | Build args don't read `env_file` — `set -a; source deploy/compose/.env.generated; set +a` before building |

## Roadmap

- Add `FreqAiStrategy.py` / `HybridStrategy.py` (+ optional `freqaimodels/`, `freqtrade/Dockerfile.freqai` on a CUDA base, `docs/freqai.md`) per SPEC Task 11, so `signal_source` fully switches without code changes.
- CI workflow (lint → unit → `helm lint` → `compose config` → image builds → secret scan) and the small-dataset FreqAI backtest smoke test per SPEC Task 10.
- Optional hardening: auth on the gateway, ingress/TLS for the Helm chart, alerting rules on exporter/gateway metrics.
