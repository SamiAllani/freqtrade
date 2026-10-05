# Compose stack (Tasks 4 + 9)

Local deployment path: `docker compose up` gives a working dry-run stack on
CPU, with opt-in GPU overrides. Safety default is `dry_run: true` everywhere
(see `config/app.yaml.example`); live trading additionally requires
`FT_ALLOW_LIVE=yes` (enforced by `ftctl`, Task 2).

## Prerequisites

- Docker Engine + Compose v2.
- Rendered config (git-ignored, never commit):
  ```bash
  cp config/.env.example config/.env   # fill in secrets
  python -m ftctl.cli render freqtrade --out freqtrade/user_data/config.json
  python -m ftctl.cli render compose  --out deploy/compose/.env.generated
  ```
- For GPU overrides: NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Check with `nvidia-smi` on the host first.

## Quickstart (CPU, dry-run)

From the repo root:

```bash
docker compose -f deploy/compose/docker-compose.yml up --build
```

Services:

| Service    | Host endpoint             | Notes                                          |
|------------|---------------------------|------------------------------------------------|
| freqtrade  | `127.0.0.1:8080` (FreqUI/API) | `trade` with rendered `config.json` (read-only mount) |
| inference  | `127.0.0.1:8000`          | `/healthz`, `/metrics`, `/v1/predict`          |
| freqtrade-exporter | — (in-network `freqtrade-exporter:9108`) | polls bot `/api/v1`, exposes `/metrics` (Task 9) |
| prometheus | `127.0.0.1:9090`          | scrapes `inference:8000/metrics` + exporter |
| grafana    | `127.0.0.1:3000`          | admin/admin locally; datasource + dashboards pre-provisioned |
| mcp-server | — (no host port)          | optional, only with `--profile mcp` (Task 7)   |

All host ports bind to `127.0.0.1`, never `0.0.0.0`. `freqtrade` waits for
`inference` to be healthy (`depends_on: service_healthy`); the strategy
falls back to indicators if the gateway is unreachable (Task 3).

```bash
# optional bot-control MCP server (Task 7 provides its Dockerfile)
docker compose -f deploy/compose/docker-compose.yml --profile mcp up --build
```

## GPU overrides

Gateway GPU (CUDA image + device reservation for `inference`):

```bash
docker compose -f deploy/compose/docker-compose.yml \
               -f deploy/compose/docker-compose.gpu.yml up --build
curl http://127.0.0.1:8000/healthz   # -> {"status":"ok","gpu":true,...}
```

PyTorch FreqAI models additionally need a GPU on `freqtrade`
(separate override; not needed for gateway mode or classic-ML FreqAI):

```bash
docker compose -f deploy/compose/docker-compose.yml \
               -f deploy/compose/docker-compose.gpu.yml \
               -f deploy/compose/docker-compose.freqai-gpu.yml up --build
```

## FreqAI variant

`ftctl render compose` writes `FREQTRADE_TAG` (`stable` |
`stable_freqai` | `stable_freqaitorch`) and `FREQTRADE_ARGS`
(`--strategy <Name> [--freqaimodel <Model>]`) into `.env.generated`.
Compose passes the tag as the `freqtrade` build arg and the shell expands
`FREQTRADE_ARGS` in the container command, so the strategy (and
`--freqaimodel` when FreqAI is enabled) switches without editing compose files.

Caveats:

- Build args interpolate from the shell/project `.env`, **not** from
  `env_file`, so export the rendered env before building a non-default tag:
  `set -a; source deploy/compose/.env.generated; set +a`.
- Trained models live in `user_data/models/<identifier>` on the
  `freqtrade-user-data` named volume and survive restarts. Changing the
  feature set requires a new `freqai.identifier` (Task 11).
- Stock FreqAI images may not ship CUDA-enabled PyTorch; for GPU training
  with PyTorch models use the custom CUDA-base image from Task 11
  (`freqtrade/Dockerfile.freqai`).

## Persistence

- `freqtrade-user-data` → `/freqtrade/user_data`: FreqAI models and the
  trades DB (`--db-url sqlite:////freqtrade/user_data/tradesv3.sqlite`).
- `prometheus-data`, `grafana-data`: metrics and Grafana settings.

## Observability (Task 9)

Prometheus scrapes the inference gateway (`inference:8000/metrics`:
`gateway_predict_requests_total`, `gateway_predict_latency_seconds_*`,
`gateway_backend_errors_total`, `gateway_gpu_available`) and the
`freqtrade-exporter` service (`freqtrade-exporter:9108/metrics`), which polls
the bot REST API (`/api/v1/status`, `/profit`, `/performance`, `/balance`)
and exposes `freqtrade_open_trades`, `freqtrade_*_profit_*`,
`freqtrade_win_rate`, `freqtrade_pair_*`, and `freqtrade_up`.

Grafana starts with the Prometheus datasource and both dashboards
pre-provisioned (canonical JSON in `<repo>/dashboards/`):

- **Trading overview** (`dashboards/trading.json`): open trades, open/closed
  profit, win rate, per-pair profit, scrape errors.
- **Inference gateway** (`dashboards/inference.json`): request rate by
  model/backend, latency p50/p95 (from the `/metrics` histogram), error rate,
  backend errors, GPU availability, backend usage.

Check quickly:

```bash
curl http://127.0.0.1:8000/metrics | grep gateway_
curl http://127.0.0.1:9090/api/v1/targets  # both jobs up
```

## Files

- `docker-compose.yml` — base CPU stack.
- `docker-compose.gpu.yml` — inference GPU reservation + CUDA image.
- `docker-compose.freqai-gpu.yml` — GPU reservation for `freqtrade`
  (PyTorch FreqAI only).
- `.env.example` — optional local interpolation overrides.
- `prometheus/prometheus.yml` — scrape jobs (inference + exporter).
- `grafana/provisioning/` — datasource + dashboard provider; dashboard JSON
  lives in `<repo>/dashboards/` and is mounted into Grafana read-only.
