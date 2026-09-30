# Tutorial: running the app and using Freqtrade with different strategies

This is a hands-on walkthrough for the Freqtrade AI Trading App. It starts with
a working dry-run stack, then focuses on **the part people ask about most: how to
run Freqtrade with different strategies** — the built-in AI strategy, a custom
or stock strategy, and the FreqAI / hybrid paths.

> Everything here defaults to **`dry_run: true`** (paper trading). Live trading
> requires an explicit opt-in — see [Going live](#going-live-safety) at the end.

See [README.md](../README.md) for the full reference, [STRATEGIES.md](STRATEGIES.md)
for the field-by-field strategy reference, and [SPEC.md](../SPEC.md) for the
architecture and contracts.

---

## 1. The mental model (read this first)

There is **one source of truth**: `config/app.yaml` (plus `config/.env` for
secrets). The `ftctl` CLI renders that single config into every deployment
target:

```
config/app.yaml + config/.env
        │
        ▼  python -m ftctl.cli render ...
   ┌────────────────────────────────────────────┐
   │ freqtrade/user_data/config.json            │  ← Freqtrade's own config
   │ deploy/compose/.env.generated              │  ← Compose runtime env
   │ deploy/helm/values.generated.yaml          │  ← Helm values
   └────────────────────────────────────────────┘
```

Two fields select the strategy, and they are coupled:

| `app.yaml` field                | Becomes                                             |
|---------------------------------|-----------------------------------------------------|
| `strategy.name`                 | Freqtrade `--strategy <name>` **and** `config.json` `strategy` |
| `strategy.signal_source`        | `FT_SIGNAL_SOURCE` / Helm `strategy.signalSource`   |
| `strategy.entry_*` / `exit_*`   | `config.json` → `strategy_params` (read by the strategy) |

**Key consequence:** the strategy is chosen by name from
`freqtrade/user_data/strategies/`. To run a different strategy you (1) put its
`.py` file there and (2) point `strategy.name` at it. You never edit the strategy
logic inside Compose/Helm.

---

## 2. Install and configure

Prerequisites: Python 3.11+, Docker Engine + Compose v2.

```bash
# 1. Local Python deps for the CLI + gateway
pip install -e ".[dev,ftctl,gateway]"

# 2. Secrets file (git-ignored; placeholders are fine for dry-run)
cp config/.env.example config/.env

# 3. Start from the working example config
cp config/app.yaml.example config/app.yaml
```

The config defaults use `strategy: AiSignalStrategy` and
`signal_source: gateway`. Leave them as-is for the first run.

---

## 3. Validate, render, run

```bash
# Validate interpolation + safety guards (never prints secrets)
python -m ftctl.cli validate

# Render the Freqtrade config and the Compose env
python -m ftctl.cli render freqtrade --out freqtrade/user_data/config.json
python -m ftctl.cli render compose  --out deploy/compose/.env.generated

# Boot the CPU dry-run stack
docker compose -f deploy/compose/docker-compose.yml up --build
```

Then check the services:

| Service            | URL                              |
|--------------------|----------------------------------|
| FreqUI / bot API   | http://127.0.0.1:8080            |
| Inference gateway  | http://127.0.0.1:8000/healthz    |
| Prometheus         | http://127.0.0.1:9090            |
| Grafana            | http://127.0.0.1:3000 (admin/admin) |

```bash
curl http://127.0.0.1:8000/healthz    # {"status":"ok","gpu":...,"models":[...]}
curl http://127.0.0.1:8000/metrics | grep gateway_
```

Inspect the effective config any time (secret values are redacted):

```bash
python -m ftctl.cli show
python -m ftctl.cli show --format json
```

---

## 4. Strategies, in depth

> For a field-by-field reference of the strategy class itself — `INTERFACE_VERSION`,
> `stoploss`, `minimal_roi`, `order_types`, `startup_candle_count`, the entry/exit
> columns, and the callback methods — see [STRATEGIES.md](STRATEGIES.md). This
> section covers *running* strategies; that guide covers *writing* them.

### 4.1 How Freqtrade is invoked

The `freqtrade` container runs:

```
freqtrade trade --config /freqtrade/user_data/config.json \
                --db-url sqlite:////freqtrade/user_data/tradesv3.sqlite \
                ${FREQTRADE_ARGS}
```

`FREQTRADE_ARGS` is rendered by `ftctl` as `--strategy <Name>` (plus
`--freqaimodel <Model>` when FreqAI is enabled). Strategy files are bind-mounted
read-only from `freqtrade/user_data/strategies/`.

> ⚠️ **Compose gotcha:** `FREQTRADE_ARGS` is interpolated by Compose from the
> **shell / project `.env`**, *not* from `env_file`. So changing
> `.env.generated` alone does **not** change the running strategy — you must
> export it (or restart with it exported). This is also why the default is
> `--strategy AiSignalStrategy`: if you don't export anything, that's what runs,
> regardless of `config.json`. Helm does not have this issue because the args are
> rendered into values.

### 4.2 The built-in AI strategy (`AiSignalStrategy`)

`freqtrade/user_data/strategies/AiSignalStrategy.py` is a long-only spot
strategy (`INTERFACE_VERSION = 3`) driven by the inference gateway:

- **Entry:** `signal > entry_signal_min and confidence > entry_confidence_min`
- **Exit:** `signal < exit_signal_max`
- **Fallback** (gateway unreachable / unusable response): pure-indicator rule —
  enter when `RSI < 30`, exit when `RSI > 70`. The bot loop never crashes; it
  just logs a warning and keeps trading on indicators.

It makes **one `POST /v1/predict` call per pair** (last 50 candles), caches the
answer for the bot loop, and reads its thresholds from `strategy_params`.

Tune it **without touching code** by editing `config/app.yaml`:

```yaml
strategy:
  signal_source: gateway
  name: AiSignalStrategy
  timeframe: 5m
  entry_signal_min: 0.4       # enter above this signal
  entry_confidence_min: 0.6   # ...and above this confidence
  exit_signal_max: -0.2       # exit below this signal
```

Which model the strategy asks for comes from the `inference:` section
(`default_model`, per-model `backend: local|mcp`). See
[§4.6](#46-switching-the-inference-model-backend) for details.

### 4.3 The three `signal_source` values

| `signal_source` | Strategy it expects | Status in this repo |
|-----------------|---------------------|---------------------|
| `gateway`       | `AiSignalStrategy`  | ✅ Implemented |
| `freqai`        | `FreqAiStrategy`    | ⚠️ Plumbing only (config/image render, but the `.py` is not included yet) |
| `hybrid`        | `HybridStrategy`    | ⚠️ Same as FreqAI — reuses `_inference_client.py` when added |

`common/contracts.py` **rejects** `signal_source: freqai|hybrid` unless
`freqai.enabled: true`:

```yaml
strategy: {signal_source: freqai}
freqai:   {enabled: true, ...}
```

To actually run `freqai`/`hybrid` today you must add the corresponding strategy
file (see [§4.5](#45-adding-your-own-strategy)); otherwise Freqtrade will fail to
resolve the strategy name at startup.

### 4.4 Running a different (stock or custom) strategy

Any standard Freqtrade `IStrategy` works. Two steps:

**Step 1 — drop the `.py` into the strategies directory.** Freqtrade resolves
strategies from `user_data/strategies/`, which Compose mounts read-only:

```
freqtrade/user_data/strategies/MyRsiStrategy.py
```

A minimal, self-contained example (no TA-Lib needed — pure pandas):

```python
"""Simple RSI mean-reversion strategy (example)."""

import pandas as pd
from freqtrade.strategy import IStrategy


class MyRsiStrategy(IStrategy):
    """Long-only RSI oversold/overbought example."""

    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False
    startup_candle_count = 30
    process_only_new_candles = True
    stoploss = -0.10
    minimal_roi: dict[str, float] = {}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        delta = dataframe["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = gain / loss.replace(0, float("nan"))
        dataframe["rsi"] = (100 - 100 / (1 + rs)).fillna(100.0)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["rsi"] < 30, ["enter_long", "enter_tag"]] = (1, "rsi_oversold")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["rsi"] > 70, ["exit_long", "exit_tag"]] = (1, "rsi_overbought")
        return dataframe
```

**Step 2 — point `strategy.name` at it** in `config/app.yaml`:

```yaml
strategy:
  signal_source: gateway   # this example doesn't call the gateway, but the value
  name: MyRsiStrategy      # is still required by the contract
  timeframe: 5m
```

**Step 3 — re-render and restart Compose** (remember the interpolation gotcha):

```bash
python -m ftctl.cli validate
python -m ftctl.cli render freqtrade --out freqtrade/user_data/config.json
python -m ftctl.cli render compose  --out deploy/compose/.env.generated

# Export FREQTRADE_ARGS so Compose picks up the new --strategy
set -a; source deploy/compose/.env.generated; set +a

docker compose -f deploy/compose/docker-compose.yml up -d --force-recreate freqtrade
```

Confirm Freqtrade picked it up:

```bash
docker compose -f deploy/compose/docker-compose.yml logs freqtrade | grep -i strategy
```

> Use a stock Freqtrade strategy (from the Freqtrade repo or a community repo)
> the same way. If it lives in a subdirectory, either move it directly into
> `user_data/strategies/` or pass `--strategy-path` via `FREQTRADE_ARGS`.

### 4.5 Adding your own strategy

For anything beyond the example above, follow the Freqtrade strategy contract:

1. **Create the file** in `freqtrade/user_data/strategies/`.
2. **Declare** `INTERFACE_VERSION = 3`, `timeframe`, `stoploss`, `startup_candle_count`.
3. **Implement** `populate_indicators`, `populate_entry_trend`
   (`enter_long`/`enter_tag`) and `populate_exit_trend` (`exit_long`/`exit_tag`).
4. **Read tunables from config** instead of hard-coding them, so `ftctl` stays
   the single source of truth:

   ```python
   def __init__(self, config):
       super().__init__(config)
       params = config.get("strategy_params", {})
       self.entry_signal_min = float(params.get("entry_signal_min", 0.4))
   ```

   Any key you add under `strategy:` in `app.yaml` lands in
   `config.json` → `strategy_params` only if the renderer forwards it — see
   `ftctl/render/freqtrade.py` to add new keys.
5. **To call the inference gateway**, import the shared client the same way the
   built-in strategy does:

   ```python
   from _inference_client import query_inference_gateway, dataframe_to_candles

   result = query_inference_gateway(
       pair,
       dataframe_to_candles(dataframe),
       timeframe=self.timeframe,
       model="local-gru",
       url="http://inference:8000",
   )  # -> InferenceResult | None; never raises
   ```

6. **Select it** by name in `app.yaml` and re-render as in
   [§4.4](#44-running-a-different-stock-or-custom-strategy).

The shared client (`_inference_client.py`) is deliberately fail-safe: a 2 s
timeout, one retry on transport errors and 502/503/504 (no retry on 404/422),
and `None` on any failure so your strategy can fall back instead of raising.

### 4.6 Switching the inference model / backend

`AiSignalStrategy` asks for a model name resolved by the gateway's registry.
Change which backend serves it in `config/app.yaml`:

```yaml
inference:
  url: http://inference:8000
  default_model: local-gru
  models:
    local-gru:  {backend: local, device: auto}                  # in-process PyTorch GRU
    remote-llm: {backend: mcp, server: llm-tools, tool: predict} # external model via MCP
mcp:
  servers:
    llm-tools: {transport: stdio, command: ["python", "-m", "some_mcp_server"]}
```

- `backend: local` → in-process PyTorch GRU (falls back to a deterministic
  heuristic when torch isn't installed, e.g. the CPU image).
- `backend: mcp` → calls an external tool on a configured MCP server. The tool
  must return JSON `{signal, confidence}`; free text or non-conforming output is
  rejected and logged (never traded on), with lazy connect, a 10 s per-call
  timeout and a circuit breaker (3 failures → 60 s recovery).

To point the strategy at a non-default model, set `inference.default_model`, or
have your custom strategy pass a specific `model=` to `query_inference_gateway`.

### 4.7 FreqAI and hybrid strategies

The config, image tag and CLI args already render:

```yaml
strategy: {signal_source: freqai}
freqai:
  enabled: true
  identifier: ft-freqai-v1        # bump whenever the feature set changes
  model: LightGBMRegressor        # → --freqaimodel; "torch" ⇒ stable_freqaitorch image
  train_period_days: 30
  backtest_period_days: 7
```

`ftctl` then renders `FREQTRADE_TAG` and `--freqaimodel` for you. **What's
missing in this repo:** the `FreqAiStrategy.py` / `HybridStrategy.py` files
themselves (see the README Roadmap). Until you add them:

- `signal_source: freqai|hybrid` passes config validation (when
  `freqai.enabled: true`) but Freqtrade will fail to resolve the strategy name.
- Add a FreqAI strategy file under `user_data/strategies/` (subclassing
  `IStrategy` and implementing `feature_engineering_*` / `populate_*` per the
  Freqtrade FreqAI docs), name it in `strategy.name`, and re-render.
- Build the image with the right tag — export the rendered env before building:

  ```bash
  set -a; source deploy/compose/.env.generated; set +a
  docker compose -f deploy/compose/docker-compose.yml up --build
  ```

For GPU-trained PyTorch FreqAI models, also add the FreqAI GPU override (needs
the NVIDIA Container Toolkit):

```bash
docker compose -f deploy/compose/docker-compose.yml \
               -f deploy/compose/docker-compose.gpu.yml \
               -f deploy/compose/docker-compose.freqai-gpu.yml up --build
```

Trained models persist in `user_data/models/<identifier>` on the named volume.

---

## 5. Backtesting and hyperopt per strategy

Run Freqtrade CLI subcommands against the **same** rendered config and strategy
directory. With Compose, override the command (the image entrypoint is
`freqtrade`, so pass subcommands directly):

```bash
# Backtest MyRsiStrategy over a fixed range
docker compose -f deploy/compose/docker-compose.yml run --rm freqtrade \
  backtesting \
  --config /freqtrade/user_data/config.json \
  --strategy MyRsiStrategy \
  --timerange 20240101-20240601

# Hyperopt the same strategy
docker compose -f deploy/compose/docker-compose.yml run --rm freqtrade \
  hyperopt \
  --config /freqtrade/user_data/config.json \
  --strategy MyRsiStrategy \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces buy sell --epochs 100

# List every strategy Freqtrade can see
docker compose -f deploy/compose/docker-compose.yml run --rm freqtrade \
  list-strategies --config /freqtrade/user_data/config.json
```

Running Freqtrade installed locally instead of in Docker? Use the same commands
without the `docker compose ... run --rm freqtrade` prefix and point
`--strategy-path` at the folder:

```bash
freqtrade backtesting --config freqtrade/user_data/config.json \
  --strategy MyRsiStrategy --strategy-path freqtrade/user_data/strategies \
  --timerange 20240101-20240601
```

> Backtesting needs historical data. Download it first with
> `freqtrade download-data --config ... --timeframe 5m --timerange ...`
> (add `--pairs BTC/USDT ETH/USDT`), or let the command prompt you.

---

## 6. Optional: bot-control MCP server

An assistant can inspect the running bot through the MCP server (no host port;
read-only unless writes are enabled):

```bash
docker compose -f deploy/compose/docker-compose.yml --profile mcp up --build
```

Read tools: `get_status`, `get_open_trades`, `get_profit`, `get_balance`,
`get_performance`. Write tools (`pause_trading`, `force_exit`) exist **only**
when `MCP_ALLOW_WRITE=yes`. There is no tool that opens new positions.

---

## 7. Troubleshooting strategies

| Symptom | Likely cause / fix |
|---------|--------------------|
| Trades tagged `fallback_rsi` | Gateway unreachable — check `INFERENCE_URL`, `docker compose logs inference`, `curl 127.0.0.1:8000/healthz` |
| Freqtrade exits: `invalid choice` / unknown strategy | `strategy.name` doesn't match a file in `user_data/strategies/`, or `FREQTRADE_ARGS` wasn't exported before restart |
| Still running the old strategy after `render` | Compose interpolates `FREQTRADE_ARGS` from the shell — `set -a; source deploy/compose/.env.generated; set +a` then recreate |
| `signal_source=freqai` rejected | Set `freqai.enabled: true` in `app.yaml` |
| `/v1/predict` → 404 / 422 / 502 | 404 unknown model; 422 too few candles; 502 backend down / MCP breaker open |
| `gpu: false` on `/healthz` despite a GPU | CPU image in use — use the `.gpu.yml` override and the CUDA image |

---

## 8. Going live (safety)

Live trading is opt-in in **two** independent places and cannot be enabled by
accident:

1. `mode: live` in `config/app.yaml`, **and**
2. `FT_ALLOW_LIVE=yes` in the environment.

Live mode additionally rejects missing or placeholder-looking secrets
(`placeholder`/`changeme`/`example`, empty, or `${...}`), and
`stake_amount <= max_stake` is enforced in **every** mode.

```bash
python -m ftctl.cli validate          # fails loudly if live isn't fully configured
python -m ftctl.cli render freqtrade --out freqtrade/user_data/config.json
```

> Not trading advice. Backtest and paper-trade before risking capital.
