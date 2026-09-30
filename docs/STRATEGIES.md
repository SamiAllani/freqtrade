# Freqtrade strategy fields — a reference guide

A Freqtrade **strategy** is a Python class that subclasses `IStrategy`. It
declares a handful of **class attributes ("fields")** that tell Freqtrade how to
run the bot, and implements a few **methods** that produce trading signals.

This guide documents the fields you can set and what each one does. For the
hands-on walkthrough (running, switching, backtesting strategies), see
[TUTORIAL.md](TUTORIAL.md#4-strategies-in-depth). For the config schema that
selects the strategy, see the `strategy:` section in
[README.md](../README.md#strategy).

> Fields marked with **★** are the ones this repo's
> `freqtrade/user_data/strategies/AiSignalStrategy.py` actually sets. Everything
> else is optional and available to any strategy you add.

---

## 1. Where strategies live and how one is selected

| Thing | Value |
|-------|-------|
| Directory | `freqtrade/user_data/strategies/` (bind-mounted read-only by Compose) |
| Selected by | `strategy.name` in `config/app.yaml` → `--strategy <Name>` and `config.json` `strategy` |
| Timeframe | the strategy's `timeframe` field (falls back to config `timeframe`) |
| Gateway client | `_inference_client.py` (shared; see [TUTORIAL §4.5](TUTORIAL.md#45-adding-your-own-strategy)) |

Freqtrade resolves the class **by filename and class name**, so the file
`MyRsiStrategy.py` must define `class MyRsiStrategy(IStrategy)`.

---

## 2. Class-level fields

### 2.1 Interface and identity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `INTERFACE_VERSION` **★** | `int` | `3` | Strategy interface version. `3` is the current long/short-capable interface — use it for new strategies. |
| `timeframe` **★** | `str` | `"5m"` | Candle timeframe the bot runs on (`5m`, `15m`, `1h`, …). Must match a downloaded-data timeframe for backtesting. |
| `can_short` **★** | `bool` | `False` | Enables short entries. Requires `trading_mode = futures`; spot is long-only. |
| `startup_candle_count` **★** | `int` | `0` | Extra candles fetched before signals so indicators are warmed up. Raise it when your indicator lookback exceeds 1 candle (the repo uses `30`). |
| `process_only_new_candles` **★** | `bool` | `True` | Evaluate only the newest candle instead of recomputing every row each loop. Leave `True` unless you need full-history recalculation. |
| `position_adjustment_enable` | `bool` | `False` | Enables `adjust_trade_position` (DCA/scaling). |
| `max_entry_position_adjustment` | `int` | `-1` | Max DCA orders per position (`-1` = unlimited). |

### 2.2 Entry / exit signal handling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_entry_signal` | `bool` | `True` | If `False`, the entry columns are ignored (only `minimal_roi`/stoploss exits remain). |
| `use_exit_signal` **★** | `bool` | `True` | If `False`, you need a custom exit/stoploss to leave trades. |
| `exit_profit_only` **★** | `bool` | `False` | Only allow exit signals while in profit. |
| `exit_profit_offset` | `float` | `0.0` | Minimum profit ratio required by `exit_profit_only`. |
| `ignore_roi_if_entry_signal` **★** | `bool` | `False` | Keep a position open while an entry signal persists, even if `minimal_roi` says exit. |
| `ignore_buying_expired_candle_after` | `int` | `0` | Seconds after candle close during which a buy is still valid (`0` = always). |

### 2.3 Risk management (stoploss / ROI / trailing)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stoploss` **★** | `float` | `-0.10` | Hard stop as a ratio of entry (`-0.10` = −10 %). Use `-0.99` if you want `minimal_roi`/custom exits to dominate. |
| `minimal_roi` **★** | `dict[str, float]` | `{"0": 0.10}` | Time→profit map, e.g. `{"0": 0.05, "60": 0.02, "120": 0}`. An empty `{}` disables ROI (the repo does this and relies on gateway/fallback exits). |
| `trailing_stop` **★** | `bool` | `False` | Enable trailing stoploss. |
| `trailing_stop_positive` | `float \| None` | `None` | Offset used once the trail activates. |
| `trailing_stop_positive_offset` | `float` | `0.0` | Profit ratio that activates the trailing stop. |
| `trailing_only_offset_is_reached` | `bool` | `False` | Start trailing only after the offset is reached. |
| `use_custom_stoploss` | `bool` | `False` | Enables the `custom_stoploss()` callback (see §4). |
| `protections` | `list` | `[]` | Cooldowns/stops (e.g. `StoplossGuard`, `MaxDrawdown`). |
| `use_exit_signal` | `bool` | `True` | See §2.2. |
| `custom_stoploss` | method | — | Per-trade dynamic stoploss callback. |

### 2.4 Order types and execution

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `order_types` **★** | `dict[str, str \| bool]` | see below | Per-action order type: `entry`, `exit`, `stoploss`, `stoploss_on_exchange`. |
| `order_time_in_force` | `dict[str, str]` | `{"entry": "gtc", "exit": "gtc"}` | Time-in-force per action: `gtc`, `fok`, `ioc`. |
| `unfilledtimeout` | `dict` | config default | How long unfilled orders live (usually set in `config.json`, not the strategy). |

The repo's `AiSignalStrategy` uses plain market orders:

```python
order_types = {
    "entry": "market",
    "exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": False,
}
```

> Market orders require `price_side = "other"`. `ftctl` already renders
> `entry_pricing`/`exit_pricing` with `price_side: "other"` for this reason
> (see `ftctl/render/freqtrade.py`).

### 2.5 Informative pairs / multi-timeframe

| Field / method | Description |
|----------------|-------------|
| `informative_pairs()` | Return `[("ETH/USDT", "1h"), …]` to load extra timeframes. |
| `informative_timeframe()` | Removed in interface v3 — use `informative_pairs()`. |
| FreqAI `feature_engineering_*` | Feature hooks used when `signal_source: freqai` (see [TUTORIAL §4.7](TUTORIAL.md#47-freqai-and-hybrid-strategies)). |

---

## 3. Entry/exit columns (the dataframe contract)

`populate_entry_trend` and `populate_exit_trend` set columns on the dataframe.
Freqtrade reads these exact names:

| Column | Where | Meaning |
|--------|-------|---------|
| `enter_long` | entry trend | `1` to open a long |
| `enter_short` | entry trend | `1` to open a short (needs `can_short` + futures) |
| `enter_tag` | entry trend | free-text reason, shown in the UI/logs |
| `exit_long` | exit trend | `1` to close a long |
| `exit_short` | exit trend | `1` to close a short |
| `exit_tag` | exit trend | free-text exit reason |

`AiSignalStrategy` tags entries `ai_<model>` when the gateway answers and
`fallback_rsi` when it falls back — useful for spotting the fallback path in
Freqtrade logs and Grafana.

Signals are normally produced **per row** from indicators. `AiSignalStrategy` is
unusual: it evaluates only the latest candle and caches one
`POST /v1/predict` call per pair per bot loop.

---

## 4. Methods (callbacks)

### Required in practice

| Method | Responsibility |
|--------|----------------|
| `populate_indicators(dataframe, metadata)` | Add indicator columns (RSI, EMA, …). Return the dataframe. |
| `populate_entry_trend(dataframe, metadata)` | Set `enter_long`/`enter_short` (+ tags). |
| `populate_exit_trend(dataframe, metadata)` | Set `exit_long`/`exit_short` (+ tags). |

### Optional

| Method | Purpose |
|--------|---------|
| `custom_stoploss(...)` | Dynamic stoploss (requires `use_custom_stoploss = True`). |
| `custom_exit(...)` | Extra exit reasons beyond ROI/stoploss. |
| `custom_stake_amount(...)` | Override stake per trade (still bounded by `max_stake` guard). |
| `confirm_trade_entry/exit(...)` | Final veto on a trade. |
| `custom_entry_price/exit_price(...)` | Override the order price. |
| `adjust_entry_price(...)` | Re-price an unfilled entry order. |
| `adjust_trade_position(...)` | DCA/scaling (requires `position_adjustment_enable = True`). |
| `leverage(...)` | Per-pair leverage for futures. |

---

## 5. The repo's strategy fields at a glance

`AiSignalStrategy.py` sets exactly these fields:

```python
INTERFACE_VERSION = 3
can_short = False
timeframe = "5m"
startup_candle_count = 30
process_only_new_candles = True  # only evaluate the newest candle

minimal_roi = {}  # disabled — exits come from signals
stoploss = -0.10
trailing_stop = False
use_exit_signal = True
exit_profit_only = False
ignore_roi_if_entry_signal = False

order_types = {
    "entry": "market",
    "exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": False,
}
```

| Field | Value | Why |
|-------|-------|-----|
| `timeframe` | `5m` | Matches `strategy.timeframe` / `FT_TIMEFRAME`. |
| `startup_candle_count` | `30` | RSI(14) + EMA(26) warm-up. |
| `process_only_new_candles` | `True` | The gateway is called once per pair per loop, not per row. |
| `minimal_roi` | `{}` | The AI/fallback exit rule is the only profit exit. |
| `stoploss` | `-0.10` | Hard −10 % backstop. |
| `order_types` | market | Simple, immediate fills for a signal-driven strategy. |

### Tunable thresholds (`strategy_params`)

Rather than hard-code numbers, the strategy reads them from the rendered
`config.json` → `strategy_params` block (populated from `config/app.yaml`):

| `app.yaml` key | `strategy_params` key | Used for |
|----------------|----------------------|----------|
| `strategy.entry_signal_min` | `entry_signal_min` | enter when `signal >` this |
| `strategy.entry_confidence_min` | `entry_confidence_min` | enter when `confidence >` this |
| `strategy.exit_signal_max` | `exit_signal_max` | exit when `signal <` this |

Add new tunables by (1) reading them in the strategy's `__init__` and (2)
forwarding them from `ftctl/render/freqtrade.py` — see
[TUTORIAL §4.5](TUTORIAL.md#45-adding-your-own-strategy).

---

## 6. Minimal skeleton

```python
"""MyStrategy — short description."""

import pandas as pd
from freqtrade.strategy import IStrategy


class MyStrategy(IStrategy):
    """One-line description of the edge."""

    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False
    startup_candle_count = 30
    process_only_new_candles = True
    stoploss = -0.10
    minimal_roi = {"0": 0.05, "120": 0}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["rsi"] = ...  # add indicators
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["rsi"] < 30, ["enter_long", "enter_tag"]] = (1, "rsi_oversold")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["rsi"] > 70, ["exit_long", "exit_tag"]] = (1, "rsi_overbought")
        return dataframe
```

Drop it in `freqtrade/user_data/strategies/`, set `strategy.name: MyStrategy` in
`config/app.yaml`, re-render, and restart — the full steps (including the
`FREQTRADE_ARGS` Compose gotcha) are in
[TUTORIAL §4.4](TUTORIAL.md#44-running-a-different-stock-or-custom-strategy).

---

## 7. Further reading

- [TUTORIAL.md](TUTORIAL.md) — running, switching, backtesting and hyperopt.
- [README.md](../README.md#strategies) — strategy selection and config reference.
- [SPEC.md](../SPEC.md) — architecture, contracts and the shared gateway client.
- Upstream Freqtrade docs — [Strategy basics](https://www.freqtrade.io/en/stable/strategy-advanced/) for the authoritative field list; this guide focuses on the fields that matter for this repo.
