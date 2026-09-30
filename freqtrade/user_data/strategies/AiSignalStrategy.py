"""AI-signal strategy.

Long-only spot strategy (``INTERFACE_VERSION = 3``). Entry/exit decisions
come from the inference gateway (``POST /v1/predict`` via
:mod:`_inference_client`); when the gateway is unreachable or returns an
unusable response the strategy degrades to a pure-indicator fallback rule
(RSI-based) and logs a warning. The bot loop is never crashed by a
gateway failure.

Thresholds are configurable without editing code: they are read from the
``strategy_params`` block of the rendered Freqtrade ``config.json``
(see ``ftctl/render/freqtrade.py``)::

    "strategy_params": {
        "entry_signal_min": 0.4,
        "entry_confidence_min": 0.6,
        "exit_signal_max": -0.2
    }

Gateway entry rule (applied to the latest candle, one ``/v1/predict``
call per pair with the last ``N_CANDLES``)::

    enter long if signal > entry_signal_min and confidence > entry_confidence_min
    exit       if signal < exit_signal_max

Fallback rule (per-row indicators, used when the gateway gives no signal)::

    enter long if rsi < 30 (oversold bounce)
    exit       if rsi > 70 (overbought)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

try:  # pragma: no cover - import path when loaded by Freqtrade as a package
    from ._inference_client import (
        DEFAULT_N_CANDLES,
        InferenceResult,
        dataframe_to_candles,
        query_inference_gateway,
    )
except ImportError:  # Freqtrade adds the strategy directory to sys.path
    from _inference_client import (  # type: ignore[no-redef]
        DEFAULT_N_CANDLES,
        InferenceResult,
        dataframe_to_candles,
        query_inference_gateway,
    )

try:  # pragma: no cover - real Freqtrade runtime
    from freqtrade.strategy import IStrategy
except ImportError:  # unit-test env without Freqtrade installed
    IStrategy = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Defaults mirror config/app.yaml.example so the strategy behaves sanely
# even when strategy_params is absent from the Freqtrade config.
DEFAULT_ENTRY_SIGNAL_MIN = 0.4
DEFAULT_ENTRY_CONFIDENCE_MIN = 0.6
DEFAULT_EXIT_SIGNAL_MAX = -0.2

RSI_PERIOD = 14
EMA_SHORT = 12
EMA_LONG = 26
N_CANDLES = DEFAULT_N_CANDLES

# Fallback rule thresholds (pure-indicator mode when the gateway is down).
FALLBACK_ENTRY_RSI = 30.0
FALLBACK_EXIT_RSI = 70.0


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI implemented with pure pandas (no pandas-ta/talib dep)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(100.0 * (avg_gain > 0).astype(float))


def add_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the RSI/EMA columns used by the fallback rule (pure function)."""
    df = dataframe.copy()
    df["rsi"] = compute_rsi(df["close"])
    df["ema_short"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()
    return df


def fallback_entries(dataframe: pd.DataFrame) -> pd.Series:
    """Pure-indicator entry rule: oversold bounce (``rsi < 30``)."""
    return (dataframe["rsi"] < FALLBACK_ENTRY_RSI).fillna(False)


def fallback_exits(dataframe: pd.DataFrame) -> pd.Series:
    """Pure-indicator exit rule: overbought (``rsi > 70``)."""
    return (dataframe["rsi"] > FALLBACK_EXIT_RSI).fillna(False)


class AiSignalStrategy(IStrategy):  # type: ignore[valid-type,misc]
    """Long-only spot strategy driven by the inference gateway."""

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "5m"
    startup_candle_count = 30
    process_only_new_candles = True

    minimal_roi: dict[str, float] = {}
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

    def __init__(self, config: dict[str, Any]) -> None:
        # IStrategy.__init__ (via HyperStrategyMixin) initialises
        # _ft_params_from_file / _ft_hyper_params etc. Skipping it crashes
        # StrategyResolver.load_strategy with
        # AttributeError: '_ft_params_from_file'. Fall back to a plain
        # attribute set when running in the unit-test env where IStrategy
        # is stubbed as `object` (object.__init__ takes no arguments).
        try:
            super().__init__(config)  # type: ignore[arg-type]
        except TypeError:
            self.config = config
        params = config.get("strategy_params", {})
        self.entry_signal_min = float(params.get("entry_signal_min", DEFAULT_ENTRY_SIGNAL_MIN))
        self.entry_confidence_min = float(
            params.get("entry_confidence_min", DEFAULT_ENTRY_CONFIDENCE_MIN)
        )
        self.exit_signal_max = float(params.get("exit_signal_max", DEFAULT_EXIT_SIGNAL_MAX))
        inference_cfg = config.get("inference", {})
        self.inference_url = str(
            inference_cfg.get("url", os.getenv("INFERENCE_URL", "http://inference:8000"))
        )
        self.inference_model = str(
            inference_cfg.get("model", os.getenv("INFERENCE_MODEL", "local-gru"))
        )
        # Cache of the latest gateway answer per pair so entry and exit
        # evaluation within the same bot loop share a single HTTP call.
        self._signal_cache: dict[str, tuple[Any, InferenceResult | None]] = {}

    # -- gateway access -------------------------------------------------

    def _gateway_signal(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any]
    ) -> InferenceResult | None:
        pair = metadata.get("pair", "unknown")
        if dataframe is None or len(dataframe) == 0:
            return None
        last_ts = dataframe["date"].iloc[-1] if "date" in dataframe.columns else len(dataframe)
        cache_key = f"{pair}"
        cached = self._signal_cache.get(cache_key)
        if cached is not None and cached[0] == last_ts:
            return cached[1]
        try:
            candles = dataframe_to_candles(dataframe, limit=N_CANDLES)
            result = query_inference_gateway(
                pair,
                candles,
                timeframe=self.timeframe,
                model=self.inference_model,
                url=self.inference_url,
            )
        except Exception as exc:  # never crash the bot loop
            logger.warning("Gateway signal failed for %s, using fallback: %s", pair, exc)
            result = None
        if result is None:
            logger.warning("Gateway gave no signal for %s — using indicator fallback", pair)
        self._signal_cache[cache_key] = (last_ts, result)
        return result

    # -- IStrategy API --------------------------------------------------

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any]
    ) -> pd.DataFrame:
        return add_indicators(dataframe)

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any]
    ) -> pd.DataFrame:
        df = dataframe.copy()
        df["enter_long"] = 0
        df["enter_tag"] = ""
        if df.empty:
            return df
        if "rsi" not in df.columns:
            df = add_indicators(df)
        try:
            result = self._gateway_signal(df, metadata)
        except Exception as exc:  # never crash the bot loop
            logger.warning("Entry evaluation failed, using fallback: %s", exc)
            result = None
        if result is not None:
            if (
                result.signal > self.entry_signal_min
                and result.confidence > self.entry_confidence_min
            ):
                df.loc[df.index[-1], "enter_long"] = 1
                df.loc[df.index[-1], "enter_tag"] = f"ai_{result.model}"
        else:
            mask = fallback_entries(df)
            df.loc[mask, "enter_long"] = 1
            df.loc[mask, "enter_tag"] = "fallback_rsi"
        return df

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any]
    ) -> pd.DataFrame:
        df = dataframe.copy()
        df["exit_long"] = 0
        df["exit_tag"] = ""
        if df.empty:
            return df
        if "rsi" not in df.columns:
            df = add_indicators(df)
        try:
            result = self._gateway_signal(df, metadata)
        except Exception as exc:  # never crash the bot loop
            logger.warning("Exit evaluation failed, using fallback: %s", exc)
            result = None
        if result is not None:
            if result.signal < self.exit_signal_max:
                df.loc[df.index[-1], "exit_long"] = 1
                df.loc[df.index[-1], "exit_tag"] = f"ai_exit_{result.model}"
        else:
            mask = fallback_exits(df)
            df.loc[mask, "exit_long"] = 1
            df.loc[mask, "exit_tag"] = "fallback_rsi_exit"
        return df
