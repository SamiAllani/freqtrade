"""Shared Pydantic v2 contracts for the Freqtrade AI trading app.

All tasks must import shared models from here. Do not rename fields
without updating this file (see SPEC.md — Shared Contracts).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExchangeConfig(BaseModel):
    """Exchange and stake settings: name, pair whitelist, and stake limits."""

    name: str = "binance"
    pairs: list[str] = Field(default_factory=list)
    stake_currency: str = "USDT"
    stake_amount: float = 50.0
    max_stake: float = 100.0


class StrategyConfig(BaseModel):
    """Signal strategy settings: name, timeframe, entry/exit thresholds."""

    name: str = "AiSignalStrategy"
    timeframe: str = "5m"
    entry_signal_min: float = 0.4
    entry_confidence_min: float = 0.6
    exit_signal_max: float = -0.2


class FeatureParameters(BaseModel):
    """FreqAI feature parameters: timeframes, correlation pairs, label horizon."""

    include_timeframes: list[str] = Field(default_factory=lambda: ["5m", "1h"])
    include_corr_pairlist: list[str] = Field(default_factory=list)
    label_period_candles: int = 24


class DataSplitParameters(BaseModel):
    """Train/test split settings for FreqAI model fitting."""

    test_size: float = 0.33
    shuffle: bool = False


class FreqAiConfig(BaseModel):
    """FreqAI block: enable flag, identifier, model class, windows and params."""

    enabled: bool = False
    identifier: str = "ft-freqai-v1"
    model: str = "LightGBMRegressor"
    train_period_days: int = 30
    backtest_period_days: int = 7
    live_retrain_hours: int = 1
    device: Literal["auto", "cuda", "cpu"] = "auto"
    feature_parameters: FeatureParameters = Field(default_factory=FeatureParameters)
    data_split_parameters: DataSplitParameters = Field(default_factory=DataSplitParameters)
    model_training_parameters: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    """One inference model entry: backend plus its device/server/tool options."""

    backend: Literal["local", "mcp"]
    device: Literal["auto", "cuda", "cpu"] | None = None
    server: str | None = None
    tool: str | None = None


class InferenceConfig(BaseModel):
    """Inference gateway settings: base URL, default model, model registry."""

    url: str = "http://inference:8000"
    default_model: str = "local-gru"
    models: dict[str, ModelSpec] = Field(default_factory=dict)


class McpServerSpec(BaseModel):
    """One MCP server entry: transport plus its stdio ``command`` or http ``url``."""

    transport: Literal["stdio", "http"]
    command: list[str] | None = None
    url: str | None = None


class McpConfig(BaseModel):
    """Collection of MCP servers keyed by name."""

    servers: dict[str, McpServerSpec] = Field(default_factory=dict)


class SecretsConfig(BaseModel):
    """Secret fields, always ``${ENV_VAR}`` references (never literals)."""

    binance_key: str = ""
    binance_secret: str = ""
    ft_api_username: str = ""
    ft_api_password: str = ""


class AppConfig(BaseModel):
    """Root application config validated from the central ``app.yaml``."""

    mode: Literal["dry_run", "live"] = "dry_run"
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    freqai: FreqAiConfig = Field(default_factory=FreqAiConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)


class Candle(BaseModel):
    """One OHLCV candle as sent to the inference API."""

    t: int
    o: float
    h: float
    l: float  # noqa: E741  # OHLCV wire name fixed by inference API contract
    c: float
    v: float


class PredictRequest(BaseModel):
    """Inference request: pair/timeframe/model plus a candle window."""

    pair: str
    timeframe: str = "5m"
    model: str = "local-gru"
    candles: list[Candle] = Field(min_length=1)


class PredictResponse(BaseModel):
    """Inference response: signal, confidence, model/backend and latency."""

    signal: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model: str
    backend: Literal["local", "mcp"]
    latency_ms: float
