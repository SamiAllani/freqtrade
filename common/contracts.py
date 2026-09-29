"""Shared Pydantic v2 contracts for the Freqtrade AI trading app.

All tasks must import shared models from here. Do not rename fields
without updating this file (see SPEC.md — Shared Contracts).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ExchangeConfig(BaseModel):
    name: str = "binance"
    pairs: list[str] = Field(default_factory=list)
    stake_currency: str = "USDT"
    stake_amount: float = 50.0
    max_stake: float = 100.0


class StrategyConfig(BaseModel):
    signal_source: Literal["gateway", "freqai", "hybrid"] = "gateway"
    name: str = "AiSignalStrategy"
    timeframe: str = "5m"
    entry_signal_min: float = 0.4
    entry_confidence_min: float = 0.6
    exit_signal_max: float = -0.2


class FeatureParameters(BaseModel):
    include_timeframes: list[str] = Field(default_factory=lambda: ["5m", "1h"])
    include_corr_pairlist: list[str] = Field(default_factory=list)
    label_period_candles: int = 24


class DataSplitParameters(BaseModel):
    test_size: float = 0.33
    shuffle: bool = False


class FreqAiConfig(BaseModel):
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
    backend: Literal["local", "mcp"]
    device: Literal["auto", "cuda", "cpu"] | None = None
    server: str | None = None
    tool: str | None = None


class InferenceConfig(BaseModel):
    url: str = "http://inference:8000"
    default_model: str = "local-gru"
    models: dict[str, ModelSpec] = Field(default_factory=dict)


class McpServerSpec(BaseModel):
    transport: Literal["stdio", "http"]
    command: list[str] | None = None
    url: str | None = None


class McpConfig(BaseModel):
    servers: dict[str, McpServerSpec] = Field(default_factory=dict)


class SecretsConfig(BaseModel):
    binance_key: str = ""
    binance_secret: str = ""
    ft_api_username: str = ""
    ft_api_password: str = ""


class AppConfig(BaseModel):
    mode: Literal["dry_run", "live"] = "dry_run"
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    freqai: FreqAiConfig = Field(default_factory=FreqAiConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)

    @model_validator(mode="after")
    def _check_freqai_enabled(self) -> "AppConfig":
        if self.strategy.signal_source in ("freqai", "hybrid") and not self.freqai.enabled:
            raise ValueError(
                f"strategy.signal_source={self.strategy.signal_source!r} "
                "requires freqai.enabled=true"
            )
        return self


class Candle(BaseModel):
    t: int
    o: float
    h: float
    l: float  # noqa: E741  # OHLCV wire name fixed by inference API contract
    c: float
    v: float


class PredictRequest(BaseModel):
    pair: str
    timeframe: str = "5m"
    model: str = "local-gru"
    candles: list[Candle] = Field(min_length=1)


class PredictResponse(BaseModel):
    signal: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model: str
    backend: Literal["local", "mcp"]
    latency_ms: float
