"""Render :class:`AppConfig` into Helm ``values`` data.

Secret *values* never appear here: credentials are referenced through
``existingSecret`` (a pre-created Kubernetes Secret) instead.
"""

from __future__ import annotations

from typing import Any

from common.contracts import AppConfig
from ftctl.render.freqtrade import freqtrade_args, freqtrade_tag

SECRET_NAME = "freqtrade-secrets"


def render_helm(cfg: AppConfig) -> dict[str, Any]:
    """Build the Helm values dict for the given app config."""
    return {
        "mode": cfg.mode,
        "dryRun": cfg.mode == "dry_run",
        "exchange": {
            "name": cfg.exchange.name,
            "pairs": list(cfg.exchange.pairs),
            "stakeCurrency": cfg.exchange.stake_currency,
            "stakeAmount": cfg.exchange.stake_amount,
            "maxStake": cfg.exchange.max_stake,
        },
        "strategy": {
            "name": cfg.strategy.name,
            "timeframe": cfg.strategy.timeframe,
            "entrySignalMin": cfg.strategy.entry_signal_min,
            "entryConfidenceMin": cfg.strategy.entry_confidence_min,
            "exitSignalMax": cfg.strategy.exit_signal_max,
            "args": freqtrade_args(cfg),
        },
        "freqtrade": {
            "strategy": cfg.strategy.name,
            "freqaimodel": cfg.freqai.model if cfg.freqai.enabled else None,
            "tag": freqtrade_tag(cfg),
        },
        "freqai": {
            "enabled": cfg.freqai.enabled,
            "identifier": cfg.freqai.identifier,
            "model": cfg.freqai.model,
            "trainPeriodDays": cfg.freqai.train_period_days,
            "backtestPeriodDays": cfg.freqai.backtest_period_days,
            "liveRetrainHours": cfg.freqai.live_retrain_hours,
            "featureParameters": {
                "includeTimeframes": list(cfg.freqai.feature_parameters.include_timeframes),
                "includeCorrPairlist": list(cfg.freqai.feature_parameters.include_corr_pairlist),
                "labelPeriodCandles": cfg.freqai.feature_parameters.label_period_candles,
            },
            "dataSplitParameters": {
                "testSize": cfg.freqai.data_split_parameters.test_size,
                "shuffle": cfg.freqai.data_split_parameters.shuffle,
            },
            "modelTrainingParameters": dict(cfg.freqai.model_training_parameters),
        },
        "inference": {
            "url": cfg.inference.url,
            "defaultModel": cfg.inference.default_model,
        },
        "apiServer": {"enabled": True},
        # Reference only: the chart mounts these keys from a K8s Secret.
        # Never put secret values in this file.
        "existingSecret": {
            "name": SECRET_NAME,
            "keys": {
                "binanceKey": "BINANCE_API_KEY",
                "binanceSecret": "BINANCE_API_SECRET",
                "apiUsername": "FT_API_USERNAME",
                "apiPassword": "FT_API_PASSWORD",
            },
        },
    }
