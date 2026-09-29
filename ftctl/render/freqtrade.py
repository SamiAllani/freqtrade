"""Render :class:`AppConfig` into a Freqtrade ``config.json`` document.

``freqai.model`` is deliberately *not* part of the JSON: Freqtrade takes it
as the ``--freqaimodel`` CLI flag, so the compose and Helm renderers carry
it as a command argument next to ``--strategy`` (see :func:`freqtrade_args`).
"""

from __future__ import annotations

from typing import Any

from common.contracts import AppConfig

API_LISTEN_IP = "0.0.0.0"
API_PORT = 8080


def freqtrade_args(cfg: AppConfig) -> list[str]:
    """CLI args selecting the strategy (and FreqAI model when enabled)."""
    args = ["--strategy", cfg.strategy.name]
    if cfg.freqai.enabled:
        args += ["--freqaimodel", cfg.freqai.model]
    return args


def freqtrade_tag(cfg: AppConfig) -> str:
    """Docker image variant: ``stable`` by default, FreqAI variants when enabled."""
    if not cfg.freqai.enabled:
        return "stable"
    model = cfg.freqai.model.lower()
    if "pytorch" in model or "torch" in model:
        return "stable_freqaitorch"
    return "stable_freqai"


def render_freqtrade(cfg: AppConfig) -> dict[str, Any]:
    """Build the Freqtrade configuration dict for the given app config."""
    dry_run = cfg.mode == "dry_run"
    config: dict[str, Any] = {
        "max_open_trades": 3,
        "stake_currency": cfg.exchange.stake_currency,
        "stake_amount": cfg.exchange.stake_amount,
        "dry_run": dry_run,
        "dry_run_wallet": 1000,
        "cancel_open_orders_on_exit": True,
        "timeframe": cfg.strategy.timeframe,
        "strategy": cfg.strategy.name,
        # Custom block read by the AI strategies; Freqtrade ignores unknown
        # top-level keys, so this passes thresholds without code changes.
        "strategy_params": {
            "entry_signal_min": cfg.strategy.entry_signal_min,
            "entry_confidence_min": cfg.strategy.entry_confidence_min,
            "exit_signal_max": cfg.strategy.exit_signal_max,
        },
        "exchange": {
            "name": cfg.exchange.name,
            "enabled": True,
            "key": cfg.secrets.binance_key,
            "secret": cfg.secrets.binance_secret,
            "pair_whitelist": list(cfg.exchange.pairs),
            "pair_blacklist": [],
        },
        "pairlists": [{"method": "StaticPairList"}],
        # --- Order execution defaults (required by Exchange.validate_config).
        # The upstream image reads config["entry_pricing"] / config["exit_pricing"]
        # with direct dict access, so omitting them crashes the bot with
        # KeyError: 'exit_pricing' before schema defaults apply.
        # price_side="other" (not the stock template's "same") because
        # AiSignalStrategy uses market orders, which freqtrade requires to
        # price from the opposite side of the book.
        "trading_mode": "spot",
        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "exit_timeout_count": 0,
            "unit": "minutes",
        },
        "entry_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1,
            },
        },
        "exit_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1,
            },
        },
        "api_server": {
            "enabled": True,
            "listen_ip_address": API_LISTEN_IP,
            "listen_port": API_PORT,
            "username": cfg.secrets.ft_api_username,
            "password": cfg.secrets.ft_api_password,
        },
        "bot_name": "freqtrade",
        "initial_state": "running",
        "db_url": "sqlite:///tradesv3.sqlite",
    }
    if cfg.freqai.enabled:
        config["freqai"] = {
            "enabled": True,
            "identifier": cfg.freqai.identifier,
            "train_period_days": cfg.freqai.train_period_days,
            "backtest_period_days": cfg.freqai.backtest_period_days,
            "live_retrain_hours": cfg.freqai.live_retrain_hours,
            "feature_parameters": {
                "include_timeframes": list(cfg.freqai.feature_parameters.include_timeframes),
                "include_corr_pairlist": list(cfg.freqai.feature_parameters.include_corr_pairlist),
                "label_period_candles": cfg.freqai.feature_parameters.label_period_candles,
            },
            "data_split_parameters": {
                "test_size": cfg.freqai.data_split_parameters.test_size,
                "shuffle": cfg.freqai.data_split_parameters.shuffle,
            },
            "model_training_parameters": dict(cfg.freqai.model_training_parameters),
        }
    return config
