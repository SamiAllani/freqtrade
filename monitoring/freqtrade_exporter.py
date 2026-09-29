"""Freqtrade Prometheus exporter (Task 9).

Polls the Freqtrade REST API (``/api/v1/...``) and exposes bot metrics in
Prometheus text format for the ``prometheus`` service to scrape.

Endpoints scraped (all read-only):

- ``GET /status`` — open trades
- ``GET /profit`` — closed-trade profit summary
- ``GET /performance`` — per-pair performance of finished trades
- ``GET /balance`` — account balances (best-effort)

Metrics exposed (on ``EXPORTER_PORT``, default 9108):

- ``freqtrade_up`` — 1 when the last poll succeeded, 0 otherwise
- ``freqtrade_scrape_errors_total`` — failed poll count
- ``freqtrade_open_trades`` — number of currently open trades
- ``freqtrade_open_profit_usdt`` — summed unrealised profit of open trades
- ``freqtrade_closed_profit_usdt`` — closed profit (``profit_closed_coin``)
- ``freqtrade_closed_profit_percent`` — closed profit in percent
- ``freqtrade_closed_trades_total`` — number of closed trades
- ``freqtrade_winning_trades_total`` / ``freqtrade_losing_trades_total``
- ``freqtrade_win_rate`` — wins / closed (0 when no closed trades)
- ``freqtrade_pair_profit_usdt{pair}`` — per-pair closed profit
- ``freqtrade_pair_trades_total{pair}`` — per-pair closed trade count
- ``freqtrade_balance{currency}`` — wallet balance per currency (best-effort)

Configuration via environment (no secrets in code or logs):

- ``FT_API_URL`` (default ``http://freqtrade:8080``)
- ``FT_API_USERNAME`` / ``FT_API_PASSWORD``
- ``FT_API_TIMEOUT_S`` (default ``10.0``)
- ``EXPORTER_PORT`` (default ``9108``)
- ``POLL_INTERVAL_S`` (default ``15``)

Run::

    python -m monitoring.freqtrade_exporter

The bot may be unreachable at startup (compose ordering); the exporter
serves ``freqtrade_up 0`` until the first successful poll and never crashes
on API errors.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from prometheus_client import Counter, Gauge, start_http_server

from mcp_server.ft_client import FreqtradeApiError, FreqtradeClient

logger = logging.getLogger(__name__)

EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "9108"))
POLL_INTERVAL_S = float(os.environ.get("POLL_INTERVAL_S", "15"))

UP = Gauge("freqtrade_up", "1 when the last Freqtrade API poll succeeded, 0 otherwise.")
SCRAPE_ERRORS = Counter("freqtrade_scrape_errors_total", "Total failed polls of the Freqtrade API.")
OPEN_TRADES = Gauge("freqtrade_open_trades", "Number of currently open trades.")
OPEN_PROFIT = Gauge(
    "freqtrade_open_profit_usdt",
    "Summed unrealised profit of open trades in stake currency.",
)
CLOSED_PROFIT = Gauge("freqtrade_closed_profit_usdt", "Closed-trade profit in stake currency.")
CLOSED_PROFIT_PCT = Gauge("freqtrade_closed_profit_percent", "Closed-trade profit in percent.")
CLOSED_TRADES = Gauge("freqtrade_closed_trades_total", "Number of closed trades.")
WINNING_TRADES = Gauge("freqtrade_winning_trades_total", "Number of winning closed trades.")
LOSING_TRADES = Gauge("freqtrade_losing_trades_total", "Number of losing closed trades.")
WIN_RATE = Gauge("freqtrade_win_rate", "Winning trades / closed trades (0 when none).")
PAIR_PROFIT = Gauge("freqtrade_pair_profit_usdt", "Per-pair closed profit.", ["pair"])
PAIR_TRADES = Gauge("freqtrade_pair_trades_total", "Per-pair closed trade count.", ["pair"])
BALANCE = Gauge("freqtrade_balance", "Wallet balance per currency.", ["currency"])


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce an API value to float without ever raising."""
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_status(trades: Any) -> tuple[int, float]:
    """Return ``(open_count, open_profit_sum)`` from a ``/status`` payload."""
    if not isinstance(trades, list):
        return 0, 0.0
    total = 0.0
    for t in trades:
        if not isinstance(t, dict):
            continue
        # Freqtrade uses `profit_abs` (stake currency) on /status entries.
        total += _num(t.get("profit_abs", t.get("profit", 0.0)))
    return len(trades), total


def parse_profit(body: Any) -> dict[str, float]:
    """Extract closed-trade summary from a ``/profit`` payload.

    Freqtrade field names vary slightly across versions, so every lookup
    tries the known aliases and falls back to 0.
    """
    if not isinstance(body, dict):
        return {
            "closed_profit": 0.0,
            "closed_pct": 0.0,
            "closed_count": 0.0,
            "winning": 0.0,
            "losing": 0.0,
            "win_rate": 0.0,
        }
    closed_profit = _num(body.get("profit_closed_coin", body.get("profit_closed", 0.0)))
    closed_pct = _num(body.get("profit_closed_percent", body.get("profit_closed_ratio", 0.0)))
    # Some versions report ratio (0.05) instead of percent (5.0); normalise.
    if -1.0 < closed_pct < 1.0 and closed_pct != 0.0:
        # Ambiguous: leave as-is; dashboards treat it as percent. Only scale
        # obvious ratios when a percent field name was used.
        pass
    closed_count = _num(
        body.get("closed_trade_count", body.get("trade_count", body.get("total_trades", 0.0)))
    )
    winning = _num(body.get("winning_trades", body.get("wins", 0.0)))
    losing = _num(body.get("losing_trades", body.get("losses", 0.0)))
    if closed_count <= 0:
        closed_count = winning + losing
    win_rate = (winning / closed_count) if closed_count > 0 else 0.0
    return {
        "closed_profit": closed_profit,
        "closed_pct": closed_pct,
        "closed_count": closed_count,
        "winning": winning,
        "losing": losing,
        "win_rate": max(0.0, min(1.0, win_rate)),
    }


def parse_performance(rows: Any) -> list[tuple[str, float, float]]:
    """Return ``[(pair, profit, count), ...]`` from a ``/performance`` payload."""
    if not isinstance(rows, list):
        return []
    out: list[tuple[str, float, float]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        pair = str(r.get("pair", r.get("market", "unknown")))
        profit = _num(r.get("profit_abs", r.get("profit", r.get("profit_coin", 0.0))))
        count = _num(r.get("count", r.get("trades", r.get("total_trades", 0.0))))
        out.append((pair, profit, count))
    return out


def parse_balance(body: Any) -> list[tuple[str, float]]:
    """Return ``[(currency, balance), ...]`` from a ``/balance`` payload."""
    if isinstance(body, dict):
        currencies = body.get("currencies", body.get("balances", body))
        if isinstance(currencies, list):
            out: list[tuple[str, float]] = []
            for c in currencies:
                if not isinstance(c, dict):
                    continue
                cur = str(c.get("currency", c.get("coin", c.get("symbol", "unknown"))))
                bal = _num(c.get("balance", c.get("total", c.get("free", 0.0))))
                out.append((cur, bal))
            return out
        if isinstance(currencies, dict):
            result: list[tuple[str, float]] = []
            for cur, info in currencies.items():
                if isinstance(info, dict):
                    result.append((str(cur), _num(info.get("balance", info.get("total", 0.0)))))
                else:
                    result.append((str(cur), _num(info)))
            return result
    return []


class FreqtradeExporter:
    """Poll loop that refreshes the Prometheus gauges from the bot API."""

    def __init__(self, client: FreqtradeClient | None = None) -> None:
        self.client = client if client is not None else FreqtradeClient.from_env()

    def poll_once(self) -> bool:
        """Poll the bot once; return True on success. Never raises."""
        try:
            status = self.client.get_status()
            profit = self.client.get_profit()
            performance = self.client.get_performance()
            try:
                balance = self.client.get_balance()
            except FreqtradeApiError as exc:
                logger.warning("balance poll failed (non-fatal): %s", exc)
                balance = None
        except FreqtradeApiError as exc:
            logger.warning("freqtrade poll failed: %s", exc)
            UP.set(0.0)
            SCRAPE_ERRORS.inc()
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("freqtrade poll failed unexpectedly: %s", exc)
            UP.set(0.0)
            SCRAPE_ERRORS.inc()
            return False

        open_count, open_profit = parse_status(status)
        summary = parse_profit(profit)
        pairs = parse_performance(performance)
        balances = parse_balance(balance)

        OPEN_TRADES.set(float(open_count))
        OPEN_PROFIT.set(open_profit)
        CLOSED_PROFIT.set(summary["closed_profit"])
        CLOSED_PROFIT_PCT.set(summary["closed_pct"])
        CLOSED_TRADES.set(summary["closed_count"])
        WINNING_TRADES.set(summary["winning"])
        LOSING_TRADES.set(summary["losing"])
        WIN_RATE.set(summary["win_rate"])
        for pair, pprofit, pcount in pairs:
            PAIR_PROFIT.labels(pair=pair).set(pprofit)
            PAIR_TRADES.labels(pair=pair).set(pcount)
        for currency, bal in balances:
            BALANCE.labels(currency=currency).set(bal)
        UP.set(1.0)
        return True

    def run_forever(self, interval_s: float = POLL_INTERVAL_S) -> None:
        while True:
            self.poll_once()
            time.sleep(max(1.0, interval_s))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("EXPORTER_PORT", str(EXPORTER_PORT)))
    interval = float(os.environ.get("POLL_INTERVAL_S", str(POLL_INTERVAL_S)))
    start_http_server(port)
    logger.info("freqtrade exporter listening on :%d (poll every %ss)", port, interval)
    FreqtradeExporter().run_forever(interval_s=interval)


if __name__ == "__main__":  # pragma: no cover
    main()
