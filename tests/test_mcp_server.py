"""Task 7 tests: bot monitoring/control MCP server.

Covers the acceptance criteria: read tools work against a (fake) dry-run
bot, write tools are absent unless ``MCP_ALLOW_WRITE=yes``, and no tool can
open new positions. Plus client edge cases: JWT re-login on 401, bot
unreachable, invalid trade id.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from mcp_server import (
    READ_TOOL_NAMES,
    WRITE_TOOL_NAMES,
    create_server,
    is_write_enabled,
)
from mcp_server.ft_client import (
    AuthError,
    BotUnreachableError,
    FreqtradeApiError,
    FreqtradeClient,
    InvalidTradeIdError,
)

FAKE_TRADES = [
    {"trade_id": 1, "pair": "BTC/USDT", "is_open": True, "profit_abs": 1.5},
]
FAKE_PROFIT = {"profit_closed_coin": 3.0, "trade_count": 2}
FAKE_BALANCE = {"currencies": [{"currency": "USDT", "free": 100.0}]}
FAKE_PERF = [{"pair": "BTC/USDT", "profit_abs": 3.0}]


def make_transport(state: dict[str, Any] | None = None) -> httpx.MockTransport:
    """Fake dry-run Freqtrade REST API (JWT login + canned endpoints)."""
    state = state if state is not None else {}
    state.setdefault("logins", 0)
    state.setdefault("expire_once", False)
    state.setdefault("calls", [])

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        state["calls"].append((request.method, path))

        if path == "/api/v1/token/login":
            state["logins"] += 1
            if state.get("bad_credentials"):
                return httpx.Response(401, json={"detail": "unauthorized"})
            return httpx.Response(
                200, json={"access_token": "access-1", "refresh_token": "refresh-1"}
            )

        if path == "/api/v1/token/refresh":
            return httpx.Response(200, json={"access_token": "access-2"})

        auth = request.headers.get("Authorization", "")
        if auth not in ("Bearer access-1", "Bearer access-2"):
            return httpx.Response(401, json={"detail": "unauthorized"})
        if state["expire_once"] and auth == "Bearer access-1" and path == "/api/v1/status":
            state["expire_once"] = False
            return httpx.Response(401, json={"detail": "token expired"})

        if path == "/api/v1/status" and request.method == "GET":
            return httpx.Response(200, json=FAKE_TRADES)
        if path == "/api/v1/profit":
            return httpx.Response(200, json=FAKE_PROFIT)
        if path == "/api/v1/balance":
            return httpx.Response(200, json=FAKE_BALANCE)
        if path == "/api/v1/performance":
            return httpx.Response(200, json=FAKE_PERF)
        if path == "/api/v1/stopbuy" and request.method == "POST":
            if state.get("no_stopbuy"):
                return httpx.Response(404, json={"detail": "not found"})
            return httpx.Response(200, json={"status": "entries stopped"})
        if path == "/api/v1/pause" and request.method == "POST":
            return httpx.Response(200, json={"status": "paused"})
        if path == "/api/v1/forceexit" and request.method == "POST":
            body = json.loads(request.content or b"{}")
            if str(body.get("tradeid")) == "1":
                return httpx.Response(200, json={"result": "exited", "trade_id": 1})
            return httpx.Response(404, json={"detail": "Trade with id 999 not found"})
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


def make_client(state: dict[str, Any] | None = None, **kwargs: Any) -> FreqtradeClient:
    return FreqtradeClient(
        base_url="http://fake-bot:8080",
        username="u",
        password="p",  # noqa: S106 - test-only credential
        transport=make_transport(state),
        **kwargs,
    )


def call(server: Any, name: str, args: dict[str, Any] | None = None) -> Any:
    """Call an MCP tool in-memory and decode the JSON payload.

    Handles every ``call_tool`` shape across SDK versions: a plain block
    list, a ``(blocks, structured)`` tuple (typed tools), or a bare dict.
    """
    res = asyncio.run(server.call_tool(name, args or {}))
    if isinstance(res, tuple):
        blocks, structured = res
        if isinstance(structured, dict):
            return structured
        res = blocks
    if isinstance(res, dict):
        return res
    assert res, f"tool {name} returned no content"
    texts = [b.text for b in res if hasattr(b, "text")]
    assert texts, f"tool {name} returned no text content"
    if len(texts) == 1:
        return json.loads(texts[0])
    return [json.loads(t) for t in texts]


def tool_names(server: Any) -> set[str]:
    return {t.name for t in asyncio.run(server.list_tools())}


@pytest.fixture(autouse=True)
def _read_only_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_ALLOW_WRITE", raising=False)


# -- acceptance: read tools work -------------------------------------------


def test_read_tools_work_against_dry_run_bot() -> None:
    server = create_server(make_client())
    assert call(server, "get_status") == {"status": FAKE_TRADES}
    assert call(server, "get_open_trades") == {"open_trades": FAKE_TRADES, "count": 1}
    assert call(server, "get_profit") == FAKE_PROFIT
    assert call(server, "get_balance") == FAKE_BALANCE
    assert call(server, "get_performance") == {"performance": FAKE_PERF}


def test_client_read_methods_return_raw_bot_payloads() -> None:
    """Client methods pass the bot payloads through unwrapped."""
    client = make_client()
    assert client.get_status() == FAKE_TRADES
    assert client.get_open_trades() == FAKE_TRADES
    assert client.get_profit() == FAKE_PROFIT
    assert client.get_balance() == FAKE_BALANCE
    assert client.get_performance() == FAKE_PERF


def test_read_tool_names_match_spec() -> None:
    assert set(READ_TOOL_NAMES) == {
        "get_status",
        "get_open_trades",
        "get_profit",
        "get_balance",
        "get_performance",
    }
    assert set(WRITE_TOOL_NAMES) == {"pause_trading", "force_exit"}


# -- acceptance: write tools gated ------------------------------------------


def test_write_tools_absent_by_default() -> None:
    server = create_server(make_client())
    assert tool_names(server) == set(READ_TOOL_NAMES)


@pytest.mark.parametrize("flag", ["yes", "YES", "1", "true"])
def test_write_tools_present_when_enabled(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setenv("MCP_ALLOW_WRITE", flag)
    assert is_write_enabled()
    server = create_server(make_client())
    assert tool_names(server) == set(READ_TOOL_NAMES) | set(WRITE_TOOL_NAMES)


def test_write_tools_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_ALLOW_WRITE", "yes")
    server = create_server(make_client())
    assert call(server, "pause_trading") == {"status": "entries stopped"}
    assert call(server, "force_exit", {"trade_id": "1"}) == {
        "result": "exited",
        "trade_id": 1,
    }


# -- acceptance: no tool can open new positions -------------------------------


def test_no_tool_opens_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_ALLOW_WRITE", "yes")
    names = tool_names(create_server(make_client()))
    assert "force_exit" in names  # sanity: exit tool exists
    for forbidden in ("forceenter", "force_enter", "forcebuy", "force_buy", "start"):
        assert forbidden not in names, f"position-opening tool present: {forbidden}"


def test_source_has_no_position_entry_calls() -> None:
    for path in sorted((Path(__file__).parent.parent / "mcp_server").glob("*.py")):
        text = path.read_text()
        for forbidden in ("forceenter", "force_enter", "forcebuy", "force_buy"):
            assert forbidden not in text.lower(), f"{path.name} mentions {forbidden}"


# -- client: auth + session ---------------------------------------------------


def test_client_logs_in_once_and_reuses_token() -> None:
    state: dict[str, Any] = {}
    client = make_client(state)
    client.get_profit()
    client.get_balance()
    assert state["logins"] == 1


def test_client_relogin_on_token_expiry() -> None:
    state: dict[str, Any] = {"expire_once": True}
    client = make_client(state)
    assert client.get_status() == FAKE_TRADES  # 401 -> refresh -> retry
    assert state["logins"] == 1  # only the initial login; refresh sufficed


def test_client_login_failure_is_auth_error() -> None:
    state: dict[str, Any] = {"bad_credentials": True}
    with pytest.raises(AuthError):
        make_client(state).get_profit()


def test_client_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FT_API_URL", "http://bot:8080")
    monkeypatch.setenv("FT_API_USERNAME", "trader")
    monkeypatch.setenv("FT_API_PASSWORD", "s3cret")  # noqa: S105 - test-only
    client = FreqtradeClient.from_env()
    assert client.base_url == "http://bot:8080"
    assert client.username == "trader"


# -- client: edge cases ---------------------------------------------------------


def test_bot_unreachable() -> None:
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = FreqtradeClient(
        base_url="http://down:8080",
        username="u",
        password="p",  # noqa: S106 - test-only credential
        transport=httpx.MockTransport(_boom),
    )
    with pytest.raises(BotUnreachableError):
        client.get_status()


def test_tool_reports_unreachable_as_error_dict() -> None:
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = FreqtradeClient(base_url="http://down:8080", transport=httpx.MockTransport(_boom))
    server = create_server(client)
    result = call(server, "get_status")
    assert result["ok"] is False
    assert "unreachable" in result["error"]


def test_force_exit_invalid_trade_id_never_signals() -> None:
    state: dict[str, Any] = {}
    client = make_client(state)
    with pytest.raises(InvalidTradeIdError):
        client.force_exit("999")  # bot answers 404
    with pytest.raises(InvalidTradeIdError):
        client.force_exit("   ")  # rejected before any HTTP call
    assert issubclass(InvalidTradeIdError, FreqtradeApiError)


def test_force_exit_error_surfaced_as_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_ALLOW_WRITE", "yes")
    server = create_server(make_client())
    result = call(server, "force_exit", {"trade_id": "999"})
    assert result["ok"] is False
    assert "999" in result["error"]


def test_pause_trading_calls_stopbuy() -> None:
    state: dict[str, Any] = {}
    client = make_client(state)
    assert client.pause_trading() == {"status": "entries stopped"}
    assert ("POST", "/api/v1/stopbuy") in state["calls"]


def test_pause_trading_falls_back_to_pause_endpoint() -> None:
    state: dict[str, Any] = {"no_stopbuy": True}
    client = make_client(state)
    assert client.pause_trading() == {"status": "paused"}
    assert ("POST", "/api/v1/pause") in state["calls"]


def test_no_forceenter_endpoint_is_ever_called(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, Any] = {}
    monkeypatch.setenv("MCP_ALLOW_WRITE", "yes")
    server = create_server(make_client(state))
    call(server, "pause_trading")
    call(server, "force_exit", {"trade_id": "1"})
    for method, path in state["calls"]:
        assert "forceenter" not in path and "forcebuy" not in path, (method, path)
