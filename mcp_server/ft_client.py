"""Freqtrade REST client for the bot monitoring/control MCP server (Task 7).

Thin wrapper over the Freqtrade REST API (``/api/v1/...``) with JWT login.
Credentials and the base URL come from the environment so no secret is ever
hard-coded or logged:

- ``FT_API_URL`` (default ``http://127.0.0.1:8080``)
- ``FT_API_USERNAME`` / ``FT_API_PASSWORD``
- ``FT_API_TIMEOUT_S`` (default ``10.0``)

The access token expires after ~15 minutes; the client transparently
refreshes it (``POST /token/refresh``) or re-logs in when the bot answers
401, and retries the original request once.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_S = 10.0

__all__ = [
    "FreqtradeClient",
    "FreqtradeApiError",
    "BotUnreachableError",
    "AuthError",
    "InvalidTradeIdError",
    "API_PREFIX",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_S",
]


class FreqtradeApiError(Exception):
    """Base error for Freqtrade REST failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BotUnreachableError(FreqtradeApiError):
    """The bot API could not be reached (connection refused, timeout, ...)."""


class AuthError(FreqtradeApiError):
    """Login failed (bad credentials) or the session could not be renewed."""


class InvalidTradeIdError(FreqtradeApiError, ValueError):
    """The bot rejected a trade id (unknown trade, bad format)."""


class FreqtradeClient:
    """Sync client for the Freqtrade REST API with JWT session handling."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        username: str = "",
        password: str = "",
        timeout: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.timeout = timeout
        self._password = password
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._http = httpx.Client(transport=transport, timeout=timeout)

    @classmethod
    def from_env(cls, transport: httpx.BaseTransport | None = None) -> FreqtradeClient:
        """Build a client from ``FT_API_*`` environment variables."""
        try:
            timeout = float(os.environ.get("FT_API_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_S
        return cls(
            base_url=os.environ.get("FT_API_URL", DEFAULT_BASE_URL),
            username=os.environ.get("FT_API_USERNAME", ""),
            password=os.environ.get("FT_API_PASSWORD", ""),
            timeout=timeout,
            transport=transport,
        )

    # -- session handling -------------------------------------------------

    def login(self) -> None:
        """Authenticate with username/password and store the JWT pair."""
        url = f"{self.base_url}{API_PREFIX}/token/login"
        try:
            resp = self._http.post(url, auth=(self.username, self._password))
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise BotUnreachableError(f"bot at {self.base_url} unreachable: {exc}") from exc
        if resp.status_code == 401:
            raise AuthError("freqtrade login rejected (bad username/password)")
        if resp.status_code >= 400:
            raise FreqtradeApiError(
                f"freqtrade login failed: HTTP {resp.status_code} {resp.text[:200]}",
                status_code=resp.status_code,
            )
        try:
            body = resp.json()
            access = body["access_token"]
            refresh = body.get("refresh_token")
        except (ValueError, KeyError, TypeError) as exc:
            raise FreqtradeApiError("freqtrade login returned a non-JSON payload") from exc
        if not isinstance(access, str) or not access:
            raise FreqtradeApiError("freqtrade login returned no access token")
        self._access_token = access
        self._refresh_token = refresh if isinstance(refresh, str) else None

    def _try_refresh(self) -> bool:
        """Swap the refresh token for a fresh access token. Never raises."""
        if not self._refresh_token:
            return False
        url = f"{self.base_url}{API_PREFIX}/token/refresh"
        try:
            resp = self._http.post(url, headers={"Authorization": f"Bearer {self._refresh_token}"})
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning("token refresh failed (bot unreachable): %s", exc)
            return False
        if resp.status_code != 200:
            return False
        try:
            access = resp.json().get("access_token")
        except ValueError:
            return False
        if not isinstance(access, str) or not access:
            return False
        self._access_token = access
        return True

    def _reauth(self) -> None:
        """Renew the session: refresh token first, full login as fallback."""
        if self._try_refresh():
            return
        self.login()

    # -- low-level request -------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        token = self._access_token or ""
        return {"Authorization": f"Bearer {token}"}

    def _request(
        self, method: str, path: str, *, json: Any = None, _retried: bool = False
    ) -> httpx.Response:
        if self._access_token is None:
            self.login()
        url = f"{self.base_url}{API_PREFIX}{path}"
        try:
            resp = self._http.request(method, url, headers=self._auth_headers(), json=json)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise BotUnreachableError(f"bot at {self.base_url} unreachable: {exc}") from exc
        if resp.status_code == 401 and not _retried:
            logger.info("access token expired; renewing session and retrying %s %s", method, path)
            try:
                self._reauth()
            except FreqtradeApiError as exc:
                raise AuthError(f"session renewal failed: {exc}") from exc
            return self._request(method, path, json=json, _retried=True)
        return resp

    def _check(self, resp: httpx.Response, method: str, path: str) -> None:
        if resp.status_code == 401:
            raise AuthError(f"unauthorized calling {method} {path} (check credentials)")
        if resp.status_code >= 400:
            raise FreqtradeApiError(
                f"freqtrade {method} {path} failed: HTTP {resp.status_code} {resp.text[:300]}",
                status_code=resp.status_code,
            )

    def _call(self, method: str, path: str, *, json: Any = None) -> Any:
        resp = self._request(method, path, json=json)
        self._check(resp, method, path)
        try:
            return resp.json()
        except ValueError as exc:
            raise FreqtradeApiError(f"freqtrade {method} {path} returned non-JSON") from exc

    # -- read API ----------------------------------------------------------

    def get_status(self) -> Any:
        """Bot status: the list of open trades (``GET /status``)."""
        return self._call("GET", "/status")

    def get_open_trades(self) -> Any:
        """Open trades (``GET /status``)."""
        return self._call("GET", "/status")

    def get_profit(self) -> Any:
        """Profit/loss summary over closed trades (``GET /profit``)."""
        return self._call("GET", "/profit")

    def get_balance(self) -> Any:
        """Account balance per currency (``GET /balance``)."""
        return self._call("GET", "/balance")

    def get_performance(self) -> Any:
        """Per-pair performance of finished trades (``GET /performance``)."""
        return self._call("GET", "/performance")

    # -- write API (gated behind MCP_ALLOW_WRITE in server.py) --------------

    def pause_trading(self) -> Any:
        """Stop opening new trades; open trades keep running their rules.

        Uses ``POST /stopbuy`` (falling back to ``POST /pause`` on bots that
        only expose the newer endpoint).
        """
        try:
            return self._call("POST", "/stopbuy")
        except FreqtradeApiError as exc:
            if exc.status_code == 404:
                logger.warning("/stopbuy missing (404); falling back to /pause")
                return self._call("POST", "/pause")
            raise

    def force_exit(self, trade_id: str | int) -> Any:
        """Instantly exit one open trade, ignoring ``minimum_roi``."""
        if trade_id is None or (isinstance(trade_id, str) and not trade_id.strip()):
            raise InvalidTradeIdError(f"invalid trade id: {trade_id!r}")
        tid = str(trade_id).strip()
        try:
            return self._call("POST", "/forceexit", json={"tradeid": tid})
        except FreqtradeApiError as exc:
            if exc.status_code == 404:
                raise InvalidTradeIdError(f"unknown trade id: {tid!r}") from exc
            raise

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FreqtradeClient:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
