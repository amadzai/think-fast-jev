"""This module fetches the coin sector data from the CoinGecko Demo API (`bench fetch`)."""

import json
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from bench.arm import CONNECT_TIMEOUT_S, ERROR_BODY_CHARS, TIMEOUT_S
from bench.coins import Coin
from bench.config import CoinGeckoConfig

MARKETS_PER_PAGE_MAX = 250
# The Demo plan allows 30 calls per minute. One call every 2.5 s stays below this limit.
CALL_INTERVAL_S = 2.5
# CoinGecko sends Retry-After with a 429. This wait applies when it does not.
RATE_LIMIT_WAIT_S = 60.0
MAX_ATTEMPTS = 3
HTTP_TOO_MANY_REQUESTS = 429
# Only the fields that the coin sector needs. The other parts of the response are large.
COIN_PARAMS = {
    "localization": "false",
    "tickers": "false",
    "market_data": "false",
    "community_data": "false",
    "developer_data": "false",
    "sparkline": "false",
}


class FetchError(Exception):
    """The CoinGecko API sent a response that the command cannot read."""


def fetch_coins(config: CoinGeckoConfig, limit: int, out: Path, done: set[str]) -> Iterator[Coin]:
    """Add the top `limit` coins by market cap to `out`, one line per coin, and yield each added coin.

    The function skips the coins in `done`, so a rerun continues after an error.
    """
    with httpx.Client(
        base_url=config.base_url,
        headers={"x-cg-demo-api-key": config.api_key},
        timeout=httpx.Timeout(TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
    ) as http:
        markets = _get(http, "/coins/markets", {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit})
        if not isinstance(markets, list) or not all(isinstance(m, dict) and "id" in m for m in markets):
            raise FetchError(f"unexpected markets response: {str(markets)[:ERROR_BODY_CHARS]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as file:
            for coin_id in [m["id"] for m in markets if m["id"] not in done]:
                time.sleep(CALL_INTERVAL_S)
                coin = _coin(_get(http, f"/coins/{coin_id}", COIN_PARAMS))
                # Write each coin at once, so that an error later keeps the coins before it.
                file.write(json.dumps(asdict(coin)) + "\n")
                file.flush()
                yield coin


def _get(http: httpx.Client, path: str, params: dict[str, Any]) -> Any:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = http.get(path, params=params)
        if response.status_code == HTTP_TOO_MANY_REQUESTS and attempt < MAX_ATTEMPTS:
            time.sleep(_retry_after(response))
            continue
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as error:
            raise FetchError(f"{path}: response is not JSON: {response.text[:ERROR_BODY_CHARS]}") from error
    raise AssertionError("unreachable: the last attempt returns or raises")


def _retry_after(response: httpx.Response) -> float:
    # Retry-After can also be an HTTP date. Then the command waits the default time.
    try:
        return float(response.headers.get("retry-after", RATE_LIMIT_WAIT_S))
    except ValueError:
        return RATE_LIMIT_WAIT_S


def _coin(data: Any) -> Coin:
    unexpected = FetchError(f"unexpected coin response: {str(data)[:ERROR_BODY_CHARS]}")
    try:
        description = (data.get("description") or {}).get("en") or ""
        # The categories list can contain null.
        categories = tuple(c for c in data.get("categories") or [] if isinstance(c, str))
        coin = Coin(
            id=data["id"], name=data["name"], symbol=data["symbol"], description=description, categories=categories
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise unexpected from error
    if not all(isinstance(v, str) for v in (coin.id, coin.name, coin.symbol, coin.description)):
        raise unexpected
    return coin
