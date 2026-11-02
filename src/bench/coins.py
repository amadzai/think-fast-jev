"""This module reads the coin sector data (JSONL, one coin per line) and builds the state for each coin."""

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

from bench.nodes import SECTORS

COINS_PATH = Path("data/coingecko/coins.jsonl")
# Kev works best with short states. 500 characters is about 120 tokens.
DESCRIPTION_MAX_CHARS = 500
HTML_TAG = re.compile(r"<[^>]+>")


class CoinFileError(Exception):
    """A line in the coin file is not a valid coin."""


@dataclass(frozen=True)
class Coin:
    id: str
    name: str
    symbol: str
    description: str
    categories: tuple[str, ...]

    @property
    def text(self) -> str:
        """Return the state that every arm gets: the name, the symbol and the cut description."""
        return f"Name: {self.name}\nSymbol: {self.symbol.upper()}\nDescription: {_cut(self.description)}"

    @property
    def has_description(self) -> bool:
        # Without a description, the arms see only the name and the symbol. The run skips these coins.
        return bool(_cut(self.description))

    @property
    def sectors(self) -> frozenset[str]:
        """Return the sectors that the coin's CoinGecko categories give. These are the labels."""
        categories = set(self.categories)
        return frozenset(name for name, sector in SECTORS.items() if categories & set(sector.categories))


def load_coins(path: Path) -> list[Coin]:
    if not path.exists():
        raise CoinFileError(f"{path}: file not found. Run `bench fetch` first.")
    coins = []
    names = {field.name for field in fields(Coin)}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except ValueError as error:
            raise CoinFileError(f"{path}:{number}: not valid JSON: {error}") from error
        if not isinstance(data, dict) or set(data) != names:
            raise CoinFileError(f"{path}:{number}: a coin must have exactly the fields {sorted(names)}")
        strings = [data["id"], data["name"], data["symbol"], data["description"]]
        categories = data["categories"]
        if not all(isinstance(v, str) for v in strings) or not isinstance(categories, list):
            raise CoinFileError(f"{path}:{number}: the fields must be text, and categories must be a list")
        if not all(isinstance(c, str) for c in categories):
            raise CoinFileError(f"{path}:{number}: each category must be text")
        coins.append(Coin(*strings, categories=tuple(categories)))
    ids = [coin.id for coin in coins]
    if len(ids) != len(set(ids)):
        raise CoinFileError(f"{path}: coin IDs are not unique")
    return coins


def _cut(description: str) -> str:
    # CoinGecko descriptions contain HTML links and line breaks.
    text = " ".join(HTML_TAG.sub("", description).split())
    if len(text) <= DESCRIPTION_MAX_CHARS:
        return text
    # Cut at the last space, so that no word is cut in half.
    return text[:DESCRIPTION_MAX_CHARS].rsplit(" ", 1)[0]
