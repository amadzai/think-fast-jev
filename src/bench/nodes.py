"""This module defines the decision nodes as data. The same definition builds the request for every arm."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Node:
    name: str
    question_id: str
    instructions: str
    # Option name to one-line description. The order is fixed and the same for every arm.
    options: dict[str, str]

    def system_one_questions(self) -> dict[str, dict[str, Any]]:
        return {
            self.question_id: {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": dict(self.options),
            }
        }


ROUTER = Node(
    name="router",
    question_id="route",
    instructions=(
        "The state is a question from a user of a crypto market-data assistant. "
        "Select the one data tool that can answer the question."
    ),
    options={
        "price": "The current price of one or more coins in a currency.",
        "coin_details": "Facts about one coin: description, links, market cap, supply, all-time high.",
        "market_chart": "Historical price, market cap or volume of a coin over a time range.",
        "coin_markets": "A ranked list of coins by market cap or volume.",
        "trending": "The coins, NFTs and categories that users search most right now.",
        "categories": "Coin categories (sectors) and their market cap and volume.",
        "exchanges": "Centralised exchanges: volume, trust score, trading pairs.",
        "derivatives": "Perpetual and futures contracts: open interest, funding rate, derivatives exchanges.",
        "nfts": "NFT collections: floor price, market cap, volume.",
        "global": "The total crypto market: total market cap, volume, Bitcoin dominance.",
        "onchain_dex": "On-chain DEX data: liquidity pools, token prices on a network, new pools.",
        "treasury": "Crypto held by public companies and governments.",
        "unsupported": "No data tool above can answer the question.",
    },
)

NODES = {node.name: node for node in [ROUTER]}
