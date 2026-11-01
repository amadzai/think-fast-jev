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

    def chat_system_prompt(self) -> str:
        """Return the system prompt for a chat arm. It has the same instructions and options as the System One request."""
        options = "\n".join(f"- {name}: {description}" for name, description in self.options.items())
        return (
            f"{self.instructions}\n\nOptions:\n{options}\n\n"
            "Return a probability from 0 to 1 for each option. The probabilities sum to 1."
        )

    def probability_schema(self) -> dict[str, Any]:
        """Return the structured output schema for a chat arm. It has one probability field per option."""
        return {
            "type": "object",
            "properties": {name: {"type": "number"} for name in self.options},
            "required": list(self.options),
            "additionalProperties": False,
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
        "tickers": "Where one coin trades: the exchanges and trading pairs for it, with price and volume.",
        "exchanges": "Centralised exchanges: volume, trust score, the trading pairs that one exchange lists.",
        "derivatives": "Perpetual and futures contracts: open interest, funding rate, derivatives exchanges.",
        "nfts": "NFT collections: floor price, market cap, volume.",
        "global": "The total crypto market: total market cap, volume, Bitcoin dominance.",
        "onchain_dex": "On-chain DEX data: liquidity pools, token prices on a network, new pools.",
        "treasury": "Crypto held by public companies and governments.",
        "unsupported": "No data tool above can answer the question.",
    },
)

GUARDRAIL = Node(
    name="guardrail",
    question_id="guard",
    instructions=(
        "The state is a question from a user of a crypto market-data assistant. "
        "Decide if the assistant may answer the question."
    ),
    options={
        # `allow` does not mean that a data tool can answer the question. The router decides this with `unsupported`.
        "allow": "A question about crypto markets or crypto data that asks for facts.",
        "advice": "A request for personal financial advice: what the user should buy, sell or hold, or when.",
        "off_topic": "A question that is not about crypto markets.",
    },
)

NODES = {node.name: node for node in [ROUTER, GUARDRAIL]}
