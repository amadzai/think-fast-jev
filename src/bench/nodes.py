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
    # True: the node asks one yes/no question per option. The probabilities do not sum to 1. `question_id` is not used.
    multi_label: bool = False

    def system_one_questions(self) -> dict[str, dict[str, Any]]:
        if self.multi_label:
            # All questions for one item go in one request.
            return {
                # The text has the option name, because the chat prompt shows it too.
                name: {"type": "noul", "instructions": f"{self.instructions} {name}: {description}"}
                for name, description in self.options.items()
            }
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
        if self.multi_label:
            return (
                f"{self.instructions}\n\nOptions:\n{options}\n\n"
                "For each option, return the probability from 0 to 1 that the answer is yes. "
                "Each option is a separate question, so the probabilities do not sum to 1."
            )
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


@dataclass(frozen=True)
class Sector:
    description: str
    # The CoinGecko categories that make a coin a yes for this sector. Only scoring uses them. The arms never see them.
    categories: tuple[str, ...]


# Broad sectors with reliable CoinGecko categories. `bench fetch` prints the number of coins per sector. Each sector has 6 or
# more of the 200 fetched coins (2026-11-01). Sectors with fewer coins give an unstable F1.
SECTORS = {
    "layer_1": Sector("A base blockchain with its own validators or miners.", ("Layer 1 (L1)",)),
    "layer_2": Sector("A network that runs on top of a base blockchain to scale it.", ("Layer 2 (L2)",)),
    "smart_contract_platform": Sector(
        "A blockchain that runs programs (smart contracts) for applications.", ("Smart Contract Platform",)
    ),
    "defi": Sector("Financial services that run on a blockchain without a bank.", ("Decentralized Finance (DeFi)",)),
    "dex": Sector("An exchange where users trade tokens from their own wallets.", ("Decentralized Exchange (DEX)",)),
    "lending": Sector("A protocol where users lend or borrow crypto.", ("Lending/Borrowing Protocols",)),
    "stablecoin": Sector("A token with a value that is fixed to a currency or an asset.", ("Stablecoins",)),
    "exchange_token": Sector(
        "The token of a centralised exchange.", ("Exchange-based Tokens", "Centralized Exchange (CEX) Token")
    ),
    "meme": Sector("A coin based on a joke, a meme or internet culture.", ("Meme",)),
    "ai": Sector("A project about artificial intelligence or AI agents.", ("Artificial Intelligence (AI)",)),
    "rwa": Sector("Tokens for real-world assets such as bonds, property or commodities.", ("Real World Assets (RWA)",)),
    "governance": Sector("A token that gives its holders a vote on how a protocol changes.", ("Governance",)),
    "depin": Sector("A network of physical devices or hardware that users run for rewards.", ("DePIN",)),
    "infrastructure": Sector("Tools and services that other blockchain projects build on.", ("Infrastructure",)),
}

COIN_SECTOR = Node(
    name="coin_sector",
    question_id="sector",
    instructions="The state describes one crypto coin. Does the coin belong to this sector?",
    options={name: sector.description for name, sector in SECTORS.items()},
    multi_label=True,
)

NODES = {node.name: node for node in [ROUTER, GUARDRAIL, COIN_SECTOR]}
