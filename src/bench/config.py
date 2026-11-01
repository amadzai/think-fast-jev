"""This module reads the settings from environment variables. The CLI loads them from `.env` first."""

import os
from dataclasses import dataclass
from typing import Literal

DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_KEV_MODEL = "kev-latest"
# The arm name of Kev in the records. Only Kev has the permutation endpoint and a self-hosted cost.
KEV_ARM = "kev"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_HAIKU_MODEL = "claude-haiku-5-5"
# Haiku has no logprobs and no temperature setting, so each item runs 3 times to measure agreement.
HAIKU_REPEATS = 3
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai"
# The generator is neither Claude nor Qwen. This model supports structured outputs on OpenRouter.
DEFAULT_GENERATOR_MODEL = "google/gemini-3.8-flash"
EXCLUDED_GENERATOR_FAMILIES = ("anthropic", "claude", "qwen")
DEFAULT_COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input_per_1m: float
    output_per_1m: float


# Prices by model ID. A model without a price gets no cost, so a changed HAIKU_MODEL never gets the wrong price.
# Kev has no price per token: its cost comes from a GPU hourly price and the measured throughput.
PRICES = {
    DEFAULT_JEV_MODEL: Price(input_per_1m=0.042, output_per_1m=0.0),
    # Through the Anthropic API, prompts up to 100K tokens.
    DEFAULT_HAIKU_MODEL: Price(input_per_1m=0.10, output_per_1m=0.50),
}

# Self-hosted Kev: the on-demand price of an AWS g6.xlarge (1 NVIDIA L4) in us-east-1, USD per hour.
SELF_HOSTED_HOURLY_USD = 0.8048


class ConfigError(Exception):
    """A required setting is missing."""


@dataclass(frozen=True)
class ArmConfig:
    """Connection settings for one arm."""

    name: str
    # "system_one" for Jev and Kev, "anthropic" for Haiku.
    kind: Literal["system_one", "anthropic"]
    base_url: str
    api_key: str
    model: str
    # The number of calls for each item.
    repeats: int = 1
    # Extra text for the error message when the server cannot be reached.
    unreachable_hint: str = ""
    # True: the runner sends one unrecorded call right before each recorded call. The recorded call never follows an
    # idle gap.
    keep_warm: bool = False


def jev_config() -> ArmConfig:
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("TYPESAFE_API_KEY is not set. Add it to .env (get a key at console.typesafe.ai).")
    return ArmConfig(
        name="jev",
        kind="system_one",
        base_url=os.environ.get("JEV_BASE_URL", DEFAULT_JEV_BASE_URL),
        api_key=api_key,
        model=os.environ.get("JEV_MODEL", DEFAULT_JEV_MODEL),
    )


def kev_config() -> ArmConfig | None:
    """Return None when KEV_BASE_URL is not set. Kev is an optional arm."""
    base_url = os.environ.get("KEV_BASE_URL", "").strip()
    if not base_url:
        return None
    api_key = os.environ.get("KEV_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "KEV_BASE_URL is set but KEV_API_KEY is not. Add the key from ~/.config/kev/api-key on the PC."
        )
    return ArmConfig(
        name=KEV_ARM,
        kind="system_one",
        base_url=base_url,
        api_key=api_key,
        model=os.environ.get("KEV_MODEL", DEFAULT_KEV_MODEL),
        unreachable_hint="Is kev-serve running on the PC, and is KEV_BASE_URL the PC's current IP address?",
        # Kev's model time rises from about 60 ms to about 200 ms after 8 s or more of idle on the PC. Production
        # traffic keeps a GPU busy. The benchmark measures the warm path.
        keep_warm=True,
    )


def haiku_config() -> ArmConfig:
    # Haiku is the incumbent. A run without it compares the candidates with nothing.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("ANTHROPIC_API_KEY is not set. Add it to .env (get a key at platform.claude.com).")
    return ArmConfig(
        name="haiku",
        kind="anthropic",
        base_url=os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
        api_key=api_key,
        model=os.environ.get("HAIKU_MODEL", DEFAULT_HAIKU_MODEL),
        repeats=HAIKU_REPEATS,
    )


@dataclass(frozen=True)
class GeneratorConfig:
    """Settings for the model that writes query variations (`bench generate`)."""

    base_url: str
    api_key: str
    model: str


def generator_config() -> GeneratorConfig:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("OPENROUTER_API_KEY is not set. Add it to .env (get a key at openrouter.ai).")
    model = os.environ.get("GENERATOR_MODEL", "").strip() or DEFAULT_GENERATOR_MODEL
    # Claude phrasing could favour Haiku, and Qwen phrasing could favour the Qwen-based Jev and Kev.
    if any(family in model.lower() for family in EXCLUDED_GENERATOR_FAMILIES):
        raise ConfigError(f"GENERATOR_MODEL {model!r} is a Claude or Qwen model. Use a model of another family.")
    return GeneratorConfig(
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL), api_key=api_key, model=model
    )


@dataclass(frozen=True)
class CoinGeckoConfig:
    """Settings for the CoinGecko Demo API (`bench fetch`)."""

    base_url: str
    api_key: str


def coingecko_config() -> CoinGeckoConfig:
    api_key = os.environ.get("COINGECKO_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("COINGECKO_API_KEY is not set. Add your Demo key to .env (get one at coingecko.com/en/api).")
    return CoinGeckoConfig(base_url=os.environ.get("COINGECKO_BASE_URL", DEFAULT_COINGECKO_BASE_URL), api_key=api_key)


def arm_configs() -> list[ArmConfig]:
    """Return Jev, Kev when it is configured, and Haiku."""
    jev = jev_config()
    kev = kev_config()
    haiku = haiku_config()
    if kev is None:
        return [jev, haiku]
    return [jev, kev, haiku]
