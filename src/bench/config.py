"""This module reads the settings from environment variables. The CLI loads them from `.env` first."""

import os
from dataclasses import dataclass
from typing import Literal

DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_KEV_MODEL = "kev-latest"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_HAIKU_MODEL = "claude-haiku-5-5"
# Haiku has no logprobs and no temperature setting, so each item runs 3 times to measure agreement.
HAIKU_REPEATS = 3


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
        name="kev",
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


def arm_configs() -> list[ArmConfig]:
    """Return Jev, Kev when it is configured, and Haiku."""
    jev = jev_config()
    kev = kev_config()
    haiku = haiku_config()
    if kev is None:
        return [jev, haiku]
    return [jev, kev, haiku]
