"""This module reads the settings from environment variables. The CLI loads them from `.env` first."""

import os
from dataclasses import dataclass

DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_KEV_MODEL = "kev-latest"


class ConfigError(Exception):
    """A required setting is missing."""


@dataclass(frozen=True)
class SystemOneArmConfig:
    """Connection settings for one arm that speaks the System One API (Jev or Kev)."""

    name: str
    base_url: str
    api_key: str
    model: str
    # Extra text for the error message when the server cannot be reached.
    unreachable_hint: str = ""


def jev_config() -> SystemOneArmConfig:
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("TYPESAFE_API_KEY is not set. Add it to .env (get a key at console.typesafe.ai).")
    return SystemOneArmConfig(
        name="jev",
        base_url=os.environ.get("JEV_BASE_URL", DEFAULT_JEV_BASE_URL),
        api_key=api_key,
        model=os.environ.get("JEV_MODEL", DEFAULT_JEV_MODEL),
    )


def kev_config() -> SystemOneArmConfig | None:
    """Return None when KEV_BASE_URL is not set. Kev is an optional arm."""
    base_url = os.environ.get("KEV_BASE_URL", "").strip()
    if not base_url:
        return None
    api_key = os.environ.get("KEV_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "KEV_BASE_URL is set but KEV_API_KEY is not. Add the key from ~/.config/kev/api-key on the PC."
        )
    return SystemOneArmConfig(
        name="kev",
        base_url=base_url,
        api_key=api_key,
        model=os.environ.get("KEV_MODEL", DEFAULT_KEV_MODEL),
        unreachable_hint="Is kev-serve running on the PC, and is KEV_BASE_URL the PC's current IP address?",
    )


def arm_configs() -> list[SystemOneArmConfig]:
    """Return Jev, and Kev when it is configured."""
    jev = jev_config()
    kev = kev_config()
    if kev is None:
        return [jev]
    return [jev, kev]
