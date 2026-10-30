"""Configuration from environment variables, with `.env` in the working directory as the default source."""

import os
from dataclasses import dataclass


class ConfigError(Exception):
    """A required setting is missing."""


@dataclass(frozen=True)
class SystemOneArmConfig:
    """Connection settings for one arm that speaks the System One API (Jev or Kev)."""

    name: str
    base_url: str
    api_key: str
    model: str


def jev_config() -> SystemOneArmConfig:
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("TYPESAFE_API_KEY is not set. Add it to .env (get a key at console.typesafe.ai).")
    return SystemOneArmConfig(
        name="jev",
        base_url=os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai"),
        api_key=api_key,
        model=os.environ.get("JEV_MODEL", "jev-latest"),
    )
