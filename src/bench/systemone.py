"""Jev and Kev use this client to call the System One API (POST /v1/systemone)."""

import time
from dataclasses import dataclass
from typing import Any, Self

import httpx

from bench.config import SystemOneArmConfig

TIMEOUT_S = 30.0
ERROR_BODY_CHARS = 300


class SystemOneError(Exception):
    """The API sent a response that the client cannot read."""


@dataclass(frozen=True)
class SystemOneResult:
    answers: dict[str, Any]
    model: str
    model_ms: float | None
    e2e_ms: float
    input_tokens: int | None
    output_tokens: int | None


class SystemOneClient:
    def __init__(self, config: SystemOneArmConfig, transport: httpx.BaseTransport | None = None) -> None:
        self._config = config
        # One client per arm keeps the connection open between calls (keep-alive).
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=TIMEOUT_S,
            transport=transport,
        )

    @property
    def config(self) -> SystemOneArmConfig:
        return self._config

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._http.close()

    def decide(self, state: str, questions: dict[str, dict[str, Any]]) -> SystemOneResult:
        body = {"state": state, "model": self._config.model, "questions": questions}
        start = time.perf_counter()
        response = self._http.post("/v1/systemone", json=body)
        e2e_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        try:
            data = response.json()
            answers = data["answers"]
        except (ValueError, KeyError) as error:
            raise SystemOneError(f"unexpected response: {response.text[:ERROR_BODY_CHARS]}") from error
        missing = set(questions) - set(answers)
        if missing:
            raise SystemOneError(f"response has no answer for {sorted(missing)}")
        usage = data.get("usage") or {}
        return SystemOneResult(
            answers=answers,
            model=data.get("model", self._config.model),
            # Kev returns the model time. Hosted Jev does not.
            model_ms=data.get("latency_ms"),
            e2e_ms=e2e_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


def describe_error(error: Exception) -> str:
    """Return a short message for a failed call."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}: {error.response.text[:ERROR_BODY_CHARS]}"
    return f"{type(error).__name__}: {error}"
