"""Client for the System One API (POST /v1/systemone), used by Jev and Kev."""

import time
from dataclasses import dataclass
from typing import Any

import httpx

from bench.config import SystemOneArmConfig

TIMEOUT_S = 30.0


@dataclass(frozen=True)
class SystemOneResult:
    answers: dict[str, Any]
    model: str
    model_ms: float | None
    e2e_ms: float
    input_tokens: int | None
    output_tokens: int | None


class SystemOneClient:
    def __init__(self, config: SystemOneArmConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        # One client per arm keeps the connection open between calls (keep-alive).
        self.http = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=TIMEOUT_S,
            transport=transport,
        )

    def decide(self, state: str, questions: dict[str, dict[str, Any]]) -> SystemOneResult:
        body = {"state": state, "model": self.config.model, "questions": questions}
        start = time.perf_counter()
        response = self.http.post("/v1/systemone", json=body)
        e2e_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return SystemOneResult(
            answers=data["answers"],
            model=data.get("model", self.config.model),
            model_ms=data.get("latency_ms"),
            e2e_ms=e2e_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def close(self) -> None:
        self.http.close()
