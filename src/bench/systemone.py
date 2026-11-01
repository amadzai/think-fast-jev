"""Jev and Kev use this client to call the System One API (POST /v1/systemone)."""

import time
from typing import Any, Self

import httpx

from bench.arm import CONNECT_TIMEOUT_S, ERROR_BODY_CHARS, TIMEOUT_S, Answer, ArmResponseError, in_option_order
from bench.config import ArmConfig
from bench.nodes import Node


class SystemOneError(ArmResponseError):
    """The System One API sent a response that the client cannot read."""


class SystemOneClient:
    def __init__(self, config: ArmConfig, transport: httpx.BaseTransport | None = None) -> None:
        self._config = config
        # One client per arm keeps the connection open between calls (keep-alive).
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            transport=transport,
        )

    @property
    def config(self) -> ArmConfig:
        return self._config

    @property
    def request_settings(self) -> dict[str, Any]:
        return {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._http.close()

    def models(self) -> dict[str, Any]:
        """Return the GET /v1/models response. Kev reports its server settings here."""
        response = self._http.get("/v1/models")
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as error:
            raise SystemOneError(f"unexpected models response: {response.text[:ERROR_BODY_CHARS]}") from error
        if not isinstance(data, dict):
            raise SystemOneError(f"unexpected models response: {response.text[:ERROR_BODY_CHARS]}")
        return data

    def answer(self, node: Node, state: str) -> Answer:
        body = {"state": state, "model": self._config.model, "questions": node.system_one_questions()}
        start = time.perf_counter()
        response = self._http.post("/v1/systemone", json=body)
        e2e_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        try:
            data = response.json()
            probabilities = _probabilities(node, data["answers"])
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise SystemOneError(f"unexpected response: {response.text[:ERROR_BODY_CHARS]}") from error
        usage = data.get("usage") or {}
        return Answer(
            probabilities=probabilities,
            model_version=data.get("model", self._config.model),
            # Kev returns the model time. Hosted Jev does not.
            model_ms=data.get("latency_ms"),
            e2e_ms=e2e_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


def _probabilities(node: Node, answers: dict[str, Any]) -> dict[str, float]:
    """Raise KeyError, TypeError, ValueError or AttributeError when the answers do not have the node's shape."""
    if node.multi_label:
        # One noul answer per option. `noul` is the probability of yes.
        probabilities = {name: answer["noul"] for name, answer in answers.items()}
    else:
        answer = answers[node.question_id]
        probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    # Keep the node's option order, not the order of the response.
    return {name: float(p) for name, p in in_option_order(node, probabilities, str(answers)).items()}
