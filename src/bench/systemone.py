"""Jev and Kev use this client to call the System One API (POST /v1/systemone)."""

import time
from dataclasses import dataclass
from typing import Any, Self

import httpx

from bench.arm import CONNECT_TIMEOUT_S, ERROR_BODY_CHARS, TIMEOUT_S, Answer, ArmResponseError, in_option_order
from bench.config import ArmConfig
from bench.nodes import Node

# The Kev server takes at most 64 requests in one batch. `bench throughput` sends at most this many calls at a time,
# and the client keeps this many connections open. Fewer open connections add connection setup to the burst time.
MAX_CONNECTIONS = 64


class SystemOneError(ArmResponseError):
    """The System One API sent a response that the client cannot read."""


@dataclass(frozen=True)
class PermuteAnswer:
    """The answer of POST /v1/systemone/permute: one choice question under several option orders."""

    # True when every option order selects the same option.
    stable: bool
    # The probability change of each option between the option orders: the largest value minus the smallest value.
    spread: dict[str, float]
    # One entry per option order: the order, the probabilities, the choice and the model time.
    runs: list[dict[str, Any]]


class SystemOneClient:
    def __init__(self, config: ArmConfig, transport: httpx.BaseTransport | None = None) -> None:
        self._config = config
        # One client per arm keeps the connection open between calls (keep-alive).
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_CONNECTIONS),
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

    def permute(self, node: Node, state: str, n_perm: int, seed: int) -> PermuteAnswer:
        """Ask the node's choice question under `n_perm` option orders. Only Kev has this endpoint.

        The first order is the node's order. The server shuffles the other orders with `seed`.
        """
        request = {"state": state, "model": self._config.model, "questions": node.system_one_questions()}
        body = {"request": request, "question": node.question_id, "n_perm": n_perm, "seed": seed}
        response = self._http.post("/v1/systemone/permute", json=body)
        response.raise_for_status()
        try:
            data = response.json()
            spread = {name: float(p) for name, p in in_option_order(node, data["spread"], response.text).items()}
            stable, runs = data["argmax_stable"], data["runs"]
        except (ValueError, KeyError, TypeError) as error:
            raise SystemOneError(f"unexpected permute response: {response.text[:ERROR_BODY_CHARS]}") from error
        if not isinstance(stable, bool) or not isinstance(runs, list) or len(runs) != n_perm:
            raise SystemOneError(f"unexpected permute response: {response.text[:ERROR_BODY_CHARS]}")
        return PermuteAnswer(stable=stable, spread=spread, runs=runs)


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
