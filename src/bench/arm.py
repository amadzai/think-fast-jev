"""This module defines what every arm client returns, and the errors that the runner and the CLI handle."""

import json
from dataclasses import dataclass
from typing import Any, Protocol, Self

import anthropic
import httpx

from bench.config import ArmConfig
from bench.nodes import Node

ERROR_BODY_CHARS = 300
# A multi-label node (coin sector) says yes to an option at this probability or more, for every arm.
YES_THRESHOLD = 0.5
TIMEOUT_S = 30.0
# A short connect timeout gives a fast, clear error when the Kev PC is off or its IP address changed.
CONNECT_TIMEOUT_S = 5.0
# A sum this close to 1 does not count as scaled.
SUM_TOLERANCE = 1e-6


class ArmResponseError(Exception):
    """The API sent a response that the client cannot read."""


# A call that fails with one of these is retried, then written as an error record.
CALL_ERRORS = (httpx.HTTPError, anthropic.APIError, ArmResponseError)
# A call that fails with one of these did not reach the server.
UNREACHABLE_ERRORS = (httpx.TransportError, anthropic.APIConnectionError)


@dataclass(frozen=True)
class Answer:
    # Option name to probability, in the node's option order. The probabilities sum to 1.
    probabilities: dict[str, float]
    model_version: str
    model_ms: float | None
    e2e_ms: float
    input_tokens: int | None
    output_tokens: int | None
    # True when the client scaled the probabilities to sum 1. None for arms that never need it.
    normalised: bool | None = None


class ArmClient(Protocol):
    @property
    def config(self) -> ArmConfig: ...

    @property
    def request_settings(self) -> dict[str, Any]:
        """The request settings that the run metadata records, for example Haiku's thinking setting."""
        ...

    def answer(self, node: Node, state: str) -> Answer: ...

    def models(self) -> dict[str, Any]:
        """Return the server's description of the model. The run metadata records it."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc_info: object) -> None: ...


def describe_error(error: Exception) -> str:
    """Return a short message for a failed call."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}: {error.response.text[:ERROR_BODY_CHARS]}"
    if isinstance(error, anthropic.APIStatusError):
        return f"HTTP {error.status_code}: {error.message[:ERROR_BODY_CHARS]}"
    return f"{type(error).__name__}: {error}"


def in_option_order(node: Node, values: object, raw: str) -> dict[str, Any]:
    """Return the values in the node's option order. Raise when the keys are not exactly the node's options."""
    if not isinstance(values, dict) or set(values) != set(node.options):
        raise ArmResponseError(f"answer has no probability for each option: {raw[:ERROR_BODY_CHARS]}")
    return {name: values[name] for name in node.options}


def scaled_probabilities(node: Node, text: str) -> tuple[dict[str, float], bool | None]:
    """Read an LLM's structured output and scale the probabilities to sum 1. Also return True if they needed it.

    A multi-label node has one yes/no question per option. The function does not scale its probabilities and
    returns None for the flag.
    """
    try:
        data = json.loads(text)
    except ValueError as error:
        raise ArmResponseError(f"output is not JSON: {text[:ERROR_BODY_CHARS]}") from error
    values = in_option_order(node, data, text)
    if any(not isinstance(v, int | float) or isinstance(v, bool) or v < 0 for v in values.values()):
        raise ArmResponseError(f"output has a probability that is not a number of 0 or more: {text[:ERROR_BODY_CHARS]}")
    if node.multi_label:
        if any(v > 1 for v in values.values()):
            raise ArmResponseError(f"output has a probability over 1: {text[:ERROR_BODY_CHARS]}")
        return {name: float(v) for name, v in values.items()}, None
    total = sum(values.values())
    if total <= 0:
        raise ArmResponseError("all probabilities are 0")
    return {name: v / total for name, v in values.items()}, abs(total - 1) > SUM_TOLERANCE


def argmax(probabilities: dict[str, float]) -> str:
    # The same decision rule for every arm. A tie goes to the option that comes first in the node.
    return max(probabilities, key=probabilities.__getitem__)


def yes_options(probabilities: dict[str, float]) -> list[str]:
    return [name for name, p in probabilities.items() if p >= YES_THRESHOLD]
