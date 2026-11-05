"""Haiku uses this client to call the Anthropic Messages API with structured output."""

import re
import time
from typing import Any, Self

import anthropic
import httpx2
from anthropic.types import ThinkingConfigDisabledParam

from bench.arm import CONNECT_TIMEOUT_S, TIMEOUT_S, Answer, ArmResponseError, scaled_probabilities
from bench.config import ArmConfig
from bench.nodes import Node

# One probability per option fits in about 150 tokens. A larger limit only matters if the model goes wrong.
MAX_TOKENS = 1024
# The lowest reasoning setting. Haiku 5.5 runs adaptive thinking at effort "medium" by default.
THINKING: ThinkingConfigDisabledParam = {"type": "disabled"}
EFFORT = "low"
# Cloudflare reports the time that Anthropic's servers took (it does not include the network to the edge).
ORIGIN_TIME = re.compile(r"x-originResponse;dur=([0-9.]+)")


class AnthropicClient:
    def __init__(self, config: ArmConfig, transport: httpx2.BaseTransport | None = None) -> None:
        self._config = config
        # The runner owns retries and the warm-up count, so the SDK must not retry.
        self._client = anthropic.Anthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=0,
            timeout=anthropic.Timeout(TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            http_client=anthropic.DefaultHttpxClient(transport=transport) if transport else None,
        )

    @property
    def config(self) -> ArmConfig:
        return self._config

    @property
    def request_settings(self) -> dict[str, Any]:
        # The request sends no temperature, because Haiku 5.5 rejects a value that is not the default.
        return {"thinking": THINKING, "effort": EFFORT, "max_tokens": MAX_TOKENS}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._client.close()

    def models(self) -> dict[str, Any]:
        return self._client.models.retrieve(self._config.model).to_dict(mode="json")

    def answer(self, node: Node, state: str) -> Answer:
        start = time.perf_counter()
        raw = self._client.messages.with_raw_response.create(
            model=self._config.model,
            max_tokens=MAX_TOKENS,
            system=node.chat_system_prompt(),
            thinking=THINKING,
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": node.probability_schema()}},
            messages=[{"role": "user", "content": state}],
        )
        e2e_ms = (time.perf_counter() - start) * 1000
        message = raw.parse()
        if message.stop_reason != "end_turn":
            raise ArmResponseError(f"stop_reason {message.stop_reason!r}, not 'end_turn'")
        text = next((block.text for block in message.content if block.type == "text"), "")
        probabilities, normalised = scaled_probabilities(node, text)
        origin = ORIGIN_TIME.search(raw.headers.get("server-timing", ""))
        return Answer(
            probabilities=probabilities,
            model_version=message.model,
            model_ms=float(origin.group(1)) if origin else None,
            e2e_ms=e2e_ms,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            normalised=normalised,
        )
