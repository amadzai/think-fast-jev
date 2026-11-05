"""Luna uses this client to call the OpenAI Chat Completions API with structured output."""

import time
from typing import Any, Self

import httpx2
import openai
from openai.types.shared import ReasoningEffort

from bench.arm import CONNECT_TIMEOUT_S, ERROR_BODY_CHARS, TIMEOUT_S, Answer, ArmResponseError, scaled_probabilities
from bench.config import ArmConfig
from bench.nodes import Node

# One probability per option fits in about 150 tokens. A larger limit only matters if the model goes wrong.
MAX_TOKENS = 1024
# The lowest reasoning setting, to match Haiku's lowest setting.
REASONING_EFFORT: ReasoningEffort = "none"
# The standard tier. The flex tier is cheaper and slower, and the priority tier is more expensive and faster.
SERVICE_TIER = "default"
# OpenAI reports the time that its servers took.
PROCESSING_TIME_HEADER = "openai-processing-ms"


class OpenAIClient:
    def __init__(self, config: ArmConfig, transport: httpx2.BaseTransport | None = None) -> None:
        self._config = config
        # The runner owns retries and the warm-up count, so the SDK must not retry.
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=0,
            timeout=openai.Timeout(TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            http_client=openai.DefaultHttpxClient(transport=transport) if transport else None,
        )

    @property
    def config(self) -> ArmConfig:
        return self._config

    @property
    def request_settings(self) -> dict[str, Any]:
        # The request sends no temperature, to match Haiku's request.
        return {"reasoning_effort": REASONING_EFFORT, "service_tier": SERVICE_TIER, "max_tokens": MAX_TOKENS}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._client.close()

    def models(self) -> dict[str, Any]:
        return self._client.models.retrieve(self._config.model).to_dict(mode="json")

    def answer(self, node: Node, state: str) -> Answer:
        start = time.perf_counter()
        raw = self._client.chat.completions.with_raw_response.create(
            model=self._config.model,
            max_completion_tokens=MAX_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            service_tier=SERVICE_TIER,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "probabilities", "strict": True, "schema": node.probability_schema()},
            },
            messages=[
                {"role": "system", "content": node.chat_system_prompt()},
                {"role": "user", "content": state},
            ],
        )
        e2e_ms = (time.perf_counter() - start) * 1000
        completion = raw.parse()
        if not completion.choices:
            raise ArmResponseError("response has no choices")
        choice = completion.choices[0]
        if choice.message.refusal:
            raise ArmResponseError(f"refusal: {choice.message.refusal[:ERROR_BODY_CHARS]}")
        if choice.finish_reason != "stop":
            raise ArmResponseError(f"finish_reason {choice.finish_reason!r}, not 'stop'")
        probabilities, normalised = scaled_probabilities(node, choice.message.content or "")
        processing_ms = raw.headers.get(PROCESSING_TIME_HEADER)
        return Answer(
            probabilities=probabilities,
            model_version=completion.model,
            model_ms=float(processing_ms) if processing_ms else None,
            e2e_ms=e2e_ms,
            input_tokens=completion.usage.prompt_tokens if completion.usage else None,
            output_tokens=completion.usage.completion_tokens if completion.usage else None,
            normalised=normalised,
        )
