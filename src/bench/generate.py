"""This module writes query variations from the seed queries with a model on OpenRouter (`bench generate`)."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bench.arm import CONNECT_TIMEOUT_S, ERROR_BODY_CHARS, describe_error
from bench.config import EXCLUDED_GENERATOR_FAMILIES, GeneratorConfig
from bench.nodes import GUARDRAIL, ROUTER
from bench.queries import Query, QueryFileError

# A generation call writes much more text than a decision call.
GENERATE_TIMEOUT_S = 120.0
GENERATED_ID_PREFIX = "gen-"


class GenerateError(Exception):
    """The generator model sent a response that the command cannot read, or it is a Claude or Qwen model."""


@dataclass(frozen=True)
class Generation:
    queries: list[Query]
    # Variations that the command did not add: a wrong label, empty text, or the same text as another query.
    dropped: int
    # The model ID that OpenRouter reports in the responses.
    versions: list[str]
    # The error that stopped the command before the last seed, or None. The command keeps the queries before the error.
    error: str | None = None


def generate_variations(
    config: GeneratorConfig, seeds: list[Query], existing: list[Query], per_seed: int
) -> Generation:
    """Return new, unreviewed queries. Their IDs and texts differ from all queries in `existing`."""
    taken = {_normalised(q.text) for q in existing}
    next_number = 1 + max((_generated_number(q.id) for q in existing), default=0)
    queries: list[Query] = []
    dropped = 0
    versions: set[str] = set()
    error = None
    with httpx.Client(
        base_url=config.base_url,
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=httpx.Timeout(GENERATE_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
    ) as http:
        for seed in seeds:
            try:
                proposals, version = _call(http, config.model, seed, per_seed)
            except (httpx.HTTPError, GenerateError) as exc:
                error = f"{seed.id}: {describe_error(exc)}"
                break
            versions.add(version)
            # A model can return more variations than the prompt asks for.
            for proposal in proposals[:per_seed]:
                query = _to_query(proposal, f"{GENERATED_ID_PREFIX}{next_number + len(queries):03d}", taken)
                if query is None:
                    dropped += 1
                    continue
                taken.add(_normalised(query.text))
                queries.append(query)
    return Generation(queries=queries, dropped=dropped, versions=sorted(versions), error=error)


def append_queries(path: Path, queries: list[Query]) -> None:
    """Add the queries at the end of the query file. The existing lines do not change."""
    text = path.read_text() if path.exists() else ""
    with path.open("a") as file:
        if text and not text.endswith("\n"):
            file.write("\n")
        file.writelines(json.dumps(asdict(q)) + "\n" for q in queries)


def generator_models(queries: Path) -> list[str]:
    """Return the models that generated queries in the query file."""
    return sorted({g["model"] for g in load_generations(queries)})


def load_generations(queries: Path) -> list[dict[str, Any]]:
    """Return the entries of `<queries>.meta.json`. Return an empty list when the file does not exist."""
    meta = queries.with_suffix(".meta.json")
    if not meta.exists():
        return []
    try:
        generations = json.loads(meta.read_text())["generations"]
    except (ValueError, KeyError, TypeError) as error:
        raise QueryFileError(f"{meta}: not a valid generation metadata file ({error})") from error
    if not isinstance(generations, list) or not all(isinstance(g, dict) and "model" in g for g in generations):
        raise QueryFileError(f"{meta}: each generation must be an object with a model")
    return generations


def append_generation_metadata(
    queries: Path, previous: list[dict[str, Any]], config: GeneratorConfig, seeds: list[Query], result: Generation
) -> None:
    """Write `<queries>.meta.json` with one more entry. The entry records the generator model of the new queries."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_url": config.base_url,
        "model": config.model,
        "versions": result.versions,
        "seeds": [q.id for q in seeds],
        "added": [q.id for q in result.queries],
        "dropped": result.dropped,
        "error": result.error,
    }
    queries.with_suffix(".meta.json").write_text(json.dumps({"generations": [*previous, entry]}, indent=2) + "\n")


def _to_query(proposal: dict[str, Any], query_id: str, taken: set[str]) -> Query | None:
    """Return None for empty text, a text in `taken`, or a wrong label."""
    text, router, guardrail = proposal.get("text"), proposal.get("router"), proposal.get("guardrail")
    # Not every provider enforces the schema, so this function checks the labels again.
    if not isinstance(text, str) or not text.strip() or _normalised(text) in taken:
        return None
    if not isinstance(router, str) or router not in ROUTER.options:
        return None
    if not isinstance(guardrail, str) or guardrail not in GUARDRAIL.options:
        return None
    return Query(id=query_id, text=text.strip(), router=router, guardrail=guardrail, source="generated", reviewed=False)


def _call(http: httpx.Client, model: str, seed: Query, per_seed: int) -> tuple[list[dict[str, Any]], str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt(per_seed)},
            {
                "role": "user",
                "content": json.dumps({"text": seed.text, "router": seed.router, "guardrail": seed.guardrail}),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "variations", "strict": True, "schema": _schema()},
        },
        # OpenRouter sends the request only to providers that support the JSON schema.
        "provider": {"require_parameters": True},
    }
    response = http.post("/api/v1/chat/completions", json=body)
    response.raise_for_status()
    try:
        data = response.json()
        proposals = json.loads(data["choices"][0]["message"]["content"])["queries"]
        if not isinstance(proposals, list) or not all(isinstance(p, dict) for p in proposals):
            raise TypeError("queries is not a list of objects")
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise GenerateError(f"unexpected response for {seed.id}: {response.text[:ERROR_BODY_CHARS]}") from error
    version = str(data.get("model", model))
    # A router model ID such as openrouter/auto can send the request to any model.
    if any(family in version.lower() for family in EXCLUDED_GENERATOR_FAMILIES):
        raise GenerateError(f"OpenRouter answered with {version}, a Claude or Qwen model. Use a fixed model ID.")
    return proposals, version


def _system_prompt(per_seed: int) -> str:
    def listing(options: dict[str, str]) -> str:
        return "\n".join(f"- {name}: {description}" for name, description in options.items())

    return (
        "You write test questions for a crypto market-data assistant. "
        f"The user message is one seed question with its labels. Write {per_seed} new questions like it.\n"
        "Rules:\n"
        "- Change the coin, the exchange, the chain, the time range or the phrasing. Do not copy the seed.\n"
        "- Write like real users: some short, some with typos, some casual.\n"
        "- Give each question its correct labels. A label can differ from the seed label.\n\n"
        f"Router labels (the data tool that answers the question):\n{listing(ROUTER.options)}\n\n"
        f"Guardrail labels (whether the assistant may answer):\n{listing(GUARDRAIL.options)}"
    )


def _schema() -> dict[str, Any]:
    query = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "router": {"type": "string", "enum": list(ROUTER.options)},
            "guardrail": {"type": "string", "enum": list(GUARDRAIL.options)},
        },
        "required": ["text", "router", "guardrail"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"queries": {"type": "array", "items": query}},
        "required": ["queries"],
        "additionalProperties": False,
    }


def _normalised(text: str) -> str:
    return " ".join(text.lower().split())


def _generated_number(query_id: str) -> int:
    suffix = query_id.removeprefix(GENERATED_ID_PREFIX)
    return int(suffix) if query_id.startswith(GENERATED_ID_PREFIX) and suffix.isdigit() else 0
