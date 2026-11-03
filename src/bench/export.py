"""This module writes the reviewed router and guardrail queries as a Hugging Face dataset (`bench export`)."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from bench.nodes import GUARDRAIL, ROUTER
from bench.queries import Query

DATA_FILE = "queries.jsonl"
CARD_FILE = "README.md"
# Only the query text and the labels. The run settings and the reviewed flag stay in the repo.
FIELDS = ("id", "text", "router", "guardrail", "source")


def export_dataset(queries: list[Query], generators: list[str], out: Path) -> int:
    """Write the reviewed queries and a dataset card to `out`. Return the number of queries."""
    reviewed = [q for q in queries if q.reviewed]
    out.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({name: value for name, value in asdict(q).items() if name in FIELDS}) for q in reviewed]
    (out / DATA_FILE).write_text("".join(f"{line}\n" for line in lines))
    (out / CARD_FILE).write_text(_card(reviewed, generators))
    return len(reviewed)


def _card(queries: list[Query], generators: list[str]) -> str:
    sources = Counter(q.source for q in queries)
    router = Counter(q.router for q in queries)
    guardrail = Counter(q.guardrail for q in queries)
    generated_by = ", ".join(f"`{model}`" for model in generators) or "no model"

    def table(options: dict[str, str], counts: Counter[str]) -> str:
        rows = [f"| `{name}` | {description} | {counts[name]} |" for name, description in options.items()]
        return "\n".join(["| Option | Description | Queries |", "|---|---|---|", *rows])

    return f"""---
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
pretty_name: think-fast-jev router and guardrail queries
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: {DATA_FILE}
---

# think-fast-jev router and guardrail queries

{len(queries)} user questions about crypto markets. Each question has two labels: the data tool that can answer it
(router) and whether it may be answered (guardrail). The
[think-fast-jev](https://github.com/amadzai/think-fast-jev) benchmark uses this set to compare single-pass decision
models with a frontier LLM at these two decisions.

## Fields

| Field | Description |
|---|---|
| `id` | The query ID. |
| `text` | The user question. |
| `router` | The router label: one of the {len(ROUTER.options)} options below. |
| `guardrail` | The guardrail label: one of the {len(GUARDRAIL.options)} options below. |
| `source` | `seed` or `generated`. |

## Sources

- `seed` ({sources["seed"]} queries): Claude (Anthropic) drafted these queries. Public example prompts for crypto data
  tools were the guide.
- `generated` ({sources["generated"]} queries): {generated_by} on OpenRouter wrote these variations of the seeds. Each
  variation starts with the labels of its seed.

These queries are not real user traffic.

## Router options

{table(ROUTER.options, router)}

## Guardrail options

`allow` does not mean that a data tool can answer the question. A crypto fact question with no data tool is `allow`
and router `unsupported`.

{table(GUARDRAIL.options, guardrail)}

## License

CC-BY-4.0.
"""
