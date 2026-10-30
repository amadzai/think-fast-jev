"""This module runs the arms on one decision node and writes one JSONL record per decision."""

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bench.nodes import Node
from bench.queries import Query
from bench.systemone import SystemOneClient, SystemOneError

WARMUP_CALLS = 3
MAX_ATTEMPTS = 3
BACKOFF_S = 1.0
ERROR_BODY_CHARS = 300


class ResultsFileError(Exception):
    """The existing results file has a line that the runner cannot read."""


@dataclass
class RunSummary:
    written: int = 0
    skipped: int = 0
    errors: int = 0
    # Arm name to the model versions that its responses report.
    model_versions: dict[str, set[str]] = field(default_factory=dict)


def run_node(node: Node, queries: list[Query], clients: dict[str, SystemOneClient], out: Path) -> RunSummary:
    """Call the arms for one reviewed query, then go to the next query. Skip decisions that have a success record in `out`."""
    done = _finished_keys(out)
    summary = RunSummary(model_versions={arm: set() for arm in clients})
    calls = dict.fromkeys(clients, 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as file:
        for query in (q for q in queries if q.reviewed):
            for arm, client in clients.items():
                if (node.name, query.id, arm, 0) in done:
                    summary.skipped += 1
                    continue
                # Warm-up is per process: a resumed run starts cold again.
                record = _decide(node, query, arm, client, is_warmup=calls[arm] < WARMUP_CALLS)
                calls[arm] += 1
                file.write(json.dumps(record) + "\n")
                file.flush()
                summary.written += 1
                if record["error"]:
                    summary.errors += 1
                else:
                    summary.model_versions[arm].add(record["model_version"])
    return summary


def _finished_keys(out: Path) -> set[tuple[str, str, str, int]]:
    # Error records do not count, so a rerun retries them. Scoring keeps the last record per key.
    if not out.exists():
        return set()
    keys = set()
    for number, line in enumerate(out.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if not r["error"]:
                keys.add((r["node"], r["item_id"], r["arm"], r["repeat"]))
        except (ValueError, KeyError, TypeError) as error:
            raise ResultsFileError(f"{out}:{number}: not a valid record ({error}). Fix or remove the line.") from error
    return keys


def _decide(node: Node, query: Query, arm: str, client: SystemOneClient, is_warmup: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "arm": arm,
        "model": client.model,
        "model_version": None,
        "node": node.name,
        "item_id": query.id,
        "repeat": 0,
        "selected": None,
        "probabilities": None,
        "model_ms": None,
        "e2e_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "warmup": is_warmup,
        "error": None,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = client.decide(query.text, node.system_one_questions())
            probabilities = _probabilities(node, result.answers[node.question_id])
        except (httpx.HTTPError, SystemOneError) as error:
            last_error = _describe(error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S * 2 ** (attempt - 1))
            continue
        return record | {
            "model_version": result.model,
            # The same decision rule for every arm: the argmax of the probabilities.
            "selected": max(probabilities, key=probabilities.__getitem__),
            "probabilities": probabilities,
            "model_ms": result.model_ms,
            "e2e_ms": result.e2e_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
    return record | {"error": last_error}


def _probabilities(node: Node, answer: Any) -> dict[str, float]:
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict) or set(probabilities) != set(node.options):
        raise SystemOneError(f"answer has no probability for each option: {str(answer)[:ERROR_BODY_CHARS]}")
    # Keep the node's option order, not the order of the response.
    return {name: float(probabilities[name]) for name in node.options}


def _describe(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}: {error.response.text[:ERROR_BODY_CHARS]}"
    return f"{type(error).__name__}: {error}"
