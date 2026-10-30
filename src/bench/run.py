"""This module runs the arms on one decision node and writes one JSONL record per decision."""

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import httpx

from bench.nodes import Node
from bench.queries import Query
from bench.systemone import ERROR_BODY_CHARS, SystemOneClient, SystemOneError, describe_error

WARMUP_CALLS = 3
MAX_ATTEMPTS = 3
BACKOFF_S = 1.0
# Only Haiku runs more than one repeat. Jev and Kev have repeat 0 only.
FIRST_REPEAT = 0

type DecisionKey = tuple[str, str, str, int]


class ResultsFileError(Exception):
    """An existing results file or run metadata file has content that the runner cannot read."""


@dataclass(frozen=True)
class Decision:
    """One line of the results file."""

    arm: str
    model: str
    node: str
    item_id: str
    repeat: int
    warmup: bool
    timestamp: str
    model_version: str | None = None
    selected: str | None = None
    probabilities: dict[str, float] | None = None
    model_ms: float | None = None
    e2e_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    @property
    def key(self) -> DecisionKey:
        return (self.node, self.item_id, self.arm, self.repeat)


@dataclass(frozen=True)
class ArmSettings:
    base_url: str
    model: str
    # The model versions that the responses of this run report.
    versions: tuple[str, ...]


@dataclass(frozen=True)
class RunSummary:
    node: str
    started: str
    finished: str
    written: int
    errors: int
    skipped: int
    arms: dict[str, ArmSettings]


def run_node(node: Node, queries: list[Query], clients: dict[str, SystemOneClient], out: Path) -> RunSummary:
    """Call the arms for one reviewed query, then go to the next query. Skip decisions that have a success record in `out`."""
    started = datetime.now(UTC).isoformat()
    done = _finished_keys(out)
    todo = [(q, arm) for q in queries if q.reviewed for arm in clients]
    pending = [(q, arm) for q, arm in todo if (node.name, q.id, arm, FIRST_REPEAT) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as file:
        decisions = [_write(file, decision) for decision in _decisions(node, pending, clients)]
    return RunSummary(
        node=node.name,
        started=started,
        finished=datetime.now(UTC).isoformat(),
        written=len(decisions),
        errors=sum(d.error is not None for d in decisions),
        skipped=len(todo) - len(pending),
        arms={
            arm: ArmSettings(
                base_url=client.config.base_url,
                model=client.config.model,
                versions=tuple(sorted({d.model_version for d in decisions if d.arm == arm and d.model_version})),
            )
            for arm, client in clients.items()
        },
    )


def append_run_metadata(out: Path, summary: RunSummary, queries: Path) -> None:
    """Add one entry to `<out>.meta.json`, so a resumed run keeps the settings of the runs before it."""
    path = out.with_suffix(".meta.json")
    try:
        runs = json.loads(path.read_text())["runs"] if path.exists() else []
    except (ValueError, KeyError, TypeError) as error:
        raise ResultsFileError(f"{path}: not a valid run metadata file ({error})") from error
    entry = asdict(summary) | {"queries": str(queries), "warmup_calls": WARMUP_CALLS, "max_attempts": MAX_ATTEMPTS}
    path.write_text(json.dumps({"runs": [*runs, entry]}, indent=2) + "\n")


def _decisions(node: Node, pending: list[tuple[Query, str]], clients: dict[str, SystemOneClient]) -> Iterator[Decision]:
    # Warm-up counts HTTP calls per arm in this process, retries included. A resumed run starts cold again.
    calls = dict.fromkeys(clients, 0)
    for query, arm in pending:
        decision, attempts = _decide(node, query, arm, clients[arm], calls[arm])
        calls = calls | {arm: calls[arm] + attempts}
        yield decision


def _write(file: TextIO, decision: Decision) -> Decision:
    file.write(json.dumps(asdict(decision)) + "\n")
    file.flush()
    return decision


def _finished_keys(out: Path) -> set[DecisionKey]:
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


def _decide(node: Node, query: Query, arm: str, client: SystemOneClient, calls_before: int) -> tuple[Decision, int]:
    """Return the decision and the number of HTTP calls it used."""
    base = Decision(
        arm=arm,
        model=client.config.model,
        node=node.name,
        item_id=query.id,
        repeat=FIRST_REPEAT,
        warmup=calls_before < WARMUP_CALLS,
        timestamp=datetime.now(UTC).isoformat(),
    )
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = client.decide(query.text, node.system_one_questions())
            probabilities = _probabilities(node, result.answers[node.question_id])
        except (httpx.HTTPError, SystemOneError) as error:
            last_error = describe_error(error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S * 2 ** (attempt - 1))
            continue
        decision = replace(
            base,
            warmup=calls_before + attempt <= WARMUP_CALLS,
            model_version=result.model,
            # The same decision rule for every arm: the argmax of the probabilities.
            selected=max(probabilities, key=probabilities.__getitem__),
            probabilities=probabilities,
            model_ms=result.model_ms,
            e2e_ms=result.e2e_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        return decision, attempt
    return replace(base, error=last_error), MAX_ATTEMPTS


def _probabilities(node: Node, answer: Any) -> dict[str, float]:
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict) or set(probabilities) != set(node.options):
        raise SystemOneError(f"answer has no probability for each option: {str(answer)[:ERROR_BODY_CHARS]}")
    # Keep the node's option order, not the order of the response.
    return {name: float(probabilities[name]) for name in node.options}
