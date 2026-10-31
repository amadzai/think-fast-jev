"""This module runs the arms on one decision node and writes one JSONL record per decision."""

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from bench.arm import CALL_ERRORS, ArmClient, argmax, describe_error
from bench.nodes import Node
from bench.queries import Query

WARMUP_CALLS = 3
MAX_ATTEMPTS = 3
BACKOFF_S = 1.0
# Repeat numbers start at 1. Jev and Kev have repeat 1 only, Haiku has repeats 1 to 3.
FIRST_REPEAT = 1

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
    # True when the client scaled the probabilities to sum 1 (Haiku). None for Jev and Kev.
    normalised: bool | None = None
    error: str | None = None

    @property
    def key(self) -> DecisionKey:
        return (self.node, self.item_id, self.arm, self.repeat)


@dataclass(frozen=True)
class ArmSettings:
    base_url: str
    model: str
    repeats: int
    # For example Haiku's thinking setting. Empty for Jev and Kev.
    request_settings: dict[str, Any]
    # The model versions that the responses of this run report.
    versions: tuple[str, ...]
    # The server's model description before and after the run, or {"error": ...}. For Kev it has the run, the LoRA,
    # the temperature and the prefix cache counters. The counters count from server start, so compare the two.
    models_before: dict[str, Any]
    models_after: dict[str, Any]


@dataclass(frozen=True)
class RunSummary:
    node: str
    started: str
    finished: str
    written: int
    errors: int
    skipped: int
    arms: dict[str, ArmSettings]


def run_node(node: Node, queries: list[Query], clients: dict[str, ArmClient], out: Path) -> RunSummary:
    """Call the arms for one reviewed query, then go to the next query. Skip decisions that have a success record in `out`."""
    started = datetime.now(UTC).isoformat()
    models_before = {arm: _models(client) for arm, client in clients.items()}
    done = _finished_keys(out)
    reviewed = [q for q in queries if q.reviewed]
    # An arm's repeats stay together, so the arm order rotates by arm, not by call.
    todo = [
        (q, arm, repeat)
        for i, q in enumerate(reviewed)
        for arm in _rotated(list(clients), i)
        for repeat in range(FIRST_REPEAT, FIRST_REPEAT + clients[arm].config.repeats)
    ]
    pending = [(q, arm, repeat) for q, arm, repeat in todo if (node.name, q.id, arm, repeat) not in done]
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
                repeats=client.config.repeats,
                request_settings=client.request_settings,
                versions=tuple(sorted({d.model_version for d in decisions if d.arm == arm and d.model_version})),
                models_before=models_before[arm],
                models_after=_models(client),
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


def _rotated(arms: list[str], index: int) -> list[str]:
    # The arm order rotates from query to query, so that no arm always gets the call right after the same other arm.
    shift = index % len(arms)
    return arms[shift:] + arms[:shift]


def _decisions(node: Node, pending: list[tuple[Query, str, int]], clients: dict[str, ArmClient]) -> Iterator[Decision]:
    # Warm-up counts HTTP calls per arm in this process, retries included. A resumed run starts cold again.
    calls = dict.fromkeys(clients, 0)
    for query, arm, repeat in pending:
        decision, attempts = _decide(node, query, arm, repeat, clients[arm], calls[arm])
        calls = calls | {arm: calls[arm] + attempts}
        yield decision


def _models(client: ArmClient) -> dict[str, Any]:
    # The response is for the metadata only, so a failed call does not stop the run. The metadata keeps the error.
    try:
        return client.models()
    except CALL_ERRORS as error:
        return {"error": describe_error(error)}


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


def _decide(
    node: Node, query: Query, arm: str, repeat: int, client: ArmClient, calls_before: int
) -> tuple[Decision, int]:
    """Return the decision and the number of HTTP calls it used."""
    base = Decision(
        arm=arm,
        model=client.config.model,
        node=node.name,
        item_id=query.id,
        repeat=repeat,
        warmup=calls_before < WARMUP_CALLS,
        timestamp=datetime.now(UTC).isoformat(),
    )
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            answer = client.answer(node, query.text)
        except CALL_ERRORS as error:
            last_error = describe_error(error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S * 2 ** (attempt - 1))
            continue
        decision = replace(
            base,
            warmup=calls_before + attempt <= WARMUP_CALLS,
            model_version=answer.model_version,
            selected=argmax(answer.probabilities),
            probabilities=answer.probabilities,
            model_ms=answer.model_ms,
            e2e_ms=answer.e2e_ms,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            normalised=answer.normalised,
        )
        return decision, attempt
    return replace(base, error=last_error), MAX_ATTEMPTS
