"""This module runs the arms on one decision node and writes one JSONL record per decision."""

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TextIO

from bench.arm import CALL_ERRORS, ArmClient, argmax, describe_error, yes_options
from bench.nodes import Node

WARMUP_CALLS = 3
MAX_ATTEMPTS = 3
BACKOFF_S = 1.0
# Repeat numbers start at 1. Jev and Kev have repeat 1 only, Haiku has repeats 1 to 3.
FIRST_REPEAT = 1
# The state of a keep-warm call. It differs from every query. Kev caches by state, so the recorded call gets no cache hit.
KEEP_WARM_STATE = "Keep-warm call. Not recorded."
# An error record with this prefix has no recorded call: the keep-warm call failed before it.
KEEP_WARM_ERROR_PREFIX = "keep-warm call: "

type DecisionKey = tuple[str, str, str, int]


class Item(Protocol):
    """One input of a decision node: a query or a coin."""

    @property
    def id(self) -> str: ...

    @property
    def text(self) -> str:
        """The state that every arm gets."""
        ...


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
    # The selected option. For a multi-label node (coin sector), the list of options with yes.
    selected: str | list[str] | None = None
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
    # True: one unrecorded keep-warm call comes right before each recorded call.
    keep_warm: bool
    # For example Haiku's thinking setting. Empty for Jev and Kev.
    request_settings: dict[str, Any]
    # The model versions that the responses of this run report.
    versions: tuple[str, ...]
    # The server's model description before and after the run, or {"error": ...}. For Kev it has the run, the LoRA,
    # the temperature and the prefix cache counters. The counters count from server start, so compare the two.
    # With keep_warm, the counters include the keep-warm calls.
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


def run_node(node: Node, items: Sequence[Item], clients: dict[str, ArmClient], out: Path) -> RunSummary:
    """Call the arms for one item, then go to the next item. Skip decisions that have a success record in `out`."""
    started = datetime.now(UTC).isoformat()
    models_before = {arm: _models(client) for arm, client in clients.items()}
    done = _finished_keys(out)
    # An arm's repeats stay together, so the arm order rotates by arm, not by call.
    todo = [
        (q, arm, repeat)
        for i, q in enumerate(items)
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
                keep_warm=client.config.keep_warm,
                request_settings=client.request_settings,
                versions=tuple(sorted({d.model_version for d in decisions if d.arm == arm and d.model_version})),
                models_before=models_before[arm],
                models_after=_models(client),
            )
            for arm, client in clients.items()
        },
    )


def append_run_metadata(out: Path, summary: RunSummary, items: Path, generators: list[str] | None = None) -> None:
    """Add one entry to `<out>.meta.json`, so a resumed run keeps the settings of the runs before it."""
    path = out.with_suffix(".meta.json")
    try:
        runs = json.loads(path.read_text())["runs"] if path.exists() else []
    except (ValueError, KeyError, TypeError) as error:
        raise ResultsFileError(f"{path}: not a valid run metadata file ({error})") from error
    entry = asdict(summary) | {
        # The query file or the coin file.
        "items": str(items),
        # The models that wrote the generated queries in the query file.
        "generators": generators or [],
        "warmup_calls": WARMUP_CALLS,
        "max_attempts": MAX_ATTEMPTS,
    }
    path.write_text(json.dumps({"runs": [*runs, entry]}, indent=2) + "\n")


def _rotated(arms: list[str], index: int) -> list[str]:
    # The arm order rotates from item to item, so that no arm always gets the call right after the same other arm.
    shift = index % len(arms)
    return arms[shift:] + arms[:shift]


def _decisions(node: Node, pending: list[tuple[Item, str, int]], clients: dict[str, ArmClient]) -> Iterator[Decision]:
    # Warm-up counts HTTP calls per arm in this process, retries included. A resumed run starts cold again.
    calls = dict.fromkeys(clients, 0)
    for item, arm, repeat in pending:
        decision, attempts = _decide(node, item, arm, repeat, clients[arm], calls[arm])
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


def load_records(path: Path) -> list[Decision]:
    """Read a results file. Return an empty list when the file does not exist."""
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(Decision(**json.loads(line)))
        except (ValueError, TypeError) as error:
            raise ResultsFileError(f"{path}:{number}: not a valid record ({error}). Fix or remove the line.") from error
    return records


def _finished_keys(out: Path) -> set[DecisionKey]:
    # Error records do not count, so a rerun retries them. Scoring keeps the last record per key.
    return {r.key for r in load_records(out) if not r.error}


def _decide(
    node: Node, item: Item, arm: str, repeat: int, client: ArmClient, calls_before: int
) -> tuple[Decision, int]:
    """Return the decision and the number of HTTP calls it sent, keep-warm calls included."""
    base = Decision(
        arm=arm,
        model=client.config.model,
        node=node.name,
        item_id=item.id,
        repeat=repeat,
        warmup=calls_before < WARMUP_CALLS,
        timestamp=datetime.now(UTC).isoformat(),
    )
    calls = 0
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prefix = ""
        try:
            if client.config.keep_warm:
                calls, prefix = calls + 1, KEEP_WARM_ERROR_PREFIX
                client.answer(node, KEEP_WARM_STATE)
            calls, prefix = calls + 1, ""
            answer = client.answer(node, item.text)
        except CALL_ERRORS as error:
            last_error = prefix + describe_error(error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S * 2 ** (attempt - 1))
            continue
        decision = replace(
            base,
            warmup=calls_before + calls <= WARMUP_CALLS,
            model_version=answer.model_version,
            selected=yes_options(answer.probabilities) if node.multi_label else argmax(answer.probabilities),
            probabilities=answer.probabilities,
            model_ms=answer.model_ms,
            e2e_ms=answer.e2e_ms,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            normalised=answer.normalised,
        )
        return decision, calls
    return replace(base, error=last_error), calls
