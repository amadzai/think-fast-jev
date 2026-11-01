"""This module runs Kev's option-order pass (`bench permute`) and writes one JSONL record per item."""

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.arm import CALL_ERRORS, describe_error
from bench.nodes import Node
from bench.run import BACKOFF_S, MAX_ATTEMPTS, Item, load_jsonl
from bench.systemone import SystemOneClient

# The server default. The first order is the node's order. The server shuffles the other 5 orders.
PERMUTATIONS = 6
# The server accepts at most 64 option orders.
MAX_PERMUTATIONS = 64
# A fixed seed gives the same shuffled orders on every run.
SEED = 0


@dataclass(frozen=True)
class Permutation:
    """One line of the permutation file."""

    arm: str
    model: str
    node: str
    item_id: str
    n_perm: int
    seed: int
    timestamp: str
    # True when every option order selects the same option.
    stable: bool | None = None
    # The probability change of each option between the option orders: the largest value minus the smallest value.
    spread: dict[str, float] | None = None
    runs: list[dict[str, Any]] | None = None
    error: str | None = None


def permutation_path(results: Path) -> Path:
    """Return the permutation file that belongs to a results file: results/router.jsonl -> results/router.permute.jsonl."""
    return results.with_suffix(".permute.jsonl")


def permute_node(
    node: Node, items: Sequence[Item], client: SystemOneClient, out: Path, n_perm: int = PERMUTATIONS
) -> tuple[int, int, int]:
    """Ask each item under `n_perm` option orders. Skip items that have a success record in `out`.

    Return the number of written records, of error records, and of skipped items.
    """
    done = {r.item_id for r in load_permutations(out) if r.node == node.name and not r.error}
    pending = [item for item in items if item.id not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with out.open("a") as file:
        for item in pending:
            record = _permute(node, item, client, n_perm)
            errors += record.error is not None
            file.write(json.dumps(asdict(record)) + "\n")
            file.flush()
    return len(pending), errors, len(items) - len(pending)


def load_permutations(path: Path) -> list[Permutation]:
    """Read a permutation file. Return an empty list when the file does not exist."""
    return load_jsonl(path, Permutation)


def _permute(node: Node, item: Item, client: SystemOneClient, n_perm: int) -> Permutation:
    base = Permutation(
        arm=client.config.name,
        model=client.config.model,
        node=node.name,
        item_id=item.id,
        n_perm=n_perm,
        seed=SEED,
        timestamp=datetime.now(UTC).isoformat(),
    )
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            answer = client.permute(node, item.text, n_perm, SEED)
        except CALL_ERRORS as error:
            last_error = describe_error(error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S * 2 ** (attempt - 1))
            continue
        return replace(base, stable=answer.stable, spread=answer.spread, runs=answer.runs)
    return replace(base, error=last_error)
