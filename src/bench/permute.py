"""This module runs the option-order pass (`bench permute`) and writes one JSONL record per item and arm.

The pass asks the node's choice question once for each option order, through POST /v1/systemone. Hosted Jev has no
permute endpoint, so the client makes the orders. It makes them the same way as Kev's /v1/systemone/permute, so every
arm gets the same orders.
"""

import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.arm import CALL_ERRORS, ArmClient, argmax, describe_error
from bench.nodes import Node
from bench.run import BACKOFF_S, MAX_ATTEMPTS, Item, load_jsonl

# The default of Kev's permute endpoint. The first order is the node's order. The other 5 orders are shuffled.
PERMUTATIONS = 6
# Each order is one call. Kev's permute endpoint has the same limit.
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
    node: Node, items: Sequence[Item], client: ArmClient, out: Path, n_perm: int = PERMUTATIONS
) -> tuple[int, int, int]:
    """Ask each item under `n_perm` option orders. Skip items that have a success record of this arm in `out`.

    Return the number of written records, of error records, and of skipped items.
    """
    arm = client.config.name
    done = {r.item_id for r in load_permutations(out) if r.node == node.name and r.arm == arm and not r.error}
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


def option_orders(node: Node, n_perm: int, seed: int) -> list[list[str]]:
    """Return the node's order, then `n_perm - 1` shuffled orders. This is the method of Kev's permute endpoint."""
    rng = random.Random(seed)
    orders = []
    for i in range(n_perm):
        order = list(node.options)
        if i > 0:
            rng.shuffle(order)
        orders.append(order)
    return orders


def _permute(node: Node, item: Item, client: ArmClient, n_perm: int) -> Permutation:
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
            runs = [_run(node, order, item, client) for order in option_orders(node, n_perm, SEED)]
        except CALL_ERRORS as error:
            last_error = describe_error(error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S * 2 ** (attempt - 1))
            continue
        spread = {
            name: max(r["probabilities"][name] for r in runs) - min(r["probabilities"][name] for r in runs)
            for name in node.options
        }
        return replace(base, stable=len({r["choice"] for r in runs}) == 1, spread=spread, runs=runs)
    return replace(base, error=last_error)


def _run(node: Node, order: list[str], item: Item, client: ArmClient) -> dict[str, Any]:
    answer = client.answer(replace(node, options={name: node.options[name] for name in order}), item.text)
    return {
        "order": order,
        "probabilities": answer.probabilities,
        "choice": argmax(answer.probabilities),
        "model_ms": answer.model_ms,
        "e2e_ms": answer.e2e_ms,
    }
