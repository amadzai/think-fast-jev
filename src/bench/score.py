"""This module scores the result records of one decision node against the labels."""

import csv
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from bench.config import Price
from bench.run import FIRST_REPEAT, Decision

TOKENS_PER_PRICE_UNIT = 1_000_000
DECISIONS_PER_COST_UNIT = 1_000


@dataclass(frozen=True)
class Score:
    """One row of the metrics table. None means that the metric does not apply or has no data."""

    node: str
    arm: str
    # The number of items in the quality scores: items that every arm answered validly in every repeat.
    items: int
    accuracy: float | None
    # Multi-class Brier score of the first repeat: 0 is perfect, 2 is the worst.
    brier: float | None = None
    # The share of items where all repeats selected the same option. Only for arms with more than one repeat.
    agreement: float | None = None
    model_p50: float | None = None
    model_p95: float | None = None
    e2e_p50: float | None = None
    e2e_p95: float | None = None
    # USD per 1k decisions. None for a model without a price per token (Kev).
    cost_per_1k: float | None = None
    error_rate: float | None = None


def score_node(node: str, records: list[Decision], labels: dict[str, str], prices: dict[str, Price]) -> list[Score]:
    """Return one row per arm. An arm with more than one repeat also gets a majority-vote row."""
    # A rerun appends a new record for the same key. The last record wins.
    latest = list({r.key: r for r in records if r.node == node}.values())
    arms = list(dict.fromkeys(r.arm for r in latest))
    repeats = {arm: max(r.repeat for r in latest if r.arm == arm) for arm in arms}
    valid = {r.key: r for r in latest if r.error is None and r.probabilities is not None}

    def repeats_of(item: str, arm: str) -> list[Decision]:
        # Empty unless every repeat of the item has a valid record.
        keys = [(node, item, arm, repeat) for repeat in range(FIRST_REPEAT, FIRST_REPEAT + repeats[arm])]
        return [valid[key] for key in keys] if all(key in valid for key in keys) else []

    candidates = [item for item in dict.fromkeys(r.item_id for r in latest) if item in labels]
    answered = {item: {arm: repeats_of(item, arm) for arm in arms} for item in candidates}
    items = [item for item in candidates if all(answered[item].values())]
    rows = []
    for arm in arms:
        arm_repeats = [answered[item][arm] for item in items]
        arm_records = [r for r in latest if r.arm == arm]
        price = prices.get(arm_records[0].model)
        rows.append(_arm_score(node, arm, arm_repeats, arm_records, labels, price))
        if repeats[arm] > 1:
            majority = _mean(_majority(group) == labels[group[0].item_id] for group in arm_repeats)
            rows.append(
                Score(node=node, arm=f"{arm} (majority of {repeats[arm]})", items=len(items), accuracy=majority)
            )
    return rows


def write_scores(rows: list[Score], stem: Path) -> str:
    """Write `<stem>.csv` and `<stem>.md`. Return the Markdown table."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(Score)]
    with stem.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    lines = [
        "| " + " | ".join(names) + " |",
        "|" + "---|" * len(names),
        *("| " + " | ".join(_cell(value) for value in asdict(row).values()) + " |" for row in rows),
    ]
    markdown = "\n".join(lines) + "\n"
    stem.with_suffix(".md").write_text(markdown)
    return markdown


def _arm_score(
    node: str,
    arm: str,
    repeats: list[list[Decision]],
    records: list[Decision],
    labels: dict[str, str],
    price: Price | None,
) -> Score:
    # The main quality scores use the first repeat only, so an arm with repeats gets no voting advantage.
    firsts = [group[0] for group in repeats]
    has_repeats = any(len(group) > 1 for group in repeats)
    # Latency uses every valid call that is not warm-up, also from items outside the quality scores.
    calls = [r for r in records if r.error is None and not r.warmup]
    model = [r.model_ms for r in calls if r.model_ms is not None]
    e2e = [r.e2e_ms for r in calls if r.e2e_ms is not None]
    cost = _mean(_cost(r, price) for r in records if r.error is None) if price else None
    return Score(
        node=node,
        arm=arm,
        items=len(repeats),
        accuracy=_mean(r.selected == labels[r.item_id] for r in firsts),
        brier=_mean(_brier(r, labels[r.item_id]) for r in firsts),
        agreement=_mean(len({r.selected for r in group}) == 1 for group in repeats) if has_repeats else None,
        model_p50=_percentile(model, 50),
        model_p95=_percentile(model, 95),
        e2e_p50=_percentile(e2e, 50),
        e2e_p95=_percentile(e2e, 95),
        cost_per_1k=cost * DECISIONS_PER_COST_UNIT if cost is not None else None,
        error_rate=_mean(r.error is not None for r in records),
    )


def _majority(group: list[Decision]) -> str | None:
    # On a tie, Counter keeps the first option it saw, which is the answer of the first repeat.
    return Counter(r.selected for r in group).most_common(1)[0][0]


def _brier(record: Decision, label: str) -> float:
    probabilities = record.probabilities or {}
    return sum((p - (option == label)) ** 2 for option, p in probabilities.items())


def _cost(record: Decision, price: Price) -> float:
    tokens_cost = (record.input_tokens or 0) * price.input_per_1m + (record.output_tokens or 0) * price.output_per_1m
    return tokens_cost / TOKENS_PER_PRICE_UNIT


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _percentile(values: list[float], q: float) -> float | None:
    # Linear interpolation between the closest ranks, the same as the numpy default.
    if not values:
        return None
    ordered = sorted(values)
    position = q / 100 * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _cell(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
