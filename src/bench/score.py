"""This module scores the result records of one decision node against the labels."""

import csv
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from bench.config import SELF_HOSTED_HOURLY_USD, Price
from bench.permute import Permutation
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
    # Router and guardrail.
    accuracy: float | None = None
    # Coin sector: the mean F1 over the sectors, yes at 0.5.
    macro_f1: float | None = None
    # Brier score of the first repeat. 0 is perfect. Router and guardrail: multi-class, 2 is the worst.
    # Coin sector: the binary score per sector question, 1 is the worst.
    brier: float | None = None
    # The share of items where all repeats selected the same option (coin sector: the same sectors). Only for arms
    # with more than one repeat.
    agreement: float | None = None
    # Kev, router and guardrail (`bench permute`): the share of items where every option order selects the same option.
    order_stable: float | None = None
    # Kev: the mean over the items of the largest probability change of one option between the option orders.
    order_spread: float | None = None
    model_p50: float | None = None
    model_p95: float | None = None
    e2e_p50: float | None = None
    e2e_p95: float | None = None
    # USD per 1k decisions, from the tokens and the price per token. Self-hosted Kev: the GPU hourly price divided by
    # the decisions per hour of `bench throughput`. None when there is no price or no throughput.
    cost_per_1k: float | None = None
    error_rate: float | None = None


@dataclass(frozen=True)
class _Scored:
    """The records of one node, and the repeat groups of the items that every arm answered validly."""

    arms: list[str]
    repeats: dict[str, int]
    # Arm to one list of repeat records per scored item, in the same item order for every arm.
    groups: dict[str, list[list[Decision]]]
    # Arm to all its latest records, also the errors and the items outside the quality scores.
    records: dict[str, list[Decision]]


def score_node(node: str, records: list[Decision], labels: dict[str, str], prices: dict[str, Price]) -> list[Score]:
    """Return one row per arm. An arm with more than one repeat also gets a majority-vote row."""
    scored = _scored(node, records, labels.keys())
    rows = []
    for arm in scored.arms:
        groups = scored.groups[arm]
        firsts = [group[0] for group in groups]
        rows.append(
            _arm_score(
                node,
                arm,
                scored,
                prices,
                accuracy=_mean(r.selected == labels[r.item_id] for r in firsts),
                brier=_mean(_brier(r, frozenset({labels[r.item_id]})) for r in firsts),
            )
        )
        if scored.repeats[arm] > 1:
            majority = _mean(_majority(group) == labels[group[0].item_id] for group in groups)
            rows.append(_majority_row(node, arm, scored, accuracy=majority))
    return rows


def score_sectors(
    node: str, records: list[Decision], labels: dict[str, frozenset[str]], sectors: list[str], prices: dict[str, Price]
) -> list[Score]:
    """Return one row per arm for a multi-label node. An arm with more than one repeat also gets a majority-vote row.

    The majority vote is per sector question: yes when more than half of the repeats say yes.
    """
    scored = _scored(node, records, labels.keys())
    rows = []
    for arm in scored.arms:
        groups = scored.groups[arm]
        firsts = [group[0] for group in groups]
        first_yes = [(set(r.selected or []), labels[r.item_id]) for r in firsts]
        rows.append(
            _arm_score(
                node,
                arm,
                scored,
                prices,
                macro_f1=_macro_f1(first_yes, sectors),
                # The mean per sector question, so that the score does not grow with the number of sectors.
                brier=_mean(_brier(r, labels[r.item_id]) / len(sectors) for r in firsts),
            )
        )
        if scored.repeats[arm] > 1:
            majority = [(_majority_yes(group, sectors), labels[group[0].item_id]) for group in groups]
            rows.append(_majority_row(node, arm, scored, macro_f1=_macro_f1(majority, sectors)))
    return rows


def with_kev_scores(
    rows: list[Score],
    arm: str,
    permutations: list[Permutation],
    decisions_per_hour: float | None,
) -> list[Score]:
    """Add the option-order stability and the self-hosted cost to the rows of `arm`.

    The stability uses every item with a valid permutation record, also items outside the quality scores.
    """

    def with_scores(row: Score) -> Score:
        if row.arm != arm:
            return row
        # A rerun appends a new record for the same item. The last record wins.
        latest = {p.item_id: p for p in permutations if p.node == row.node and p.arm == arm}.values()
        valid = [p for p in latest if p.error is None and p.spread]
        cost = SELF_HOSTED_HOURLY_USD / decisions_per_hour * DECISIONS_PER_COST_UNIT if decisions_per_hour else None
        return replace(
            row,
            order_stable=_mean(bool(p.stable) for p in valid),
            order_spread=_mean(max((p.spread or {}).values()) for p in valid),
            cost_per_1k=cost if cost is not None else row.cost_per_1k,
        )

    return [with_scores(row) for row in rows]


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


def _scored(node: str, records: list[Decision], labelled: Iterable[str]) -> _Scored:
    # A rerun appends a new record for the same key. The last record wins.
    latest = list({r.key: r for r in records if r.node == node}.values())
    arms = list(dict.fromkeys(r.arm for r in latest))
    repeats = {arm: max(r.repeat for r in latest if r.arm == arm) for arm in arms}
    valid = {r.key: r for r in latest if r.error is None and r.probabilities is not None}

    def repeats_of(item: str, arm: str) -> list[Decision]:
        # Empty unless every repeat of the item has a valid record.
        keys = [(node, item, arm, repeat) for repeat in range(FIRST_REPEAT, FIRST_REPEAT + repeats[arm])]
        return [valid[key] for key in keys] if all(key in valid for key in keys) else []

    labelled_items = set(labelled)
    candidates = [item for item in dict.fromkeys(r.item_id for r in latest) if item in labelled_items]
    answered = {item: {arm: repeats_of(item, arm) for arm in arms} for item in candidates}
    items = [item for item in candidates if all(answered[item].values())]
    return _Scored(
        arms=arms,
        repeats=repeats,
        groups={arm: [answered[item][arm] for item in items] for arm in arms},
        records={arm: [r for r in latest if r.arm == arm] for arm in arms},
    )


def _arm_score(
    node: str,
    arm: str,
    scored: _Scored,
    prices: dict[str, Price],
    accuracy: float | None = None,
    macro_f1: float | None = None,
    brier: float | None = None,
) -> Score:
    # The quality scores use the first repeat only, so an arm with repeats gets no voting advantage.
    groups, records = scored.groups[arm], scored.records[arm]
    price = prices.get(records[0].model)
    has_repeats = scored.repeats[arm] > 1
    # Latency uses every valid call that is not warm-up, also from items outside the quality scores.
    calls = [r for r in records if r.error is None and not r.warmup]
    model = [r.model_ms for r in calls if r.model_ms is not None]
    e2e = [r.e2e_ms for r in calls if r.e2e_ms is not None]
    cost = _mean(_cost(r, price) for r in records if r.error is None) if price else None
    return Score(
        node=node,
        arm=arm,
        items=len(groups),
        accuracy=accuracy,
        macro_f1=macro_f1,
        brier=brier,
        agreement=_mean(all(r.selected == group[0].selected for r in group) for group in groups)
        if has_repeats
        else None,
        model_p50=_percentile(model, 50),
        model_p95=_percentile(model, 95),
        e2e_p50=_percentile(e2e, 50),
        e2e_p95=_percentile(e2e, 95),
        cost_per_1k=cost * DECISIONS_PER_COST_UNIT if cost is not None else None,
        error_rate=_mean(r.error is not None for r in records),
    )


def _majority_row(
    node: str, arm: str, scored: _Scored, accuracy: float | None = None, macro_f1: float | None = None
) -> Score:
    return Score(
        node=node,
        arm=f"{arm} (majority of {scored.repeats[arm]})",
        items=len(scored.groups[arm]),
        accuracy=accuracy,
        macro_f1=macro_f1,
    )


def _majority(group: list[Decision]) -> str | list[str] | None:
    # On a tie, Counter keeps the first option it saw, which is the answer of the first repeat.
    return Counter(r.selected for r in group).most_common(1)[0][0]


def _majority_yes(group: list[Decision], sectors: list[str]) -> set[str]:
    return {s for s in sectors if sum(s in (r.selected or []) for r in group) * 2 > len(group)}


def _macro_f1(predictions: list[tuple[set[str], frozenset[str]]], sectors: list[str]) -> float | None:
    """Return the mean F1 over the sectors. Each prediction is (the yes sectors, the label sectors) of one item.

    A sector with no label yes and no predicted yes has no F1, so it is left out of the mean.
    """
    scores = []
    for s in sectors:
        true_yes = sum(s in yes and s in label for yes, label in predictions)
        false_yes = sum(s in yes and s not in label for yes, label in predictions)
        false_no = sum(s not in yes and s in label for yes, label in predictions)
        if true_yes + false_yes + false_no:
            scores.append(2 * true_yes / (2 * true_yes + false_yes + false_no))
    return _mean(scores)


def _brier(record: Decision, label: frozenset[str]) -> float:
    """Return the sum of (p - 1)² over the label options and p² over the other options."""
    probabilities = record.probabilities or {}
    return sum((p - (option in label)) ** 2 for option, p in probabilities.items())


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
