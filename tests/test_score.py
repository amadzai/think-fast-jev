from dataclasses import replace

import pytest

from bench.config import Price
from bench.permute import Permutation
from bench.run import Decision
from bench.score import score_node, score_sectors, with_order_scores, with_self_hosted_cost

LABELS = {"q1": "price", "q2": "market_chart", "q3": "price"}
MODELS = {"jev": "jev-latest", "kev": "kev-latest", "haiku": "claude-haiku-5-5"}
PRICES = {
    "jev-latest": Price(input_per_1m=0.042, output_per_1m=0.0),
    "claude-haiku-5-5": Price(input_per_1m=0.10, output_per_1m=0.50),
}


def rec(arm: str, item: str, repeat: int, p_price: float | None, e2e: float, warmup: bool = False) -> Decision:
    """A record for options price and market_chart. `p_price` is the probability of price. None makes an error record."""
    tokens = (400, 0) if arm == "jev" else (1000, 100)
    if p_price is None:
        return Decision(
            arm=arm, model=MODELS[arm], node="router", item_id=item, repeat=repeat, warmup=warmup, timestamp="t",
            error="HTTP 500: busy",
        )  # fmt: skip
    return Decision(
        arm=arm,
        model=MODELS[arm],
        node="router",
        item_id=item,
        repeat=repeat,
        warmup=warmup,
        timestamp="t",
        selected="price" if p_price >= 0.5 else "market_chart",
        probabilities={"price": p_price, "market_chart": round(1 - p_price, 10)},
        model_ms=None if arm == "jev" else e2e - 200,
        e2e_ms=e2e,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
    )


def test_score_uses_valid_items_only_and_reports_haiku_majority_and_agreement() -> None:
    # Arrange: q3 has a Haiku error on repeat 2, so q3 leaves the quality scores of both arms.
    # Jev's q3 error is followed by a success from a rerun, so only the success counts.
    records = [
        rec("jev", "q1", 1, 0.8, 900, warmup=True), rec("jev", "q2", 1, 0.6, 200),
        rec("jev", "q3", 1, None, 0), rec("jev", "q3", 1, 0.9, 300),
        rec("haiku", "q1", 1, 1.0, 1000), rec("haiku", "q1", 2, 1.0, 1100), rec("haiku", "q1", 3, 1.0, 1200),
        rec("haiku", "q2", 1, 0.7, 1300), rec("haiku", "q2", 2, 0.2, 1400), rec("haiku", "q2", 3, 0.1, 1500),
        rec("haiku", "q3", 1, 0.9, 1600), rec("haiku", "q3", 2, None, 0), rec("haiku", "q3", 3, 0.9, 1700),
    ]  # fmt: skip

    # Act
    rows = {row.arm: row for row in score_node("router", records, LABELS, PRICES)}

    # Assert
    jev, haiku, majority = rows["jev"], rows["haiku"], rows["haiku (majority of 3)"]
    assert (jev.items, haiku.items, majority.items) == (2, 2, 2)
    assert (jev.accuracy, haiku.accuracy, majority.accuracy) == (0.5, 0.5, 1.0)
    # Jev: q1 (0.2² + 0.2²) = 0.08, q2 (0.6² + 0.6²) = 0.72. Haiku repeat 1: q1 0, q2 (0.7² + 0.7²) = 0.98.
    assert jev.brier == pytest.approx(0.40)
    assert haiku.brier == pytest.approx(0.49)
    assert (jev.agreement, haiku.agreement) == (None, 0.5)
    # Latency uses every valid record that is not warm-up, also from items left out of the quality scores.
    assert (jev.e2e_p50, jev.e2e_p95) == (pytest.approx(250), pytest.approx(295))
    assert (jev.model_p50, jev.model_p95) == (None, None)
    assert haiku.model_p50 == pytest.approx(1150)
    assert haiku.e2e_p95 == pytest.approx(1665)
    assert jev.cost_per_1k == pytest.approx(0.0168)
    assert haiku.cost_per_1k == pytest.approx(0.15)
    assert (jev.error_rate, haiku.error_rate) == (0.0, pytest.approx(1 / 9))


def test_arm_without_a_price_has_no_cost() -> None:
    # Act
    rows = score_node("router", [rec("kev", "q1", 1, 0.8, 70)], LABELS, PRICES)

    # Assert
    assert [(row.arm, row.accuracy, row.cost_per_1k) for row in rows] == [("kev", 1.0, None)]


def sector(arm: str, item: str, repeat: int, defi: float, meme: float) -> Decision:
    """A coin sector record for the sectors defi, meme and ai. The ai probability is always 0."""
    probabilities = {"defi": defi, "meme": meme, "ai": 0.0}
    return Decision(
        arm=arm, model=MODELS[arm], node="coin_sector", item_id=item, repeat=repeat, warmup=False, timestamp="t",
        selected=[name for name, p in probabilities.items() if p >= 0.5], probabilities=probabilities, e2e_ms=100,
    )  # fmt: skip


def test_coin_sector_score_gives_macro_f1_over_sectors_and_binary_brier_per_question() -> None:
    # Arrange: no coin is in ai and no arm says yes to ai, so ai has no F1 and stays out of the macro F1.
    labels = {"c1": frozenset({"defi"}), "c2": frozenset({"meme"}), "c3": frozenset()}
    records = [
        sector("jev", "c1", 1, 0.8, 0.6), sector("jev", "c2", 1, 0.2, 0.9), sector("jev", "c3", 1, 0.4, 0.0),
        sector("haiku", "c1", 1, 0.9, 0.1), sector("haiku", "c1", 2, 0.9, 0.1), sector("haiku", "c1", 3, 0.9, 0.7),
        sector("haiku", "c2", 1, 0.1, 0.8), sector("haiku", "c2", 2, 0.1, 0.8), sector("haiku", "c2", 3, 0.1, 0.8),
        sector("haiku", "c3", 1, 0.6, 0.1), sector("haiku", "c3", 2, 0.2, 0.1), sector("haiku", "c3", 3, 0.3, 0.1),
    ]  # fmt: skip

    # Act
    rows = {row.arm: row for row in score_sectors("coin_sector", records, labels, ["defi", "meme", "ai"], PRICES)}

    # Assert
    jev, haiku, majority = rows["jev"], rows["haiku"], rows["haiku (majority of 3)"]
    assert (jev.items, haiku.items, majority.items) == (3, 3, 3)
    # Jev: defi F1 1, meme F1 2/3 (c1 is a false yes). Haiku repeat 1: defi F1 2/3 (c3 is a false yes), meme F1 1.
    assert jev.macro_f1 == pytest.approx(5 / 6)
    assert haiku.macro_f1 == pytest.approx(5 / 6)
    # Majority: c1 defi, c2 meme, c3 nothing.
    assert majority.macro_f1 == pytest.approx(1.0)
    assert (jev.accuracy, majority.accuracy) == (None, None)
    # Mean of (p - label)² over 9 questions. Jev: 0.04 + 0.36 + 0.04 + 0.01 + 0.16. Haiku: 0.02 + 0.05 + 0.37.
    assert jev.brier == pytest.approx(0.61 / 9)
    assert haiku.brier == pytest.approx(0.44 / 9)
    # Only c2 has the same yes sectors in all 3 repeats.
    assert haiku.agreement == pytest.approx(1 / 3)


def perm(arm: str, item: str, stable: bool | None, spread: float = 0.0) -> Permutation:
    """A permutation record. `stable` None makes an error record. The price spread is twice the other spread."""
    base = Permutation(arm=arm, model=MODELS[arm], node="router", item_id=item, n_perm=6, seed=0, timestamp="t")
    if stable is None:
        return replace(base, error="HTTP 500")
    return replace(base, stable=stable, spread={"price": spread, "market_chart": spread / 2}, runs=[])


def test_order_scores_per_arm_and_self_hosted_cost_for_kev_only() -> None:
    # Arrange: Kev's q2 has an error record, then a success from a rerun. Only the success counts.
    rows = score_node("router", [rec("jev", "q1", 1, 0.8, 200), rec("kev", "q1", 1, 0.8, 70)], LABELS, PRICES)
    permutations = [
        perm("kev", "q1", True, 0.1), perm("kev", "q2", None), perm("kev", "q2", False, 0.4), perm("kev", "q3", True),
        perm("jev", "q1", False, 0.2), perm("jev", "q2", True, 0.0),
    ]  # fmt: skip

    # Act
    rows = with_self_hosted_cost(with_order_scores(rows, permutations), "kev", decisions_per_hour=36_000)

    # Assert
    kev, jev = {row.arm: row for row in rows}["kev"], {row.arm: row for row in rows}["jev"]
    assert (kev.order_stable, jev.order_stable) == (pytest.approx(2 / 3), 0.5)
    # The largest spread of each item. Kev: 0.1, 0.4 and 0. Jev: 0.2 and 0.
    assert (kev.order_spread, jev.order_spread) == (pytest.approx(0.5 / 3), pytest.approx(0.1))
    # $0.8048 per hour for 36,000 decisions: $0.8048 / 36 per 1k decisions. Jev keeps its price per token.
    assert kev.cost_per_1k == pytest.approx(0.8048 / 36)
    assert jev.cost_per_1k == pytest.approx(0.0168)


def test_arm_without_permutations_or_throughput_keeps_empty_scores() -> None:
    # Act
    rows = score_node("router", [rec("kev", "q1", 1, 0.8, 70)], LABELS, PRICES)
    rows = with_self_hosted_cost(with_order_scores(rows, []), "kev", None)

    # Assert
    assert [(row.order_stable, row.order_spread, row.cost_per_1k) for row in rows] == [(None, None, None)]


def test_router_reports_accuracy_without_advice_and_answers_at_each_confidence_level() -> None:
    # Arrange: q2 is an advice question. Jev selects price with these top probabilities: q1 0.95 (right), q2 0.75
    # (wrong), q3 0.5 (right).
    records = [rec("jev", "q1", 1, 0.95, 200), rec("jev", "q2", 1, 0.75, 200), rec("jev", "q3", 1, 0.5, 200)]

    # Act
    [jev] = score_node("router", records, LABELS, PRICES, advice_items=frozenset({"q2"}))

    # Assert
    assert jev.accuracy == pytest.approx(2 / 3)
    assert jev.accuracy_without_advice == 1.0
    assert (jev.answered_at_90, jev.correct_at_90) == (1, 1)
    assert (jev.answered_at_70, jev.correct_at_70) == (2, 1)
