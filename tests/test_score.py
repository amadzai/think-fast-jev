import pytest

from bench.config import Price
from bench.run import Decision
from bench.score import score_node

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
