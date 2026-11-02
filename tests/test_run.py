import json
import threading
import time
from pathlib import Path

import httpx
import httpx2
import pytest

from bench.anthropic_client import AnthropicClient
from bench.coins import DESCRIPTION_MAX_CHARS, Coin
from bench.config import ArmConfig
from bench.nodes import COIN_SECTOR, ROUTER
from bench.permute import permute_node
from bench.queries import Query
from bench.run import KEEP_WARM_ERROR_PREFIX, KEEP_WARM_STATE, append_run_metadata, run_node
from bench.systemone import SystemOneClient
from bench.throughput import measure_throughput

JEV = ArmConfig(name="jev", kind="system_one", base_url="https://jev.test", api_key="test-key", model="jev-latest")
KEV = ArmConfig(
    name="kev",
    kind="system_one",
    base_url="http://kev.test:8009",
    api_key="test-key",
    model="kev-latest",
    keep_warm=True,
)
HAIKU = ArmConfig(
    name="haiku",
    kind="anthropic",
    base_url="https://anthropic.test",
    api_key="test-key",
    model="claude-haiku-5-5",
    repeats=3,
)
PROBABILITIES = {name: 0.0 for name in ROUTER.options} | {"price": 0.9, "market_chart": 0.1}
# The shape of a recorded hosted Jev response: no latency_ms.
RECORDED = {
    "model": "jev-1.14.0",
    "usage": {"input_tokens": 412, "output_tokens": 0},
    "answers": {"route": {"choice": "price", "probabilities": PROBABILITIES}},
}
# The shape of a recorded Kev response: it has latency_ms, and model is only the configured name.
KEV_RECORDED = {
    "model": "kev-latest",
    "answers": {"route": {"type": "choice", "choice": "price", "confidence": 0.8, "probabilities": PROBABILITIES}},
    "usage": {"input_tokens": 405, "output_tokens": 52},
    "latency_ms": 13.4,
}
# The shape of a recorded Anthropic Messages response with structured output. The probabilities sum to 1.1.
HAIKU_RECORDED = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {
            "type": "text",
            "text": json.dumps({name: 0.0 for name in ROUTER.options} | {"price": 0.66, "market_chart": 0.44}),
        }
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1015, "output_tokens": 146, "output_tokens_details": {"thinking_tokens": 0}},
}
HAIKU_MODEL = {
    "type": "model",
    "id": "claude-haiku-5-5",
    "display_name": "Claude Haiku 5.5",
    "created_at": "2026-09-01T00:00:00Z",
}
KEV_MODELS = {"models": [{"name": "kev-latest", "run": "jaredpalmer/kev-4b", "prefix_cache": {"size": 1, "hits": 7}}]}


QUERIES = [
    Query(id="q1", text="What is the price of ETH?", router="price", guardrail="allow", source="seed", reviewed=True),
    Query(id="q2", text="BTC price in EUR", router="price", guardrail="allow", source="seed", reviewed=True),
    Query(id="q4", text="ADA price now", router="price", guardrail="allow", source="seed", reviewed=True),
    Query(id="q5", text="Price of XRP", router="price", guardrail="allow", source="seed", reviewed=True),
]


def fake_system_one(recorded: dict, models: dict | None = None, seen: list | None = None) -> httpx.MockTransport:
    """Answer POST /v1/systemone with `recorded` and GET /v1/models with `models`. Keep each decision body in `seen`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=models or {"models": []})
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json=recorded)

    return httpx.MockTransport(handler)


def test_router_run_writes_one_record_per_query_and_rerun_skips_them(tmp_path: Path) -> None:
    # Arrange
    requests = []
    out = tmp_path / "router.jsonl"

    # Act
    for _ in range(2):
        with SystemOneClient(JEV, transport=fake_system_one(RECORDED, seen=requests)) as client:
            run_node(ROUTER, QUERIES, {"jev": client}, out)

    # Assert
    assert len(requests) == 4
    assert requests[0]["state"] == "What is the price of ETH?"
    assert set(requests[0]["questions"]["route"]["criteria"]) == set(ROUTER.options)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["item_id"] for r in records] == ["q1", "q2", "q4", "q5"]
    assert [r["warmup"] for r in records] == [True, True, True, False]
    first = records[0]
    assert first["arm"] == "jev"
    assert (first["model"], first["model_version"]) == ("jev-latest", "jev-1.14.0")
    assert first["node"] == "router"
    assert first["selected"] == "price"
    assert first["probabilities"] == PROBABILITIES
    assert first["model_ms"] is None
    assert first["e2e_ms"] >= 0
    assert (first["input_tokens"], first["output_tokens"]) == (412, 0)
    assert first["error"] is None


def test_retried_call_counts_every_attempt_toward_warmup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the first 2 calls fail, so q1 succeeds on the 3rd call and q2 is the 4th call.
    monkeypatch.setattr("bench.run.BACKOFF_S", 0.0)
    responses = iter([httpx.Response(500, text="busy"), httpx.Response(500, text="busy")])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": []})
        return next(responses, httpx.Response(200, json=RECORDED))

    out = tmp_path / "router.jsonl"

    # Act
    with SystemOneClient(JEV, transport=httpx.MockTransport(handler)) as client:
        run_node(ROUTER, QUERIES[:2], {"jev": client}, out)

    # Assert
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["item_id"], r["error"], r["warmup"]) for r in records] == [("q1", None, True), ("q2", None, False)]


def test_two_arms_alternate_order_and_kev_gets_keep_warm_call_model_time_and_server_settings(tmp_path: Path) -> None:
    # Arrange
    kev_requests: list = []
    out = tmp_path / "router.jsonl"

    # Act
    with (
        SystemOneClient(JEV, transport=fake_system_one(RECORDED)) as jev,
        SystemOneClient(KEV, transport=fake_system_one(KEV_RECORDED, models=KEV_MODELS, seen=kev_requests)) as kev,
    ):
        summary = run_node(ROUTER, QUERIES, {"jev": jev, "kev": kev}, out)

    # Assert
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["item_id"], r["arm"]) for r in records] == [
        ("q1", "jev"), ("q1", "kev"),
        ("q2", "kev"), ("q2", "jev"),
        ("q4", "jev"), ("q4", "kev"),
        ("q5", "kev"), ("q5", "jev"),
    ]  # fmt: skip
    kev_record = records[1]
    assert (kev_record["model_ms"], kev_record["output_tokens"]) == (13.4, 52)
    assert summary.arms["kev"].models_before == summary.arms["kev"].models_after == KEV_MODELS
    # Kev gets one unrecorded keep-warm call right before each recorded call.
    assert [r["state"] for r in kev_requests] == [
        KEEP_WARM_STATE, "What is the price of ETH?",
        KEEP_WARM_STATE, "BTC price in EUR",
        KEEP_WARM_STATE, "ADA price now",
        KEEP_WARM_STATE, "Price of XRP",
    ]  # fmt: skip
    assert [r["warmup"] for r in records if r["arm"] == "kev"] == [True, False, False, False]
    assert (summary.arms["kev"].keep_warm, summary.arms["jev"].keep_warm) == (True, False)


def test_failed_keep_warm_call_counts_once_toward_warmup_and_is_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: q1's first keep-warm call fails, so q1 uses 3 calls and is still warm-up. All keep-warm calls for q2 fail.
    monkeypatch.setattr("bench.run.BACKOFF_S", 0.0)
    keep_warm_statuses = iter([500, 200, 500, 500, 500])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=KEV_MODELS)
        if json.loads(request.content)["state"] == KEEP_WARM_STATE and next(keep_warm_statuses) == 500:
            return httpx.Response(500, text="busy")
        return httpx.Response(200, json=KEV_RECORDED)

    out = tmp_path / "router.jsonl"

    # Act
    with SystemOneClient(KEV, transport=httpx.MockTransport(handler)) as kev:
        run_node(ROUTER, QUERIES[:2], {"kev": kev}, out)

    # Assert
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["item_id"], r["warmup"]) for r in records] == [("q1", True), ("q2", False)]
    assert records[0]["error"] is None
    assert records[1]["error"].startswith(KEEP_WARM_ERROR_PREFIX)


def fake_anthropic(seen: list, recorded: dict = HAIKU_RECORDED) -> httpx2.MockTransport:
    """Answer POST /v1/messages with `recorded` and GET /v1/models/<id> with HAIKU_MODEL."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return httpx2.Response(200, json=HAIKU_MODEL)
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=recorded, headers={"server-timing": "x-originResponse;dur=1250"})

    return httpx2.MockTransport(handler)


def test_haiku_runs_three_repeats_per_query_with_normalised_probabilities(tmp_path: Path) -> None:
    # Arrange
    requests: list = []
    out = tmp_path / "router.jsonl"

    # Act: the second run must skip all 3 Haiku repeats of each query.
    for _ in range(2):
        with (
            SystemOneClient(JEV, transport=fake_system_one(RECORDED)) as jev,
            AnthropicClient(HAIKU, transport=fake_anthropic(requests)) as haiku,
        ):
            summary = run_node(ROUTER, QUERIES[:2], {"jev": jev, "haiku": haiku}, out)
        append_run_metadata(out, summary, Path("queries.jsonl"))

    # Assert
    assert len(requests) == 6
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["item_id"], r["arm"], r["repeat"]) for r in records] == [
        ("q1", "jev", 1), ("q1", "haiku", 1), ("q1", "haiku", 2), ("q1", "haiku", 3),
        ("q2", "haiku", 1), ("q2", "haiku", 2), ("q2", "haiku", 3), ("q2", "jev", 1),
    ]  # fmt: skip
    haiku_record = records[1]
    assert haiku_record["selected"] == "price"
    assert haiku_record["probabilities"]["price"] == pytest.approx(0.6)
    assert haiku_record["probabilities"]["market_chart"] == pytest.approx(0.4)
    assert haiku_record["normalised"] is True
    assert (haiku_record["model_ms"], haiku_record["input_tokens"], haiku_record["output_tokens"]) == (
        1250.0,
        1015,
        146,
    )
    assert haiku_record["e2e_ms"] >= 0
    assert records[0]["normalised"] is None
    body = requests[0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"]["effort"] == "low"
    assert set(body["output_config"]["format"]["schema"]["required"]) == set(ROUTER.options)
    assert body["messages"] == [{"role": "user", "content": "What is the price of ETH?"}]
    assert "temperature" not in body
    haiku_meta = json.loads(out.with_suffix(".meta.json").read_text())["runs"][0]["arms"]["haiku"]
    assert haiku_meta["request_settings"]["thinking"] == {"type": "disabled"}
    assert haiku_meta["models_before"]["created_at"] == "2026-09-01T00:00:00Z"


COIN = Coin(
    id="ethereum",
    name="Ethereum",
    symbol="eth",
    description='Ethereum is a <a href="https://example.test">smart contract platform</a>.\r\n' + "More text. " * 200,
    categories=("Smart Contract Platform", "Layer 1 (L1)"),
)
# The probabilities do not sum to 1, because each sector is its own yes/no question.
SECTOR_PROBABILITIES = {name: 0.1 for name in COIN_SECTOR.options} | {"layer_1": 0.9, "smart_contract_platform": 0.5}
# The shape of a System One response with noul questions: one answer per sector, `noul` is the probability of yes.
JEV_SECTORS = {
    "model": "jev-1.14.0",
    "usage": {"input_tokens": 900, "output_tokens": 0},
    "answers": {name: {"type": "noul", "noul": p} for name, p in SECTOR_PROBABILITIES.items()},
}
HAIKU_SECTORS = HAIKU_RECORDED | {"content": [{"type": "text", "text": json.dumps(SECTOR_PROBABILITIES)}]}


def test_coin_sector_run_asks_every_sector_in_one_call_and_keeps_probabilities_unscaled(tmp_path: Path) -> None:
    # Arrange
    jev_requests: list = []
    haiku_requests: list = []
    out = tmp_path / "coin_sector.jsonl"

    # Act
    with (
        SystemOneClient(JEV, transport=fake_system_one(JEV_SECTORS, seen=jev_requests)) as jev,
        AnthropicClient(HAIKU, transport=fake_anthropic(haiku_requests, HAIKU_SECTORS)) as haiku,
    ):
        run_node(COIN_SECTOR, [COIN], {"jev": jev, "haiku": haiku}, out)

    # Assert
    assert len(jev_requests) == 1
    assert len(haiku_requests) == 3
    questions = jev_requests[0]["questions"]
    assert list(questions) == list(COIN_SECTOR.options)
    assert {q["type"] for q in questions.values()} == {"noul"}
    state = jev_requests[0]["state"]
    assert state.startswith("Name: Ethereum\nSymbol: ETH\nDescription: Ethereum is a smart contract platform. More")
    assert "<a" not in state
    assert len(state.split("Description: ")[1]) <= DESCRIPTION_MAX_CHARS
    assert haiku_requests[0]["messages"] == [{"role": "user", "content": state}]
    assert haiku_requests[0]["output_config"]["format"]["schema"]["required"] == list(COIN_SECTOR.options)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["arm"], r["repeat"]) for r in records] == [("jev", 1), ("haiku", 1), ("haiku", 2), ("haiku", 3)]
    for record in records:
        assert record["probabilities"] == pytest.approx(SECTOR_PROBABILITIES)
        # Yes at 0.5 for every arm, in the node's option order.
        assert record["selected"] == ["layer_1", "smart_contract_platform"]
    assert records[1]["normalised"] is None


def test_permute_pass_asks_jev_under_shuffled_option_orders_and_rerun_skips_finished_items(tmp_path: Path) -> None:
    # Arrange: Jev selects price when price is the first option, and market_chart for every other order.
    orders: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        order = list(body["questions"]["route"]["criteria"])
        orders.append((body["state"], order))
        price = 0.9 if order[0] == "price" else 0.3
        answer = {name: 0.0 for name in ROUTER.options} | {"price": price, "market_chart": round(1 - price, 10)}
        return httpx.Response(200, json=RECORDED | {"answers": {"route": {"probabilities": answer}}})

    out = tmp_path / "router.permute.jsonl"

    # Act
    for _ in range(2):
        with SystemOneClient(JEV, transport=httpx.MockTransport(handler)) as jev:
            permute_node(ROUTER, QUERIES[:2], jev, out, n_perm=3)

    # Assert: 3 orders per item, and the rerun sends nothing.
    assert [state for state, _ in orders] == ["What is the price of ETH?"] * 3 + ["BTC price in EUR"] * 3
    first, *shuffled = [order for _, order in orders[:3]]
    assert first == list(ROUTER.options)
    assert all(sorted(order) == sorted(ROUTER.options) and order != first for order in shuffled)
    # The same seed gives the same orders for every item and every arm.
    assert [order for _, order in orders[3:]] == [order for _, order in orders[:3]]
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["arm"], r["node"], r["item_id"]) for r in records] == [("jev", "router", "q1"), ("jev", "router", "q2")]
    assert records[0]["stable"] is False
    assert records[0]["spread"]["price"] == pytest.approx(0.6)
    assert [run["choice"] for run in records[0]["runs"]] == ["price", "market_chart", "market_chart"]
    assert records[0]["error"] is None


def test_throughput_burst_sends_concurrent_calls_after_a_warmup_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr("bench.throughput.CAPTURE_PAUSE_S", 0.0)
    states: list = []
    in_flight = {"now": 0, "max": 0}
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=KEV_MODELS)
        with lock:
            states.append(json.loads(request.content)["state"])
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
        time.sleep(0.05)
        with lock:
            in_flight["now"] -= 1
        return httpx.Response(200, json=KEV_RECORDED)

    # Act
    with SystemOneClient(KEV, transport=httpx.MockTransport(handler)) as kev:
        result = measure_throughput(ROUTER, [q.text for q in QUERIES], kev, rounds=2, concurrency=4)

    # Assert: a warm-up burst of 8 calls, then the measured burst of 8 calls: each state once per round.
    assert len(states) == 16
    assert sorted(states[8:]) == sorted([q.text for q in QUERIES] * 2)
    assert in_flight["max"] == 4
    assert (result.arm, result.node, result.requests, result.concurrency) == ("kev", "router", 8, 4)
    assert result.models_before == result.models_after == KEV_MODELS
    # 8 calls of 50 ms, 4 at a time, take 0.1 s or more.
    assert result.seconds >= 0.1
