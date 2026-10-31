import json
from pathlib import Path

import httpx
import httpx2
import pytest

from bench.anthropic_client import AnthropicClient
from bench.config import ArmConfig
from bench.nodes import ROUTER
from bench.queries import Query
from bench.run import append_run_metadata, run_node
from bench.systemone import SystemOneClient

JEV = ArmConfig(name="jev", kind="system_one", base_url="https://jev.test", api_key="test-key", model="jev-latest")
KEV = ArmConfig(name="kev", kind="system_one", base_url="http://kev.test:8009", api_key="test-key", model="kev-latest")
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
    Query(id="q3", text="SOL price?", router="price", guardrail="allow", source="generated", reviewed=False),
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


def test_router_run_writes_one_record_per_reviewed_query_and_rerun_skips_them(tmp_path: Path) -> None:
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


def test_two_arms_alternate_order_per_query_and_kev_records_model_time_and_server_settings(tmp_path: Path) -> None:
    # Arrange
    out = tmp_path / "router.jsonl"

    # Act
    with (
        SystemOneClient(JEV, transport=fake_system_one(RECORDED)) as jev,
        SystemOneClient(KEV, transport=fake_system_one(KEV_RECORDED, models=KEV_MODELS)) as kev,
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


def fake_anthropic(seen: list) -> httpx2.MockTransport:
    """Answer POST /v1/messages with HAIKU_RECORDED and GET /v1/models/<id> with HAIKU_MODEL."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return httpx2.Response(200, json=HAIKU_MODEL)
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=HAIKU_RECORDED, headers={"server-timing": "x-originResponse;dur=1250"})

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
