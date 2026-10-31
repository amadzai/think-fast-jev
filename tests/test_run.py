import json
from pathlib import Path

import httpx
import pytest

from bench.config import SystemOneArmConfig
from bench.nodes import ROUTER
from bench.queries import Query
from bench.run import run_node
from bench.systemone import SystemOneClient

JEV = SystemOneArmConfig(name="jev", base_url="https://jev.test", api_key="test-key", model="jev-latest")
KEV = SystemOneArmConfig(name="kev", base_url="http://kev.test:8009", api_key="test-key", model="kev-latest")
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
