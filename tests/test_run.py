import json
from pathlib import Path

import httpx

from bench.config import SystemOneArmConfig
from bench.nodes import ROUTER
from bench.queries import Query
from bench.run import run_node
from bench.systemone import SystemOneClient

JEV = SystemOneArmConfig(name="jev", base_url="https://jev.test", api_key="test-key", model="jev-latest")
PROBABILITIES = {name: 0.0 for name in ROUTER.options} | {"price": 0.9, "market_chart": 0.1}
# The shape of a recorded hosted Jev response: no latency_ms.
RECORDED = {
    "model": "jev-1.14.0",
    "usage": {"input_tokens": 412, "output_tokens": 0},
    "answers": {"route": {"choice": "price", "probabilities": PROBABILITIES}},
}
QUERIES = [
    Query(id="q1", text="What is the price of ETH?", router="price", guardrail="allow", source="seed", reviewed=True),
    Query(id="q2", text="BTC price in EUR", router="price", guardrail="allow", source="seed", reviewed=True),
    Query(id="q3", text="SOL price?", router="price", guardrail="allow", source="generated", reviewed=False),
    Query(id="q4", text="ADA price now", router="price", guardrail="allow", source="seed", reviewed=True),
    Query(id="q5", text="Price of XRP", router="price", guardrail="allow", source="seed", reviewed=True),
]


def test_router_run_writes_one_record_per_reviewed_query_and_rerun_skips_them(tmp_path: Path) -> None:
    # Arrange
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=RECORDED)

    out = tmp_path / "router.jsonl"

    # Act
    for _ in range(2):
        with SystemOneClient(JEV, transport=httpx.MockTransport(handler)) as client:
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
