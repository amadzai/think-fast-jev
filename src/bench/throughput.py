"""This module measures Kev's decisions per hour in a burst of concurrent calls (`bench throughput`).

The Kev server puts concurrent requests in one batch. Sequential calls make self-hosted Kev look more expensive than
it is. The self-hosted cost per 1k decisions is a GPU hourly price divided by this throughput.
"""

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from bench.nodes import Node
from bench.systemone import SystemOneClient

SECONDS_PER_HOUR = 3600
# The Kev server captures CUDA graphs for new batch sizes when it is idle. This pause after the warm-up burst lets it
# finish. It is shorter than the 8 s idle that slows Kev down.
CAPTURE_PAUSE_S = 2.0


@dataclass(frozen=True)
class Throughput:
    arm: str
    model: str
    node: str
    started: str
    concurrency: int
    # The number of calls in the measured burst. Each state is called once per round.
    requests: int
    seconds: float
    decisions_per_hour: float
    # The server's model description before and after the measured burst. For Kev, the batch counters show the
    # mean batch size, and the prefix cache counters show the cache hits.
    models_before: dict[str, Any]
    models_after: dict[str, Any]

    @property
    def mean_batch(self) -> float | None:
        """Return the mean number of requests in one server batch during the burst. None without Kev's counters."""
        try:
            start, end = self.models_before["models"][0]["batches"], self.models_after["models"][0]["batches"]
            return (end["requests"] - start["requests"]) / (end["count"] - start["count"])
        except (KeyError, IndexError, TypeError, ZeroDivisionError):
            return None


def measure_throughput(
    node: Node, states: Sequence[str], client: SystemOneClient, rounds: int, concurrency: int
) -> Throughput:
    """Send a warm-up burst, then the measured burst, `concurrency` calls at a time.

    A failed call raises, because a burst with errors does not give a valid throughput.
    """
    # Every state comes once per round, after all other states. With more states than concurrent calls, one batch
    # never has the same state twice. A second copy of a state could get a prefix cache hit.
    calls = [state for _ in range(rounds) for state in states]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:

        def burst() -> None:
            list(pool.map(lambda state: client.answer(node, state), calls))

        burst()
        time.sleep(CAPTURE_PAUSE_S)
        started = datetime.now(UTC).isoformat()
        models_before = client.models()
        start = time.perf_counter()
        burst()
        seconds = time.perf_counter() - start
    return Throughput(
        arm=client.config.name,
        model=client.config.model,
        node=node.name,
        started=started,
        concurrency=concurrency,
        requests=len(calls),
        seconds=seconds,
        decisions_per_hour=len(calls) / seconds * SECONDS_PER_HOUR,
        models_before=models_before,
        models_after=client.models(),
    )
