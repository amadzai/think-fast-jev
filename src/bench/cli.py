"""This module defines the `bench` command line."""

from contextlib import ExitStack
from pathlib import Path
from typing import Annotated

import httpx
import typer
from dotenv import find_dotenv, load_dotenv

from bench.config import ConfigError, SystemOneArmConfig, arm_configs
from bench.nodes import NODES
from bench.queries import QueryFileError, load_queries
from bench.run import ResultsFileError, append_run_metadata, run_node
from bench.systemone import SystemOneClient, SystemOneError, describe_error

app = typer.Typer(no_args_is_help=True, help="Benchmark single-pass decision models against a frontier LLM.")

CHECK_STATE = "Bitcoin price: up 5% in the last 24 hours."
CHECK_QUESTION = {
    "direction": {
        "type": "choice",
        "instructions": "Did the price go up or down?",
        "criteria": {"up": "The price went up", "down": "The price went down"},
    }
}


def fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


@app.callback()
def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))


@app.command()
def check() -> None:
    """Call each configured arm once and print its answer and latency."""
    try:
        configs = arm_configs()
    except ConfigError as error:
        raise fail(str(error))
    # A list, not all(), so that every arm is called even after one fails.
    results = [_check_arm(config) for config in configs]
    if not all(results):
        raise typer.Exit(1)


def _arm_error(config: SystemOneArmConfig, error: httpx.HTTPError | SystemOneError) -> str:
    if isinstance(error, httpx.HTTPStatusError | SystemOneError):
        return f"{config.name}: {describe_error(error)}"
    return f"{config.name}: cannot reach {config.base_url}: {describe_error(error)}. {config.unreachable_hint}".strip()


def _check_arm(config: SystemOneArmConfig) -> bool:
    try:
        with SystemOneClient(config) as client:
            result = client.decide(CHECK_STATE, CHECK_QUESTION)
    except (httpx.HTTPError, SystemOneError) as error:
        typer.echo(f"error: {_arm_error(config, error)}", err=True)
        return False

    answer = result.answers["direction"]
    model_ms = f"{result.model_ms:.1f} ms" if result.model_ms is not None else "n/a"
    typer.echo(
        f"{config.name} ({result.model}): {answer.get('choice')} {answer.get('probabilities')}  "
        f"model {model_ms}, end-to-end {result.e2e_ms:.1f} ms"
    )
    return True


@app.command()
def run(
    node: Annotated[str, typer.Option(help=f"The decision node: {', '.join(NODES)}.")],
    queries: Annotated[Path, typer.Option(help="The query file (JSONL).")] = Path("data/queries.jsonl"),
    out: Annotated[Path | None, typer.Option(help="The results file (JSONL). Default: results/<node>.jsonl.")] = None,
) -> None:
    """Run the arms on one decision node. A rerun with the same output skips finished decisions."""
    if node not in NODES:
        raise fail(f"unknown node {node!r}. Use one of: {', '.join(NODES)}.")
    definition = NODES[node]
    out = out or Path("results") / f"{node}.jsonl"
    try:
        configs = arm_configs()
        items = load_queries(queries, definition.options)
    except (ConfigError, QueryFileError, OSError) as error:
        raise fail(str(error))

    try:
        with ExitStack() as stack:
            clients = {c.name: stack.enter_context(SystemOneClient(c)) for c in configs}
            _require_reachable(clients)
            summary = run_node(definition, items, clients, out)
        append_run_metadata(out, summary, queries)
    except (ResultsFileError, OSError) as error:
        raise fail(str(error))
    typer.echo(f"{out}: {summary.written} written ({summary.errors} errors), {summary.skipped} skipped")


def _require_reachable(clients: dict[str, SystemOneClient]) -> None:
    # Without this check, an unreachable arm costs about 18 s of retries per query before its error record.
    for client in clients.values():
        try:
            client.models()
        except (httpx.HTTPError, SystemOneError) as error:
            raise fail(_arm_error(client.config, error))
    typer.echo(f"arms: {', '.join(clients)}")
