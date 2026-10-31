"""This module defines the `bench` command line."""

from contextlib import ExitStack
from pathlib import Path
from typing import Annotated

import typer
from dotenv import find_dotenv, load_dotenv

from bench.anthropic_client import AnthropicClient
from bench.arm import CALL_ERRORS, UNREACHABLE_ERRORS, ArmClient, argmax, describe_error
from bench.config import ArmConfig, ConfigError, arm_configs
from bench.nodes import NODES, Node
from bench.queries import QueryFileError, load_queries
from bench.run import ResultsFileError, append_run_metadata, run_node
from bench.systemone import SystemOneClient

app = typer.Typer(no_args_is_help=True, help="Benchmark single-pass decision models against a frontier LLM.")

CHECK_STATE = "Bitcoin price: up 5% in the last 24 hours."
CHECK_NODE = Node(
    name="check",
    question_id="direction",
    instructions="Did the price go up or down?",
    options={"up": "The price went up", "down": "The price went down"},
)


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


def _client(config: ArmConfig) -> ArmClient:
    if config.kind == "anthropic":
        return AnthropicClient(config)
    return SystemOneClient(config)


def _arm_error(config: ArmConfig, error: Exception) -> str:
    if isinstance(error, UNREACHABLE_ERRORS):
        return (
            f"{config.name}: cannot reach {config.base_url}: {describe_error(error)}. {config.unreachable_hint}".strip()
        )
    return f"{config.name}: {describe_error(error)}"


def _check_arm(config: ArmConfig) -> bool:
    try:
        with _client(config) as client:
            answer = client.answer(CHECK_NODE, CHECK_STATE)
    except CALL_ERRORS as error:
        typer.echo(f"error: {_arm_error(config, error)}", err=True)
        return False

    model_ms = f"{answer.model_ms:.1f} ms" if answer.model_ms is not None else "n/a"
    typer.echo(
        f"{config.name} ({answer.model_version}): {argmax(answer.probabilities)} {answer.probabilities}  "
        f"model {model_ms}, end-to-end {answer.e2e_ms:.1f} ms"
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
            clients = {c.name: stack.enter_context(_client(c)) for c in configs}
            _require_reachable(clients)
            summary = run_node(definition, items, clients, out)
        append_run_metadata(out, summary, queries)
    except (ResultsFileError, OSError) as error:
        raise fail(str(error))
    typer.echo(f"{out}: {summary.written} written ({summary.errors} errors), {summary.skipped} skipped")


def _require_reachable(clients: dict[str, ArmClient]) -> None:
    # Without this check, an unreachable arm costs about 18 s of retries per query before its error record.
    for client in clients.values():
        try:
            client.models()
        except CALL_ERRORS as error:
            raise fail(_arm_error(client.config, error))
    typer.echo(f"arms: {', '.join(clients)}")
