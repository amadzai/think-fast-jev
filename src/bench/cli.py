"""This module defines the `bench` command line."""

from pathlib import Path
from typing import Annotated

import httpx
import typer
from dotenv import find_dotenv, load_dotenv

from bench.config import ConfigError, jev_config
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
    """Call Jev once and print its answer and latency."""
    try:
        config = jev_config()
    except ConfigError as error:
        raise fail(str(error))

    try:
        with SystemOneClient(config) as client:
            result = client.decide(CHECK_STATE, CHECK_QUESTION)
    except httpx.HTTPStatusError as error:
        raise fail(f"{config.name}: {describe_error(error)}")
    except httpx.HTTPError as error:
        raise fail(f"{config.name}: cannot reach {config.base_url}: {error}")
    except SystemOneError as error:
        raise fail(f"{config.name}: {error}")

    answer = result.answers["direction"]
    model_ms = f"{result.model_ms:.1f} ms" if result.model_ms is not None else "n/a"
    typer.echo(
        f"{config.name} ({result.model}): {answer.get('choice')} {answer.get('probabilities')}  "
        f"model {model_ms}, end-to-end {result.e2e_ms:.1f} ms"
    )


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
        config = jev_config()
        items = load_queries(queries, definition.options)
    except (ConfigError, QueryFileError, OSError) as error:
        raise fail(str(error))

    try:
        with SystemOneClient(config) as client:
            summary = run_node(definition, items, {config.name: client}, out)
        append_run_metadata(out, summary, queries)
    except (ResultsFileError, OSError) as error:
        raise fail(str(error))
    typer.echo(f"{out}: {summary.written} written ({summary.errors} errors), {summary.skipped} skipped")
