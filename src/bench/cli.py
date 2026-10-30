"""This module defines the `bench` command line."""

import httpx
import typer
from dotenv import find_dotenv, load_dotenv

from bench.config import ConfigError, jev_config
from bench.systemone import SystemOneClient, SystemOneError

app = typer.Typer(no_args_is_help=True, help="Benchmark single-pass decision models against a frontier LLM.")

ERROR_BODY_CHARS = 300
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
        raise fail(f"{config.name}: HTTP {error.response.status_code}: {error.response.text[:ERROR_BODY_CHARS]}")
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
