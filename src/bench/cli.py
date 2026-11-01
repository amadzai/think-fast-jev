"""This module defines the `bench` command line."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from dotenv import find_dotenv, load_dotenv

from bench.anthropic_client import AnthropicClient
from bench.arm import CALL_ERRORS, UNREACHABLE_ERRORS, ArmClient, argmax, describe_error
from bench.coins import COINS_PATH, CoinFileError, load_coins
from bench.config import (
    KEV_ARM,
    PRICES,
    ArmConfig,
    ConfigError,
    arm_configs,
    coingecko_config,
    generator_config,
    kev_config,
)
from bench.fetch import MARKETS_PER_PAGE_MAX, FetchError, fetch_coins
from bench.generate import append_generation_metadata, append_queries, generate_variations, load_generations
from bench.nodes import COIN_SECTOR, NODES, SECTORS, Node
from bench.permute import MAX_PERMUTATIONS, PERMUTATIONS, Permutation, load_permutations, permutation_path, permute_node
from bench.queries import QueryFileError, load_queries
from bench.run import (
    Item,
    ResultsFileError,
    append_metadata,
    append_run_metadata,
    load_records,
    read_metadata,
    run_node,
)
from bench.score import Score, score_node, score_sectors, with_kev_scores, write_scores
from bench.systemone import MAX_CONNECTIONS, SystemOneClient
from bench.throughput import measure_throughput

app = typer.Typer(no_args_is_help=True, help="Benchmark single-pass decision models against a frontier LLM.")

CHECK_STATE = "Bitcoin price: up 5% in the last 24 hours."
# The number of categories that `bench fetch` prints, to help pick the sectors.
TOP_CATEGORIES = 30
# `bench throughput`: each item 3 times, 32 calls at a time.
THROUGHPUT_ROUNDS = 3
THROUGHPUT_CONCURRENCY = 32
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


def _default_results(node: str) -> Path:
    return Path("results") / f"{node}.jsonl"


@app.command()
def run(
    node: Annotated[str, typer.Option(help=f"The decision node: {', '.join(NODES)}.")],
    queries: Annotated[Path, typer.Option(help="The query file (JSONL).")] = Path("data/queries.jsonl"),
    coins: Annotated[Path, typer.Option(help="The coin file of `bench fetch` (JSONL).")] = COINS_PATH,
    out: Annotated[Path | None, typer.Option(help="The results file (JSONL). Default: results/<node>.jsonl.")] = None,
) -> None:
    """Run the arms on one decision node. A rerun with the same output skips finished decisions."""
    if node not in NODES:
        raise fail(f"unknown node {node!r}. Use one of: {', '.join(NODES)}.")
    definition = NODES[node]
    out = out or _default_results(node)
    try:
        configs = arm_configs()
        items, source, generators = _items(definition, queries, coins)
    except (ConfigError, QueryFileError, CoinFileError, OSError) as error:
        raise fail(str(error))

    try:
        with ExitStack() as stack:
            clients = {c.name: stack.enter_context(_client(c)) for c in configs}
            _require_reachable(clients)
            summary = run_node(definition, items, clients, out)
        append_run_metadata(out, summary, source, generators)
    except (ResultsFileError, OSError) as error:
        raise fail(str(error))
    typer.echo(f"{out}: {summary.written} written ({summary.errors} errors), {summary.skipped} skipped")


def _items(node: Node, queries: Path, coins: Path) -> tuple[list[Item], Path, list[str]]:
    """Return the items, the file that they come from, and the models that generated queries in it."""
    if node is COIN_SECTOR:
        loaded = load_coins(coins)
        described = [coin for coin in loaded if coin.has_description]
        typer.echo(f"coins: {len(described)}, {len(loaded) - len(described)} without a description (skipped)")
        return list(described), coins, []
    all_queries = load_queries(queries)
    generators = sorted({g["model"] for g in load_generations(queries)})
    reviewed = [q for q in all_queries if q.reviewed]
    typer.echo(f"queries: {len(reviewed)} reviewed, {len(all_queries) - len(reviewed)} not reviewed (skipped)")
    return list(reviewed), queries, generators


def _require_reachable(clients: dict[str, ArmClient]) -> None:
    # Without this check, an unreachable arm costs about 18 s of retries per query before its error record.
    for client in clients.values():
        try:
            client.models()
        except CALL_ERRORS as error:
            raise fail(_arm_error(client.config, error))
    typer.echo(f"arms: {', '.join(clients)}")


def _require_kev_config() -> ArmConfig:
    try:
        config = kev_config()
    except ConfigError as error:
        raise fail(str(error))
    if config is None:
        raise fail("KEV_BASE_URL is not set. This command needs the Kev server.")
    return config


@app.command()
def permute(
    node: Annotated[str, typer.Option(help="The decision node: router or guardrail.")],
    queries: Annotated[Path, typer.Option(help="The query file (JSONL).")] = Path("data/queries.jsonl"),
    results: Annotated[
        Path | None,
        typer.Option(help="The results file of `bench run`. The records go to <results>.permute.jsonl next to it."),
    ] = None,
    n_perm: Annotated[
        int, typer.Option(min=2, max=MAX_PERMUTATIONS, help="The number of option orders.")
    ] = PERMUTATIONS,
) -> None:
    """Ask Kev each reviewed query under several option orders. A rerun skips finished queries."""
    choice_nodes = [name for name, definition in NODES.items() if not definition.multi_label]
    if node not in choice_nodes:
        raise fail(f"node {node!r} has no choice question. Use one of: {', '.join(choice_nodes)}.")
    definition = NODES[node]
    out = permutation_path(results or _default_results(node))
    config = _require_kev_config()
    try:
        items, _, _ = _items(definition, queries, COINS_PATH)
    except (QueryFileError, OSError) as error:
        raise fail(str(error))
    try:
        with SystemOneClient(config) as client:
            _require_reachable({config.name: client})
            written, errors, skipped = permute_node(definition, items, client, out, n_perm)
    except (ResultsFileError, OSError) as error:
        raise fail(str(error))
    typer.echo(f"{out}: {written} written ({errors} errors), {skipped} skipped")


@app.command()
def throughput(
    node: Annotated[str, typer.Option(help=f"The decision node: {', '.join(NODES)}.")],
    queries: Annotated[Path, typer.Option(help="The query file (JSONL).")] = Path("data/queries.jsonl"),
    coins: Annotated[Path, typer.Option(help="The coin file of `bench fetch` (JSONL).")] = COINS_PATH,
    results: Annotated[
        Path | None,
        typer.Option(help="The results file of `bench run`. The throughput goes to <results>.meta.json."),
    ] = None,
    rounds: Annotated[int, typer.Option(min=1, help="The number of calls for each item.")] = THROUGHPUT_ROUNDS,
    concurrency: Annotated[
        int, typer.Option(min=1, max=MAX_CONNECTIONS, help="The number of calls at the same time.")
    ] = THROUGHPUT_CONCURRENCY,
) -> None:
    """Measure Kev's decisions per hour in a burst of concurrent calls, after a warm-up burst.

    `bench score` divides the GPU hourly price by this number to get Kev's self-hosted cost.
    """
    if node not in NODES:
        raise fail(f"unknown node {node!r}. Use one of: {', '.join(NODES)}.")
    definition = NODES[node]
    out = results or _default_results(node)
    config = _require_kev_config()
    try:
        items, _, _ = _items(definition, queries, coins)
    except (QueryFileError, CoinFileError, OSError) as error:
        raise fail(str(error))
    if len(items) < concurrency:
        raise fail(f"{len(items)} items, fewer than --concurrency {concurrency}. One state would be in a batch twice.")
    try:
        with SystemOneClient(config) as client:
            _require_reachable({config.name: client})
            result = measure_throughput(definition, [item.text for item in items], client, rounds, concurrency)
    except CALL_ERRORS as error:
        raise fail(f"{_arm_error(config, error)}. The throughput is not saved.")
    try:
        append_metadata(out, "throughput", asdict(result))
    except (ResultsFileError, OSError) as error:
        raise fail(str(error))
    batch = f"{result.mean_batch:.1f}" if result.mean_batch is not None else "n/a"
    typer.echo(
        f"{config.name} {node}: {result.requests} calls, {concurrency} at a time, in {result.seconds:.2f} s "
        f"(mean batch {batch}): {result.decisions_per_hour:,.0f} decisions per hour. Saved in {out.with_suffix('.meta.json')}"
    )


@app.command()
def generate(
    seed_ids: Annotated[
        list[str] | None,
        typer.Option("--seed", help="A seed query ID. Repeat the option for more seeds. Default: all seeds."),
    ] = None,
    per_seed: Annotated[int, typer.Option(min=1, help="The number of variations for each seed.")] = 2,
    queries: Annotated[Path, typer.Option(help="The query file (JSONL).")] = Path("data/queries.jsonl"),
) -> None:
    """Write query variations from the seeds and add them to the query file, not reviewed.

    A second run on the same seeds adds more variations. It does not replace the first ones.
    """
    try:
        config = generator_config()
        existing = load_queries(queries)
        # Read before the paid calls, so that a damaged metadata file stops the command early.
        previous = load_generations(queries)
    except (ConfigError, QueryFileError, OSError) as error:
        raise fail(str(error))
    # Generated labels start from the seed labels, so only reviewed seeds count.
    seeds = [q for q in existing if q.source == "seed" and q.reviewed and (not seed_ids or q.id in seed_ids)]
    unknown = sorted(set(seed_ids or []) - {q.id for q in seeds})
    if unknown:
        raise fail(f"no reviewed seed query with ID {unknown[0]!r} in {queries}.")
    typer.echo(f"{config.model}: {len(seeds)} seeds, {per_seed} variations each")
    result = generate_variations(config, seeds, existing, per_seed)
    try:
        append_queries(queries, result.queries)
        append_generation_metadata(queries, previous, config, seeds, result)
    except OSError as error:
        raise fail(str(error))
    unsupported = sum(q.router == "unsupported" for q in result.queries)
    typer.echo(
        f"{queries}: {len(result.queries)} added (not reviewed, {unsupported} unsupported), {result.dropped} dropped"
    )
    if result.error:
        raise fail(f"stopped at seed {result.error}")


@app.command()
def fetch(
    limit: Annotated[
        int, typer.Option(min=1, max=MARKETS_PER_PAGE_MAX, help="The number of top coins by market cap.")
    ] = 200,
) -> None:
    """Fetch name, symbol, description and categories of the top coins from the CoinGecko Demo API.

    The data goes to data/coingecko/ (gitignored). A rerun continues after an error. At the end, the command prints
    the number of coins per sector and the most common categories. Use them to pick the sectors.
    """
    try:
        config = coingecko_config()
        done = {coin.id for coin in load_coins(COINS_PATH)} if COINS_PATH.exists() else set()
    except (ConfigError, CoinFileError, OSError) as error:
        raise fail(str(error))
    added, error = 0, None
    try:
        for coin in fetch_coins(config, limit, COINS_PATH, done):
            added += 1
            typer.echo(f"{added}: {coin.id}")
    except (httpx.HTTPError, FetchError, OSError) as exc:
        error = describe_error(exc)
    coins = load_coins(COINS_PATH) if COINS_PATH.exists() else []
    empty = sum(not coin.has_description for coin in coins)
    typer.echo(f"{COINS_PATH}: {len(coins)} coins ({added} added, {empty} without a description)")
    sectors = Counter(name for coin in coins for name in coin.sectors)
    typer.echo("coins per sector:")
    for name in SECTORS:
        typer.echo(f"  {sectors[name]:4}  {name}")
    typer.echo("most common categories:")
    for category, count in Counter(c for coin in coins for c in coin.categories).most_common(TOP_CATEGORIES):
        typer.echo(f"  {count:4}  {category}")
    if error:
        raise fail(f"stopped after {added} coins: {error}. Run the command again to continue.")


@app.command()
def score(
    results: Annotated[
        list[Path] | None,
        typer.Argument(help="Results files (JSONL). Default: the results/<node>.jsonl files that exist."),
    ] = None,
    queries: Annotated[Path, typer.Option(help="The query file with the labels (JSONL).")] = Path("data/queries.jsonl"),
    coins: Annotated[Path, typer.Option(help="The coin file with the coin sector labels (JSONL).")] = COINS_PATH,
    out: Annotated[Path, typer.Option(help="Writes <out>.csv and <out>.md.")] = Path("results/scores"),
) -> None:
    """Score the results files against the labels and write the metrics table."""
    paths = results or [path for path in map(_default_results, NODES) if path.exists()]
    if not paths:
        raise fail("no results files in results/. Pass the results file of `bench run --out` as an argument.")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise fail(f"{missing[0]}: file not found. Pass the results file of `bench run --out` as an argument.")
    try:
        items = load_queries(queries)
        records = [record for path in paths for record in load_records(path)]
    except (QueryFileError, ResultsFileError, OSError) as error:
        raise fail(str(error))
    labels = {"router": {q.id: q.router for q in items}, "guardrail": {q.id: q.guardrail for q in items}}
    nodes = list(dict.fromkeys(r.node for r in records))
    if not nodes:
        raise fail("no records to score.")
    unknown = [node for node in nodes if node not in labels and node != COIN_SECTOR.name]
    if unknown:
        raise fail(f"no labels for node {unknown[0]!r}.")
    try:
        sector_labels = {c.id: c.sectors for c in load_coins(coins)} if COIN_SECTOR.name in nodes else {}
    except (CoinFileError, OSError) as error:
        raise fail(str(error))
    # The labels and the Brier score use the current sector list. Records from an older list do not match it.
    sectors = set(COIN_SECTOR.options)
    if any(r.node == COIN_SECTOR.name and r.probabilities and set(r.probabilities) != sectors for r in records):
        raise fail("coin sector records have other sectors than nodes.py. Delete the results file and run again.")

    permutations, throughputs = _kev_data(paths)

    def rows_of(node: str) -> list[Score]:
        if node == COIN_SECTOR.name:
            rows = score_sectors(node, records, sector_labels, list(COIN_SECTOR.options), PRICES)
        else:
            rows = score_node(node, records, labels[node], PRICES)
        has_kev = any(r.node == node and r.arm == KEV_ARM for r in records)
        return with_kev_scores(rows, KEV_ARM, permutations, _decisions_per_hour(node, throughputs, has_kev))

    rows = [row for node in nodes for row in rows_of(node)]
    try:
        typer.echo(write_scores(rows, out), nl=False)
    except OSError as error:
        raise fail(str(error))
    typer.echo(f"{out}.csv, {out}.md: {len(rows)} rows")


def _kev_data(paths: list[Path]) -> tuple[list[Permutation], list[dict[str, Any]]]:
    """Return Kev's permutation records and throughput entries that belong to the results files."""
    try:
        permutations = [p for path in paths for p in load_permutations(permutation_path(path))]
        throughputs = [t for path in paths for t in read_metadata(path, "throughput") if t.get("arm") == KEV_ARM]
    except (ResultsFileError, OSError) as error:
        raise fail(str(error))
    return permutations, throughputs


def _decisions_per_hour(node: str, throughputs: list[dict[str, Any]], has_kev: bool) -> float | None:
    """Return the throughput of the last `bench throughput` on the node, or None when there is none."""
    entries = [t for t in throughputs if t.get("node") == node]
    if not entries:
        if has_kev:
            typer.echo(
                f"{KEV_ARM} {node}: no throughput, so no self-hosted cost. Run `bench throughput --node {node}`."
            )
        return None
    last = entries[-1]
    value = last.get("decisions_per_hour")
    if not isinstance(value, int | float) or value <= 0:
        raise fail(f"{node}: the throughput in the run metadata is not valid: {value!r}")
    typer.echo(
        f"{KEV_ARM} {node}: self-hosted cost from {value:,.0f} decisions per hour "
        f"({last.get('concurrency')} at a time, {last.get('started')})"
    )
    return value
