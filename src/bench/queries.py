"""This module reads the router and guardrail query file (JSONL, one query per line)."""

import json
from dataclasses import dataclass, fields
from pathlib import Path


class QueryFileError(Exception):
    """A line in the query file is not a valid query."""


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    router: str
    guardrail: str
    source: str
    reviewed: bool


def load_queries(path: Path, router_options: dict[str, str]) -> list[Query]:
    queries = []
    names = {field.name for field in fields(Query)}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            data = json.loads(line)
        except ValueError as error:
            raise QueryFileError(f"{where}: not valid JSON: {error}") from error
        if not isinstance(data, dict) or set(data) != names:
            raise QueryFileError(f"{where}: a query must have exactly the fields {sorted(names)}")
        query = Query(**data)
        if not isinstance(query.reviewed, bool):
            raise QueryFileError(f"{where}: reviewed must be true or false")
        if query.router not in router_options:
            raise QueryFileError(f"{where}: unknown router label {query.router!r}")
        queries.append(query)
    ids = [query.id for query in queries]
    if len(ids) != len(set(ids)):
        raise QueryFileError(f"{path}: query IDs are not unique")
    return queries
