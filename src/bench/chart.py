"""This module draws the README charts from the metrics table of `bench score` (`scores.csv`)."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from bench.nodes import NODES


class ArmStyle(NamedTuple):
    label: str
    color: str


# Arm name in scores.csv to its style. The 3 colors pass the colorblind checks for all pairs on the light surface. The
# green has less than 3:1 contrast on the surface, so each dot has its value as a text label, and the README has the
# table.
ARMS = {
    "jev": ArmStyle("Jev (hosted)", "#2a78d6"),
    "kev": ArmStyle("Kev-4B (self-hosted)", "#eb6834"),
    "haiku": ArmStyle("Claude Haiku 5.5 (incumbent)", "#1baf7a"),
}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DPI = 200
MARKER_SIZE = 9
# The surface-colored ring around each dot.
MARKER_RING = 2
LINE_WIDTH = 2
FONT_SIZE = 10
TITLE_SIZE = 13
FOOTER_SIZE = 9
LINE_SPACING = 1.5
POINTS_PER_INCH = 72
# Space above the plots for the title and the legend.
HEADER_INCHES = 0.8
# Latency axis ticks: 50 ms, 100 ms, 200 ms, 500 ms, 1 s and so on.
LATENCY_TICKS = (1.0, 2.0, 5.0)
# The quality axis ends a little above 1, so that a dot at 1 is not cut.
QUALITY_TOP = 1.02
QUALITY_MARGIN = 0.05
FIGURE_WIDTH = 7.2
# The height of one node panel, and of one row of the cost chart.
PANEL_HEIGHT = 2.6
COST_ROW_HEIGHT = 0.45
# The height of the title, legend and footer around the plots.
LATENCY_EXTRA_HEIGHT = 1.2
COST_EXTRA_HEIGHT = 2.2
# Text offsets from a dot, in points: the value above a latency dot, and to the right of a cost dot.
ABOVE_DOT = (0, 9)
RIGHT_OF_DOT = (8, 0)
# The cost axis gets this much more room on the right, for the value labels.
COST_RIGHT_ROOM = 3
LATENCY_FILE = "quality_latency.png"
COST_FILE = "cost.png"


class ChartError(Exception):
    """The scores file does not have the values that the charts need."""


@dataclass(frozen=True)
class Point:
    node: str
    arm: str
    quality: float | None
    e2e_p50: float | None
    e2e_p95: float | None
    cost_per_1k: float | None


def load_points(path: Path) -> list[Point]:
    """Return the rows of the arms in `ARMS`. Other rows, such as the Haiku majority vote, are not in the charts."""
    if not path.exists():
        raise ChartError(f"{path}: file not found. Run `bench score` first.")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    points = [_point(row, number) for number, row in enumerate(rows, start=2) if row.get("arm") in ARMS]
    if not points:
        raise ChartError(f"{path}: no rows for the arms {', '.join(ARMS)}.")
    return points


def draw_charts(points: list[Point], out: Path, note: str) -> list[Path]:
    """Save the two charts in `out` and return their paths. `note` goes in the footer, for example the run date."""
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / LATENCY_FILE, out / COST_FILE]
    _quality_latency(points, note).savefig(paths[0], dpi=DPI, facecolor=SURFACE)
    _cost(points, note).savefig(paths[1], dpi=DPI, facecolor=SURFACE)
    return paths


def _point(row: dict[str, str], number: int) -> Point:
    def value(name: str) -> float | None:
        # `bench score` writes None as an empty cell.
        text = row.get(name, "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError as error:
            raise ChartError(f"line {number}: {name} is not a number: {text!r}") from error

    if row.get("node") not in NODES:
        raise ChartError(f"line {number}: unknown node {row.get('node')!r}")
    # Router and guardrail have accuracy. Coin sector has macro F1.
    quality = value("accuracy")
    return Point(
        node=row["node"],
        arm=row["arm"],
        quality=quality if quality is not None else value("macro_f1"),
        e2e_p50=value("e2e_p50"),
        e2e_p95=value("e2e_p95"),
        cost_per_1k=value("cost_per_1k"),
    )


def _nodes(points: list[Point]) -> list[str]:
    # The order of nodes.py, not the order of the file.
    present = {p.node for p in points}
    return [name for name in NODES if name in present]


def _node_title(name: str) -> str:
    node = NODES[name]
    kind = "yes/no questions" if node.multi_label else "options"
    return f"{name.replace('_', ' ').capitalize()} ({len(node.options)} {kind})"


def _figure(width: float, height: float, title: str, subtitle: str, note: str, arms: list[str]) -> Figure:
    """Return a figure with the title and the legend at the top and the footer at the bottom."""
    figure = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    top = 1 - HEADER_INCHES / height
    figure.text(0.02, 0.99, title, va="top", fontsize=TITLE_SIZE, color=TEXT_PRIMARY, fontweight="bold")
    figure.text(0.98, 0.99, "Powered by CoinGecko", ha="right", va="top", fontsize=FONT_SIZE, color=TEXT_SECONDARY)
    _legend(figure, arms, top)
    footer = f"{subtitle}\n{note}".strip()
    figure.text(0.02, 0.01, footer, fontsize=FOOTER_SIZE, color=TEXT_SECONDARY, va="bottom")
    # Keep the plots above the footer. The footer height is its line count times the line height.
    bottom = (footer.count("\n") + 1) * FOOTER_SIZE * LINE_SPACING / POINTS_PER_INCH / height + 0.02
    figure.get_layout_engine().set(rect=(0, bottom, 1, top - bottom))  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
    return figure


def _style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_SECONDARY, labelsize=FONT_SIZE)


def _legend(figure: Figure, arms: list[str], top: float) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="", color=ARMS[arm].color, markersize=MARKER_SIZE, label=ARMS[arm].label)
        for arm in arms
    ]
    figure.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.01, top), ncols=len(handles), frameon=False,
        fontsize=FONT_SIZE, labelcolor=TEXT_PRIMARY, handletextpad=0.2, borderaxespad=0,
    )  # fmt: skip


def _arms(points: list[Point]) -> list[str]:
    present = {p.arm for p in points}
    return [arm for arm in ARMS if arm in present]


def _milliseconds(value: float, _position: object) -> str:
    return f"{value / 1000:g} s" if value >= 1000 else f"{value:g} ms"


def _quality_latency(points: list[Point], note: str) -> Figure:
    nodes = _nodes(points)
    figure = _figure(
        FIGURE_WIDTH,
        PANEL_HEIGHT * len(nodes) + LATENCY_EXTRA_HEIGHT,
        "Quality against latency per decision",
        "Dot: median end-to-end latency. Line: to the 95th percentile. Latency axis is logarithmic.\n"
        "Quality: accuracy (router, guardrail), macro F1 over the sectors (coin sector).",
        note,
        _arms(points),
    )
    # One quality scale for all panels, so that a gap of 0.05 looks the same in every panel.
    lowest = min((p.quality for p in points if p.quality is not None), default=0.0)
    panels = figure.subplots(len(nodes), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    for axes, node in zip(panels, nodes, strict=True):
        _style(axes)
        axes.set_xscale("log")
        axes.xaxis.set_major_locator(LogLocator(subs=LATENCY_TICKS))
        axes.xaxis.set_major_formatter(FuncFormatter(_milliseconds))
        axes.xaxis.set_minor_formatter(NullFormatter())
        axes.set_title(_node_title(node), loc="left", fontsize=FONT_SIZE + 1, color=TEXT_PRIMARY)
        for p in (p for p in points if p.node == node):
            if p.quality is None or p.e2e_p50 is None:
                continue
            if p.e2e_p95 is not None:
                axes.plot([p.e2e_p50, p.e2e_p95], [p.quality, p.quality], color=ARMS[p.arm].color, linewidth=LINE_WIDTH)
            _dot(axes, p.e2e_p50, p.quality, p.arm, f"{p.quality:.2f}", ABOVE_DOT, ha="center", va="bottom")
    panels[0].set_ylim(max(0.0, lowest - QUALITY_MARGIN), QUALITY_TOP)
    # No tick above 1: no quality is above 1.
    panels[0].set_yticks([t for t in panels[0].get_yticks() if lowest - QUALITY_MARGIN <= t <= 1])
    panels[-1].set_xlabel("End-to-end latency per decision", color=TEXT_SECONDARY, fontsize=FONT_SIZE)
    return figure


def _cost(points: list[Point], note: str) -> Figure:
    nodes = _nodes(points)
    arms = _arms(points)
    figure = _figure(
        FIGURE_WIDTH,
        COST_ROW_HEIGHT * len(nodes) * len(arms) + COST_EXTRA_HEIGHT,
        "Cost per 1,000 decisions (USD)",
        "Cost axis is logarithmic. Jev and Haiku: list price per token. Kev-4B: AWS g6.xlarge (1 x L4) on-demand price\n"
        "per hour, divided by the measured decisions per hour.",
        note,
        arms,
    )
    axes = figure.subplots()
    _style(axes)
    axes.set_xscale("log")
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    axes.grid(False, axis="y")
    labels: list[str] = []
    for row, (node, arm) in enumerate((node, arm) for node in nodes for arm in arms):
        # The node name is on the middle row of its group.
        labels.append(_node_title(node).split(" (")[0] if arm == arms[len(arms) // 2] else "")
        if arm == arms[0] and row:
            axes.axhline(row - 0.5, color=AXIS, linewidth=1)
        p = next((p for p in points if p.node == node and p.arm == arm), None)
        if p is None or p.cost_per_1k is None:
            axes.text(
                0.01, row, f"{ARMS[arm].label}: no data", transform=axes.get_yaxis_transform(), va="center",
                fontsize=FONT_SIZE - 1, color=TEXT_MUTED,
            )  # fmt: skip
            continue
        _dot(axes, p.cost_per_1k, row, arm, f"${p.cost_per_1k:.3g}", RIGHT_OF_DOT, ha="left", va="center")
    axes.set_yticks(range(len(labels)), labels)
    axes.tick_params(axis="y", length=0, labelcolor=TEXT_PRIMARY)
    axes.invert_yaxis()
    low, high = axes.get_xlim()
    axes.set_xlim(low, high * COST_RIGHT_ROOM)
    return figure


def _dot(axes: Axes, x: float, y: float, arm: str, text: str, offset: tuple[int, int], ha: str, va: str) -> None:
    """Draw the arm's dot with a surface ring, and its value as a text label at `offset` points from the dot."""
    axes.plot(
        x, y, marker="o", color=ARMS[arm].color, markersize=MARKER_SIZE, markeredgecolor=SURFACE,
        markeredgewidth=MARKER_RING, linestyle="",
    )  # fmt: skip
    axes.annotate(
        text, (x, y), xytext=offset, textcoords="offset points", fontsize=FONT_SIZE, color=TEXT_PRIMARY, ha=ha, va=va
    )
