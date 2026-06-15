"""Post-hoc rendering of recorded training history."""

from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from babygrad.types import History, Step, Tag

_RIDGE_BINS = 50
_RIDGE_ROW_HEIGHT = 1.6


class PlotVisualizer:
    def __init__(self, history: History):
        self.history = history

    def plot_scalar(self, tags: list[Tag], save_path: str | None = None):
        """Line-plots a scalar series (loss, accuracy, ...) over steps."""
        assert isinstance(tags, list), "plot_scalar() takes a list of tags"

        fig, ax = plt.subplots()

        for tag in tags:
            series = self.history[tag]
            steps = sorted(series)
            values = [series[step] for step in steps]
            ax.plot(steps, values, label=tag)

        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.legend()
        _show_or_save(fig, save_path)

    def plot_ridge(
        self,
        tag: Tag,
        save_path: str | None = None,
        clip_quantiles: tuple[float, float] | None = None,
    ):
        """Ridgeline of a distribution series: one row per recorded step.

        All rows share one set of bin edges so their shapes are comparable
        across steps; each row is peak-normalised and offset upward by its
        row index, earliest step at the bottom.

        For heavy-tailed data (gradients), pass clip_quantiles such as
        (0.01, 0.99) to span the bins over that quantile range instead of
        min/max; outliers beyond it pile into the edge bins.
        """
        series = self.history[tag]
        steps = sorted(series)
        rows = cast("list[list[float]]", [series[step] for step in steps])

        edges = _shared_bin_edges(rows, _RIDGE_BINS, clip_quantiles)
        centers = [(left + right) / 2 for left, right in zip(edges, edges[1:])]

        fig, ax = plt.subplots()
        for row_index, values in enumerate(rows):
            counts = _bin_counts(values, edges)
            peak = max(counts)
            tops = [row_index + count / peak * _RIDGE_ROW_HEIGHT for count in counts]
            # lower rows draw in front so their peaks overlap rows above
            depth = len(rows) - row_index
            ax.fill_between(
                centers, tops, row_index, color="C0", alpha=0.7, zorder=depth
            )
            ax.plot(centers, tops, color="C0", linewidth=0.8, zorder=depth)

        _label_ridge_axes(ax, tag, steps)
        _show_or_save(fig, save_path)


def _shared_bin_edges(
    rows: list[list[float]],
    bin_count: int,
    clip_quantiles: tuple[float, float] | None = None,
) -> list[float]:
    """Bin edges spanning every row, so all ridge rows share one x-axis."""
    if clip_quantiles is None:
        lo = min(min(row) for row in rows)
        hi = max(max(row) for row in rows)
    else:
        pooled = sorted(value for row in rows for value in row)
        lo = _quantile(pooled, clip_quantiles[0])
        hi = _quantile(pooled, clip_quantiles[1])
    if hi == lo:
        # all values identical: widen so the single spike still has a bin
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / bin_count
    return [lo + index * width for index in range(bin_count + 1)]


def _quantile(ordered: list[float], quantile: float) -> float:
    """Value at the given quantile of an already-sorted list."""
    position = min(int(quantile * len(ordered)), len(ordered) - 1)
    return ordered[position]


def _bin_counts(values: list[float], edges: list[float]) -> list[float]:
    """Histogram counts of values against the given bin edges."""
    counts = [0.0] * (len(edges) - 1)
    lo = edges[0]
    width = (edges[-1] - lo) / len(counts)
    for value in values:
        # clamp both ends: with a clipped range, outliers land in the
        # edge bins instead of wrapping to a negative index
        index = max(0, min(int((value - lo) / width), len(counts) - 1))
        counts[index] += 1.0
    return counts


def _label_ridge_axes(ax: Axes, tag: Tag, steps: list[Step]) -> None:
    """Y ticks show step numbers, thinned so long runs stay readable."""
    stride = max(1, len(steps) // 10)
    positions = list(range(0, len(steps), stride))
    ax.set_yticks(positions, [str(steps[position]) for position in positions])
    ax.set_title(tag)
    ax.set_xlabel("value")
    ax.set_ylabel("step")


def _show_or_save(fig: Figure, save_path: str | None) -> None:
    """Show interactively when no path is given, otherwise save and close."""
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")
