"""Summarise each tensor node's current values for the graph popup.

Walks the autograd graph from a root tensor and, for every tensor it reaches,
records a small ``{mean, min, max, n}`` summary of its data and gradient. Ops
carry no values, so only tensors get an entry. Keyed by ``str(id(tensor))`` to
match the node ids in the graph JSON, so the frontend can look a node up on tap.

This is a point-in-time snapshot — the values a node holds right now. During a
live run it is re-read each epoch; on a finished run it is the final state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from babygrad.ops import Op
from babygrad.tensor import Tensor

if TYPE_CHECKING:
    from babygrad.types import Number


def to_node_stats(root: Tensor) -> dict[str, dict]:
    """Map every tensor reachable from ``root`` to its current value/grad summary."""
    stats: dict[str, dict] = {}
    stack: list[Tensor | Op] = [root]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Tensor):
            stats[str(id(node))] = {
                "value": _summary(node.data),
                "grad": _summary(node.grad),
            }
            if node.producer is not None:
                stack.append(node.producer)
        elif isinstance(node, Op):
            stack.extend(node.inputs)
    return stats


def _summary(values: list[Number]) -> dict | None:
    """Mean/min/max/count of a flat value list, or None when empty."""
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }
