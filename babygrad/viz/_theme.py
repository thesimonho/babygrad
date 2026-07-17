"""Presentation for the graph renderer: graphviz attributes, node styling, cluster
colours. Kept apart from graph.py so changing a colour never touches traversal code.
"""

from __future__ import annotations

import zlib

import graphviz

from babygrad.types import NodeKind

GRAPH_ATTR = {
    "rankdir": "TB",
    "bgcolor": "white",
    "splines": "spline",
    "nodesep": "0.35",
    "ranksep": "0.6",
}
NODE_ATTR = {
    "fontname": "Helvetica",
    "penwidth": "0",
    "margin": "0.18,0.07",
}
EDGE_ATTR = {
    "color": "#9aa3ab",
    "arrowsize": "0.7",
    "penwidth": "1.2",
    "fontname": "Helvetica",
    "fontsize": "9",
    "fontcolor": "#6c757d",
}

# (shape, fill, font) per role
_DEFAULT_STYLE = ("box", "#edf2f4", "#495057")
_NODE_STYLE: dict[NodeKind, tuple[str, str, str]] = {
    NodeKind.INPUT: ("box", "#e9c46a", "#3d3d3d"),
    NodeKind.TARGET: ("box", "#e9c46a", "#3d3d3d"),
    NodeKind.PARAMETER: ("box", "#a8dadc", "#1d3557"),
    NodeKind.OP: ("ellipse", "#e63946", "white"),
    NodeKind.OP_RESULT: ("box", "#edf2f4", "#495057"),
    NodeKind.LAYER_OUTPUT: ("box", "#457b9d", "white"),
    NodeKind.LOSS: ("box", "#1d3557", "white"),
    NodeKind.CONSTANT: ("box", "#8a6bb5", "white"),
}

# (fill, border) pairs — Picked per cluster by a stable hash of its scope path, so a given scope keeps its colour across renders and sibling boxes read as distinct.
_CLUSTER_PALETTE: list[tuple[str, str]] = [
    ("#eef4fb", "#c3d9ee"),  # blue
    ("#eef7f0", "#c6e4cd"),  # green
    ("#fbf4e9", "#ecd9b8"),  # amber
    ("#f5eefb", "#dcc6ee"),  # purple
    ("#eafaf7", "#bfe6dd"),  # teal
    ("#fbeef2", "#eec6d2"),  # rose
    ("#f2f3f5", "#dadde1"),  # slate
    ("#f6f8ea", "#dde6bb"),  # lime
]


def node_attrs(kind: NodeKind, scope: str | None) -> dict[str, str]:
    """Graphviz attributes for one node, by the role it plays."""
    shape, fill, font = _NODE_STYLE.get(kind, _DEFAULT_STYLE)
    attrs = {
        "shape": shape,
        "style": "filled,rounded" if shape == "box" else "filled",
        "fillcolor": fill,
        "fontcolor": font,
    }
    if scope:  # align a layer's nodes into a column
        attrs["group"] = scope
    return attrs


def style_cluster(cluster: graphviz.Digraph, label: str, path_key: str) -> None:
    """Colour a scope's box, keyed by a stable hash of its path."""
    fill, border = _CLUSTER_PALETTE[
        zlib.crc32(path_key.encode()) % len(_CLUSTER_PALETTE)
    ]
    cluster.attr(
        label=label,
        labelloc="t",
        style="rounded,filled",
        color=border,
        fillcolor=fill,
        fontname="Helvetica",
        fontcolor="#495057",
        fontsize="12",
        penwidth="1.4",
        margin="14",
    )
