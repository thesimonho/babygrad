"""Presentation for the graph renderer: graphviz attributes, node styling, cluster
colours. Kept apart from graph.py so changing a colour never touches traversal code.

The palette (background, per-role node colours, cluster colours) lives in
``static/theme.json`` so this renderer and the browser dashboard read one source
of colour truth. Only the graphviz-structural attributes (spacing, splines,
penwidth) stay here — they have no Cytoscape equivalent.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import graphviz

from babygrad.types import NodeKind

_THEME = json.loads((Path(__file__).parent / "static" / "theme.json").read_text())

GRAPH_ATTR = {
    "rankdir": "TB",
    "bgcolor": _THEME["background"],
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
    "color": _THEME["edge"]["color"],
    "arrowsize": "0.7",
    "penwidth": "1.2",
    "fontname": "Helvetica",
    "fontsize": "9",
    "fontcolor": _THEME["edge"]["fontColor"],
}


def _style(entry: dict) -> tuple[str, str, str]:
    """A theme node entry as the (shape, fill, font) tuple the renderer wants."""
    return (entry["shape"], entry["fill"], entry["font"])


# (shape, fill, font) per role, plus the fallback — read from the shared theme.
_DEFAULT_STYLE = _style(_THEME["nodes"]["_default"])
_NODE_STYLE: dict[NodeKind, tuple[str, str, str]] = {
    NodeKind[name]: _style(entry)
    for name, entry in _THEME["nodes"].items()
    if name != "_default"
}

# (fill, border) pairs — Picked per cluster by a stable hash of its scope path, so a given scope keeps its colour across renders and sibling boxes read as distinct.
_CLUSTER_PALETTE: list[tuple[str, str]] = [
    (cluster["fill"], cluster["border"]) for cluster in _THEME["clusters"]
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
