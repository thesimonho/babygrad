"""Serialize the neutral autograd graph to the dashboard's JSON contract.

A pure structural dump: the frontend (Cytoscape) derives every coordinate from the
scope nesting, so no geometry is emitted here — only nodes, edges, and the scope
tree. This adds a wire format over the existing ``Graph`` and ``Scope`` structures;
it introduces no graph logic of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from babygrad.types import NodeKind, series_tag

if TYPE_CHECKING:
    from babygrad.types import Scope
    from babygrad.viz.graph import Edge, Graph, Node

# Roles the recorder tracks over time; their nodes carry a series_tag so the
# frontend can pull a node's history for a popup sparkline.
_RECORDED_KINDS = frozenset({NodeKind.PARAMETER, NodeKind.LAYER_OUTPUT})


def to_graph_json(graph: Graph, scopes: dict[str, Scope]) -> dict:
    """Emit the §04 graph payload: nodes and edges from the walked ``graph``, and
    the full ``scopes`` tree (empty containers included, since compound boxes need
    them). Scopes come in separately because pass-through containers own no graph
    nodes yet must still render as boxes.
    """
    return {
        "nodes": [_node_json(node_id, node) for node_id, node in graph.nodes.items()],
        "edges": [_edge_json(edge) for edge in graph.edges],
        "scopes": [_scope_json(scope) for scope in scopes.values()],
    }


def _node_json(node_id: int, node: Node) -> dict:
    """One node's contract fields. The id crosses as a string because it is an
    ``id(obj)`` pointer that would overflow a JS number; ``kind`` is the enum name
    so the frontend styles by role; ``scope`` is the owning scope's id or null;
    ``series_tag`` links a recorded node to its history, or null."""
    return {
        "id": str(node_id),
        "label": node.label,
        "kind": node.kind.name,
        "shape": list(node.shape) if node.shape is not None else None,
        "scope": node.scope.id if node.scope is not None else None,
        "series_tag": _series_tag(node),
    }


def _series_tag(node: Node) -> str | None:
    """The recorder tag for a tracked node, mirroring the recorder's own
    ``"<scope label>/<role>"`` convention: the node's label already carries the
    role (``weights``/``bias``/``result``) and its scope carries the label. Nodes
    the recorder does not track return None."""
    if node.kind not in _RECORDED_KINDS or node.scope is None:
        return None
    return series_tag(node.scope.label, node.label)


def _edge_json(edge: Edge) -> dict:
    """One edge, endpoints stringified to match the node ids they reference."""
    return {
        "source": str(edge.source),
        "target": str(edge.target),
        "label": edge.label,
    }


def _scope_json(scope: Scope) -> dict:
    """One scope. ``outer`` names the parent scope directly so the frontend nests
    compound nodes by pointer rather than by splitting the id path."""
    return {
        "id": scope.id,
        "label": scope.label,
        "outer": scope.outer_scope,
        "collapsed": scope.collapsed,
    }
