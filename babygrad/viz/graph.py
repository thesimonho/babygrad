"""
Render an autograd graph: extract a neutral bipartite structure, then draw it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import graphviz

from babygrad.ops import Op
from babygrad.tensor import Tensor
from babygrad.types import NodeKind, Shape


@dataclass
class Node:
    kind: NodeKind
    label: str
    shape: Shape | None
    scope: str | None


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    label: str | None = field(default=None, compare=False)


@dataclass
class Graph:
    nodes: dict[int, Node]  # keyed by id(obj)
    edges: set[Edge]


_GRAPH_ATTR = {
    "rankdir": "TB",
    "bgcolor": "white",
    "splines": "spline",
    "nodesep": "0.35",
    "ranksep": "0.6",
}
_NODE_ATTR = {
    "fontname": "Helvetica",
    "penwidth": "0",
    "margin": "0.18,0.07",
}
_EDGE_ATTR = {
    "color": "#9aa3ab",
    "arrowsize": "0.7",
    "penwidth": "1.2",
    "fontname": "Helvetica",
    "fontsize": "9",
    "fontcolor": "#6c757d",
}

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
_DEFAULT_STYLE = ("box", "#edf2f4", "#495057")


def _format_shape(shape: Shape) -> str:
    return "×".join(str(dim) for dim in shape)


class GraphVisualizer:
    """Walk a tensor's autograd graph into a neutral bipartite structure, then draw it.

    walk(root) -> Graph(nodes, edges) -> draw_*(): computation (flat),
    combined (clustered by layer scope), architecture (collapsed).
    """

    def __init__(self, root: Tensor):
        self.root = root
        self.graph = Graph(nodes=dict(), edges=set())
        self._walk(self.root, set(), self.graph)

    def draw_computation(self, save_path: str | None = None) -> graphviz.Digraph:
        """The full bipartite graph, drawn flat (no clustering)."""
        dot = self._new_digraph("computation")
        for node_id, node in self.graph.nodes.items():
            self._add_node(dot, node_id, node)
        self._add_edges(dot, self.graph)
        return self._render(dot, save_path)

    def draw_combined(self, save_path: str | None = None) -> graphviz.Digraph:
        """The full graph with each layer scope wrapped in a cluster box."""
        dot = self._new_digraph("combined")

        by_scope: dict[str | None, list[tuple[int, Node]]] = defaultdict(list)
        for node_id, node in self.graph.nodes.items():
            by_scope[node.scope].append((node_id, node))

        for scope, members in by_scope.items():
            if scope is None:
                for node_id, node in members:
                    self._add_node(dot, node_id, node)
            else:
                cluster = graphviz.Digraph(name=f"cluster_{scope}")
                self._style_cluster(cluster, scope)
                for node_id, node in members:
                    self._add_node(cluster, node_id, node)
                dot.subgraph(cluster)

        self._add_edges(dot, self.graph)
        return self._render(dot, save_path)

    def draw_architecture(self, save_path: str | None = None) -> graphviz.Digraph:
        """Each layer scope collapsed to one box, edges labelled with the
        shape of the tensor crossing between layers."""
        scopes = {n.scope for n in self.graph.nodes.values() if n.scope is not None}
        collapsed = self._collapse(scopes)

        dot = self._new_digraph("architecture")
        for node_id, node in collapsed.nodes.items():
            self._add_node(dot, node_id, node)
        self._add_edges(dot, collapsed)
        return self._render(dot, save_path)

    def _walk(self, item: Tensor | Op, visited: set[int], graph: Graph):
        """
        Walk the bipartite graph, storing nodes and edges as we go.
        """
        if id(item) in visited:
            return

        assert item.kind is not None

        visited.add(id(item))
        graph.nodes[id(item)] = Node(
            kind=item.kind,
            label=self._display(item),
            shape=item.shape if isinstance(item, Tensor) else None,
            scope=item.scope,
        )

        if isinstance(item, Tensor) and item.producer is not None:
            graph.edges.add(Edge(source=id(item.producer), target=id(item)))
            self._walk(item.producer, visited, graph)
        elif isinstance(item, Op):
            for input in item.inputs:
                graph.edges.add(Edge(source=id(input), target=id(item)))
                self._walk(input, visited, graph)

    def _display(self, item: Tensor | Op) -> str:
        """
        Return a display string for the node.
        """
        # assert item.kind is not None

        if isinstance(item, Op):
            return item.label

        if item.kind in (NodeKind.CONSTANT, NodeKind.PARAMETER, NodeKind.LAYER_OUTPUT):
            assert item.name is not None
            return item.name

        if item.kind is None:
            return ""
        return item.kind.name.lower()

    def _new_digraph(self, name: str) -> graphviz.Digraph:
        return graphviz.Digraph(
            name,
            graph_attr=_GRAPH_ATTR,
            node_attr=_NODE_ATTR,
            edge_attr=_EDGE_ATTR,
        )

    def _add_node(self, target: graphviz.Digraph, node_id: int, node: Node) -> None:
        shape, fill, font = _NODE_STYLE.get(node.kind, _DEFAULT_STYLE)
        style = "filled,rounded" if shape == "box" else "filled"
        label = node.label
        if node.shape is not None:
            label = f"{node.label}\n({_format_shape(node.shape)})"
        attrs = {"shape": shape, "style": style, "fillcolor": fill, "fontcolor": font}
        if node.scope:  # align a layer's nodes into a column
            attrs["group"] = node.scope
        target.node(str(node_id), label, **attrs)

    def _add_edges(self, dot: graphviz.Digraph, graph: Graph) -> None:
        for edge in graph.edges:
            into_op = graph.nodes[edge.target].kind == NodeKind.OP
            dot.edge(
                str(edge.source),
                str(edge.target),
                label=edge.label or "",
                style="dashed" if into_op else "solid",
            )

    def _collapse(self, scopes: set[str]) -> Graph:
        """Fold every node in a collapsed scope into one box node, rerouting
        edges across the boundary and dropping edges internal to a scope.

        Each surviving edge keeps the crossing tensor's shape as its label: in
        the bipartite graph a cross-scope edge always runs tensor -> op, so the
        source's shape is the data flowing between layers.
        """
        rep: dict[int, int] = {}
        nodes: dict[int, Node] = {}
        box_ids: dict[str, int] = {}

        for node_id, node in self.graph.nodes.items():
            scope = node.scope
            if scope is not None and scope in scopes:
                if scope not in box_ids:
                    box_ids[scope] = -len(box_ids) - 1  # negative: never an id()
                    nodes[box_ids[scope]] = Node(
                        kind=NodeKind.LAYER_OUTPUT, label=scope, shape=None, scope=None
                    )
                rep[node_id] = box_ids[scope]
            else:
                rep[node_id] = node_id
                nodes[node_id] = node

        edges: set[Edge] = set()
        for edge in self.graph.edges:
            source, target = rep[edge.source], rep[edge.target]
            if source == target:
                continue  # edge internal to a collapsed scope
            crossing = self.graph.nodes[edge.source].shape
            label = _format_shape(crossing) if crossing is not None else None
            edges.add(Edge(source, target, label=label))

        return Graph(nodes=nodes, edges=edges)

    def _style_cluster(self, cluster: graphviz.Digraph, scope: str) -> None:
        cluster.attr(
            label=scope,
            labelloc="t",
            style="rounded,filled",
            color="#dee2e6",
            fillcolor="#f6f7f9",
            fontname="Helvetica",
            fontcolor="#6c757d",
            fontsize="12",
            penwidth="1.4",
            margin="14",
        )

    def _render(self, dot: graphviz.Digraph, save_path: str | None) -> graphviz.Digraph:
        """Save to save_path when given. Always return the dot so notebooks
        render it inline (Jupyter shows a Digraph via its rich SVG repr)."""
        if save_path is not None:
            dot.render(outfile=save_path, format="svg", cleanup=True)
        return dot
