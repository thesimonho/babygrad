"""
Render an autograd graph: walk it into a neutral bipartite structure, colour each
node by the scope a tracer attributed to it, then draw it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import graphviz

from babygrad.ops import Op
from babygrad.tensor import Tensor
from babygrad.types import NodeKind, Shape
from babygrad.viz import _theme

if TYPE_CHECKING:
    from babygrad.types import Scope
    from babygrad.viz.attribution import TraceResult


@dataclass
class Node:
    kind: NodeKind
    label: str
    shape: Shape | None
    scope: Scope | None


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    label: str | None = field(default=None, compare=False)


@dataclass
class Graph:
    nodes: dict[int, Node]  # keyed by id(obj)
    edges: set[Edge]


def _format_shape(shape: Shape) -> str:
    return "×".join(str(dim) for dim in shape)


class GraphVisualizer:
    """Walk a tensor's autograd graph into a neutral bipartite structure, then draw it.

    Scope comes from a tracer's ``TraceResult`` — the node->scope map colours each
    node, and the scope tree nests the cluster boxes. draw_*(): computation (flat),
    combined (clustered by scope), architecture (every scope collapsed to one box).
    """

    def __init__(self, root: Tensor, trace: TraceResult):
        self.root = root
        self.trace = trace
        self.graph = Graph(nodes=dict(), edges=set())
        self._walk(self.root, set(), self.graph)

    def draw_computation(self, save_path: str | None = None) -> graphviz.Digraph:
        """The full bipartite graph, drawn flat (no clustering)."""
        return self._draw_flat("computation", self.graph, save_path)

    def draw_combined(self, save_path: str | None = None) -> graphviz.Digraph:
        """The full graph with each scope wrapped in a cluster box, nested to mirror
        the module hierarchy. Scopes flagged collapsed are drawn as one box, hiding
        their internals but keeping their place in the hierarchy."""
        collapse_ids = {s.id for s in self.trace.scopes.values() if s.collapsed}
        graph = self.graph
        if collapse_ids:
            graph = self._collapse(graph, collapse_ids, descendants=True, nest=True)

        dot = self._new_digraph("combined")
        self._add_scope_clusters(dot, graph, collapse_ids)
        self._add_edges(dot, graph)
        return self._render(dot, save_path)

    def draw_architecture(self, save_path: str | None = None) -> graphviz.Digraph:
        """Every scope collapsed to one box, edges labelled with the shape of the
        tensor crossing between scopes."""
        scoped = {
            node.scope.id for node in self.graph.nodes.values() if node.scope is not None
        }
        return self._draw_flat(
            "architecture", self._collapse(self.graph, scoped, nest=False), save_path
        )

    def _draw_flat(
        self, name: str, graph: Graph, save_path: str | None
    ) -> graphviz.Digraph:
        """Render every node at the top level, no clustering. Shared by the
        computation and architecture views, which differ only in their graph."""
        dot = self._new_digraph(name)
        for node_id, node in graph.nodes.items():
            self._add_node(dot, node_id, node)
        self._add_edges(dot, graph)
        return self._render(dot, save_path)

    def _walk(self, item: Tensor | Op, visited: set[int], graph: Graph):
        """Walk the bipartite graph, storing nodes and edges as we go. Each node's
        scope is looked up from the tracer's attribution by object identity."""
        if id(item) in visited:
            return

        visited.add(id(item))
        graph.nodes[id(item)] = Node(
            kind=item.kind,
            label=self._display(item),
            shape=item.shape if isinstance(item, Tensor) else None,
            scope=self.trace.node_scope.get(id(item)),
        )

        if isinstance(item, Tensor) and item.producer is not None:
            graph.edges.add(Edge(source=id(item.producer), target=id(item)))
            self._walk(item.producer, visited, graph)
        elif isinstance(item, Op):
            for input in item.inputs:
                graph.edges.add(Edge(source=id(input), target=id(item)))
                self._walk(input, visited, graph)

    def _display(self, item: Tensor | Op) -> str:
        """Return a display string for the node."""
        if isinstance(item, Op):
            return item.label

        if item.kind in (NodeKind.CONSTANT, NodeKind.PARAMETER, NodeKind.LAYER_OUTPUT):
            if item.name is None:
                return f"{item.data[0]:.2f}"
            return item.name

        return item.kind.name.lower()

    def _new_digraph(self, name: str) -> graphviz.Digraph:
        return graphviz.Digraph(
            name,
            graph_attr=_theme.GRAPH_ATTR,
            node_attr=_theme.NODE_ATTR,
            edge_attr=_theme.EDGE_ATTR,
        )

    def _add_node(self, target: graphviz.Digraph, node_id: int, node: Node) -> None:
        label = node.label
        if node.shape is not None:
            label = f"{node.label}\n({_format_shape(node.shape)})"
        group = node.scope.id if node.scope is not None else None
        target.node(str(node_id), label, **_theme.node_attrs(node.kind, group))

    def _add_edges(self, dot: graphviz.Digraph, graph: Graph) -> None:
        for edge in graph.edges:
            into_op = graph.nodes[edge.target].kind == NodeKind.OP
            dot.edge(
                str(edge.source),
                str(edge.target),
                label=edge.label or "",
                style="dashed" if into_op else "solid",
            )

    def _box_scope(
        self, scope: Scope, collapse_ids: set[str], descendants: bool
    ) -> Scope | None:
        """The collapsed scope that swallows ``scope``, or None if it survives.

        With ``descendants``, a collapsed scope also claims everything nested
        beneath it, and the outermost claimant wins — so collapsing a Residual
        hides its whole block rather than fighting a collapsed layer inside it.
        """
        if not descendants:
            return scope if scope.id in collapse_ids else None
        outermost = scope if scope.id in collapse_ids else None
        current = scope
        while current.outer_scope is not None:
            current = self.trace.scopes[current.outer_scope]
            if current.id in collapse_ids:
                outermost = current
        return outermost

    def _collapse(
        self,
        graph: Graph,
        collapse_ids: set[str],
        *,
        descendants: bool = False,
        nest: bool = False,
    ) -> Graph:
        """Fold every node in a collapsed scope into one box node, rerouting edges
        across the boundary and dropping edges internal to a scope."""
        nodes, rep = self._fold_nodes(graph, collapse_ids, descendants, nest)
        return Graph(nodes=nodes, edges=self._reroute_edges(graph, rep))

    def _fold_nodes(
        self, graph: Graph, collapse_ids: set[str], descendants: bool, nest: bool
    ) -> tuple[dict[int, Node], dict[int, int]]:
        """Map every node to what survives it: its scope's box, or itself.

        Returns the surviving nodes and ``rep``, the old-id -> surviving-id map that
        _reroute_edges needs to redirect the edges.
        """
        nodes: dict[int, Node] = {}
        rep: dict[int, int] = {}
        box_ids: dict[str, int] = {}

        for node_id, node in graph.nodes.items():
            box = (
                self._box_scope(node.scope, collapse_ids, descendants)
                if node.scope is not None
                else None
            )
            if box is None:
                rep[node_id] = node_id
                nodes[node_id] = node
                continue
            if box.id not in box_ids:
                box_ids[box.id] = -len(box_ids) - 1  # negative: never an id()
                nodes[box_ids[box.id]] = self._make_box(box, nest)
            rep[node_id] = box_ids[box.id]

        return nodes, rep

    def _make_box(self, scope: Scope, nest: bool) -> Node:
        """The single node standing in for a collapsed scope.

        ``nest`` places the box in its parent scope so the combined view clusters it
        where the module sat; the architecture view wants it free-standing and labels
        it with the full path since it has no cluster for context.
        """
        if nest:
            parent = (
                self.trace.scopes[scope.outer_scope]
                if scope.outer_scope is not None
                else None
            )
            return Node(kind=NodeKind.LAYER_OUTPUT, label=scope.label, shape=None, scope=parent)
        return Node(kind=NodeKind.LAYER_OUTPUT, label=scope.id, shape=None, scope=None)

    def _reroute_edges(self, graph: Graph, rep: dict[int, int]) -> set[Edge]:
        """Redirect edges onto surviving nodes, dropping those that end up inside a
        single box. Each survivor keeps the crossing tensor's shape as its label: in
        the bipartite graph a cross-scope edge always runs tensor -> op, so the
        source's shape is the data flowing between scopes.
        """
        edges: set[Edge] = set()
        for edge in graph.edges:
            source, target = rep[edge.source], rep[edge.target]
            if source == target:
                continue  # both ends folded into the same box: internal, drop it
            crossing = graph.nodes[edge.source].shape
            label = _format_shape(crossing) if crossing is not None else None
            edges.add(Edge(source, target, label=label))
        return edges

    def _add_scope_clusters(
        self, dot: graphviz.Digraph, graph: Graph, collapse_ids: set[str]
    ) -> None:
        """Nest one graphviz cluster per surviving scope, mirroring the scope tree.
        Nodes with no scope sit ungrouped at the top; a folded scope contributes no
        cluster (its box lives in its parent's)."""
        nodes_by_scope: dict[str, list[tuple[int, Node]]] = defaultdict(list)
        for node_id, node in graph.nodes.items():
            if node.scope is None:
                self._add_node(dot, node_id, node)
            else:
                nodes_by_scope[node.scope.id].append((node_id, node))

        children_of: dict[str | None, list[Scope]] = defaultdict(list)
        for scope in self.trace.scopes.values():
            if not self._folded(scope, collapse_ids):
                children_of[scope.outer_scope].append(scope)

        self._emit_clusters(dot, None, children_of, nodes_by_scope)

    def _emit_clusters(
        self,
        parent: graphviz.Digraph,
        outer_id: str | None,
        children_of: dict[str | None, list[Scope]],
        nodes_by_scope: dict[str, list[tuple[int, Node]]],
    ) -> None:
        """Emit the scopes directly inside ``outer_id`` as clusters, recursing into
        their own children."""
        for scope in children_of.get(outer_id, []):
            cluster = graphviz.Digraph(name="cluster_" + scope.id)
            _theme.style_cluster(cluster, scope.label, scope.id)
            for node_id, node in nodes_by_scope.get(scope.id, []):
                self._add_node(cluster, node_id, node)
            self._emit_clusters(cluster, scope.id, children_of, nodes_by_scope)
            parent.subgraph(cluster)

    def _folded(self, scope: Scope, collapse_ids: set[str]) -> bool:
        """True if ``scope`` or any ancestor is collapsed — so it should not render
        as its own cluster."""
        current: Scope | None = scope
        while current is not None:
            if current.id in collapse_ids:
                return True
            current = (
                self.trace.scopes[current.outer_scope]
                if current.outer_scope is not None
                else None
            )
        return False

    def _render(self, dot: graphviz.Digraph, save_path: str | None) -> graphviz.Digraph:
        """Save to save_path when given. Always return the dot so notebooks render it
        inline (Jupyter shows a Digraph via its rich SVG repr)."""
        if save_path is not None:
            dot.render(outfile=save_path, format="svg", cleanup=True)
        return dot
