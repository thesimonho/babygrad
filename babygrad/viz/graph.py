"""
Render an autograd graph: extract a neutral bipartite structure, then draw it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import graphviz

from babygrad.ops import Op
from babygrad.tensor import Tensor
from babygrad.types import NodeKind, Shape
from babygrad.viz import _theme


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


@dataclass
class _ScopeTree:
    """One level of the scope hierarchy: the graph nodes stamped with this exact
    scope, plus the child scopes nested one segment deeper."""

    nodes: list[tuple[int, Node]] = field(default_factory=list)
    children: dict[str, "_ScopeTree"] = field(default_factory=dict)


def _format_shape(shape: Shape) -> str:
    return "×".join(str(dim) for dim in shape)


class GraphVisualizer:
    """Walk a tensor's autograd graph into a neutral bipartite structure, then draw it.

    walk(root) -> Graph(nodes, edges) -> draw_*(): computation (flat),
    combined (clustered by layer scope), architecture (collapsed).
    """

    def __init__(self, root: Tensor, collapsed_scopes: set[str] | None = None):
        self.root = root
        # scope strings to fold into one box in the combined view, with their
        # descendants. Model.collapsed_scopes supplies these; a bare tensor graph
        # has none, which is why this defaults empty rather than requiring a Model.
        self.collapsed_scopes = collapsed_scopes or set()
        self.graph = Graph(nodes=dict(), edges=set())
        self._walk(self.root, set(), self.graph)

    def draw_computation(self, save_path: str | None = None) -> graphviz.Digraph:
        """The full bipartite graph, drawn flat (no clustering)."""
        return self._draw_flat("computation", self.graph, save_path)

    def draw_combined(self, save_path: str | None = None) -> graphviz.Digraph:
        """The full graph with each layer scope wrapped in a cluster box, nested to
        mirror the module hierarchy. Modules marked collapse=True are drawn as one
        box, hiding their internals but keeping their place in the hierarchy."""
        graph = self._marked_collapsed()

        dot = self._new_digraph("combined")
        ungrouped, tree = self._build_scope_tree(graph)
        for node_id, node in ungrouped:
            self._add_node(dot, node_id, node)
        self._add_scope_clusters(dot, tree, [])
        self._add_edges(dot, graph)
        return self._render(dot, save_path)

    def draw_architecture(self, save_path: str | None = None) -> graphviz.Digraph:
        """Every layer scope collapsed to one box, edges labelled with the
        shape of the tensor crossing between layers."""
        every_scope = {
            n.scope for n in self.graph.nodes.values() if n.scope is not None
        }
        return self._draw_flat(
            "architecture", self._collapse(self.graph, every_scope), save_path
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

    def _marked_collapsed(self) -> Graph:
        """The graph with only the collapse=True scopes folded, descendants included."""
        if not self.collapsed_scopes:
            return self.graph
        return self._collapse(
            self.graph, self.collapsed_scopes, descendants=True, nest=True
        )

    def _walk(self, item: Tensor | Op, visited: set[int], graph: Graph):
        """
        Walk the bipartite graph, storing nodes and edges as we go.
        """
        if id(item) in visited:
            return

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
        target.node(str(node_id), label, **_theme.node_attrs(node.kind, node.scope))

    def _add_edges(self, dot: graphviz.Digraph, graph: Graph) -> None:
        for edge in graph.edges:
            into_op = graph.nodes[edge.target].kind == NodeKind.OP
            dot.edge(
                str(edge.source),
                str(edge.target),
                label=edge.label or "",
                style="dashed" if into_op else "solid",
            )

    def _box_scope(self, scope: str, scopes: set[str], descendants: bool) -> str | None:
        """The marked scope that swallows `scope`, or None if it survives intact.

        With `descendants`, a marked scope also claims everything nested beneath it,
        and the outermost claimant wins — so collapsing a Residual hides its whole
        block rather than fighting with a collapsed layer inside it.
        """
        if scope in scopes:
            return scope
        if not descendants:
            return None
        enclosing = [s for s in scopes if scope.startswith(s + "/")]
        return min(enclosing, key=len) if enclosing else None

    def _collapse(
        self,
        graph: Graph,
        scopes: set[str],
        *,
        descendants: bool = False,
        nest: bool = False,
    ) -> Graph:
        """Fold every node in a collapsed scope into one box node, rerouting
        edges across the boundary and dropping edges internal to a scope.
        """
        nodes, rep = self._fold_nodes(graph, scopes, descendants, nest)
        return Graph(nodes=nodes, edges=self._reroute_edges(graph, rep))

    def _fold_nodes(
        self, graph: Graph, scopes: set[str], descendants: bool, nest: bool
    ) -> tuple[dict[int, Node], dict[int, int]]:
        """Map every node to what survives it: its scope's box, or itself.

        Returns the surviving nodes and `rep`, the old-id -> surviving-id map that
        _reroute_edges needs to redirect the edges.
        """
        nodes: dict[int, Node] = {}
        rep: dict[int, int] = {}
        box_ids: dict[str, int] = {}

        for node_id, node in graph.nodes.items():
            box = (
                self._box_scope(node.scope, scopes, descendants)
                if node.scope is not None
                else None
            )
            if box is None:
                rep[node_id] = node_id
                nodes[node_id] = node
                continue
            if box not in box_ids:
                box_ids[box] = -len(box_ids) - 1  # negative: never an id()
                nodes[box_ids[box]] = self._make_box(box, nest)
            rep[node_id] = box_ids[box]

        return nodes, rep

    def _make_box(self, scope: str, nest: bool) -> Node:
        """The single node standing in for a collapsed scope.

        `nest` hands the box its parent's scope so the combined view clusters it
        where the module sat; the architecture view wants it free-standing, and
        labels it with the full path since it has no cluster for context.
        """
        parent, _, leaf = scope.rpartition("/")
        return Node(
            kind=NodeKind.LAYER_OUTPUT,
            label=leaf if nest else scope,
            shape=None,
            scope=(parent or None) if nest else None,
        )

    def _reroute_edges(self, graph: Graph, rep: dict[int, int]) -> set[Edge]:
        """Redirect edges onto surviving nodes, dropping those that end up inside
        a single box.

        Each survivor keeps the crossing tensor's shape as its label: in the
        bipartite graph a cross-scope edge always runs tensor -> op, so the
        source's shape is the data flowing between layers.
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

    def _build_scope_tree(
        self, graph: Graph | None = None
    ) -> tuple[list[tuple[int, Node]], _ScopeTree]:
        """Split each node's slash-delimited scope into a nested tree. Nodes with
        no scope are returned separately to sit ungrouped at the top level."""
        graph = self.graph if graph is None else graph
        ungrouped: list[tuple[int, Node]] = []
        root = _ScopeTree()
        for node_id, node in graph.nodes.items():
            if node.scope is None:
                ungrouped.append((node_id, node))
                continue
            cursor = root
            for segment in node.scope.split("/"):
                cursor = cursor.children.setdefault(segment, _ScopeTree())
            cursor.nodes.append((node_id, node))
        return ungrouped, root

    def _add_scope_clusters(
        self, parent: graphviz.Digraph, tree: _ScopeTree, path: list[str]
    ) -> None:
        """Emit one nested cluster per scope segment. Each cluster holds the nodes
        stamped with its exact scope and embeds its child scopes as sub-clusters."""
        for segment, subtree in tree.children.items():
            full_path = path + [segment]
            cluster = graphviz.Digraph(name="cluster_" + "/".join(full_path))
            _theme.style_cluster(cluster, segment, "/".join(full_path))
            for node_id, node in subtree.nodes:
                self._add_node(cluster, node_id, node)
            self._add_scope_clusters(cluster, subtree, full_path)
            parent.subgraph(cluster)

    def _render(self, dot: graphviz.Digraph, save_path: str | None) -> graphviz.Digraph:
        """Save to save_path when given. Always return the dot so notebooks
        render it inline (Jupyter shows a Digraph via its rich SVG repr)."""
        if save_path is not None:
            dot.render(outfile=save_path, format="svg", cleanup=True)
        return dot
