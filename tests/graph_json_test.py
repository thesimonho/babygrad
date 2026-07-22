"""The graph JSON emitter is the wire format M1 hands the frontend: a pure
structural dump of the neutral graph + scope tree, no coordinates. These tests
pin the §04 contract shape so the JS side can rely on it."""

from babygrad.nn.model import Model
from babygrad.nn.modules import Linear, Residual, Sequential
from babygrad.observers import Tracer
from babygrad.tensor import Tensor
from babygrad.tracing import tracing
from babygrad.types import NodeKind
from babygrad.viz.attribution import attribute
from babygrad.viz.graph import GraphVisualizer
from babygrad.viz.graph_json import to_graph_json


def _emit(root) -> dict:
    """Trace a forward pass through ``Model(root)``, build the visualizer, and emit
    its JSON — the same dance a real server route performs."""
    model = Model(root)
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)
    tracer = Tracer()
    with tracing(tracer):
        output = model.forward(x)
    trace = attribute(tracer.records)
    visualizer = GraphVisualizer(output, trace)
    return to_graph_json(visualizer.graph, trace.scopes)


def test_emit_has_the_three_top_level_keys():
    """The contract is exactly nodes, edges, scopes — nothing more, nothing less."""
    payload = _emit(Sequential([Linear(2, 2)]))

    assert set(payload) == {"nodes", "edges", "scopes"}


def test_every_node_carries_the_contract_fields():
    """Each node object answers id/label/kind/shape/scope/series_tag — the fields the
    frontend maps onto a Cytoscape element and its click popup."""
    payload = _emit(Sequential([Linear(2, 2)]))

    for node in payload["nodes"]:
        assert set(node) == {"id", "label", "kind", "shape", "scope", "series_tag"}


def test_recorded_param_node_carries_its_series_tag():
    """A parameter node's series_tag mirrors the recorder's ``<scope label>/<role>``
    tag, so the popup can pull that node's history for a sparkline."""
    payload = _emit(Sequential([Linear(2, 2)]))

    weights = next(
        node
        for node in payload["nodes"]
        if node["kind"] == "PARAMETER" and node["label"] == "weights"
    )
    assert weights["series_tag"] == "Linear_0/weights"


def test_untracked_node_has_null_series_tag():
    """An op is not something the recorder tracks over time, so it carries no tag."""
    payload = _emit(Sequential([Linear(2, 2)]))

    op = next(node for node in payload["nodes"] if node["kind"] == "OP")
    assert op["series_tag"] is None


def test_node_ids_are_strings_and_unique():
    """ids come from ``id(obj)`` but cross the wire as strings (JS numbers can't
    hold a 64-bit pointer); they must stay unique so edges resolve."""
    payload = _emit(Sequential([Linear(2, 2)]))

    ids = [node["id"] for node in payload["nodes"]]
    assert all(isinstance(node_id, str) for node_id in ids)
    assert len(ids) == len(set(ids))


def test_kind_is_the_enum_name():
    """kind serializes as the ``NodeKind`` name so the frontend can style by role
    without knowing enum values. A Linear's weights are a PARAMETER node."""
    payload = _emit(Sequential([Linear(2, 2)]))

    kinds = {node["kind"] for node in payload["nodes"]}
    assert NodeKind.PARAMETER.name in kinds
    assert all(kind in NodeKind.__members__ for kind in kinds)


def test_shape_is_a_list_or_null():
    """Tensor shapes cross as JSON arrays; ops (which have no shape) carry null."""
    payload = _emit(Sequential([Linear(2, 2)]))

    for node in payload["nodes"]:
        shape = node["shape"]
        assert shape is None or (
            isinstance(shape, list) and all(isinstance(dim, int) for dim in shape)
        )


def test_node_scope_is_a_scope_id_or_null():
    """A node's scope is the id of its compound parent, or null when it belongs to
    no scope (graph-entry leaves like the input view)."""
    payload = _emit(Sequential([Linear(2, 2)]))

    scope_ids = {scope["id"] for scope in payload["scopes"]}
    for node in payload["nodes"]:
        assert node["scope"] is None or node["scope"] in scope_ids


def test_edges_reference_declared_node_ids():
    """Every edge endpoint must be a node the payload also declares, or the frontend
    would draw a dangling edge."""
    payload = _emit(Sequential([Linear(2, 2)]))

    node_ids = {node["id"] for node in payload["nodes"]}
    for edge in payload["edges"]:
        assert set(edge) == {"source", "target", "label"}
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids


def test_scopes_carry_the_nesting_fields():
    """Each scope answers id/label/outer/collapsed — outer names its parent scope so
    the frontend nests compound nodes without parsing the id path."""
    payload = _emit(Sequential([Linear(2, 2)]))

    for scope in payload["scopes"]:
        assert set(scope) == {"id", "label", "outer", "collapsed"}


def test_scope_outer_links_resolve_within_the_payload():
    """A scope's outer is either null (root) or the id of another scope present in
    the same payload — the compound-node tree must be self-contained."""
    payload = _emit(Sequential([Residual(Sequential([Linear(2, 2)]))]))

    scope_ids = {scope["id"] for scope in payload["scopes"]}
    for scope in payload["scopes"]:
        assert scope["outer"] is None or scope["outer"] in scope_ids


def test_empty_container_scope_is_still_emitted():
    """A pass-through Sequential owns no graph nodes but must still appear as a
    scope, so the frontend can draw its compound box in the hierarchy."""
    payload = _emit(Sequential([Sequential([Linear(2, 2)])]))

    scope_ids = {scope["id"] for scope in payload["scopes"]}
    assert any(scope_id.endswith("/Sequential_0") for scope_id in scope_ids)


def test_collapsed_flag_rides_through_to_json():
    """A scope flagged collapsed carries collapsed=true so the frontend can render
    it folded on first paint."""
    payload = _emit(Sequential([Residual(Sequential([Linear(2, 2)]), collapse=True)]))

    collapsed = [scope for scope in payload["scopes"] if scope["collapsed"]]
    assert any(scope["label"] == "Residual_0" for scope in collapsed)
