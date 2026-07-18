from babygrad.nn.model import Model
from babygrad.nn.modules import Linear, Residual, Sequential
from babygrad.observers import Tracer
from babygrad.tensor import Tensor
from babygrad.tracing import tracing
from babygrad.types import NodeKind
from babygrad.viz.attribution import attribute
from babygrad.viz.graph import GraphVisualizer


def _visualizer(root) -> GraphVisualizer:
    """Trace a forward pass through ``Model(root)`` and build a visualizer from the
    resulting attribution — the explicit dance every real caller performs."""
    model = Model(root)
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)
    tracer = Tracer()
    with tracing(tracer):
        output = model.forward(x)
    return GraphVisualizer(output, attribute(tracer.records))


def test_computation_view_is_flat_with_no_clusters():
    """The computation view draws every node at the top level — no scope boxes."""
    graph = _visualizer(Sequential([Linear(2, 2)]))

    source = graph.draw_computation().source

    assert "subgraph" not in source
    # nodes carry only their intrinsic role now; the scope prefix that qualified
    # them (Linear_0/) is supplied by the tracer's cluster boxes, not the label
    assert "weights" in source


def test_combined_view_nests_a_cluster_per_scope():
    """Each scope becomes a cluster, nested to mirror the module tree — the inner
    Linear's cluster sits inside the Residual's, inside the root Sequential's."""
    graph = _visualizer(Sequential([Residual(Sequential([Linear(2, 2)]))]))

    source = graph.draw_combined().source

    assert "cluster_Sequential_0 {" in source
    assert '"cluster_Sequential_0/Residual_0" {' in source
    assert '"cluster_Sequential_0/Residual_0/Sequential_0/Linear_0" {' in source


def test_combined_view_keeps_an_empty_container_as_a_box():
    """A pass-through Sequential owns no nodes but must still draw as a cluster, so
    the hierarchy stays visible. It renders because it nests the inner block."""
    graph = _visualizer(Sequential([Sequential([Linear(2, 2)])]))

    source = graph.draw_combined().source

    assert '"cluster_Sequential_0/Sequential_0" {' in source


def test_architecture_view_collapses_each_scope_to_one_box():
    """The architecture view folds every scope to a single box labelled with its
    full path, with the crossing tensor's shape on the edges between them."""
    graph = _visualizer(Sequential([Linear(2, 2), Linear(2, 2)]))

    source = graph.draw_architecture().source

    assert "subgraph" not in source  # flat: no clusters
    assert 'label="Sequential_0/Linear_0"' in source
    assert 'label="Sequential_0/Linear_1"' in source
    assert 'label="1×2"' in source  # boundary edge carries the crossing shape


def test_collapsed_scope_draws_as_one_box_in_combined():
    """A scope flagged collapsed folds to a single box inside its parent cluster,
    hiding its internals while keeping its place in the hierarchy."""
    graph = _visualizer(Sequential([Residual(Sequential([Linear(2, 2)]), collapse=True)]))

    source = graph.draw_combined().source

    # the Residual is now a leaf box in the root cluster, not its own cluster
    assert "cluster_Sequential_0 {" in source
    assert '"cluster_Sequential_0/Residual_0"' not in source
    assert "label=Residual_0" in source
