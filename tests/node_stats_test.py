"""The node-stats emitter feeds the graph popup: a current value/grad summary for
every tensor node, keyed by the same id the graph JSON uses. These tests pin the
summary shape and that ops (which hold no values) are skipped."""

from babygrad.tensor import Tensor
from babygrad.types import NodeKind
from babygrad.viz.node_stats import to_node_stats


def _param(data, shape) -> Tensor:
    return Tensor(data, shape=shape, kind=NodeKind.PARAMETER)


def test_every_tensor_gets_a_value_and_grad_summary():
    """Both a leaf tensor and the tensor an op produces are summarised, keyed by
    their string id so the frontend can look them up by node id."""
    leaf = _param([1.0, 2.0, 3.0], (3,))
    produced = leaf + leaf

    stats = to_node_stats(produced)

    assert str(id(leaf)) in stats
    assert str(id(produced)) in stats
    for entry in stats.values():
        assert set(entry) == {"value", "grad"}


def test_summary_reports_mean_min_max_and_count():
    """The value summary is the mean/min/max/count of the tensor's data."""
    leaf = _param([1.0, 2.0, 3.0], (3,))

    stats = to_node_stats(leaf)

    assert stats[str(id(leaf))]["value"] == {"mean": 2.0, "min": 1.0, "max": 3.0, "n": 3}


def test_ops_are_not_summarised():
    """An op holds no values, so it gets no entry — only the tensors around it do."""
    leaf = _param([1.0], (1,))
    produced = leaf + leaf

    stats = to_node_stats(produced)

    assert produced.producer is not None
    assert str(id(produced.producer)) not in stats
