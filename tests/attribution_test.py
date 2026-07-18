from babygrad.nn.model import Model
from babygrad.nn.modules import Linear, Residual, Sequential
from babygrad.tensor import Tensor
from babygrad.tracing import tracing
from babygrad.types import NodeKind
from babygrad.viz.attribution import attribute
from babygrad.viz.tracer import Tracer


def _trace(root):
    """Trace a forward pass of ``Model(root)`` and return (model, tracer)."""
    model = Model(root)
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)
    tracer = Tracer()
    with tracing(tracer):
        model.forward(x)
    return model, tracer


def test_scope_tree_mirrors_module_nesting():
    """The scope tree comes from bracket nesting: one scope per module, each
    linked to its outer scope, with full-path ids."""
    residual = Residual(Sequential([Linear(2, 2)]))
    _, tracer = _trace(Sequential([residual]))

    result = attribute(tracer.records)

    assert set(result.scopes) == {
        "Sequential_0",
        "Sequential_0/Residual_0",
        "Sequential_0/Residual_0/Sequential_0",
        "Sequential_0/Residual_0/Sequential_0/Linear_0",
    }
    assert result.scopes["Sequential_0"].outer_scope is None
    assert result.scopes["Sequential_0/Residual_0"].outer_scope == "Sequential_0"
    assert (
        result.scopes["Sequential_0/Residual_0/Sequential_0/Linear_0"].outer_scope
        == "Sequential_0/Residual_0/Sequential_0"
    )


def test_residual_add_scoped_to_residual_not_inner_block():
    """Residual's ``input + block(input)`` add must land on the Residual, not the
    inner block. The stop-rule halts the walk at the boundary input and at the
    block output the inner module already claimed, so Residual claims only its
    own add (the op and its result tensor)."""
    residual = Residual(Sequential([Linear(2, 2)]))
    _, tracer = _trace(Sequential([residual]))

    result = attribute(tracer.records)

    residual_record = next(r for r in tracer.records if r.module is residual)
    add_result = residual_record.output
    add_op = add_result.producer

    assert result.node_scope[id(add_op)].label == "Residual_0"
    assert result.node_scope[id(add_result)].label == "Residual_0"

    residual_nodes = [n for n, s in result.node_scope.items() if s.label == "Residual_0"]
    assert len(residual_nodes) == 2  # only the add op and its result tensor


def test_parameter_scoped_to_its_owning_module():
    """A module's parameters belong to it directly."""
    linear = Linear(2, 2)
    _, tracer = _trace(Sequential([linear]))

    result = attribute(tracer.records)

    for param in linear.own_parameters():
        assert result.node_scope[id(param)].label == "Linear_0"


def test_empty_container_survives_in_the_tree():
    """A pass-through Sequential claims no ops (its output is its child's output),
    but its scope must still exist so it renders as a box."""
    _, tracer = _trace(Sequential([Sequential([Linear(2, 2)])]))

    result = attribute(tracer.records)

    assert "Sequential_0/Sequential_0" in result.scopes
    claimed = {scope.id for scope in result.node_scope.values()}
    assert "Sequential_0/Sequential_0" not in claimed
