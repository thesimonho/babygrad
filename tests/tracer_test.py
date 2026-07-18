import pytest

from babygrad.nn.model import Model
from babygrad.nn.modules import Linear, Residual, Sequential
from babygrad.tensor import Tensor
from babygrad.tracing import tracing
from babygrad.types import NodeKind
from babygrad.viz.tracer import Tracer


def test_call_brackets_forward_into_the_active_tracer():
    """__call__ records one bracket per traced module, capturing its inputs and
    output; the outer scope is None at the top level."""
    linear = Linear(2, 2)
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)

    tracer = Tracer()
    with tracing(tracer):
        output = linear(x)

    assert len(tracer.records) == 1
    record = tracer.records[0]
    assert record.module is linear
    assert len(record.inputs) == 1
    assert record.inputs[0] is x
    assert record.output is output
    assert record.outer is None


def test_no_active_tracer_is_a_passthrough():
    """Without a tracer, __call__ just runs forward — no error, no records kept."""
    linear = Linear(2, 2)
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)

    output = linear(x)  # NullTracer is active by default

    assert output.shape == (1, 2)


def test_forward_brackets_every_nested_module():
    """After call-site conversion, a model forward brackets the root and every
    nested module — verify_covers passes and each module gets one record."""
    model = Model(Sequential([Residual(Sequential([Linear(2, 2)]))]))
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)

    tracer = Tracer()
    with tracing(tracer):
        model.forward(x)

    tracer.verify_covers(model.root)  # raises if any module went unbracketed
    assert len(tracer.records) == 4  # root Seq, Residual, inner Seq, Linear


def test_verify_covers_raises_when_a_module_is_unbracketed():
    """An unbracketed module (here: nothing was traced) is caught loudly, not
    silently merged into a parent's scope."""
    model = Model(Sequential([Linear(2, 2)]))
    tracer = Tracer()  # no forward traced

    with pytest.raises(RuntimeError):
        tracer.verify_covers(model.root)
