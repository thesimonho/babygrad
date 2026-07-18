from babygrad.nn.modules import Linear
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
