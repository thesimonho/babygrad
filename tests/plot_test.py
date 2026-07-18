import matplotlib

matplotlib.use("Agg")

from babygrad.types import History
from babygrad.nn.activations import ReLU
from babygrad.nn.model import Model
from babygrad.nn.modules import Linear, Sequential
from babygrad.observers import Recorder, Tracer
from babygrad.tracing import tracing
from babygrad.viz.plot import PlotVisualizer, _bin_counts
from babygrad.tensor import Tensor
from babygrad.types import NodeKind


def _trace_forward(model: Model, x: Tensor) -> tuple[Tensor, Tracer]:
    """Run a forward pass under a tracer, returning its output and the tracer whose
    records the recorder consumes — the dance a training loop performs each batch."""
    tracer = Tracer()
    with tracing(tracer):
        output = model.forward(x)
    return output, tracer


SCALAR_HISTORY: History = {"loss": {0: 1.5, 1: 1.2, 2: 0.9}}
RIDGE_HISTORY: History = {"Linear_0/weights": {0: [0.1, 0.2, 0.3], 1: [0.2, 0.3, 0.4]}}


def test_plot_scalar_saves_figure(tmp_path):
    save_path = tmp_path / "loss.png"

    PlotVisualizer(SCALAR_HISTORY).plot_scalar(["loss"], save_path=str(save_path))

    assert save_path.exists()


def test_plot_ridge_saves_figure(tmp_path):
    save_path = tmp_path / "weights.png"

    PlotVisualizer(RIDGE_HISTORY).plot_ridge("Linear_0/weights", save_path=str(save_path))

    assert save_path.exists()


def test_plot_ridge_handles_identical_values(tmp_path):
    constant_history: History = {"w": {0: [0.5, 0.5], 1: [0.5, 0.5]}}
    save_path = tmp_path / "constant.png"

    PlotVisualizer(constant_history).plot_ridge("w", save_path=str(save_path))

    assert save_path.exists()


def test_plot_ridge_clip_quantiles_saves_figure(tmp_path):
    outlier_history: History = {
        "grads": {0: [0.001] * 98 + [-5.0, 5.0], 1: [0.002] * 98 + [-4.0, 6.0]}
    }
    save_path = tmp_path / "clipped.png"

    PlotVisualizer(outlier_history).plot_ridge(
        "grads", save_path=str(save_path), clip_quantiles=(0.01, 0.99)
    )

    assert save_path.exists()


def test_bin_counts_clamps_outliers_into_edge_bins():
    edges = [0.0, 1.0, 2.0]

    counts = _bin_counts([-10.0, 0.5, 1.5, 10.0], edges)

    assert counts == [2.0, 2.0]


def test_capture_records_named_tensors():
    recorder = Recorder()
    model = Model(Sequential([Linear(2, 3), ReLU()]))
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)

    recorder.step = 0
    _, tracer = _trace_forward(model, x)
    recorder.capture(tracer)

    assert "Linear_0/weights" in recorder.history
    assert "Linear_0/bias" in recorder.history
    assert "Linear_0/result" in recorder.history
    # every layer boundary is named now, parameterless layers included
    assert "ReLU_1/result" in recorder.history
    # exactly these value tags: the container's output is ReLU's output, already
    # claimed, so it must NOT reappear as Sequential_0/result (the double-tag trap)
    data_tags = {tag for tag in recorder.history if not tag.endswith("/grad")}
    assert data_tags == {
        "Linear_0/weights",
        "Linear_0/bias",
        "Linear_0/result",
        "ReLU_1/result",
    }


def test_captured_weights_are_snapshots():
    recorder = Recorder()
    layer = Linear(2, 3)
    model = Model(Sequential([layer]))
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)

    recorder.step = 0
    _, tracer = _trace_forward(model, x)
    recorder.capture(tracer)
    layer.weights.data[0] += 100.0
    recorder.step = 1
    _, tracer = _trace_forward(model, x)
    recorder.capture(tracer)

    recorded = recorder.history["Linear_0/weights"]
    before, after = recorded[0], recorded[1]
    assert isinstance(before, list) and isinstance(after, list)
    assert before[0] != after[0]
    assert before[0] != layer.weights.data[0]


def test_no_capture_records_nothing():
    model = Sequential([Linear(2, 3)])
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)

    result = model.forward(x)

    assert result.shape == (1, 3)


def test_capture_records_grads():
    recorder = Recorder()
    layer = Linear(2, 3)
    model = Model(Sequential([layer]))
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)

    output, tracer = _trace_forward(model, x)
    output.sum().backward()

    recorder.step = 0
    recorder.capture(tracer)

    assert "Linear_0/weights/grad" in recorder.history
    recorded = recorder.history["Linear_0/weights/grad"][0]
    assert isinstance(recorded, list)
    assert any(value != 0.0 for value in recorded)


def test_captured_grads_survive_zero_grad():
    recorder = Recorder()
    layer = Linear(2, 3)
    model = Model(Sequential([layer]))
    x = Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW)
    _, tracer = _trace_forward(model, x)
    layer.weights.grad[0] = 7.0

    recorder.step = 0
    recorder.capture(tracer)
    layer.weights.grad[0] = 0.0  # what zero_grad() does, in place

    recorded = recorder.history["Linear_0/weights/grad"][0]
    assert isinstance(recorded, list)
    assert recorded[0] == 7.0
