import matplotlib

matplotlib.use("Agg")

from babygrad.aliases import History
from babygrad.nn import Linear, ReLU, Sequential
from babygrad.recorder import Recorder
from babygrad.plot import Visualizer, _bin_counts
from babygrad.tensor import Tensor


SCALAR_HISTORY: History = {"loss": {0: 1.5, 1: 1.2, 2: 0.9}}
RIDGE_HISTORY: History = {"Linear_0/weights": {0: [0.1, 0.2, 0.3], 1: [0.2, 0.3, 0.4]}}


def test_plot_scalar_saves_figure(tmp_path):
    save_path = tmp_path / "loss.png"

    Visualizer().plot_scalar("loss", SCALAR_HISTORY, save_path=str(save_path))

    assert save_path.exists()


def test_plot_ridge_saves_figure(tmp_path):
    save_path = tmp_path / "weights.png"

    Visualizer().plot_ridge("Linear_0/weights", RIDGE_HISTORY, save_path=str(save_path))

    assert save_path.exists()


def test_plot_ridge_handles_identical_values(tmp_path):
    constant_history: History = {"w": {0: [0.5, 0.5], 1: [0.5, 0.5]}}
    save_path = tmp_path / "constant.png"

    Visualizer().plot_ridge("w", constant_history, save_path=str(save_path))

    assert save_path.exists()


def test_plot_ridge_clip_quantiles_saves_figure(tmp_path):
    outlier_history: History = {
        "grads": {0: [0.001] * 98 + [-5.0, 5.0], 1: [0.002] * 98 + [-4.0, 6.0]}
    }
    save_path = tmp_path / "clipped.png"

    Visualizer().plot_ridge(
        "grads", outlier_history, save_path=str(save_path), clip_quantiles=(0.01, 0.99)
    )

    assert save_path.exists()


def test_bin_counts_clamps_outliers_into_edge_bins():
    edges = [0.0, 1.0, 2.0]

    counts = _bin_counts([-10.0, 0.5, 1.5, 10.0], edges)

    assert counts == [2.0, 2.0]


def test_forward_fans_out_report_under_namespaced_tags():
    recorder = Recorder()
    model = Sequential([Linear(2, 3), ReLU()])
    x = Tensor([1.0, 2.0], shape=(1, 2))

    recorder.set_step(0)
    model.forward(x, recorder)

    assert "Linear_0/weights" in recorder.history
    assert "Linear_0/result" in recorder.history
    # parameterless layers report nothing, so no ReLU tags appear
    assert not any(tag.startswith("ReLU") for tag in recorder.history)


def test_recorded_weights_are_snapshots_not_references():
    recorder = Recorder()
    layer = Linear(2, 3)
    model = Sequential([layer])
    x = Tensor([1.0, 2.0], shape=(1, 2))

    recorder.set_step(0)
    model.forward(x, recorder)
    layer.weights.data[0] += 100.0
    recorder.set_step(1)
    model.forward(x, recorder)

    recorded = recorder.history["Linear_0/weights"]
    before, after = recorded[0], recorded[1]
    assert isinstance(before, list) and isinstance(after, list)
    assert before[0] != after[0]
    assert before[0] != layer.weights.data[0]


def test_forward_without_recorder_records_nothing():
    model = Sequential([Linear(2, 3)])
    x = Tensor([1.0, 2.0], shape=(1, 2))

    result = model.forward(x)

    assert result.shape == (1, 3)


def test_report_grads_fans_out_namespaced_tags():
    recorder = Recorder()
    model = Sequential([Linear(2, 3), ReLU()])

    recorder.set_step(0)
    model.report_grads(recorder)

    assert "Linear_0/grad" in recorder.history
    assert not any(tag.startswith("ReLU") for tag in recorder.history)


def test_recorded_grads_survive_zero_grad():
    recorder = Recorder()
    layer = Linear(2, 3)
    model = Sequential([layer])
    layer.weights.grad[0] = 7.0

    recorder.set_step(0)
    model.report_grads(recorder)
    layer.weights.grad[0] = 0.0  # what zero_grad() does, in place

    recorded = recorder.history["Linear_0/grad"][0]
    assert isinstance(recorded, list)
    assert recorded[0] == 7.0
