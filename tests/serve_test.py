"""The streaming helpers turn a recorder snapshot into per-epoch deltas, and the
server merges those deltas back into a run's history. These pin that split and the
merge, which is what keeps the live stream and the board correct across runs."""

from babygrad.viz.serve import (
    DashboardServer,
    _delta_for_step,
    _steps_after,
)


def test_delta_splits_scalars_from_series():
    """A step's delta routes number series to ``scalars`` and list series to
    ``series``, so the frontend sends each to the right chart."""
    snapshot = {
        "loss": {0: 1.5, 1: 1.2},
        "Linear_0/weights": {0: [0.1, 0.2], 1: [0.3, 0.4]},
    }

    delta = _delta_for_step(snapshot, 1)

    assert delta == {
        "step": 1,
        "scalars": {"loss": 1.2},
        "series": {"Linear_0/weights": [0.3, 0.4]},
    }


def test_delta_omits_series_without_that_step():
    """A tag missing the step contributes nothing, so a delta only carries what was
    actually recorded that epoch."""
    snapshot = {"loss": {0: 1.5}, "val_loss": {0: 1.4, 1: 1.1}}

    delta = _delta_for_step(snapshot, 1)

    assert delta["scalars"] == {"val_loss": 1.1}


def test_steps_after_returns_new_steps_in_order():
    """Only steps newer than the last emitted one, sorted, so each epoch streams
    once and in order."""
    snapshot = {"loss": {0: 1.0, 1: 0.9, 2: 0.8}}

    assert _steps_after(snapshot, 0) == [1, 2]
    assert _steps_after(snapshot, 2) == []


def _server() -> DashboardServer:
    """A server bound to an OS-assigned free port; caller closes it."""
    return DashboardServer(("127.0.0.1", 0))


def test_add_epoch_merges_scalars_and_series_into_history():
    """A pushed epoch lands in the history under its step, scalars and series
    alike, so the SSE stream and /history.json report the run's progress."""
    server = _server()
    try:
        server.start_run({"nodes": []}, {})
        server.add_epoch(
            {"step": 0, "scalars": {"loss": 1.2}, "series": {"L0/weights": [0.1, 0.2]}}
        )

        history, _ = server.snapshot()
        assert history == {"loss": {0: 1.2}, "L0/weights": {0: [0.1, 0.2]}}
    finally:
        server.server_close()


def test_start_run_bumps_generation_and_clears_prior_history():
    """Each run increments the generation (so open streams know to reset) and
    starts from an empty history, clearing the previous run's board."""
    server = _server()
    try:
        server.start_run({"nodes": []}, {})
        server.add_epoch({"step": 0, "scalars": {"loss": 1.0}, "series": {}})
        _, first_generation = server.snapshot()

        server.start_run({"nodes": []}, {})  # a second run connects

        history, second_generation = server.snapshot()
        assert history == {}
        assert second_generation == first_generation + 1
    finally:
        server.server_close()
