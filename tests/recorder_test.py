"""The Recorder's tag convention and its concurrent-read snapshot.

``series_tag`` is the one source of the ``"<label>/<role>"`` format the graph
JSON also references, and ``snapshot`` is what lets the dashboard's push thread
read the history while training writes it from another thread.
"""

from babygrad.observers import Recorder
from babygrad.types import grad_tag, series_tag


def test_series_tag_joins_label_and_role():
    """The canonical value-series tag is ``"<label>/<role>"``."""
    assert series_tag("Linear_0", "weights") == "Linear_0/weights"


def test_grad_tag_suffixes_its_base():
    """A series' gradient companion is its tag plus ``/grad``."""
    assert grad_tag("Linear_0/weights") == "Linear_0/weights/grad"


def test_snapshot_is_isolated_from_later_writes():
    """A snapshot copies each series, so a step recorded after the copy does not
    leak into a delta already being built from it."""
    recorder = Recorder()
    recorder.step = 0
    recorder.record("loss", 1.0)

    snapshot = recorder.snapshot()
    recorder.step = 1
    recorder.record("loss", 0.9)  # the next epoch, after the snapshot

    assert snapshot["loss"] == {0: 1.0}
