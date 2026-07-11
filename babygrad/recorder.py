from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from babygrad.types import History, HistoryValue, NodeKind, Step, Tag

if TYPE_CHECKING:
    from babygrad.tensor import Tensor


_RECORDABLE = {NodeKind.PARAMETER, NodeKind.LAYER_OUTPUT}


class Recorder:
    """Collects tagged training data, keyed by the loop-supplied step."""

    def __init__(self):
        self.step: Step = 0
        self.history: History = defaultdict(dict)

    def record(self, tag: Tag, value: HistoryValue):
        self.history[tag][self.step] = value

    def capture(self, root: Tensor):
        """Walk the graph backward from `root`, recording recordable nodes.

        A node is recordable by its kind (parameters and layer outputs), not
        by whether it happens to have a name. Each contributes two tags: its
        data under its name and its gradient under "<name>/grad". Values are
        copied because parameters mutate in place and gradients are zeroed
        between steps.

        Coverage is bounded by ancestry — only tensors that contributed to
        `root` are reachable, so capturing from the loss sees everything.
        Gradients are only meaningful when called after backward() and
        before the next zero_grad().
        """
        visited: set[int] = set()
        self._capture_walk(root, visited)

    def _capture_walk(self, tensor: Tensor, visited: set[int]):
        if id(tensor) in visited:
            return
        visited.add(id(tensor))

        if tensor.kind in _RECORDABLE and tensor.name is not None:
            self.record(tensor.name, list(tensor.data))
            self.record(f"{tensor.name}/grad", list(tensor.grad))

        if tensor.producer is not None:
            for parent in tensor.producer.inputs:
                self._capture_walk(parent, visited)
