"""The two observers of a model run.

- ``Tracer`` watches module/loss brackets during a forward pass, collecting the
  raw ``(module, inputs, output)`` records that ``attribution`` turns into the
  scope tree and node->scope map. It is fed through the ``__call__`` bracket seam.
- ``Recorder`` collects the tagged time-series of values — parameter and
  layer-output data and gradients — across training steps.

Both observe the same model without it knowing: the Tracer captures *structure*,
the Recorder captures *values over time*.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from babygrad.types import History, HistoryValue, NodeKind, Step, Tag

if TYPE_CHECKING:
    from collections.abc import Iterator

    from babygrad.nn.modules import Module
    from babygrad.tensor import Tensor
    from babygrad.tracing import Traceable


# --- Tracer: structure ------------------------------------------------------


@dataclass
class TraceRecord:
    """One bracketed forward: the module that ran, its input tensors, its output
    tensor, and the module that bracketed it (its outer scope), or None at the
    top level."""

    module: Traceable
    inputs: list[Tensor]
    output: Tensor
    outer: Traceable | None


class Tracer:
    """Collects a TraceRecord per bracketed module/loss, in exit order — innermost
    first, since a child's forward returns before its parent's. That order is what
    later lets scope attribution claim inner nodes before their containers."""

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []
        self._open: list[tuple[Traceable, list[Tensor], Traceable | None]] = []

    def enter(self, module: Traceable, inputs: tuple[Tensor, ...]) -> None:
        outer = self._open[-1][0] if self._open else None
        self._open.append((module, list(inputs), outer))

    def exit(self, module: Traceable, output: Tensor) -> None:
        entered, inputs, outer = self._open.pop()
        self.records.append(TraceRecord(entered, inputs, output, outer))

    def verify_covers(self, root: Module) -> None:
        """Raise if any module in ``root``'s static tree went unbracketed this
        trace — the signature of a ``.forward()`` call site that was not converted
        to call syntax, which would silently fold that module's ops into its
        parent's scope instead of giving it its own."""
        bracketed = {id(record.module) for record in self.records}
        missing = [m for m in _walk_modules(root) if id(m) not in bracketed]
        if missing:
            names = ", ".join(type(m).__name__ for m in missing)
            raise RuntimeError(
                f"tracer did not bracket: {names}. A module .forward() call site "
                "was not converted to call syntax."
            )


def _walk_modules(root: Module) -> Iterator[Module]:
    """Yield ``root`` and every module nested beneath it (its static tree)."""
    yield root
    for child in root.children():
        yield from _walk_modules(child)


# --- Recorder: values over time ---------------------------------------------

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
