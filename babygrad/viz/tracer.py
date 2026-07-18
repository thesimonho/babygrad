"""The Tracer watches module/loss brackets during a forward pass. For now it
only collects the raw records; the scope tree and node->scope attribution build
on these later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from babygrad.tensor import Tensor
    from babygrad.tracing import Traceable


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
