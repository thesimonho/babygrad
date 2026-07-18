"""The Tracer watches module/loss brackets during a forward pass. For now it
only collects the raw records; the scope tree and node->scope attribution build
on these later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from babygrad.nn.modules import Module
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
