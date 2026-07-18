"""The two observers of a model run.

- ``Tracer`` watches module/loss brackets during a forward pass, collecting the
  raw ``(module, inputs, output)`` records. ``build_scope_tree`` turns those
  records into a labelled scope tree (record-only, no graph walk); ``attribution``
  adds the graph-walking node->scope map on top for the graph views.
- ``Recorder`` collects the tagged time-series of values — parameter and
  layer-output data and gradients — across training steps.

Both observe the same model without it knowing: the Tracer captures *structure*,
the Recorder captures *values over time*.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from babygrad.types import History, HistoryValue, NodeKind, Scope, Step, Tag

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


def build_scope_tree(records: list[TraceRecord]) -> dict[int, Scope]:
    """One Scope per bracketed module, keyed by ``id(module)``, linked to its
    outer scope. Sibling indices come from exit order, which visits siblings in
    declaration order; ids are full paths so they stay unique and readable.

    Derived from the records alone — no autograd graph walk — so both the recorder
    (labels for its value tags) and the graph attribution (as the tree the node
    walk hangs off) can share it. Lives here, with the Tracer, so recording values
    never has to reach into the graphing layer.
    """
    labels: dict[int, str] = {}
    outers: dict[int, Traceable | None] = {}
    counts: dict[int | None, int] = {}
    for record in records:
        outer_key = id(record.outer) if record.outer is not None else None
        index = counts.get(outer_key, 0)
        counts[outer_key] = index + 1
        labels[id(record.module)] = f"{type(record.module).__name__}_{index}"
        outers[id(record.module)] = record.outer

    scopes: dict[int, Scope] = {}

    def resolve(module: Traceable) -> Scope:
        existing = scopes.get(id(module))
        if existing is not None:
            return existing
        label = labels[id(module)]
        outer = outers[id(module)]
        if outer is None:
            scope = Scope(id=label, label=label, outer_scope=None, collapsed=module.collapse)
        else:
            outer_scope = resolve(outer)
            scope = Scope(
                id=f"{outer_scope.id}/{label}",
                label=label,
                outer_scope=outer_scope.id,
                collapsed=module.collapse,
            )
        scopes[id(module)] = scope
        return scope

    for record in records:
        resolve(record.module)
    return scopes


# --- Recorder: values over time ---------------------------------------------


class Recorder:
    """Collects tagged training data, keyed by the loop-supplied step."""

    def __init__(self):
        self.step: Step = 0
        self.history: History = defaultdict(dict)

    def record(self, tag: Tag, value: HistoryValue):
        self.history[tag][self.step] = value

    def capture(self, tracer: Tracer) -> None:
        """Record parameter and layer-output values from a traced forward.

        Consumes the tracer's records instead of walking the graph: the scope tree
        gives each module its label, and each record hands over the module (for its
        parameters) and its output tensor directly. Tensors are deduped by identity
        in exit order, so a container whose output *is* its child's output does not
        re-tag it under the container's name — the producing leaf claims it first.

        Each recorded tensor contributes its data under "<label>/<role>" and its
        gradient under "<label>/<role>/grad". Values are copied because parameters
        mutate in place and gradients are zeroed between steps; call after
        backward() and before the next zero_grad() for gradients to be meaningful.
        """
        tree = build_scope_tree(tracer.records)
        recorded: set[int] = set()
        for record in tracer.records:
            label = tree[id(record.module)].label
            self._record_parameters(record.module, label, recorded)
            self._record_output(record.output, label, recorded)

    def _record_parameters(
        self, module: Traceable, label: str, recorded: set[int]
    ) -> None:
        """Record each of the module's own parameters as "<label>/<role>", where
        role is the parameter's own name (its scope prefix, if any, is dropped)."""
        own_parameters = getattr(module, "own_parameters", None)
        if own_parameters is None:
            return
        for param in own_parameters():
            if id(param) in recorded:
                continue
            recorded.add(id(param))
            role = param.name.split("/")[-1] if param.name else "parameter"
            self._record_series(f"{label}/{role}", param)

    def _record_output(self, output: Tensor, label: str, recorded: set[int]) -> None:
        """Record a module's output as "<label>/result", but only a real layer
        boundary — the loss output is not one, and a container's output was already
        claimed by the leaf that produced it."""
        if output.kind is not NodeKind.LAYER_OUTPUT:
            return
        if id(output) in recorded:
            return
        recorded.add(id(output))
        self._record_series(f"{label}/result", output)

    def _record_series(self, tag: Tag, tensor: Tensor) -> None:
        self.record(tag, list(tensor.data))
        self.record(f"{tag}/grad", list(tensor.grad))
