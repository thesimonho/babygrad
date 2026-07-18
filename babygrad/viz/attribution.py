"""Turn a tracer's bracket records into the two graphing outputs:

- the **scope tree**, from bracket *nesting* (so empty containers survive), and
- the **node->scope map**, from attributing each op to the innermost module whose
  forward produced it.

Structure and attribution are deliberately separate: the tree comes from who
bracketed whom, the map from walking the autograd graph. Building the tree from
attribution alone would drop pass-through containers, whose output is their
child's output and which therefore claim no ops of their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from babygrad.ops import Op
from babygrad.tensor import Tensor
from babygrad.types import Scope

if TYPE_CHECKING:
    from babygrad.observers import TraceRecord
    from babygrad.tracing import Traceable


@dataclass
class TraceResult:
    """The graphing view of one traced forward.

    ``scopes`` is the full containment tree keyed by scope id — empty containers
    included, since it comes from bracket nesting. ``node_scope`` maps a node's
    ``id()`` to the scope that owns it; nodes with no scope are simply absent.
    """

    scopes: dict[str, Scope]
    node_scope: dict[int, Scope]


def attribute(records: list[TraceRecord]) -> TraceResult:
    """Derive the scope tree and node->scope map from a tracer's records."""
    by_module = _build_scope_tree(records)
    node_scope = _attribute_nodes(records, by_module)
    return TraceResult(
        scopes={scope.id: scope for scope in by_module.values()},
        node_scope=node_scope,
    )


def _build_scope_tree(records: list[TraceRecord]) -> dict[int, Scope]:
    """One Scope per bracketed module, keyed by ``id(module)``, linked to its
    outer scope. Sibling indices come from exit order, which visits siblings in
    declaration order; ids are full paths so they stay unique and readable."""
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


def _attribute_nodes(
    records: list[TraceRecord], by_module: dict[int, Scope]
) -> dict[int, Scope]:
    """Map each node to its owning scope. Records are in exit order (innermost
    first), so an inner module claims its ops before its container walks."""
    node_scope: dict[int, Scope] = {}
    for record in records:
        scope = by_module[id(record.module)]
        _claim_own_parameters(record.module, scope, node_scope)
        boundary = {id(inp) for inp in record.inputs}
        _claim_reachable(record.output, scope, boundary, node_scope)
    _tag_entry_inputs(records, by_module, node_scope)
    return node_scope


def _claim_own_parameters(
    module: Traceable, scope: Scope, node_scope: dict[int, Scope]
) -> None:
    """A module owns its parameters directly — robust even if a param is consumed
    by a sub-op that a different walk might otherwise reach first."""
    own_parameters = getattr(module, "own_parameters", None)
    if own_parameters is None:
        return
    for param in own_parameters():
        node_scope.setdefault(id(param), scope)


def _claim_reachable(
    start: Tensor, scope: Scope, boundary: set[int], node_scope: dict[int, Scope]
) -> None:
    """Claim every node backward-reachable from ``start`` for ``scope``, halting
    at (a) a node an inner module already claimed, (b) one of this module's input
    boundaries, or (c) a leaf (no predecessors)."""
    stack: list[Tensor | Op] = [start]
    while stack:
        node = stack.pop()
        if id(node) in node_scope:
            continue
        if id(node) in boundary:
            continue
        node_scope[id(node)] = scope
        stack.extend(_predecessors(node))


def _predecessors(node: Tensor | Op) -> list[Tensor | Op]:
    """The nodes feeding ``node`` in the bipartite graph: a tensor's producing op,
    or an op's input tensors."""
    if isinstance(node, Tensor):
        return [node.producer] if node.producer is not None else []
    if isinstance(node, Op):
        return list(node.inputs)
    return []


def _tag_entry_inputs(
    records: list[TraceRecord],
    by_module: dict[int, Scope],
    node_scope: dict[int, Scope],
) -> None:
    """The model root's input leaves are the graph entry and belong to it. The
    root is the top-level module that owns children; the loss, also top-level, has
    none, so its inputs (the target) stay unattributed."""
    for record in records:
        if record.outer is None and hasattr(record.module, "children"):
            scope = by_module[id(record.module)]
            for inp in record.inputs:
                node_scope.setdefault(id(inp), scope)
