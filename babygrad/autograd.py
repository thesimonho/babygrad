from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING
from . import lib, aliases

if TYPE_CHECKING:
    from .tensor import Tensor


@dataclass
class BackpropMetadata:
    op: str
    parents: list[Tensor]
    propagate_to_parents: Callable


def attach_backprop_metadata(
    label: str,
    parents: list[Tensor],
    output: Tensor,
    gradient_rules: list[Callable[[int, aliases.Number], aliases.Number]],
) -> Tensor:
    """
    Attach backprop and gradient rules to a tensor with same shape parents
    """
    assert len(parents) == len(gradient_rules), (
        "each parent must have a gradient update rule"
    )

    output.backprop = BackpropMetadata(
        op=label,
        parents=parents,
        propagate_to_parents=lambda: propagate_same_shape(
            parents, output, gradient_rules
        ),
    )
    return output


def propagate_same_shape(parents, output, gradient_rules) -> None:
    """
    Propagates gradients to parent that has the same shape as the output.
    Mutates parents.
    """
    for parent, rule in zip(parents, gradient_rules):
        output_shaped_grad = []
        for i in range(len(output.grad)):
            output_shaped_grad.append(rule(i, output.grad[i]))
        assert len(output_shaped_grad) == len(output.grad)

        # parent may have been broadcasted, so we need to undo that so indexing aligns
        parent_shaped_grad = lib.unbroadcast(
            output_shaped_grad, output.shape, parent.shape
        )
        assert len(parent_shaped_grad) == len(parent.grad)
        for j in range(len(parent.grad)):
            parent.grad[j] += parent_shaped_grad[j]

def backward_walk(tensor: Tensor, visited: set[int]) -> None:
    """
    Pass the current gradient back to parents to accumulate the gradients. Visit each parent and walk recursively.
    """
    if tensor.backprop is None:
        return

    if id(tensor) in visited:
        return

    tensor.backprop.propagate_to_parents()
    visited.add(id(tensor))

    for p in tensor.backprop.parents:
        backward_walk(p, visited)
