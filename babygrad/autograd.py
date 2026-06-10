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
    propagate_to_parents: Callable[[], None],
) -> Tensor:
    """
    Attach backprop metadata to an output tensor using a prebuilt propagate closure.

    The single funnel through which every op joins the computational graph;
    BackpropMetadata is constructed nowhere else.
    """
    output.backprop = BackpropMetadata(
        op=label,
        parents=parents,
        propagate_to_parents=propagate_to_parents,
    )
    return output


def attach_same_shape(
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

    return attach_backprop_metadata(
        label,
        parents=parents,
        output=output,
        propagate_to_parents=lambda: propagate_same_shape(
            parents, output, gradient_rules
        ),
    )


def attach_spread(
    label: str,
    parent: Tensor,
    output: Tensor,
    groups: list[list[int]],
    gradient_rule: Callable,
) -> Tensor:
    """
    Attach backprop and a grouped gradient rule to a reduction output
    """
    return attach_backprop_metadata(
        label,
        parents=[parent],
        output=output,
        propagate_to_parents=lambda: propagate_spread(
            parent, output, groups, gradient_rule
        ),
    )


def propagate_same_shape(
    parents: list[Tensor],
    output: Tensor,
    gradient_rules: list[Callable[[int, aliases.Number], aliases.Number]],
) -> None:
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


def propagate_spread(
    parent: Tensor,
    output: Tensor,
    groups: list[list[int]],
    gradient_rule: Callable,
) -> None:
    """
    Propagates gradients by spreading across multiple inputs.
    Requires additional context about group breakdown.
    Mutates parent.
    """
    for idx_group, (grad, group) in enumerate(zip(output.grad, groups)):
        group_gradients = gradient_rule(idx_group, parent, output, grad, group)
        for p, g in zip(group, group_gradients):
            parent.grad[p] += g


def backward_walk(tensor: Tensor, child_counts: dict[int, int]) -> None:
    """
    Pass the current gradient back to parents to accumulate.
    Only propagate after gradients from all children have been received (count == 0)
    """
    if tensor.backprop is None:
        return

    # only send back gradient information once all
    # children grads have been received
    if child_counts[id(tensor)] == 0:
        tensor.backprop.propagate_to_parents()
        for p in tensor.backprop.parents:
            child_counts[id(p)] -= 1
            backward_walk(p, child_counts)


def count_children(tensor: Tensor, visited: set[int], counts: dict[int, int]) -> None:
    """
    Count the number of children/consumers a tensor has.
    During backprop we need to know whether a tensor has accumulated gradients from all its children
    before passing the accumulation back.
    """
    if id(tensor) in visited:
        return

    visited.add(id(tensor))

    if tensor.backprop:
        for p in tensor.backprop.parents:
            counts[id(p)] += 1
            count_children(p, visited, counts)
