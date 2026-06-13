from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from . import lib

if TYPE_CHECKING:
    from .tensor import Tensor


def propagate_same_shape(
    parents: list[Tensor],
    output: Tensor,
    gradient_rules: Callable,
) -> None:
    """
    Propagates gradients to parent that has the same shape as the output.
    Mutates parents.
    """
    for idx, parent in enumerate(parents):
        output_shaped_grad = []
        for i in range(len(output.grad)):
            grad = gradient_rules(i, output.grad[i])[idx]
            output_shaped_grad.append(grad)
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
    if tensor.producer is None:
        return

    # only send back gradient information once all
    # children grads have been received
    if child_counts[id(tensor)] == 0:
        tensor.producer.backward()
        for p in tensor.producer.inputs:
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

    if tensor.producer:
        for p in tensor.producer.inputs:
            counts[id(p)] += 1
            count_children(p, visited, counts)
