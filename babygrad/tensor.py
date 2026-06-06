from __future__ import annotations
from dataclasses import dataclass

import math

from . import aliases, ops, lib, formatting
from typing import Callable, Optional, Union


@dataclass
class BackpropMetadata:
    op: str
    parents: list[Tensor]
    propagate_to_parents: Callable


class Tensor:
    """
    A tensor is a multidimensional array of numbers represented as a list with a shape.

    Row major order: [row1, row2, row3, ...]
    Last 2 indices are (..., row, col)

    Compute ops return rank = max of input ranks. (Covers elementwise, broadcast, matmul, and reductions: one input, max = its own rank, preserved.)

    Shape ops change rank — that's what they're for (reshape, flatten, indexing.)
    """

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def ncol(self) -> int:
        return self.shape[-1]

    @property
    def nrow(self) -> int:
        return self.shape[-2]

    @property
    def rank(self) -> int:
        return self.ndim

    @property
    def numel(self) -> int:
        return len(self.data)

    def __init__(
        self,
        data: list[aliases.Number],
        shape: aliases.Shape,
        backprop: Union[BackpropMetadata, None] = None,
    ) -> None:
        assert len(data) == math.prod(shape), "Tensor data has incorrect shape"
        self.data = data
        self.shape = shape
        self.grad = [0.0 for _ in self.data]
        self.backprop = backprop

    def __repr__(self) -> str:
        """Return an aligned matrix-style preview of the tensor contents."""
        if self.ndim == 1:
            return f"{self.shape[0]} items\n{formatting.vector(self.data)}"
        if self.ndim != 2:
            return f"shape={self.shape}\n{formatting.vector(self.data)}"
        return f"{self.nrow} rows x {self.ncol} cols\n{formatting.matrix(self.data, self.nrow, self.ncol)}"

    def _inject_backprop_metadata(
        self,
        label: str,
        parents: list[Tensor],
        output: Tensor,
        gradient_rules: list[Callable[[int, aliases.Number], aliases.Number]],
    ) -> Tensor:
        assert len(parents) == len(gradient_rules), (
            "each parent must have a gradient update rule"
        )

        def propagate():
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

        output.backprop = BackpropMetadata(
            op=label, parents=parents, propagate_to_parents=propagate
        )
        return output

    def __eq__(self, t):
        if not isinstance(t, Tensor):
            return NotImplemented
        equal_content = self.data == t.data
        equal_shape = self.shape == t.shape
        return equal_content and equal_shape

    def __getitem__(self, key: int | tuple[int, ...]) -> aliases.Number | Tensor:
        """Get item at position in tensor.

        Cases:
        1D tensor, int key: return value by index
        2D tensor, int key: return row vector at that index
        2D tensor, tuple key: get value at that row:col position using offset
        """
        if isinstance(key, int):
            norm_key = (key,)
        elif isinstance(key, tuple):
            norm_key = key

        # normalize negative indices
        norm_key = tuple(
            k if k >= 0 else (self.shape[i] + k) for i, k in enumerate(norm_key)
        )

        if len(norm_key) == 2 and self.ndim == 2:
            row = norm_key[0]
            col = norm_key[1]
            if row >= 0 and row < self.nrow and col >= 0 and col < self.ncol:
                return self.data[row * self.ncol + col]
            else:
                raise IndexError(
                    f"Index is out of bounds for Tensor of shape {self.shape}"
                )
        elif len(norm_key) == 1 and self.ndim == 2:
            row = norm_key[0]
            if row >= 0 and row < self.nrow:
                data = self.data[row * self.ncol : row * self.ncol + self.ncol]
                return Tensor(data, shape=(self.ncol,))
            else:
                raise IndexError(
                    f"Row index {row} is out of bounds. Row count = {self.nrow}"
                )
        elif len(norm_key) == 1 and self.ndim == 1:
            if norm_key[0] >= 0 and norm_key[0] < self.ncol:
                return self.data[norm_key[0]]
            else:
                raise IndexError(
                    f"Index out of bounds for 1D vector of length {self.ncol}"
                )
        else:
            raise ValueError("Too many passed dimensions")

    def __add__(self, t: Tensor) -> Tensor:
        left, right, shape = lib.broadcast(self.data, t.data, self.shape, t.shape)
        output = Tensor(ops.add(left, right), shape=shape)
        return self._inject_backprop_metadata(
            label="+",
            parents=[self, t],
            output=output,
            gradient_rules=[lambda _, grad: grad * 1, lambda _, grad: grad * 1],
        )

    def __sub__(self, t: Tensor) -> Tensor:
        left, right, shape = lib.broadcast(self.data, t.data, self.shape, t.shape)
        return Tensor(ops.sub(left, right), shape=shape)

    def __abs__(self) -> Tensor:
        return Tensor(ops.absolute(self.data), shape=self.shape)

    def __neg__(self) -> Tensor:
        return Tensor(ops.neg(self.data), shape=self.shape)

    def __pow__(self, exponent: aliases.Number) -> Tensor:
        return Tensor(ops.power(self.data, exponent), shape=self.shape)

    def __truediv__(self, t: Tensor) -> Tensor:
        left, right, shape = lib.broadcast(self.data, t.data, self.shape, t.shape)
        return Tensor(ops.div(left, right), shape=shape)

    def __mul__(self, t: Tensor) -> Tensor:
        left, right, shape = lib.broadcast(self.data, t.data, self.shape, t.shape)
        return Tensor(ops.mul(left, right), shape=shape)

    def __matmul__(self, t: Tensor) -> Tensor:
        # inners must match
        if self.ncol != t.nrow:
            raise ValueError("Dimension mismatch.")

        # output shape = outers
        shape = (self.nrow, t.ncol)
        data = ops.matmul(self.data, t.data, self.shape, t.shape)

        return Tensor(
            data,
            shape=shape,
        )

    def log(self):
        return Tensor(ops.log(self.data), shape=self.shape)

    def exp(self):
        return Tensor(ops.exp(self.data), shape=self.shape)

    def sqrt(self):
        return Tensor(ops.sqrt(self.data), shape=self.shape)

    def copy(self):
        return Tensor(list(self.data), shape=self.shape)

    def reshape(self, shape: aliases.Shape):
        if math.prod(self.shape) != math.prod(shape):
            raise ValueError("New shape must contain the same number of items")
        t = self.copy()
        t.shape = shape
        return t

    def flatten(self):
        return self.reshape((len(self.data),))

    def transpose(self):
        if self.ndim != 2:
            raise ValueError("Requires a 2D tensor")

        output = []
        for c in range(self.ncol):
            for r in range(self.nrow):
                output.append(self.data[r * self.ncol + c])
        return Tensor(output, (self.ncol, self.nrow))

    def t(self):
        return self.transpose()

    def _reduce(self, op: Callable, axis: Optional[int]):
        if axis is not None:
            # normalize negative axis
            if axis < 0:
                axis = self.ndim + axis

            groups = lib._get_axis_groups(self.shape, axis=axis)

            # set the target axis to size 1
            shape = tuple(1 if i == axis else x for i, x in enumerate(self.shape))
            return Tensor(
                [op([self.data[i] for i in group]) for group in groups],
                shape=tuple(shape),
            )
        else:
            shape = tuple(1 for _ in range(len(self.shape)))

        return Tensor([op(self.data)], shape=shape)

    def sum(self, axis: Optional[int] = None):
        return self._reduce(ops.reduce_sum, axis)

    def mean(self, axis: Optional[int] = None):
        return self._reduce(ops.reduce_mean, axis)

    def max(self, axis: Optional[int] = None):
        return self._reduce(ops.reduce_max, axis)

    def min(self, axis: Optional[int] = None):
        return self._reduce(ops.reduce_min, axis)

    def backward(self):
        """
        Start backprop. Called once on the loss tensor.
        """
        # seed gradient for the root/loss tensor at the start of backprop
        self.grad = [1.0 for _ in self.grad]

        # track which tensors have been visited so we dont call propagate multiple times
        # for tensors that are reused.
        visited = set()

        self._backward_walk(visited)

    def _backward_walk(self, visited: set[int]):
        """
        Pass the current gradient back to parents to accumulate the gradients. Visit each parent and walk recursively.
        """
        if self.backprop is None:
            return

        if id(self) in visited:
            return

        self.backprop.propagate_to_parents()
        visited.add(id(self))

        for p in self.backprop.parents:
            p._backward_walk(visited)
