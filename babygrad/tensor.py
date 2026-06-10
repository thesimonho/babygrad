from __future__ import annotations

from collections import defaultdict
import math
from typing import Callable, Optional, Union

from . import aliases, autograd, formatting, lib, ops


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
        backprop: Union[autograd.BackpropMetadata, None] = None,
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
        return autograd.attach_same_shape(
            label="+",
            parents=[self, t],
            output=output,
            gradient_rules=[lambda _, grad: grad * 1, lambda _, grad: grad * 1],
        )

    def __sub__(self, t: Tensor) -> Tensor:
        left, right, shape = lib.broadcast(self.data, t.data, self.shape, t.shape)
        output = Tensor(ops.sub(left, right), shape=shape)
        return autograd.attach_same_shape(
            label="-",
            parents=[self, t],
            output=output,
            gradient_rules=[lambda _, grad: grad * 1, lambda _, grad: grad * -1],
        )

    def __abs__(self) -> Tensor:
        return Tensor(ops.absolute(self.data), shape=self.shape)

    def __neg__(self) -> Tensor:
        output = Tensor(ops.neg(self.data), shape=self.shape)
        return autograd.attach_same_shape(
            label="-",
            parents=[self],
            output=output,
            gradient_rules=[lambda _, grad: grad * -1],
        )

    def __pow__(self, exponent: aliases.Number) -> Tensor:
        output = Tensor(ops.power(self.data, exponent), shape=self.shape)
        return autograd.attach_same_shape(
            label="**",
            parents=[self],
            output=output,
            gradient_rules=[
                lambda i, grad: grad * exponent * self.data[i] ** (exponent - 1)
            ],
        )

    def __truediv__(self, t: Tensor) -> Tensor:
        left, right, shape = lib.broadcast(self.data, t.data, self.shape, t.shape)
        output = Tensor(ops.div(left, right), shape=shape)
        # c = a/b = a*1/b = a * 1/(4). dc/da: 1/4 = 1/b
        # c = ab^-1 = (4)/b = 4b^-1. dc/db: -4b^-2 = -4/b^2 = -a/b^2
        return autograd.attach_same_shape(
            label="/",
            parents=[self, t],
            output=output,
            gradient_rules=[
                lambda i, grad: grad * 1 / right[i],
                lambda i, grad: grad * -left[i] / right[i] ** 2,
            ],
        )

    def __mul__(self, t: Tensor) -> Tensor:
        left, right, shape = lib.broadcast(self.data, t.data, self.shape, t.shape)
        output = Tensor(ops.mul(left, right), shape=shape)
        return autograd.attach_same_shape(
            label="*",
            parents=[self, t],
            output=output,
            gradient_rules=[
                lambda i, grad: grad * right[i],
                lambda i, grad: grad * left[i],
            ],
        )

    def __matmul__(self, t: Tensor) -> Tensor:
        # inners must match
        if self.ncol != t.nrow:
            raise ValueError("Dimension mismatch.")

        # output shape = outers
        shape = (self.nrow, t.ncol)
        data = ops.matmul(self.data, t.data, self.shape, t.shape)
        output = Tensor(data, shape=shape)

        def propagate():
            # A = g@B.t
            # B = A.t@g
            At, At_shape = lib.transpose_flat_data(self.data, self.shape)
            Bt, Bt_shape = lib.transpose_flat_data(t.data, t.shape)
            A_grad = ops.matmul(output.grad, Bt, output.shape, Bt_shape)
            B_grad = ops.matmul(At, output.grad, At_shape, output.shape)

            for i in range(len(self.grad)):
                self.grad[i] += A_grad[i]

            for i in range(len(t.grad)):
                t.grad[i] += B_grad[i]

        return autograd.attach_backprop_metadata(
            label="@",
            parents=[self, t],
            output=output,
            propagate_to_parents=propagate,
        )

    def log(self):
        output = Tensor(ops.log(self.data), shape=self.shape)
        return autograd.attach_same_shape(
            label="log",
            parents=[self],
            output=output,
            gradient_rules=[lambda i, grad: grad * (1 / self.data[i])],
        )

    def exp(self):
        """For exp, the derivative is itself (the output)"""
        output = Tensor(ops.exp(self.data), shape=self.shape)
        return autograd.attach_same_shape(
            label="exp",
            parents=[self],
            output=output,
            gradient_rules=[lambda i, grad: grad * output.data[i]],
        )

    def sqrt(self):
        output = Tensor(ops.sqrt(self.data), shape=self.shape)
        return autograd.attach_same_shape(
            label="sqrt",
            parents=[self],
            output=output,
            gradient_rules=[
                lambda i, grad: grad * 0.5 * self.data[i] ** -0.5,
            ],
        )

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
        data, shape = lib.transpose_flat_data(self.data, self.shape)
        output = Tensor(data, shape)

        def propagate():
            parent_grad, parent_shape = lib.transpose_flat_data(
                output.grad, output.shape
            )
            assert parent_shape == self.shape

            for i in range(len(self.grad)):
                self.grad[i] += parent_grad[i]

        return autograd.attach_backprop_metadata(
            label="transpose",
            parents=[self],
            output=output,
            propagate_to_parents=propagate,
        )

    def t(self):
        return self.transpose()

    def _reduce(
        self,
        op: Callable[[list[aliases.Number]], aliases.Number],
        axis: Optional[int],
        gradient_rule: Callable,
    ) -> Tensor:
        if axis is not None:
            # normalize negative axis
            if axis < 0:
                axis = self.ndim + axis

            groups = lib._get_axis_groups(self.shape, axis=axis)

            # set the target axis to size 1
            reduced = [op([self.data[i] for i in group]) for group in groups]
            shape = tuple(1 if i == axis else x for i, x in enumerate(self.shape))
        else:
            groups = [[x for x in range(len(self.data))]]
            reduced = [op(self.data)]
            shape = tuple(1 for _ in range(len(self.shape)))

        output = Tensor(reduced, shape)
        return autograd.attach_spread(
            label=op.__name__.removeprefix("reduce_"),
            parent=self,
            output=output,
            groups=groups,
            gradient_rule=gradient_rule,
        )

    def sum(self, axis: Optional[int] = None):
        output = self._reduce(
            ops.reduce_sum,
            axis,
            lambda i, parent, output, grad, group: [grad] * len(group),
        )
        return output

    def mean(self, axis: Optional[int] = None):
        output = self._reduce(
            ops.reduce_mean,
            axis,
            lambda i, parent, output, grad, group: [grad / len(group)] * len(group),
        )
        return output

    def max(self, axis: Optional[int] = None):
        def rule(i, parent, output, grad, group):
            value = output.data[i]
            winners = [True if parent.data[g] == value else False for g in group]
            n_winners = sum(winners)

            # equally divide the gradient between tied winners
            return [grad * 1 / n_winners if w is True else 0 for w in winners]

        output = self._reduce(ops.reduce_max, axis, rule)
        return output

    def min(self, axis: Optional[int] = None):
        def rule(i, parent, output, grad, group):
            value = output.data[i]
            winners = [True if parent.data[g] == value else False for g in group]
            n_winners = sum(winners)

            # equally divide the gradient between tied winners
            return [grad * 1 / n_winners if w is True else 0 for w in winners]

        output = self._reduce(ops.reduce_min, axis, rule)
        return output

    def backward(self):
        """
        Start backprop. Called once on the loss tensor.
        """
        # seed gradient for the root/loss tensor at the start of backprop
        self.grad = [1.0 for _ in self.grad]

        # seed reference counter for number of children/consumers for a node
        child_counts = defaultdict(int)
        visited = set()
        autograd.count_children(self, visited, child_counts)

        autograd.backward_walk(self, child_counts)
