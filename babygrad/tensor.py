from __future__ import annotations

import math
from collections import defaultdict
from typing import overload

from . import autograd, formatting, ops, types
from .types import NodeKind


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
        data: list[types.Number],
        shape: types.Shape,
    ) -> None:
        assert len(data) == math.prod(shape), "Tensor data has incorrect shape"
        self.data = data
        self.shape = shape
        self.grad = [0.0 for _ in self.data]
        self.producer: ops.Op | None = None
        self.name: str | None = None
        # role in the graph; whoever creates the tensor for a purpose stamps it
        self.kind: NodeKind | None = None
        # layer the tensor belongs to, for graph clustering (None = outside any)
        self.scope: str | None = None

    def __repr__(self) -> str:
        """Return an aligned matrix-style preview of the tensor contents."""
        if self.ndim == 1:
            return f"{self.shape[0]} items\n{formatting.vector(self.data)}"
        if self.ndim != 2:
            return f"shape={self.shape}\n{formatting.vector(self.data)}"
        return f"{self.nrow} rows x {self.ncol} cols\n{formatting.matrix(self.data, self.nrow, self.ncol)}"

    def __len__(self) -> int:
        return self.nrow

    def __eq__(self, t):
        if not isinstance(t, Tensor):
            return NotImplemented
        equal_content = self.data == t.data
        equal_shape = self.shape == t.shape
        return equal_content and equal_shape

    def __iter__(self):
        if self.ndim == 1:
            for value in self.data:
                yield value
        else:
            for i in range(self.nrow):
                yield Tensor(self._get_row_data(i), shape=(self.ncol,))

    @overload
    def __getitem__(self, key: int) -> types.Number | Tensor: ...
    @overload
    def __getitem__(self, key: tuple[int, ...]) -> types.Number: ...
    @overload
    def __getitem__(self, key: slice) -> Tensor: ...
    def __getitem__(self, key):
        """Get item at position in tensor.

        Cases:
        Slice: always over rows (for batching)
        1D tensor, int key: return value by index
        2D tensor, int key: return row vector at that index
        2D tensor, tuple key: get value at that row:col position using offset
        """
        if isinstance(key, slice):
            start, end, step = key.indices(self.nrow)
            data = []
            for r in range(start, end, step):
                data.extend(self._get_row_data(r))
            return Tensor(data, (len(range(start, end, step)), self.ncol))

        if isinstance(key, int):
            norm_key = (key,)
        elif isinstance(key, tuple):
            norm_key = key
        else:
            raise ValueError("Unsupported index type")

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
                return Tensor(self._get_row_data(row), shape=(self.ncol,))
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

    def _get_row_data(self, idx: int) -> list[types.Number]:
        """
        Return an entire row of raw data by index
        """
        return self.data[idx * self.ncol : idx * self.ncol + self.ncol]

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

    def __add__(self, t: Tensor) -> Tensor:
        return ops.Add([self, t]).forward()

    def __sub__(self, t: Tensor) -> Tensor:
        return ops.Sub([self, t]).forward()

    def __abs__(self) -> Tensor:
        return ops.Abs([self]).forward()

    def __neg__(self) -> Tensor:
        return ops.Neg([self]).forward()

    def __pow__(self, exponent: types.Number) -> Tensor:
        return ops.Pow([self], exponent).forward()

    def __truediv__(self, t: Tensor) -> Tensor:
        return ops.Div([self, t]).forward()

    def __mul__(self, t: Tensor) -> Tensor:
        return ops.Mul([self, t]).forward()

    def __matmul__(self, t: Tensor) -> Tensor:
        return ops.MatMul([self, t]).forward()

    def log(self):
        return ops.Log([self]).forward()

    def exp(self):
        return ops.Exp([self]).forward()

    def sqrt(self):
        return ops.Sqrt([self]).forward()

    def copy(self):
        return ops.Copy([self]).forward()

    def reshape(self, shape: types.Shape):
        return ops.Reshape([self], shape).forward()

    def flatten(self):
        return ops.Flatten([self]).forward()

    def transpose(self):
        return ops.Transpose([self]).forward()

    def t(self):
        return self.transpose()

    def sum(self, axis: int | None = None):
        return ops.Sum([self], axis).forward()

    def mean(self, axis: int | None = None):
        return ops.Mean([self], axis).forward()

    def max(self, axis: int | None = None):
        return ops.Max([self], axis).forward()

    def min(self, axis: int | None = None):
        return ops.Min([self], axis).forward()
