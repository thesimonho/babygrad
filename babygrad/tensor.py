import math
from . import ops
from . import aliases


class Tensor:
    """
    A tensor is a multidimensional array of numbers represented as a list with a shape.

    Row major order: [row1, row2, row3, ...]
    Last 2 indices are (..., row, col)
    """

    def __init__(self, data: list[aliases.Number], shape: aliases.Shape) -> None:
        assert len(data) == math.prod(shape), "Tensor data has incorrect shape"
        self.data = data
        self.shape = shape

    def __eq__(self, t):
        if not isinstance(t, Tensor):
            return NotImplemented
        equal_content = self.data == t.data
        equal_shape = self.shape == t.shape
        return equal_content and equal_shape

    def __add__(self, t: Tensor) -> Tensor:
        if self.shape != t.shape:
            raise ValueError("Shape mismatch. Requires matching shapes.")

        return Tensor(ops.add(self.data, t.data), shape=self.shape)

    def __sub__(self, t: Tensor) -> Tensor:
        if self.shape != t.shape:
            raise ValueError("Shape mismatch. Requires matching shapes.")

        return Tensor(ops.sub(self.data, t.data), shape=self.shape)

    def __mul__(self, t: Tensor) -> Tensor:
        if self.shape != t.shape:
            raise ValueError("Shape mismatch. Requires matching shapes.")

        return Tensor(ops.mul(self.data, t.data), shape=self.shape)

    def __matmul__(self, t: Tensor) -> Tensor:
        # inners must match
        if self.shape[-1] != t.shape[-2]:
            raise ValueError("Dimension mismatch.")

        # output shape = outers
        shape = (self.shape[-2], t.shape[-1])
        data = ops.matmul(self.data, t.data, self.shape, t.shape)

        return Tensor(
            data,
            shape=shape,
        )

    def transpose(self):
        if len(self.shape) != 2:
            raise ValueError("Requires a 2D tensor")

        output = []
        for c in range(self.shape[-1]):
            for r in range(self.shape[-2]):
                output.append(self.data[r * self.shape[-1] + c])
        return Tensor(output, (self.shape[-1], self.shape[-2]))

    def t(self):
        return self.transpose()
