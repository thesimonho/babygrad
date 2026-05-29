import math
from . import ops
from . import aliases


class Tensor:
    """
    A tensor is a multidimensional array of numbers represented as a list with a shape.

    Row major order: [row1, row2, row3, ...]
    Last 2 indices are (..., row, col)
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
        if self.shape != t.shape:
            raise ValueError("Shape mismatch. Requires matching shapes.")

        return Tensor(ops.add(self.data, t.data), shape=self.shape)

    def __sub__(self, t: Tensor) -> Tensor:
        if self.shape != t.shape:
            raise ValueError("Shape mismatch. Requires matching shapes.")

        return Tensor(ops.sub(self.data, t.data), shape=self.shape)

    def __abs__(self) -> Tensor:
        return Tensor(ops.absolute(self.data), shape=self.shape)

    def __neg__(self) -> Tensor:
        return Tensor(ops.neg(self.data), shape=self.shape)

    def __pow__(self, exponent: aliases.Number) -> Tensor:
        return Tensor(ops.power(self.data, exponent), shape=self.shape)

    def __truediv__(self, t: Tensor) -> Tensor:
        if self.shape != t.shape:
            raise ValueError("Shape mismatch. Requires matching shapes.")

        return Tensor(ops.div(self.data, t.data), shape=self.shape)

    def __mul__(self, t: Tensor) -> Tensor:
        if self.shape != t.shape:
            raise ValueError("Shape mismatch. Requires matching shapes.")

        return Tensor(ops.mul(self.data, t.data), shape=self.shape)

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

    def sum(self):
        return Tensor([ops.reduce_sum(self.data)], shape=(1,))

    def mean(self):
        return Tensor([ops.reduce_mean(self.data)], shape=(1,))

    def max(self):
        return Tensor([ops.reduce_max(self.data)], shape=(1,))

    def min(self):
        return Tensor([ops.reduce_min(self.data)], shape=(1,))
