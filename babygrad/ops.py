from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

from babygrad import autograd, kernels, lib, types
from babygrad.types import NodeKind

if TYPE_CHECKING:
    from babygrad.tensor import Tensor


class Op(ABC):
    """
    Glue together tensors via an operation.
    Separate class so it can cleanly be expressed as a node in the graph.
    """

    label: ClassVar[str]
    kind: ClassVar[NodeKind] = NodeKind.OP

    def __init__(self, inputs: list[Tensor]):
        self.inputs = inputs

    def forward(self) -> Tensor:
        """
        Calculate the output value and create the resulting Tensor.
        """
        from babygrad.tensor import Tensor

        data, shape = self.compute()
        self.output = Tensor(data, shape, kind=NodeKind.OP_RESULT)
        self.output.producer = self
        return self.output

    @abstractmethod
    def backward(self) -> None:
        """
        Get gradient from the Tensor created by forward(). Send the gradient back to the parents stored as inputs to this Op.
        """
        pass

    @abstractmethod
    def compute(self) -> tuple[list[types.Number], types.Shape]:
        """
        Raw op kernel for calculating the output value.
        """
        pass


class BinaryOp(Op):
    """
    Used to give nicer names to the inputs of a binary op.
    """

    @property
    def a(self):
        return self.inputs[0]

    @property
    def b(self):
        return self.inputs[1]


class UnaryOp(Op):
    """
    Used to give nicer names to the input of a unary op.
    """

    @property
    def x(self):
        return self.inputs[0]


class MatMul(BinaryOp):
    """
    Calculate the dot product of two tensors.

    Shape of a tensor is always the last 2 dimensions. Any other dimensions are iterated dimensions: (batch, depth, row, col)
    """

    label = "@"

    def compute(self):
        # inners must match
        if self.a.ncol != self.b.nrow:
            raise ValueError("Dimension mismatch.")

        output = kernels.matmul(self.a.data, self.b.data, self.a.shape, self.b.shape)
        return output, (self.a.nrow, self.b.ncol)

    def backward(self) -> None:
        """
        A = g@B.t
        B = A.t@g
        """
        At, At_shape = lib.transpose_flat_data(self.a.data, self.a.shape)
        Bt, Bt_shape = lib.transpose_flat_data(self.b.data, self.b.shape)
        A_grad = kernels.matmul(self.output.grad, Bt, self.output.shape, Bt_shape)
        B_grad = kernels.matmul(At, self.output.grad, At_shape, self.output.shape)

        for i in range(len(self.a.grad)):
            self.a.grad[i] += A_grad[i]

        for i in range(len(self.b.grad)):
            self.b.grad[i] += B_grad[i]


class Transpose(UnaryOp):
    label = "transpose"

    def compute(self):
        data, shape = lib.transpose_flat_data(self.x.data, self.x.shape)
        return data, shape

    def backward(self):
        parent_grad, parent_shape = lib.transpose_flat_data(
            self.output.grad, self.output.shape
        )
        assert parent_shape == self.x.shape

        for i in range(len(self.x.grad)):
            self.x.grad[i] += parent_grad[i]


class Copy(UnaryOp):
    label = "copy"

    def compute(self):
        return list(self.x.data), self.x.shape

    def backward(self):
        for i in range(len(self.x.grad)):
            self.x.grad[i] += self.output.grad[i]


class Reshape(UnaryOp):
    label = "reshape"

    def __init__(self, inputs, target_shape):
        super().__init__(inputs)
        self.target_shape = target_shape

    def compute(self):
        if math.prod(self.x.shape) != math.prod(self.target_shape):
            raise ValueError("New shape must contain the same number of items")
        return list(self.x.data), self.target_shape

    def backward(self):
        for i in range(len(self.x.grad)):
            self.x.grad[i] += self.output.grad[i]


class Flatten(Reshape):
    label = "flatten"

    def __init__(self, inputs):
        target_shape = (len(inputs[0].data),)
        super().__init__(inputs, target_shape)


class Reduce(UnaryOp):
    """
    Base node to handle all the plumbing for reduce ops eg. spreading gradient back across multiple input groups
    """

    op: ClassVar[Callable]

    def __init__(self, inputs, axis: Optional[int]):
        super().__init__(inputs)
        self.axis = axis

    def compute(self):
        if self.axis is not None:
            # normalize negative axis
            if self.axis < 0:
                self.axis = self.x.ndim + self.axis

            self.groups = lib._get_axis_groups(self.x.shape, axis=self.axis)

            # set the target axis to size 1
            reduced = [
                self.op([self.x.data[i] for i in group]) for group in self.groups
            ]
            shape = tuple(
                1 if i == self.axis else x for i, x in enumerate(self.x.shape)
            )
        else:
            self.groups = [[x for x in range(len(self.x.data))]]
            reduced = [self.op(self.x.data)]
            shape = tuple(1 for _ in range(len(self.x.shape)))

        return reduced, shape

    def backward(self):
        autograd.propagate_spread(self.x, self.output, self.groups, self.gradient_rule)

    @abstractmethod
    def gradient_rule(self, i, parent, output, grad, group) -> list[types.Number]:
        pass


class Sum(Reduce):
    label = "Σ"
    op = staticmethod(kernels.reduce_sum)

    def gradient_rule(self, i, parent, output, grad, group):
        return [grad] * len(group)


class Mean(Reduce):
    label = "μ"
    op = staticmethod(kernels.reduce_mean)

    def gradient_rule(self, i, parent, output, grad, group):
        return [grad / len(group)] * len(group)


class Max(Reduce):
    label = "max"
    op = staticmethod(kernels.reduce_max)

    def gradient_rule(self, i, parent, output, grad, group):
        value = output.data[i]
        winners = [True if parent.data[g] == value else False for g in group]
        n_winners = sum(winners)

        # equally divide the gradient between tied winners
        return [grad * 1 / n_winners if w is True else 0 for w in winners]


class Min(Reduce):
    label = "min"
    op = staticmethod(kernels.reduce_min)

    def gradient_rule(self, i, parent, output, grad, group):
        value = output.data[i]
        winners = [True if parent.data[g] == value else False for g in group]
        n_winners = sum(winners)

        # equally divide the gradient between tied winners
        return [grad * 1 / n_winners if w is True else 0 for w in winners]


class Elementwise(Op):
    """
    Base node to handle all the plumbing for elementwise ops eg. broadcasting and unbroadcasting
    """

    op: ClassVar[Callable]

    def __init__(self, inputs):
        super().__init__(inputs)
        self.operands = {}

    def compute(self):
        if len(self.inputs) == 2:
            a = self.inputs[0]
            b = self.inputs[1]
            left, right, shape = lib.broadcast(a.data, b.data, a.shape, b.shape)
            self.operands = {"left": left, "right": right}
            return self.op(left, right), shape
        else:
            x = self.inputs[0]
            self.operands = {"x": x.data}
            return self.op(x.data), x.shape

    def backward(self):
        autograd.propagate_same_shape(self.inputs, self.output, self.gradient_rules)

    @abstractmethod
    def gradient_rules(self, i, grad) -> list[types.Number]:
        pass


class Add(BinaryOp, Elementwise):
    label = "+"
    op = staticmethod(kernels.add)

    def gradient_rules(self, i, grad):
        return [grad, grad]


class Sub(BinaryOp, Elementwise):
    label = "-"
    op = staticmethod(kernels.sub)

    def gradient_rules(self, i, grad):
        return [grad * 1, grad * -1]


class Div(BinaryOp, Elementwise):
    label = "/"
    op = staticmethod(kernels.div)

    def gradient_rules(self, i, grad):
        """
        c = a/b = a*1/b = a * 1/(4). dc/da: 1/4 = 1/b
        c = ab^-1 = (4)/b = 4b^-1. dc/db: -4b^-2 = -4/b^2 = -a/b^2
        """
        return [
            grad * 1 / self.operands["right"][i],
            grad * -self.operands["left"][i] / self.operands["right"][i] ** 2,
        ]


class Mul(BinaryOp, Elementwise):
    label = "*"
    op = staticmethod(kernels.mul)

    def gradient_rules(self, i, grad):
        return [
            grad * self.operands["right"][i],
            grad * self.operands["left"][i],
        ]


class Abs(UnaryOp, Elementwise):
    label = "abs"
    op = staticmethod(kernels.absolute)

    def gradient_rules(self, i, grad):
        # derivative is 1 for x>0, -1 for x<0, undefined at 0
        local = (
            1 if self.operands["x"][i] > 0 else -1 if self.operands["x"][i] < 0 else 0
        )
        return [grad * local]


class Neg(UnaryOp, Elementwise):
    label = "Neg"
    op = staticmethod(kernels.neg)

    def gradient_rules(self, i, grad):
        return [grad * -1]


class Pow(UnaryOp, Elementwise):
    label = "^"
    op = staticmethod(kernels.power)

    def __init__(self, inputs, exponent: types.Number):
        super().__init__(inputs)
        self.exponent = exponent

    def compute(self):
        x = self.inputs[0]
        self.operands = {"x": x.data}
        return self.op(self.operands["x"], self.exponent), x.shape

    def gradient_rules(self, i, grad):
        return [grad * self.exponent * self.operands["x"][i] ** (self.exponent - 1)]


class Log(UnaryOp, Elementwise):
    label = "log"
    op = staticmethod(kernels.log)

    def gradient_rules(self, i, grad):
        return [grad * (1 / self.operands["x"][i])]


class Exp(UnaryOp, Elementwise):
    label = "exp"
    op = staticmethod(kernels.exp)

    def gradient_rules(self, i, grad):
        """For exp, the derivative is itself (the output)"""
        return [grad * self.output.data[i]]


class Sqrt(UnaryOp, Elementwise):
    label = "√"
    op = staticmethod(kernels.sqrt)

    def gradient_rules(self, i, grad):
        return [grad * 0.5 * self.operands["x"][i] ** -0.5]


class ReLU(UnaryOp, Elementwise):
    label = "ReLU"
    op = staticmethod(kernels.rectify)

    def gradient_rules(self, i, grad):
        return [grad if self.operands["x"][i] > 0 else 0]


class Sigmoid(UnaryOp, Elementwise):
    label = "sigmoid"
    op = staticmethod(kernels.sigmoid)

    def gradient_rules(self, i, grad):
        return [grad * self.output.data[i] * (1 - self.output.data[i])]


class Tanh(UnaryOp, Elementwise):
    label = "tanh"
    op = staticmethod(kernels.tanh)

    def gradient_rules(self, i, grad):
        return [grad * (1 - self.output.data[i] ** 2)]
