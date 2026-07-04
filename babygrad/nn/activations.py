from babygrad.tensor import Tensor

from babygrad import ops
from babygrad.nn.modules import Module


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return ops.Sigmoid([input]).forward()


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return ops.Tanh([input]).forward()


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return ops.ReLU([input]).forward()


class Softmax(Module):
    def forward(self, input: Tensor) -> Tensor:
        z = input - input.max(axis=1)
        exps = z.exp()
        row = exps / exps.sum(axis=1)
        return row
