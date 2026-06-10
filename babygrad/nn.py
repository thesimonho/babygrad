import math
from abc import ABC, abstractmethod
from random import Random
from babygrad.aliases import Shape, Number
from babygrad.tensor import Tensor
from . import autograd
from dataclasses import dataclass


def _init_weights(shape: Shape, seed: int = 42) -> list[Number]:
    rng = Random(seed)
    weights = [rng.uniform(-0.1, 0.1) for _ in range(math.prod([shape[0], shape[1]]))]
    return weights


@dataclass
class Trace:
    data: list[Number]
    shape: Shape


class Sequential:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.trace: dict[str, Trace] = {}

    def forward(self, x: Tensor, plot: bool = False):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if plot:
                self.trace[f"{layer.name}_{i}"] = Trace(data=x.data, shape=x.shape)

        return x, self.trace if plot else None


class Layer(ABC):
    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size):
        weights = _init_weights((input_size, output_size))
        self.weights = Tensor(weights, shape=(input_size, output_size))

        # add bias for each output column
        bias = _init_weights((1, output_size))
        self.bias = Tensor(bias, shape=(1, output_size))

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weights + self.bias


class ReLU(Layer):
    def forward(self, input: Tensor) -> Tensor:
        output = Tensor([max(0, x) for x in input.data], shape=input.shape)

        # attach backprop data so the layer can participate in autograd
        return autograd.attach_same_shape(
            label="ReLU",
            parents=[input],
            output=output,
            gradient_rules=[lambda i, grad: grad if input.data[i] > 0 else 0],
        )


class Softmax(Layer):
    def forward(self, input: Tensor) -> Tensor:
        """
        Softmax row = exp(row - row_max) / sum(exp(row - row_max))

        Composition using tensor methods means the output here already
        contains autograd backprop data.
        """
        z = input - input.max(axis=1)
        exps = z.exp()
        row = exps / exps.sum(axis=1)
        return row


def CCE(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Categorical cross-entropy for one hot targets
    """
    return -(y_true * y_pred.log()).sum(axis=1).mean()


def MSE(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Mean squared error for scalar targets
    """
    return ((y_true - y_pred) ** 2).mean()
