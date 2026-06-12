import math
from abc import ABC, abstractmethod
from random import Random
from typing import Optional

from babygrad.aliases import Number, Report, Shape
from babygrad.observer import Observer
from babygrad.tensor import Tensor

from . import ops


def _init_weights(shape: Shape, seed: int = 42) -> list[Number]:
    rng = Random(seed)
    weights = [rng.uniform(-0.1, 0.1) for _ in range(math.prod([shape[0], shape[1]]))]
    return weights


class Sequential:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def parameters(self) -> list[Tensor]:
        parameter_layers = []
        for layer in self.layers:
            parameter_layers.extend(layer.parameters())

        return parameter_layers

    def forward(self, x: Tensor, observer: Optional[Observer] = None) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if observer:
                # fan-out: relative tags from the report become full,
                # layer-namespaced history tags ("Linear_0/weights")
                for relative_tag, value in layer.report(x).items():
                    observer.record(f"{layer.name}_{i}/{relative_tag}", value)

        return x

    def report_grads(self, observer: Observer) -> None:
        """Record each layer's gradient report under namespaced tags.

        Gradients are only meaningful between backward() and the next
        zero_grad(), so the training loop must call this after backward()
        rather than letting forward() record them (it runs while gradients
        are freshly zeroed).
        """
        for i, layer in enumerate(self.layers):
            for relative_tag, value in layer.report_grad().items():
                observer.record(f"{layer.name}_{i}/{relative_tag}", value)


class Optimizer(ABC):
    def __init__(self, parameters: list[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for p in self.parameters:
            for i in range(len(p.grad)):
                p.grad[i] = 0.0


class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            assert len(p.grad) == len(p.data)
            for i in range(len(p.grad)):
                p.data[i] -= p.grad[i] * self.lr


class Layer(ABC):
    """
    Layer outputs compose via op Nodes, which give them backprop data
    and edge data for the graph
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        pass

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        pass

    def report(self, result: Tensor) -> Report:
        return {}

    def report_grad(self) -> Report:
        return {}


class Linear(Layer):
    def __init__(self, input_size, output_size):
        weights = _init_weights((input_size, output_size))
        self.weights = Tensor(weights, shape=(input_size, output_size))

        # add bias for each output column
        bias = _init_weights((1, output_size))
        self.bias = Tensor(bias, shape=(1, output_size))

    def parameters(self):
        return [self.bias, self.weights]

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weights + self.bias

    def report(self, result) -> Report:
        return {"result": result.copy().data, "weights": self.weights.copy().data}

    def report_grad(self) -> Report:
        return {"grad": self.weights.grad.copy()}


class ReLU(Layer):
    def parameters(self):
        return []

    def forward(self, input: Tensor) -> Tensor:
        return ops.ReLU([input]).forward()


class Softmax(Layer):
    def parameters(self):
        return []

    def forward(self, input: Tensor) -> Tensor:
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
