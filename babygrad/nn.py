import math
from abc import ABC, abstractmethod
from random import Random

from babygrad.tensor import Tensor
from babygrad.types import NodeKind, Number, Shape

from . import ops


class WeightInitializer(ABC):
    def __init__(self, shape: Shape, seed: int = 42):
        self.shape = shape
        self.rng = Random(seed)

    @abstractmethod
    def generate(self) -> list[Number]:
        pass


class Uniform(WeightInitializer):
    def generate(self):
        weights = [self.rng.uniform(-0.1, 0.1) for _ in range(math.prod(self.shape))]
        return weights


class Glorot(WeightInitializer):
    """
    Keep variance independent of the number of inputs so the signal doesn't blow up with large layers.
    """

    def generate(self):
        bounds = math.sqrt(6) / math.sqrt(sum(self.shape))
        weights = [
            self.rng.uniform(-bounds, bounds) for _ in range(math.prod(self.shape))
        ]
        return weights


class He(WeightInitializer):
    """
    Used for ReLU layers. Negative half is zeroed out so we lose half the variance.
    Compensate by double the variance of the initial weights.
    """

    def generate(self):
        mu, sigma = (0, math.sqrt(2 / self.shape[0]))
        weights = [self.rng.gauss(mu, sigma) for _ in range(math.prod(self.shape))]
        return weights


class Sequential:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

        # stamp durable identities once: layers get an indexed name,
        # parameters get that name as their prefix ("Linear_0/weights")
        for i, layer in enumerate(layers):
            layer.name = f"{layer.name}_{i}"
            for parameter in layer.parameters():
                parameter.name = f"{layer.name}/{parameter.name}"
                parameter.scope = layer.name

    def parameters(self) -> list[Tensor]:
        parameter_layers = []
        for layer in self.layers:
            parameter_layers.extend(layer.parameters())

        return parameter_layers

    def forward(self, x: Tensor) -> Tensor:
        # whatever is fed in is the graph's entrypoint
        x.kind = NodeKind.INPUT

        for layer in self.layers:
            ops.set_scope(layer.name)
            try:
                x = layer.forward(x)
            finally:
                ops.clear_scope()
            # a named layer boundary: a more specific role than OP_RESULT
            x.name = f"{layer.name}/result"
            x.kind = NodeKind.LAYER_OUTPUT

        return x


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

    def __init__(self):
        # bare type name by default; Sequential re-stamps it with an index
        self.name = type(self).__name__

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        pass

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        pass


class Linear(Layer):
    def __init__(
        self,
        input_size,
        output_size,
        weight_init: type[WeightInitializer] | None = None,
    ):
        super().__init__()

        initializer = weight_init or Glorot

        self.weights = Tensor(
            initializer((input_size, output_size)).generate(),
            shape=(input_size, output_size),
        )
        self.weights.name = "weights"
        self.weights.kind = NodeKind.PARAMETER

        # add bias for each output column
        self.bias = Tensor([0.0] * output_size, shape=(1, output_size))
        self.bias.name = "bias"
        self.bias.kind = NodeKind.PARAMETER

    def parameters(self):
        return [self.bias, self.weights]

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weights + self.bias


class Sigmoid(Layer):
    def parameters(self):
        return []

    def forward(self, input: Tensor) -> Tensor:
        return ops.Sigmoid([input]).forward()


class Tanh(Layer):
    def parameters(self):
        return []

    def forward(self, input: Tensor) -> Tensor:
        return ops.Tanh([input]).forward()


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


class Loss(ABC):
    """Base for loss functions.

    forward() is the funnel: it stamps the supervision target and the loss
    scalar, then delegates the math to the subclass. The loss result is an
    op output, but LOSS is its more specific role, so it overrides OP_RESULT.
    """

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true.kind = NodeKind.TARGET
        # scope the loss ops so they cluster into their own box, like a layer
        ops.set_scope(type(self).__name__)
        try:
            result = self.compute(y_true, y_pred)
        finally:
            ops.clear_scope()
        result.kind = NodeKind.LOSS
        return result

    @abstractmethod
    def compute(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        pass


class CCE(Loss):
    """Categorical cross-entropy for one hot targets."""

    def compute(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return -(y_true * y_pred.log()).sum(axis=1).mean()


class MSE(Loss):
    """Mean squared error for scalar targets."""

    def compute(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return ((y_true - y_pred) ** 2).mean()
