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


class Module(ABC):
    """
    Layer outputs compose via op Nodes, which give them backprop data
    and edge data for the graph
    """

    def __init__(self):
        # bare type name by default; Sequential re-stamps it with an index
        self.name = type(self).__name__
        self.is_training = True

    def children(self) -> list[Module]:
        """
        Return a list of any Modules nested within this container. Default [] for leaf nodes.
        """
        return []

    def parameters(self) -> list[Tensor]:
        """
        Return a list of all parameter tensors of nested children.
        """
        parameter_tensors = []
        for c in self.children():
            for p in c.parameters():
                parameter_tensors.append(p)
        return self.own_parameters() + parameter_tensors

    def own_parameters(self) -> list[Tensor]:
        """
        Return a list of all parameter tensors owned by this module.
        """
        return []

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        pass


class Sequential(Module):
    def __init__(self, layers: list[Module]):
        super().__init__()
        self.layers = layers

        # stamp durable identities once: layers get an indexed name,
        # parameters get that name as their prefix ("Linear_0/weights")
        for i, layer in enumerate(layers):
            layer.name = f"{layer.name}_{i}"
            for parameter in layer.parameters():
                parameter.name = f"{layer.name}/{parameter.name}"
                parameter.scope = layer.name

    def children(self) -> list[Module]:
        return self.layers

    def eval(self, x: Tensor) -> Tensor:
        """
        Use for forward pass inference, not training.
        """
        self.is_training = False
        output = self.forward(x)
        self.is_training = True
        return output

    def forward(self, input: Tensor) -> Tensor:
        # whatever is fed in is the graph's entrypoint
        input.kind = NodeKind.INPUT

        for layer in self.layers:
            layer.is_training = self.is_training
            ops.set_scope(layer.name)
            try:
                input = layer.forward(input)
            finally:
                ops.clear_scope()
            # a named layer boundary: a more specific role than OP_RESULT
            input.name = f"{layer.name}/result"
            input.kind = NodeKind.LAYER_OUTPUT

        return input


class Linear(Module):
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

    def own_parameters(self):
        return [self.bias, self.weights]

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weights + self.bias


class Residual(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.block = block

    def children(self) -> list[Module]:
        return [self.block]

    def forward(self, input: Tensor) -> Tensor:
        output = self.block.forward(input)
        return input + output


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


class BatchNorm(Module):
    """
    Normalize per-feature mean/variance across the batch to stop the parameters from changing every batch. Allows use of larger LR etc.
    """

    def __init__(
        self,
        n_features: int,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.n_features = n_features

        self.gamma = Tensor([1.0] * n_features, shape=(1, n_features))
        self.gamma.name = "gamma"
        self.gamma.kind = NodeKind.PARAMETER

        # add bias for each output column
        self.beta = Tensor([0.0] * n_features, shape=(1, n_features))
        self.beta.name = "beta"
        self.beta.kind = NodeKind.PARAMETER

        self.running_mean = [0.0] * n_features
        self.running_var = [1.0] * n_features

    def own_parameters(self):
        return [self.beta, self.gamma]

    def forward(self, input: Tensor) -> Tensor:
        mean = (
            input.mean(axis=0)
            if self.is_training
            else Tensor(self.running_mean, shape=((1, self.n_features)))
        )
        variance = (
            ((input - mean) ** 2).mean(axis=0)
            if self.is_training
            else Tensor(self.running_var, shape=((1, self.n_features)))
        )

        epsilon = Tensor([self.epsilon], shape=(1,))
        std = (variance + epsilon).sqrt()

        if self.is_training:
            # Moving average of running and current batch stats.
            # Use raw values: running stats must never become autograd nodes,
            # otherwise backward would propagate into them
            self.running_mean = [
                (0.9 * running + 0.1 * current)
                for running, current in zip(self.running_mean, mean.data)
            ]
            self.running_var = [
                (0.9 * running + 0.1 * current)
                for running, current in zip(self.running_var, variance.data)
            ]

        normalized = (input - mean) / std
        return self.gamma * normalized + self.beta


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
