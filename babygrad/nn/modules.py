from abc import ABC, abstractmethod

from babygrad.nn.initializers import Glorot, WeightInitializer
from babygrad.tensor import Tensor
from babygrad.types import NodeKind
from babygrad.state import bound, _is_training, _scope


class Model:
    def __init__(self, root: Module):
        self.root = root
        self.stamp_name_and_scope(self.root)

    def eval(self, x: Tensor) -> Tensor:
        """
        Use for forward pass inference, not training.
        """
        with bound(_is_training, False):
            return self.root.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        with bound(_is_training, True):
            return self.root.forward(x)

    def stamp_name_and_scope(self, root: Module, prefix: str = "", idx: int = 0):
        """
        Set the name and scope of all descendent Modules and their children
        """
        root.name = f"{prefix + '/' if prefix else ''}{root.name}_{idx}"

        for idx, c in enumerate(root.children()):
            self.stamp_name_and_scope(c, root.name, idx)

        for idx, p in enumerate(root.own_parameters()):
            p.scope = root.name
            p.name = f"{root.name.split('/')[-1]}/{p.name}"


class Module(ABC):
    """
    Layer outputs compose via op Nodes, which give them backprop data
    and edge data for the graph
    """

    def __init__(self):
        # bare type name by default; Sequential re-stamps it with an index
        self.name = type(self).__name__

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

    def children(self) -> list[Module]:
        return self.layers

    def forward(self, input: Tensor) -> Tensor:
        # whatever is fed in is the graph's entrypoint
        if input.producer is None:
            input.kind = NodeKind.INPUT
            input.scope = self.name

        for layer in self.layers:
            with bound(_scope, layer.name):
                input = layer.forward(input)

            # a named layer boundary: a more specific role than OP_RESULT
            input.name = f"{layer.name.split('/')[-1]}/result"
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
            if _is_training.get()
            else Tensor(self.running_mean, shape=((1, self.n_features)))
        )

        variance = (
            ((input - mean) ** 2).mean(axis=0)
            if _is_training.get()
            else Tensor(self.running_var, shape=((1, self.n_features)))
        )

        epsilon = Tensor([self.epsilon], shape=(1,))

        std = (variance + epsilon).sqrt()

        if _is_training.get():
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
