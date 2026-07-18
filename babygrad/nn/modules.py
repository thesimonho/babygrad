import random
from abc import abstractmethod

from babygrad.nn.initializers import Glorot, WeightInitializer
from babygrad.state import _is_training, _scope, bound
from babygrad.tensor import Tensor
from babygrad.tracing import Traceable
from babygrad.types import NodeKind


class Module(Traceable):
    """
    Layer outputs compose via op Nodes, which give them backprop data
    and edge data for the graph
    """

    def __init__(self, collapse: bool = False):
        # bare type name by default; Sequential re-stamps it with an index
        self.name = type(self).__name__
        # draw this module as a single box, hiding its internals. Model.stamp_name_and_scope
        # reads it once the name is fully qualified; the graph only ever sees the scope string.
        self.collapse = collapse

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
    def __init__(self, layers: list[Module], collapse: bool = False):
        super().__init__(collapse)
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
        collapse: bool = False,
    ):
        """
        Maps input_size features to output_size nodes; output_size is the layer width.
        """
        super().__init__(collapse)
        initializer = weight_init or Glorot
        self.weights = Tensor(
            initializer((input_size, output_size)).generate(),
            shape=(input_size, output_size),
            kind=NodeKind.PARAMETER,
            name="weights",
        )

        # add bias for each output column
        self.bias = Tensor(
            [0.0] * output_size,
            shape=(1, output_size),
            kind=NodeKind.PARAMETER,
            name="bias",
        )

    def own_parameters(self):
        return [self.bias, self.weights]

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weights + self.bias


class Dropout(Module):
    def __init__(self, p: float = 0.5, collapse: bool = False):
        assert p < 1.0 and p >= 0.0
        super().__init__(collapse)
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        if _is_training.get() and self.p > 0:
            keep_values = []
            for _ in range(input.numel):
                keep = 1 if random.random() > self.p else 0
                keep_values.append(keep)

            # no need to store this as gradient will already be multipled by the mask values via autograd
            dropout_mask = Tensor(
                data=keep_values,
                shape=input.shape,
                kind=NodeKind.CONSTANT,
                scope=_scope.get(),
                name="mask",
            )
            output = (input / (1 - self.p)) * dropout_mask
            return output

        return input


class Residual(Module):
    def __init__(self, block: Module, collapse: bool = False):
        super().__init__(collapse)
        self.block = block

    def children(self) -> list[Module]:
        return [self.block]

    def forward(self, input: Tensor) -> Tensor:
        output = self.block.forward(input)
        return input + output


class BatchNorm(Module):
    """
    Normalize per-feature mean/variance across the batch to stop the parameters from changing every batch. Allows use of larger LR etc.

    Still used for image inputs.
    """

    def __init__(
        self,
        n_features: int,
        epsilon: float = 1e-5,
        collapse: bool = False,
    ):
        super().__init__(collapse)
        self.epsilon = epsilon
        self.n_features = n_features

        self.gamma = Tensor(
            [1.0] * n_features,
            shape=(1, n_features),
            kind=NodeKind.PARAMETER,
            name="gamma",
        )

        # add bias for each output column
        self.beta = Tensor(
            [0.0] * n_features,
            shape=(1, n_features),
            kind=NodeKind.PARAMETER,
            name="beta",
        )

        self.running_mean = [0.0] * n_features
        self.running_var = [1.0] * n_features

    def own_parameters(self):
        return [self.beta, self.gamma]

    def forward(self, input: Tensor) -> Tensor:
        mean = (
            input.mean(axis=0)
            if _is_training.get()
            else Tensor(
                self.running_mean,
                shape=((1, self.n_features)),
                kind=NodeKind.CONSTANT,
                scope=_scope.get(),
                name="mean",
            )
        )

        variance = (
            ((input - mean) ** 2).mean(axis=0)
            if _is_training.get()
            else Tensor(
                self.running_var,
                shape=((1, self.n_features)),
                kind=NodeKind.CONSTANT,
                scope=_scope.get(),
                name="variance",
            )
        )

        epsilon = Tensor(
            [self.epsilon],
            shape=(1,),
            kind=NodeKind.CONSTANT,
            scope=_scope.get(),
            name="epsilon",
        )

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


class LayerNorm(Module):
    """
    Normalize per-row mean/variance, with additional gain and bias parameters.

    Unlike BatchNorm, this normalizes across all the features of a single sample so:
    1) can be used with online training
    2) does not require a training-time mean and variance
    """

    def __init__(self, n_features: int, collapse: bool = False):
        super().__init__(collapse)
        self.n_features = n_features

        self.gain = Tensor(
            [1.0] * n_features,
            shape=(1, n_features),
            kind=NodeKind.PARAMETER,
            name="gain",
            scope=_scope.get(),
        )

        self.bias = Tensor(
            [0.0] * n_features,
            shape=(1, n_features),
            kind=NodeKind.PARAMETER,
            name="bias",
            scope=_scope.get(),
        )

    def own_parameters(self):
        return [self.bias, self.gain]

    def forward(self, input: Tensor) -> Tensor:
        mean = input.mean(axis=1)
        mean.scope = _scope.get()
        mean.kind = NodeKind.OP_RESULT

        variance = ((input - mean) ** 2).mean(axis=1)
        variance.kind = NodeKind.OP_RESULT
        variance.scope = _scope.get()

        # NOTE: original paper doesnt include epsilon
        std = variance.sqrt()
        centered = input - mean

        return (self.gain / std) * centered + self.bias
