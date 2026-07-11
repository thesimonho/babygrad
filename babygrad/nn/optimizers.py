from abc import ABC, abstractmethod
from math import sqrt

from babygrad.tensor import Tensor


class Optimizer(ABC):
    def __init__(self, parameters: list[Tensor]):
        self.parameters = parameters
        self._lr: float | None = None

    @abstractmethod
    def step(self):
        pass

    @property
    def lr(self):
        if self._lr is None:
            raise ValueError(
                "Learning rate unset. Get the value from a Scheduler or set .lr manually."
            )
        return self._lr

    @lr.setter
    def lr(self, value: float):
        self._lr = value

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


class Adam(Optimizer):
    """
    Adaptive moment estimator.

    Each learnable parameter (weight, bias, gamma, etc) has a exponential moving average of its first and second moments.
    The first moment (m) is mean gradient value. It tracks the magnitude and direction the gradient has recently moved in.
    The second moment (v) is the gradient**2. It removes the sign and tells you about overall magnitude of recent steps.

    Combined, this gives each parameter a pseudo "learning rate" allowing it to update more optimally; the (bounded) ratio m/v is large when the magnitude of the 2 moments agree, which then results in a larger step.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        beta1: float = 9e-1,
        beta2: float = 9.99e-1,
        epsilon: float = 1e-8,
    ):
        """
        parameters is the filtered list of learnable parameters (weights, bias, etc)
        """
        super().__init__(parameters)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        #  One slot per scalar — weights AND biases — hence `[0.0] * p.numel` (numel is the flat
        # element count, not the node count).
        self.m = [[0.0] * p.numel for p in parameters]
        self.v = [[0.0] * p.numel for p in parameters]
        self.time = 0

    def step(self):
        self.time += 1
        for t, tensor in enumerate(self.parameters):
            for g, grad in enumerate(tensor.grad):
                self.m[t][g] = (self.beta1 * self.m[t][g]) + (1 - self.beta1) * grad
                self.v[t][g] = (self.beta2 * self.v[t][g]) + (1 - self.beta2) * grad**2

                # corrected is just used to update the current weight value, not stored
                m_hat = self.m[t][g] / (1 - self.beta1**self.time)
                v_hat = self.v[t][g] / (1 - self.beta2**self.time)

                tensor.data[g] -= self.lr * m_hat / (sqrt(v_hat) + self.epsilon)
