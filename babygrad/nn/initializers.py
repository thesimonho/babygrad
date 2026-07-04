import math
from abc import ABC, abstractmethod
from random import Random

from babygrad.types import Number, Shape


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
