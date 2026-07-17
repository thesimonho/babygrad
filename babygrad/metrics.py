from abc import ABC, abstractmethod

from babygrad.tensor import Tensor
from babygrad.lib import argmax
from babygrad.types import Number


class Metric(ABC):
    name: str

    def __init__(self):
        self.name = type(self).__name__

    @abstractmethod
    def compute(self, y_true: Tensor, y_pred: Tensor) -> Number:
        pass


class Accuracy(Metric):
    """Fraction of rows whose predicted class matches the true class."""

    def compute(self, y_true: Tensor, y_pred: Tensor) -> float:
        assert y_true.shape == y_pred.shape, "y_true and y_pred must share a shape"

        ncol = y_pred.ncol
        matches = 0
        for start in range(0, len(y_pred.data), ncol):
            predicted_class = argmax(y_pred.data[start : start + ncol])
            true_class = argmax(y_true.data[start : start + ncol])
            if predicted_class == true_class:
                matches += 1

        return matches / y_true.nrow
