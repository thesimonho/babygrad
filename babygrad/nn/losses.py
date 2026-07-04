from abc import ABC, abstractmethod

from babygrad.tensor import Tensor
from babygrad.types import NodeKind

from babygrad import ops


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
            ops.set_scope(None)
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
