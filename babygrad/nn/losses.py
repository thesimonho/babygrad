from abc import ABC, abstractmethod

from babygrad.tensor import Tensor
from babygrad.types import NodeKind
from babygrad.state import bound, _scope, _is_training


class Loss(ABC):
    """Base for loss functions.

    forward() is the funnel: it stamps the supervision target and the loss
    scalar, then delegates the math to the subclass. The loss result is an
    op output, but LOSS is its more specific role, so it overrides OP_RESULT.
    """

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true.kind = NodeKind.TARGET
        # scope the loss ops so they cluster into their own box, like a layer
        with bound(_scope, type(self).__name__):
            result = self.compute(y_true, y_pred)

        result.kind = NodeKind.LOSS
        return result

    @abstractmethod
    def compute(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        pass


class CCE(Loss):
    """Categorical cross-entropy for one hot targets."""

    def __init__(self, epsilon: float = 0.0):
        self.epsilon = epsilon

    def compute(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        if _is_training.get() and self.epsilon > 0:
            epsilon = Tensor(
                [self.epsilon],
                shape=(1,),
                kind=NodeKind.CONSTANT,
                scope=_scope.get(),
                name="epsilon",
            )

            mask = (1 - epsilon) * y_true
            e_k = epsilon / y_true.ncol
            y_true = mask + e_k

        # TODO: guard log(0) from softmax underflow — surgical clamp op or log-softmax fusion, not a flat +eps (corrupts healthy values)
        return -(y_true * y_pred.log()).sum(axis=1).mean()


class MSE(Loss):
    """Mean squared error for scalar targets."""

    def compute(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return ((y_true - y_pred) ** 2).mean()
