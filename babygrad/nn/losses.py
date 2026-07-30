from abc import abstractmethod

from babygrad.state import _is_training
from babygrad.tensor import Tensor
from babygrad.tracing import Traceable
from babygrad.types import NodeKind


class Loss(Traceable):
    """Base for loss functions.

    forward() is the funnel: it stamps the supervision target and the loss
    scalar, then delegates the math to the subclass. The loss result is an
    op output, but LOSS is its more specific role, so it overrides OP_RESULT.
    """

    def __init__(self, collapse: bool = False):
        self.collapse = collapse

    def forward(self, y_true: Tensor, model_output: Tensor) -> Tensor:
        y_true.kind = NodeKind.TARGET
        result = self.compute(y_true, model_output)
        result.kind = NodeKind.LOSS
        return result

    @abstractmethod
    def compute(self, y_true: Tensor, model_output: Tensor) -> Tensor:
        pass


class CCE(Loss):
    """Categorical cross-entropy for one hot targets.

    Warning:
        Using this with a separate Softmax may produce zero probabilities
        and cause numerical underflow. Prefer SoftmaxCrossEntropy.
    """

    def __init__(self, epsilon: float = 0.0, collapse: bool = False):
        super().__init__(collapse)
        self.epsilon = epsilon

    def compute(self, y_true: Tensor, model_output: Tensor) -> Tensor:
        """Compute CCE loss.

        Args:
            y_true: ground truth labels.
            model_output: predicted class probabilities from softmax.

        Returns:
            loss values.
        """
        if _is_training.get() and self.epsilon > 0:
            epsilon = Tensor(
                [self.epsilon],
                shape=(1,),
                kind=NodeKind.CONSTANT,
                name="epsilon",
            )

            mask = (1 - epsilon) * y_true
            e_k = epsilon / y_true.ncol
            y_true = mask + e_k

        probabilities = model_output
        return -(y_true * probabilities.log()).sum(axis=1).mean()


class SoftmaxCrossEntropy(Loss):
    """Compute fused softmax and CCE directly from logits.

    Improves numerical stability by:
    1. Avoiding explicit Softmax probabilities, which may underflow to zero.
    2. Cancelling the logarithm and exponential algebraically, so the
       target logit can be used directly.

    softmax: exp(z - z_max) / (sum(exp(z - z_max)))
    CCE: -(y_true * log_probs).sum(axis=1)

    log_probs = log(softmax(z))
    = log(exp(z-z_max) / sum(exp(z-z_max)))
    = log(exp(z-z_max)) - log(sum(exp(z-z_max)))
    = z-z_max - log(sum(exp(z-z_max)))
    """

    def __init__(self, epsilon: float = 0.0, collapse: bool = False):
        super().__init__(collapse)
        self.epsilon = epsilon

    def compute(self, y_true: Tensor, model_output: Tensor) -> Tensor:
        """Compute CCE loss via fusion.
        Args:
            y_true: ground truth labels.
            model_output: predicted class logits from the model.

        Returns:
            loss values.
        """
        if _is_training.get() and self.epsilon > 0:
            epsilon = Tensor(
                [self.epsilon],
                shape=(1,),
                kind=NodeKind.CONSTANT,
                name="epsilon",
            )

            mask = (1 - epsilon) * y_true
            e_k = epsilon / y_true.ncol
            y_true = mask + e_k

        logits = model_output
        shifted = logits - logits.max(axis=1)
        log_probabilities = shifted - shifted.exp().sum(axis=1).log()
        return -(y_true * log_probabilities).sum(axis=1).mean()


class MSE(Loss):
    """Mean squared error for scalar targets."""

    def compute(self, y_true: Tensor, model_output: Tensor) -> Tensor:
        predictions = model_output
        return ((y_true - predictions) ** 2).mean()
