from pytest import approx

from babygrad.nn import Softmax
from babygrad.tensor import Tensor


def test_softmax_returns_uniform_rows():
    logits = Tensor([0, 0, 0, 2, 2, 2], shape=(2, 3))

    probabilities = Softmax().forward(logits)

    assert probabilities.shape == logits.shape
    assert probabilities.data == approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )
