from pytest import approx
import math

from babygrad.nn import CCE, MSE, Softmax
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


def test_mse():
    y_true = Tensor([1, 2, 3, 4], shape=(2, 2))
    y_pred = Tensor([1, 1, 5, 0], shape=(2, 2))

    loss = MSE(y_true, y_pred)

    assert loss == Tensor([21 / 4], shape=(1, 1))


def test_cce_one_hot_targets():
    y_true = Tensor([0, 1, 0, 1, 0, 0], shape=(2, 3))
    y_pred = Tensor([0.1, 0.8, 0.1, 0.7, 0.2, 0.1], shape=(2, 3))

    loss = CCE(y_true, y_pred)

    assert loss.shape == (1, 1)
    assert loss.data == approx([-(math.log(0.8) + math.log(0.7)) / 2])
