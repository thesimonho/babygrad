import math

import pytest
from pytest import approx

from babygrad.nn.activations import Softmax
from babygrad.nn.losses import CCE, MSE, SoftmaxCrossEntropy
from babygrad.tensor import Tensor
from babygrad.types import NodeKind


def test_softmax_returns_uniform_rows():
    logits = Tensor([0, 0, 0, 2, 2, 2], shape=(2, 3), kind=NodeKind.VIEW)

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
    y_true = Tensor([1, 2, 3, 4], shape=(2, 2), kind=NodeKind.VIEW)
    y_pred = Tensor([1, 1, 5, 0], shape=(2, 2), kind=NodeKind.VIEW)

    loss = MSE().forward(y_true, y_pred)

    assert loss == Tensor([21 / 4], shape=(1, 1), kind=NodeKind.VIEW)


def test_cce_one_hot_targets():
    y_true = Tensor([0, 1, 0, 1, 0, 0], shape=(2, 3), kind=NodeKind.VIEW)
    y_pred = Tensor([0.1, 0.8, 0.1, 0.7, 0.2, 0.1], shape=(2, 3), kind=NodeKind.VIEW)

    loss = CCE().forward(y_true, y_pred)

    assert loss.shape == (1, 1)
    assert loss.data == approx([-(math.log(0.8) + math.log(0.7)) / 2])


def test_softmax_cross_entropy_matches_separate_softmax_and_cce_for_batch():
    logits = Tensor([2, 1, 0, -1, 0, 1], shape=(2, 3), kind=NodeKind.VIEW)
    targets = Tensor([1, 0, 0, 0, 0, 1], shape=(2, 3), kind=NodeKind.VIEW)

    probabilities = Softmax().forward(logits)
    separate_loss = CCE().forward(targets, probabilities)
    fused_loss = SoftmaxCrossEntropy().forward(targets, logits)

    assert fused_loss.shape == (1, 1)
    assert fused_loss.data == approx(separate_loss.data)


def test_softmax_cross_entropy_stays_finite_when_probability_underflows():
    logits = Tensor([1000, 0, -1000], shape=(1, 3), kind=NodeKind.VIEW)
    targets = Tensor([0, 0, 1], shape=(1, 3), kind=NodeKind.VIEW)

    probabilities = Softmax().forward(logits)

    assert probabilities.data == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        CCE().forward(targets, probabilities)

    fused_loss = SoftmaxCrossEntropy().forward(targets, logits)

    assert math.isfinite(fused_loss.data[0])
    assert fused_loss.data == approx([2000.0])
