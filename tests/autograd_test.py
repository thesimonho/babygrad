import math

from pytest import approx

from babygrad.nn import ReLU, Softmax
from babygrad.tensor import Tensor


def propagate_output_grad(output: Tensor, grad: list[float]) -> None:
    output.grad = grad
    assert output.backprop is not None
    output.backprop.propagate_to_parents()


def test_add_tracks_parents():
    left = Tensor([2, 2], shape=(2, 1))
    right = Tensor([5, 1], shape=(2, 1))

    output = left + right

    assert output.backprop is not None
    assert output.backprop.op == "+"
    assert output.backprop.parents[0] is left
    assert output.backprop.parents[1] is right


def test_add_gradients():
    left = Tensor([2, 2], shape=(2, 1))
    right = Tensor([5, 1], shape=(2, 1))
    output = left + right

    propagate_output_grad(output, [1.0, 1.0])

    assert left.grad == [1.0, 1.0]
    assert right.grad == [1.0, 1.0]


def test_add_unbroadcasts_gradients():
    left = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
    right = Tensor([10, 20, 30], shape=(1, 3))
    output = left + right

    propagate_output_grad(output, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    assert left.grad == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert right.grad == [2.0, 2.0, 2.0]


def test_backward_walks_graph():
    left = Tensor([1.0], shape=(1,))
    right = Tensor([2.0], shape=(1,))
    final_right = Tensor([3.0], shape=(1,))

    partial_output = left + right
    final_output = partial_output + final_right

    final_output.backward()

    assert final_output.grad == [1.0]
    assert partial_output.grad == [1.0]
    assert left.grad == [1.0]
    assert right.grad == [1.0]
    assert final_right.grad == [1.0]


def test_backward_accumulates_gradients():
    left = Tensor([1.0], shape=(1,))
    right = Tensor([2.0], shape=(1,))

    partial_output = left + right
    final_output = partial_output + partial_output

    final_output.backward()

    assert final_output.grad == [1.0]
    assert partial_output.grad == [2.0]
    assert left.grad == [2.0]
    assert right.grad == [2.0]


def test_sub_gradient_signs():
    left = Tensor([2.0, 3.0], shape=(2,))
    right = Tensor([5.0, 7.0], shape=(2,))

    output = left - right
    output.backward()

    assert left.grad == [1.0, 1.0]
    assert right.grad == [-1.0, -1.0]


def test_sub_unbroadcasts_gradients():
    left = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))
    right = Tensor([10.0, 20.0, 30.0], shape=(1, 3))

    output = left - right
    output.backward()

    assert left.grad == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert right.grad == [-2.0, -2.0, -2.0]


def test_neg_gradient():
    tensor = Tensor([2.0, -3.0], shape=(2,))

    output = -tensor
    output.backward()

    assert tensor.grad == [-1.0, -1.0]


def test_mul_gradients():
    left = Tensor([2.0, 3.0], shape=(2,))
    right = Tensor([5.0, 7.0], shape=(2,))

    output = left * right
    output.backward()

    assert left.grad == [5.0, 7.0]
    assert right.grad == [2.0, 3.0]


def test_mul_unbroadcasts_gradients():
    left = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))
    right = Tensor([10.0, 20.0, 30.0], shape=(1, 3))

    output = left * right
    output.backward()

    assert left.grad == [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    assert right.grad == [5.0, 7.0, 9.0]


def test_pow_gradient():
    tensor = Tensor([2.0, -3.0], shape=(2,))

    output = tensor**3
    output.backward()

    assert tensor.grad == [12.0, 27.0]


def test_sum_gradient():
    tensor = Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 2))

    output = tensor.sum()
    output.backward()

    assert tensor.grad == [1.0, 1.0, 1.0, 1.0]


def test_sum_axis_gradient():
    tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))

    output = tensor.sum(axis=1)
    output.backward()

    assert tensor.grad == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_mean_gradient():
    tensor = Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 2))

    output = tensor.mean()
    output.backward()

    assert tensor.grad == [0.25, 0.25, 0.25, 0.25]


def test_mean_axis_gradient():
    tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))

    output = tensor.mean(axis=1)
    output.backward()

    assert tensor.grad == approx([1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])


def test_matmul_gradients():
    left = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))
    right = Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape=(3, 2))

    output = left @ right
    output.backward()

    assert left.grad == [15.0, 19.0, 23.0, 15.0, 19.0, 23.0]
    assert right.grad == [5.0, 5.0, 7.0, 7.0, 9.0, 9.0]


def test_transpose_gradient():
    tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))

    output = tensor.transpose()
    propagate_output_grad(output, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    assert tensor.grad == [10.0, 30.0, 50.0, 20.0, 40.0, 60.0]


def test_exp_gradient():
    tensor = Tensor([0.0, 1.0], shape=(2,))

    output = tensor.exp()
    output.backward()

    assert tensor.grad == approx([1.0, math.e])


def test_log_gradient():
    tensor = Tensor([2.0, 4.0], shape=(2,))

    output = tensor.log()
    output.backward()

    assert tensor.grad == [0.5, 0.25]


def test_div_gradients():
    left = Tensor([6.0, 8.0], shape=(2,))
    right = Tensor([2.0, 4.0], shape=(2,))

    output = left / right
    output.backward()

    assert left.grad == [0.5, 0.25]
    assert right.grad == [-1.5, -0.5]


def test_div_unbroadcasts_gradients():
    left = Tensor([6.0, 8.0, 10.0, 12.0], shape=(2, 2))
    right = Tensor([2.0, 4.0], shape=(1, 2))

    output = left / right
    output.backward()

    assert left.grad == [0.5, 0.25, 0.5, 0.25]
    assert right.grad == [-4.0, -1.25]


def test_relu_gradient():
    tensor = Tensor([-2.0, 0.0, 3.0], shape=(3,))

    output = ReLU().forward(tensor)
    output.backward()

    assert tensor.grad == [0.0, 0.0, 1.0]


def test_max_gradient():
    tensor = Tensor([1.0, 4.0, 2.0], shape=(3,))

    output = tensor.max()
    output.backward()

    assert tensor.grad == [0.0, 1.0, 0.0]


def test_max_axis_gradient():
    tensor = Tensor([1.0, 5.0, 2.0, 9.0, 3.0, 4.0], shape=(2, 3))

    output = tensor.max(axis=1)
    output.backward()

    assert tensor.grad == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]


def test_softmax_row_gradients():
    logits = Tensor([0.0, 1.0, 2.0, 1.0, 1.0, 1.0], shape=(2, 3))
    weights = Tensor([1.0, 0.0, -1.0, 2.0, -1.0, 0.5], shape=(2, 3))

    loss = (Softmax().forward(logits) * weights).sum()
    loss.backward()

    first_probability = 1 / (1 + math.e + math.e**2)
    second_probability = math.e / (1 + math.e + math.e**2)
    third_probability = math.e**2 / (1 + math.e + math.e**2)
    first_weighted_average = first_probability - third_probability
    uniform_probability = 1 / 3
    second_weighted_average = uniform_probability * (2 - 1 + 0.5)

    assert logits.grad == approx(
        [
            first_probability * (1 - first_weighted_average),
            second_probability * (0 - first_weighted_average),
            third_probability * (-1 - first_weighted_average),
            uniform_probability * (2 - second_weighted_average),
            uniform_probability * (-1 - second_weighted_average),
            uniform_probability * (0.5 - second_weighted_average),
        ]
    )
