import math

from pytest import approx

from babygrad.nn import ReLU, Softmax
from babygrad.tensor import Tensor


def propagate_output_grad(output: Tensor, grad: list[float]) -> None:
    output.grad = grad
    assert output.producer is not None
    output.producer.backward()


def test_add_tracks_parents():
    left = Tensor([2, 2], shape=(2, 1))
    right = Tensor([5, 1], shape=(2, 1))

    output = left + right

    assert output.producer is not None
    assert output.producer.label == "+"
    assert output.producer.inputs[0] is left
    assert output.producer.inputs[1] is right


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


def test_backward_fully_accumulates_shared():
    """A shared *intermediate* node must finish accumulating from all of its
    consumers before it propagates onward to its own parents.

    Graph (gradient flows downward to the leaves a, b):

            out = left + right
           /                   \\
        left = shared * 3     right = shared * 4
           \\                   /
              shared = a * b              <- intermediate, consumed by TWO ops
              /          \\
             a            b

    out = 3*shared + 4*shared = 7*shared = 7*a*b, so
        d(out)/da = 7 * b = 7 * 5 = 35
        d(out)/db = 7 * a = 7 * 2 = 14

    The bug propagates `shared` to a/b on its first consumer (contributing
    only the `*3` path), then the visited-guard blocks the `*4` path from
    ever reaching the leaves -> a.grad comes out as 15 instead of 35.
    """
    a = Tensor([2.0], shape=(1,))
    b = Tensor([5.0], shape=(1,))

    shared = a * b
    left = shared * Tensor([3.0], shape=(1,))
    right = shared * Tensor([4.0], shape=(1,))
    out = left + right

    out.backward()

    assert a.grad == [35.0]
    assert b.grad == [14.0]


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


def test_matmul_tracks_parents():
    left = Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 2))
    right = Tensor([5.0, 6.0, 7.0, 8.0], shape=(2, 2))

    output = left @ right

    assert output.producer is not None
    assert output.producer.inputs[0] is left
    assert output.producer.inputs[1] is right


def _weighted_matmul_loss(
    left_data: list[float],
    right_data: list[float],
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    weights: list[float],
) -> float:
    """Forward-only scalar loss: sum((left @ right) * weights).

    The element-wise weights give every output cell a distinct upstream
    gradient, so transposed/reordered gradient rules can't pass by accident.
    """
    left = Tensor(list(left_data), shape=left_shape)
    right = Tensor(list(right_data), shape=right_shape)
    weight_tensor = Tensor(list(weights), shape=(left_shape[0], right_shape[1]))
    loss = ((left @ right) * weight_tensor).sum()
    return loss.data[0]


def _finite_difference_grads(
    data: list[float], loss_of_data, epsilon: float = 1e-6
) -> list[float]:
    """Central-difference gradient of a scalar loss w.r.t. each entry of data."""
    grads = []
    for i in range(len(data)):
        bumped_up = list(data)
        bumped_up[i] += epsilon
        bumped_down = list(data)
        bumped_down[i] -= epsilon
        grads.append(
            (loss_of_data(bumped_up) - loss_of_data(bumped_down)) / (2 * epsilon)
        )
    return grads


def test_matmul_gradients_match_differences():
    """Oracle test: backprop through matmul must agree with numerically
    wiggling each input entry and measuring the loss response.

    Shapes are rectangular and non-symmetric (2x3 @ 3x4) on purpose: with
    square matrices a transposed or operand-swapped gradient rule can still
    be shape-legal and slip through; here it cannot.
    """
    left_shape, right_shape = (2, 3), (3, 4)
    left_data = [1.5, -2.0, 3.0, 0.5, 4.0, -1.0]
    right_data = [2.0, -1.0, 0.5, 3.0, 1.0, 2.5, -2.0, 0.0, -0.5, 1.5, 4.0, -3.0]
    weights = [1.0, -2.0, 3.0, 0.5, -1.5, 2.0, -0.5, 4.0]

    left = Tensor(list(left_data), shape=left_shape)
    right = Tensor(list(right_data), shape=right_shape)
    weight_tensor = Tensor(list(weights), shape=(2, 4))

    loss = ((left @ right) * weight_tensor).sum()
    loss.backward()

    expected_left_grads = _finite_difference_grads(
        left_data,
        lambda data: _weighted_matmul_loss(
            data, right_data, left_shape, right_shape, weights
        ),
    )
    expected_right_grads = _finite_difference_grads(
        right_data,
        lambda data: _weighted_matmul_loss(
            left_data, data, left_shape, right_shape, weights
        ),
    )

    assert left.grad == approx(expected_left_grads, rel=1e-4, abs=1e-4)
    assert right.grad == approx(expected_right_grads, rel=1e-4, abs=1e-4)


def test_matmul_accumulates_when_reused():
    """The same tensor fed to matmul twice must collect gradients from both
    consumers — matmul's backward has to play nice with the engine's
    fan-out accumulation, not overwrite."""
    shared = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))
    right = Tensor([2.0, -1.0, 0.5, 3.0, 1.0, 2.5], shape=(3, 2))
    weights = Tensor([1.0, -2.0, 3.0, 0.5], shape=(2, 2))

    first = (shared @ right) * weights
    second = (shared @ right) * weights
    loss = (first + second).sum()
    loss.backward()

    single_left = Tensor(list(shared.data), shape=shared.shape)
    single_right = Tensor(list(right.data), shape=right.shape)
    single_loss = ((single_left @ single_right) * weights).sum()
    single_loss.backward()

    doubled = [2 * g for g in single_left.grad]
    assert shared.grad == approx(doubled)


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


def test_max_tie_splits_gradient():
    tensor = Tensor([4.0, 4.0, 2.0], shape=(3,))

    output = tensor.max()
    output.backward()

    assert tensor.grad == approx([0.5, 0.5, 0.0])


def test_max_non_unit_upstream_gradient():
    tensor = Tensor([1.0, 4.0, 2.0], shape=(3,))

    scaled = tensor.max() * Tensor([3.0], shape=(1,))
    scaled.backward()

    assert tensor.grad == approx([0.0, 3.0, 0.0])


def test_min_gradient():
    tensor = Tensor([1.0, 4.0, 2.0], shape=(3,))

    output = tensor.min()
    output.backward()

    assert tensor.grad == [1.0, 0.0, 0.0]


def test_min_tie_splits_gradient():
    tensor = Tensor([1.0, 1.0, 2.0], shape=(3,))

    output = tensor.min()
    output.backward()

    assert tensor.grad == approx([0.5, 0.5, 0.0])


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
