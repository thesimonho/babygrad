import math

from . import types


def add(a: list[types.Number], b: list[types.Number]) -> list[types.Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return [x + y for x, y in zip(a, b)]


def sub(a: list[types.Number], b: list[types.Number]) -> list[types.Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return [x - y for x, y in zip(a, b)]


def neg(a: list[types.Number]) -> list[types.Number]:
    return [-x for x in a]


def absolute(a: list[types.Number]) -> list[types.Number]:
    return [abs(x) for x in a]


def exp(a: list[types.Number]) -> list[types.Number]:
    return [math.exp(x) for x in a]


def log(a: list[types.Number]) -> list[types.Number]:
    return [math.log(x) for x in a]


def sqrt(a: list[types.Number]) -> list[types.Number]:
    return [math.sqrt(x) for x in a]


def power(a: list[types.Number], exponent: types.Number) -> list[types.Number]:
    return [x**exponent for x in a]


def div(a: list[types.Number], b: list[types.Number]) -> list[types.Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return [x / y for x, y in zip(a, b)]


def mul(a: list[types.Number], b: list[types.Number]) -> list[types.Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return [x * y for x, y in zip(a, b)]


def dot(a: list[types.Number], b: list[types.Number]) -> types.Number:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")

    return sum(mul(a, b))


def matmul(
    a: list[types.Number],
    b: list[types.Number],
    a_shape: types.Shape,
    b_shape: types.Shape,
) -> list[types.Number]:
    """
    Calculate the dot product of two tensors.

    Shape of a tensor is always the last 2 dimensions. Any other dimensions are iterated dimensions: (batch, depth, row, col)
    """
    a_rows = a_shape[-2]
    a_cols = a_shape[-1]
    b_cols = b_shape[-1]

    output = []
    for i in range(0, a_rows * a_cols, a_cols):
        left_row = a[i : i + a_cols]

        for j in range(0, b_cols):
            right_col = b[j::b_cols]
            if left_row and right_col:
                # use dot product directly on vectors
                result = dot(left_row, right_col)
                output.append(result)

    return output


def rectify(a: list[types.Number]) -> list[types.Number]:
    return [max(0, x) for x in a]


def reduce_sum(a: list[types.Number]) -> types.Number:
    return sum(a)


def reduce_mean(a: list[types.Number]) -> types.Number:
    if len(a) == 0:
        raise ValueError
    return reduce_sum(a) / len(a)


def reduce_max(a: list[types.Number]) -> types.Number:
    return max(a)


def reduce_min(a: list[types.Number]) -> types.Number:
    return min(a)
