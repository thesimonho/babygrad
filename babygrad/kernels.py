"""Lowest level math and reduce operations."""

import math
import operator
import itertools

from babygrad.types import Number, Shape


def add(a: list[Number], b: list[Number]) -> list[Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return list(map(operator.add, a, b))


def sub(a: list[Number], b: list[Number]) -> list[Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return list(map(operator.sub, a, b))


def neg(a: list[Number]) -> list[Number]:
    # listcomp beats map(operator.neg, ...) here: no zip tuple to eliminate, and
    # UNARY_NEG is cheaper than a C-level call per element
    return [-x for x in a]


def absolute(a: list[Number]) -> list[Number]:
    return list(map(operator.abs, a))


def exp(a: list[Number]) -> list[Number]:
    return list(map(math.exp, a))


def log(a: list[Number]) -> list[Number]:
    return list(map(math.log, a))


def sqrt(a: list[Number]) -> list[Number]:
    return list(map(math.sqrt, a))


def power(a: list[Number], exponent: Number) -> list[Number]:
    return list(map(operator.pow, a, itertools.repeat(exponent)))


def div(a: list[Number], b: list[Number]) -> list[Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return list(map(operator.truediv, a, b))


def mul(a: list[Number], b: list[Number]) -> list[Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return list(map(operator.mul, a, b))


def dot(a: list[Number], b: list[Number]) -> Number:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return sum(map(operator.mul, a, b))


def matmul(
    a: list[Number],
    b: list[Number],
    a_shape: Shape,
    b_shape: Shape,
) -> list[Number]:
    """
    Calculate the dot product of two tensors.

    Shape of a tensor is always the last 2 dimensions. Any other dimensions are iterated dimensions: (batch, depth, row, col)
    """
    a_rows = a_shape[-2]
    a_cols_count = a_shape[-1]
    b_cols_count = b_shape[-1]

    b_cols = []
    for i in range(0, b_cols_count):
        b_cols.append(b[i::b_cols_count])

    output = []
    for i in range(0, a_rows * a_cols_count, a_cols_count):
        left_row = a[i : i + a_cols_count]

        for right_col in b_cols:
            if left_row and right_col:
                # use dot product directly on vectors
                result = dot(left_row, right_col)
                output.append(result)

    return output


def reduce_sum(a: list[Number]) -> Number:
    return sum(a)


def reduce_mean(a: list[Number]) -> Number:
    if len(a) == 0:
        raise ValueError
    return reduce_sum(a) / len(a)


def reduce_max(a: list[Number]) -> Number:
    return max(a)


def reduce_min(a: list[Number]) -> Number:
    return min(a)


def rectify(a: list[Number]) -> list[Number]:
    return list(map(max, itertools.repeat(0.0), a))


def sigmoid(a: list[Number]) -> list[Number]:
    return [1 / (1 + math.exp(-x)) for x in a]


def tanh(a: list[Number]) -> list[Number]:
    return list(map(math.tanh, a))
