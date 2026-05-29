from . import aliases


def add(a: list, b: list) -> list[aliases.Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return [x + y for x, y in zip(a, b)]


def sub(a: list, b: list) -> list[aliases.Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return [x - y for x, y in zip(a, b)]


def div(a: list, b: list) -> list[aliases.Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return [x / y for x, y in zip(a, b)]


def mul(a: list, b: list) -> list[aliases.Number]:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")
    return [x * y for x, y in zip(a, b)]


def dot(a: list, b: list) -> aliases.Number:
    if len(a) != len(b):
        raise ValueError(f"lists must be the same length, got {len(a)} and {len(b)}")

    return sum(mul(a, b))


def matmul(
    a: list, b: list, a_shape: aliases.Shape, b_shape: aliases.Shape
) -> list[aliases.Number]:
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
