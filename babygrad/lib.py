import math
from collections import defaultdict
from typing import Callable
from warnings import deprecated

from . import types


def broadcast(
    left: list[types.Number],
    right: list[types.Number],
    left_shape: types.Shape,
    right_shape: types.Shape,
) -> tuple[list[types.Number], list[types.Number], types.Shape]:
    """Align 2 data lists by broadcasting values across dimensions where possible.

    Compare shapes from the rightmost dimension moving left.

    For each aligned dimension, broadcasting is valid when:
    1. The dimensions are equal.
    2. One dimension is 1.
    3. One side is missing because that tensor has fewer dimensions.

    The result shape uses the larger/non-missing dimension at each position.
    If none of those are true, broadcasting is invalid.

      ┌────────────────────────┬────────────────────────┬────────┬────────────┐
      │               left dim │              right dim │ valid? │ result dim │
      ├────────────────────────┼────────────────────────┼────────┼────────────┤
      │             same value │             same value │ yes    │ that value │
      │                      1 │                      N │ yes    │          N │
      │                      N │                      1 │ yes    │          N │
      │                missing │                      N │ yes    │          N │
      │                      N │                missing │ yes    │          N │
      │ different non-1 values │ different non-1 values │ no     │      error │
      └────────────────────────┴────────────────────────┴────────┴────────────┘
    """
    if left_shape == right_shape:
        return (left, right, left_shape)

    max_ndims = max(len(left_shape), len(right_shape))

    # pad shape lengths and check that all indices can be broadcast
    output_shape = []
    for i in range(-1, -max_ndims - 1, -1):
        try:
            left_dim = left_shape[i]
        except IndexError:
            left_dim = 1

        try:
            right_dim = right_shape[i]
        except IndexError:
            right_dim = 1

        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            # if any index fails this check then the shape cannot broadcast
            raise ValueError(
                f"Incompatible dimensions for broadcasting {left_shape} and {right_shape}"
            )

        if left_dim == right_dim:
            output_shape.insert(0, left_dim)
        else:
            output_shape.insert(0, max(left_dim, right_dim))

    output_left = _expand_dims(left, left_shape, tuple(output_shape))
    output_right = _expand_dims(right, right_shape, tuple(output_shape))

    return (output_left, output_right, tuple(output_shape))


def unbroadcast(
    data: list[types.Number],
    current: types.Shape,
    target: types.Shape,
    op: Callable = sum,
) -> list[types.Number]:
    """
    Unbroadcast current data to target shape.
    Only called in tensor ops where parent broadcasting could potentially have happened.
    """
    assert len(data) == math.prod(current)
    if current == target:
        return data

    if len(target) < len(current):
        target = _pad_shape_to_rank(target, current)

    output = list(data)
    working = list(current)
    for i in range(len(working)):
        c = working[i]
        t = target[i]

        if c == t:
            continue
        if c > t:
            groups = _get_axis_groups(tuple(working), i)
            output = [op(output[i] for i in group) for group in groups]

            # update the current working shape incase multiple axes need to update
            working[i] = target[i]

    return output


def _expand_dims(
    data: list[types.Number],
    current_shape: types.Shape,
    target_shape: types.Shape,
) -> list[types.Number]:
    """
    Expand current dimensions that are == 1 to repeat towards a target shape.
    """
    assert len(data) == math.prod(current_shape)

    # pad the axes first for simplicity
    current_reshaped = _pad_shape_to_rank(current_shape, target_shape)

    # guard: broadcasting must be valid
    for c, t in zip(current_reshaped, target_shape):
        assert c == t or c == 1

    if tuple(current_reshaped) == target_shape:
        return data

    # Iterate from the left. an index that needs expansion repeats everything to the right of it.
    # The length of everything to the right is the product of their indices.
    # (1,2,3) -> (3,2,3) repeats each block of 6 (2*3) 3 times
    output = data
    for i in range(len(target_shape)):
        if current_reshaped[i] == target_shape[i]:
            continue

        if current_reshaped[i] != 1:
            raise ValueError()

        temp = []
        block_size = math.prod(current_reshaped[i + 1 :])
        n_rep = target_shape[i]

        # read from updated data incase later dimensions also need to expand
        for start in range(0, len(output), block_size):
            block = output[start : start + block_size]
            temp.extend(block * n_rep)
        output = temp

    return output


def _pad_shape_to_rank(current: types.Shape, target: types.Shape) -> types.Shape:
    """
    Left pad a shape to the same rank as a target shape.
    Example:
    current = (3,)
    target = (1,1)
    Pad current to match target rank will return (1,3)
    """
    padded = []
    for idx in range(-1, -len(target) - 1, -1):
        try:
            padded.insert(0, current[idx])
        except IndexError:
            padded.insert(0, 1)

    return tuple(padded)


def _coordinate_to_index(shape: types.Shape, coordinate: tuple[int, ...]) -> int:
    """
    Convert a tensor coordinate to the correct index location in a flat data list.
    Example: coordinate(0,0) -> list[0], coordinate(0,1) -> list[1]
    """
    if len(shape) != len(coordinate):
        raise ValueError("Target coordinate does not match target shape")

    for i in range(len(shape)):
        if coordinate[i] >= shape[i] or coordinate[i] < 0:
            raise IndexError("Index out of bounds for target shape")

    idx = 0
    for i, c in enumerate(coordinate):
        stride = math.prod(shape[i + 1 :])
        idx += c * stride

    return idx


def _index_to_coordinate(shape: types.Shape, index: int) -> tuple[int, ...]:
    """
    Convert a list index to a tensor coordinate given a specific shape.
    Example: list[0] -> coordinate(0,0)

    The insight for the divmod: if you have an index somewhere along a flatlist, you want to know how many rows can fit before that index (div). the remainder of that (mod) carries to the next shape dimension as the new index, then how many columns fit into that leftover index.
    """

    if math.prod(shape) <= index or index < 0:
        raise IndexError(f"Target index out of bounds for shape {shape}")

    coordinates = []
    for i in range(len(shape)):
        stride = math.prod(shape[i + 1 :])
        c, r = divmod(index, stride)
        index = r
        coordinates.append(c)

    return tuple(coordinates)


@deprecated("Use _get_axis_groups instead. Kept here for learning purposes.")
def _get_axis_groups_old(shape: types.Shape, axis: int):
    if axis >= len(shape):
        raise IndexError("Axis out of bounds for shape")

    stride = math.prod(shape[axis + 1 :])

    idx = 0
    groups = defaultdict(list)
    for p in range(math.prod(shape[:axis])):
        for _ in range(shape[axis]):
            for c in range(stride):
                groups[(p, c)].append(idx)
                idx += 1

    assert idx == math.prod(shape)
    return list(groups.values())


def _get_axis_groups(shape: types.Shape, axis: int) -> list[list[int]]:
    """
    Return a list of lists of flat indexes that would be reduced together.

    X = [
    1 2 3
    4 5 6
    ]
    shape = (2,3)
    axis = 0
    groups = [[0,3],[1,4],[2,5]]
    axis = 1
    groups = [[0,1,2],[3,4,5]]
    """
    if axis >= len(shape):
        raise IndexError("Axis out of bounds for shape")

    output = defaultdict(list)
    for idx in range(math.prod(shape)):
        coord = _index_to_coordinate(shape, idx)
        key = tuple(c for i, c in enumerate(coord) if i != axis)
        output[key].append(idx)

    return list(output.values())


def transpose_flat_data(data, shape: types.Shape):
    if len(shape) != 2:
        raise ValueError("Requires a 2D tensor")

    output = []
    nrow, ncol = shape
    for c in range(ncol):
        for r in range(nrow):
            output.append(data[r * ncol + c])

    return output, (ncol, nrow)


def argmax(values: list) -> int:
    """Return the index of the largest value (the first one on ties)."""
    return values.index(max(values))
