from . import aliases
import math


def broadcast(
    left: list[aliases.Number],
    right: list[aliases.Number],
    left_shape: aliases.Shape,
    right_shape: aliases.Shape,
) -> tuple[list[aliases.Number], list[aliases.Number], aliases.Shape]:
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

    # padd shape lengths and check that all indices can be broadcast
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


def _expand_dims(
    data: list[aliases.Number],
    current_shape: aliases.Shape,
    target_shape: aliases.Shape,
) -> list[aliases.Number]:
    """
    Expand current dimensions that are == 1.
    """
    assert len(data) == math.prod(current_shape)

    # pad the indices first for simplicity
    current_reshaped = []
    for t_idx in range(-1, -len(target_shape) - 1, -1):
        try:
            current_reshaped.insert(0, current_shape[t_idx])
        except IndexError:
            current_reshaped.insert(0, 1)

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
