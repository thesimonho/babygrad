import pytest
import math
from babygrad.lib import (
    broadcast,
    _coordinate_to_index,
    _index_to_coordinate,
    _get_axis_groups,
)


def test_broadcast_same():
    left, right, shape = broadcast([1, 2, 3], [4, 5, 6], (1, 3), (1, 3))
    assert left == [1, 2, 3]
    assert right == [4, 5, 6]
    assert shape == (1, 3)


def test_broadcast_1d():
    left, right, shape = broadcast([1], [4, 5, 6], (1,), (3,))
    assert left == [1, 1, 1]
    assert right == [4, 5, 6]
    assert shape == (3,)


def test_broadcast_1d_2d():
    left, right, shape = broadcast(
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30],
        (2, 3),
        (3,),
    )
    assert left == [1, 2, 3, 4, 5, 6]
    assert right == [10, 20, 30, 10, 20, 30]
    assert shape == (2, 3)


def test_broadcast_2d():
    left, right, shape = broadcast(
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30],
        (2, 3),
        (1, 3),
    )
    assert left == [1, 2, 3, 4, 5, 6]
    assert right == [10, 20, 30, 10, 20, 30]
    assert shape == (2, 3)

    left, right, shape = broadcast(
        [1, 2, 3],
        [10, 20, 30, 40],
        (3, 1),
        (1, 4),
    )
    assert left == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert right == [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]
    assert shape == (3, 4)


def test_broadcast_invalid():
    with pytest.raises(ValueError):
        broadcast([1, 2, 3, 4, 5, 6], [1, 1], (2, 3), (2,))


def test_coordinate_index():
    shape = (1,)
    idx = _coordinate_to_index(shape, (0,))
    assert idx == 0

    shape = (3, 2)
    idx = _coordinate_to_index(shape, (2, 0))
    assert idx == 4

    shape = (3, 3)
    indices = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            indices.append(_coordinate_to_index(shape, (r, c)))
    assert indices == list(range(math.prod(shape)))

    shape = (2, 3, 2)
    idx = _coordinate_to_index(shape, (1, 1, 1))
    assert idx == 9

    idx = _coordinate_to_index(shape, (0, 1, 0))
    assert idx == 2

    idx = _coordinate_to_index(shape, (1, 1, 0))
    assert idx == 8


def test_coordinate_index_invalid():
    shape = (1,)

    with pytest.raises(IndexError):
        _coordinate_to_index(shape, (1,))

    with pytest.raises(ValueError):
        _coordinate_to_index(shape, (1, 1))

    with pytest.raises(IndexError):
        _coordinate_to_index(shape, (-1,))


def test_index_coordinate():
    shape = (1,)
    coord = _index_to_coordinate(shape, 0)
    assert coord == (0,)

    shape = (3, 2)
    coord = _index_to_coordinate(shape, 4)
    assert coord == (2, 0)

    shape = (3, 3)
    coords = []
    for i in range(math.prod(shape)):
        coords.append(_index_to_coordinate(shape, i))
    assert coords == [(r, c) for r in range(shape[0]) for c in range(shape[1])]

    shape = (2, 3, 2)
    coord = _index_to_coordinate(shape, 9)
    assert coord == (1, 1, 1)

    coord = _index_to_coordinate(shape, 2)
    assert coord == (0, 1, 0)

    coord = _index_to_coordinate(shape, 8)
    assert coord == (1, 1, 0)


def test_index_coordinate_invalid():
    shape = (1,)

    with pytest.raises(IndexError):
        _index_to_coordinate(shape, 1)

    with pytest.raises(IndexError):
        _index_to_coordinate(shape, -1)


def test_get_axis_groups():
    shape = (2, 3)
    axis = 0
    res = _get_axis_groups(shape, axis)
    assert res == [[0, 3], [1, 4], [2, 5]]

    axis = 1
    res = _get_axis_groups(shape, axis)
    assert res == [[0, 1, 2], [3, 4, 5]]


def test_get_axis_groups_3d():
    shape = (2, 2, 2)
    axis = 1
    res = _get_axis_groups(shape, axis)
    assert res == [[0, 2], [1, 3], [4, 6], [5, 7]]


def test_get_axis_groups_square():
    shape = (3, 3)
    axis = 0
    res = _get_axis_groups(shape, axis)
    assert res == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

    axis = 1
    res = _get_axis_groups(shape, axis)
    assert res == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_get_axis_groups_invalid():
    with pytest.raises(IndexError):
        _get_axis_groups((1, 1), 2)
