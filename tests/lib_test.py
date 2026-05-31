import pytest
from babygrad.lib import broadcast


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
