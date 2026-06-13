import pytest
from babygrad.kernels import add, sub, mul, matmul, dot, div


def test_add():
    res = add([2, 2], [5, 1])
    assert res == [7, 3]


def test_add_wrong_length():
    with pytest.raises(ValueError):
        add([2, 2], [5, 1, 1])


def test_sub():
    res = sub([2, 2], [5, 1])
    assert res == [-3, 1]


def test_sub_wrong_length():
    with pytest.raises(ValueError):
        sub([2, 2], [5, 1, 1])


def test_div():
    res = div([2, 2], [5, 1])
    assert res == [0.4, 2]


def test_div_wrong_length():
    with pytest.raises(ValueError):
        div([2, 2], [5, 1, 1])


def test_mul():
    res = mul([2, 2], [5, 1])
    assert res == [10, 2]


def test_mul_wrong_length():
    with pytest.raises(ValueError):
        mul([2, 2], [5, 1, 1])


def test_dot():
    res = dot([2, 2], [5, 1])
    assert res == 12


def test_dot_wrong_length():
    with pytest.raises(ValueError):
        dot([2, 2], [5, 1, 1])


def test_matmul():
    x: list[float] = [1, 2, 3, 4, 5, 6]
    y: list[float] = [1, 2, 3, 4, 5, 6]
    res = matmul(x, y, (2, 3), (3, 2))
    assert res == [22, 28, 49, 64]


def test_matmul_square():
    x: list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y: list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = matmul(x, y, (3, 3), (3, 3))
    assert res == [30, 36, 42, 66, 81, 96, 102, 126, 150]


def test_matmul_identity():
    i: list[float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    x: list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = matmul(i, x, (3, 3), (3, 3))
    assert res == [1, 2, 3, 4, 5, 6, 7, 8, 9]
