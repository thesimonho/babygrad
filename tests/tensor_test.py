from babygrad.tensor import Tensor
import pytest
import math


def test_unary_ops():
    t = Tensor([1, 2, -3, -4], shape=(2, 2))
    assert abs(t) == Tensor([1, 2, 3, 4], shape=(2, 2))
    assert -t == Tensor([-1, -2, 3, 4], shape=(2, 2))
    assert t**2 == Tensor([1, 4, 9, 16], shape=(2, 2))
    assert abs(t).sqrt() == Tensor([math.sqrt(x) for x in abs(t).data], shape=(2, 2))
    assert t.exp() == Tensor([math.exp(x) for x in t.data], shape=(2, 2))
    assert abs(t).log() == Tensor([math.log(x) for x in abs(t).data], shape=(2, 2))

    with pytest.raises(ValueError):
        t.log()

    with pytest.raises(ValueError):
        t.sqrt()


def test_reduce_ops():
    t = Tensor([1, 2, -3, -4], shape=(2, 2))
    assert t.sum() == Tensor([-4], shape=(1, 1))
    assert t.mean() == Tensor([-1], shape=(1, 1))
    assert t.max() == Tensor([2], shape=(1, 1))
    assert t.min() == Tensor([-4], shape=(1, 1))


def test_getitem_1d_scalar():
    t = Tensor([1, 2, 3], shape=(3,))
    assert t[1] == 2


def test_getitem_1d_scalar_invalid():
    t = Tensor([1, 2, 3], shape=(3,))
    with pytest.raises(IndexError):
        t[3]


def test_getitem_2d_row():
    t = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
    assert t[1] == Tensor([4, 5, 6], shape=(3,))


def test_getitem_2d_row_invalid():
    t = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
    with pytest.raises(IndexError):
        t[2]


def test_getitem_2d_scalar():
    t = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
    assert t[1, 0] == 4
    assert t[1, 2] == 6


def test_getitem_2d_scalar_invalid():
    t = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
    with pytest.raises(IndexError):
        t[0, 3]
    with pytest.raises(IndexError):
        t[1, 3]


def test_add_vector_vector():
    res = Tensor([2, 2], shape=(2, 1)) + Tensor([5, 1], shape=(2, 1))
    assert res.data == [7, 3]


@pytest.mark.skip(reason="not implemented")
def test_add_vector_matrix():
    pass


def test_add_matrix_matrix():
    res = Tensor([2, 2, 1, 1, 4, 7, 9, 3, 2], shape=(3, 3)) + Tensor(
        [5, 1, 8, 2, 3, 1, 7, 5, 6], shape=(3, 3)
    )

    assert res.data == [7, 3, 9, 3, 7, 8, 16, 8, 8]


def test_sub_vector_vector():
    res = Tensor([2, 2], shape=(2, 1)) - Tensor([5, 1], shape=(2, 1))
    assert res.data == [-3, 1]


@pytest.mark.skip(reason="not implemented")
def test_sub_vector_matrix():
    pass


def test_sub_matrix_matrix():
    res = Tensor([2, 2, 1, 1, 4, 7, 9, 3, 2], shape=(3, 3)) - Tensor(
        [5, 1, 8, 2, 3, 1, 7, 5, 6], shape=(3, 3)
    )
    assert res.data == [-3, 1, -7, -1, 1, 6, 2, -2, -4]


def test_mul_vector_vector():
    res = Tensor([2, 2], shape=(2, 1)) * Tensor([5, 1], shape=(2, 1))
    assert res.data == [10, 2]


@pytest.mark.skip(reason="not implemented")
def test_mul_vector_matrix():
    pass


def test_mul_matrix_matrix():
    res = Tensor([2, 2, 1, 1, 4, 7, 9, 3, 2], shape=(3, 3)) * Tensor(
        [5, 1, 8, 2, 3, 1, 7, 5, 6], shape=(3, 3)
    )
    assert res.data == [10, 2, 8, 2, 12, 7, 63, 15, 12]


def test_dot_vector_vector():
    v1 = Tensor([2, 2], shape=(1, 2))
    v2 = Tensor([5, 1], shape=(2, 1))
    res = v1 @ v2
    assert res.shape == (1, 1)
    assert res.data == [12]


def test_matmul_vector_matrix():
    v = Tensor([2, 2], shape=(1, 2))
    m = Tensor([5, 18, 2, 1, 5, 6], shape=(2, 3))
    res = v @ m
    assert res.shape == (1, 3)


def test_matmul_matrix_vector():
    m = Tensor([5, 18, 2, 1, 5, 6], shape=(3, 2))
    v = Tensor([2, 2], shape=(2, 1))
    res = m @ v
    assert res.shape == (3, 1)


def test_matmul_matrix_matrix():
    m1 = Tensor([5, 18, 2, 1, 5, 6], shape=(2, 3))
    m2 = Tensor([5, 1, 8, 2, 3, 1, 7, 5, 6], shape=(3, 3))
    res = m1 @ m2
    assert res.shape == (2, 3)


def test_copy():
    t = Tensor([1, 2, 3], shape=(1, 3))
    c = t.copy()

    assert t is not c
    assert t.data is not c.data
    assert t.data == c.data


def test_reshape():
    t = Tensor([1, 2, 3], (1, 3))
    r = t.reshape((3, 1))
    assert t.data is not r.data  # not a copy
    assert t.data == r.data
    assert r.shape == (3, 1)


def test_reshape_valid():
    t = Tensor([1, 2, 3], (1, 3))
    with pytest.raises(ValueError):
        t.reshape((1, 4))


def test_flatten():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    f = t.flatten()
    assert t.data == f.data
    assert f.shape == (len(t.data),)


def test_axis_reduction():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    assert t.sum().shape == (1, 1)
    assert t.sum(axis=0).shape == (1, 3)
    assert t.sum(axis=1).shape == (2, 1)

    t = Tensor([1, 2, 3, 4, 5, 6], (6,))
    assert t.sum().shape == (1,)
    assert t.sum(axis=0).shape == (1,)


def test_axis_reduce_negative_axis_2d():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

    assert t.sum(axis=-1) == Tensor([6, 15], shape=(2, 1))
    assert t.sum(axis=-2) == Tensor([5, 7, 9], shape=(1, 3))


def test_axis_reduce_negative_axis_3d():
    t = Tensor([1, 2, 3, 4, 5, 6, 7, 8], (2, 2, 2))

    assert t.sum(axis=-1) == Tensor([3, 7, 11, 15], shape=(2, 2, 1))
