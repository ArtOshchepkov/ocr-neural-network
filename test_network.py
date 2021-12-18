import numpy as np

from network import Dense


def test_dense_layer_simple_test():
    dense = Dense(in_shape=1, out_shape=1)
    dense.w = np.array([[2]])

    res = dense.f(np.array([10]))

    assert res == 20


def test_dense_layer_shape_test():
    dense = Dense(in_shape=4, out_shape=2)
    input = np.array([
        1, 2, 0, 10
    ])
    dense.w = np.array([
        [0, 1, 2, 3],
        [30, 0, -4, -1]
    ])
    expected_res = np.array([32, 20])

    res = dense.f(input)
    print("Dense res", res)

    assert np.array_equal(res, expected_res)


def test_dense_layer__relu_should_drop_zero():
    dense = Dense(in_shape=4, out_shape=2)
    input = np.array([
        1, 2, 0, 10
    ])
    dense.w = np.array([
        [0, 1, 2, 3],
        [30, 0, -4, -50]
    ])
    expected_res = np.array([32, 0])

    res = dense.f(input)
    print("Dense res", res)

    assert np.array_equal(res, expected_res)
