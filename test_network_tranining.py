import random

import numpy as np
import imageio
import glob

from network import Dense, Network





def test_training_model():
    letter_samples = 500
    xs = []
    ys = []

    letter_size = {}

    for im_path in glob.glob("data/*/*.png"):
        letter = int(im_path.split('/')[1])
        if letter_size.get(letter) is None:
            letter_size[letter] = 0

        if letter_size[letter] < letter_samples:
            xs.append(imageio.imread(im_path).reshape(16 * 16))
            ys.append(letter_to_y(letter))
            letter_size[letter] += 1

    print(f'{len(xs)} images loaded')

    network = Network()
    network.teach(xs, ys)
    network.printLog()


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
