import glob

import imageio
import matplotlib.pyplot as plt
import numpy as np

from network import Network, Dense


def image_to_x(image):
    return image.reshape(16 * 16) / 255


def load_image(im_path):
    return imageio.imread(im_path)


def letter_to_y(letter: int) -> np.array:
    res = np.zeros(26)
    res[letter - 1] = 1
    return res


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def y_to_letter(y: np.array):
    # print(y)
    idx = find_nearest(y, 1)
    return chr(idx + 65)


def load_training_samples(letter_samples):
    xs = []
    ys = []

    letter_size = {}

    for im_path in glob.glob("data/*/*.png"):
        letter = int(im_path.split('/')[1])
        if letter_size.get(letter) is None:
            letter_size[letter] = 0

        if letter_size[letter] < letter_samples:
            x_loaded = image_to_x(load_image(im_path))
            xs.append(x_loaded)
            ys.append(letter_to_y(letter))
            letter_size[letter] += 1

    return xs, ys


def train_network(xs, ys, epochs=100, learn_speed=0.00000000000001):
    # Dense(in_shape=16 * 16, out_shape=16 * 16 * 2, learn_speed=learn_speed),
    # Dense(in_shape=16 * 16 * 2, out_shape=26, learn_speed=learn_speed)
    # network = Network(epochs=epochs, layers=[
    #     Dense(in_shape=16 * 16, out_shape=16*16*2, learn_speed=learn_speed),
    #     Dense(in_shape=16*16*2, out_shape=6, learn_speed=learn_speed),
    #     Dense(in_shape=6, out_shape=26, learn_speed=learn_speed)
    # ])

    network = Network(epochs=epochs, layers=[
        Dense(in_shape=16 * 16, out_shape=16 * 4, learn_speed=learn_speed),
        Dense(in_shape=16 * 4, out_shape=26, learn_speed=learn_speed),
    ])
    network.teach(xs, ys)
    return network


def validate_net(network, path):
    for im_path in glob.glob(path + "/*.png"):
        im = load_image(im_path)
        x = image_to_x(im)
        y_hat = network.predict(x)
        letter_hat = y_to_letter(y_hat)
        plt.figure()
        plt.title(letter_hat)
        plt.imshow(im)
        plt.show()


def plot_confusion_matrix(net, xs, ys):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for x, y in zip(xs, ys):
        pass


def test_validate_model():
    xs, ys = load_training_samples(700)
    print(f'{len(xs)} images loaded')
    # 0.000000000000001
    network = train_network(xs, ys, epochs=50, learn_speed=0.00000000001)
    network.plot_loss(from_epoch=10)
    validate_net(network, path="validation_data")
