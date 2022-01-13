import glob

import imageio
import matplotlib.pyplot as plt
import numpy as np

from network import Network



def image_to_x(image):
    return image.reshape(16 * 16)


def load_image(im_path):
    return imageio.imread(im_path)


def letter_to_y(letter: int) -> np.array:
    res = np.zeros(26)
    res[letter - 1] = 1
    return res


def y_to_letter(y: np.array):
    # print(y)
    return chr(y.argmax() + 65)


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


def train_network(xs, ys, epochs=100):
    network = Network(epochs)
    network.teach(xs, ys)
    network.plot_loss()
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


def test_validate_model():
    xs, ys = load_training_samples(26_000)
    print(f'{len(xs)} images loaded')
    network = train_network(xs, ys, 60)
    validate_net(network, path="validation_data")
