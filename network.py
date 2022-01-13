import sys
from random import random
from typing import List
import matplotlib.pyplot as plt

import numpy as np

input = np.zeros((16, 16))


class LearningLog:
    loss = []
    mpe = []
    mae = []


class Dense:
    w: np.array

    def __init__(self, in_shape, out_shape, learn_speed=0.00001) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.learn_speed = learn_speed
        self.w = np.random.rand(out_shape, in_shape)
        # print(self.w)
        print('Init Dense layer with shape ' + str(self.w.shape))

    def f(self, x: np.array) -> np.array:
        signal = self._z(x)
        activation = np.vectorize(relu)
        return activation(signal)

    def _z(self, x):
        return self.w.dot(x)

    def _z_derivative(self, x):
        return x

    def teach(self, x: np.array, local_grad: np.array) -> np.array:
        relu_derivative_vector = np.vectorize(relu_derivative)
        activation_derivative = relu_derivative_vector(self._z(x))
        w_grad = local_grad * activation_derivative * self._z_derivative(x)
        self.w -= self.learn_speed * w_grad
        # print(self.w)
        # print(f'Input/Output local grads shapes: {local_grad.shape} {w_grad.shape}')
        return w_grad


class Network:
    forget_speed = 15

    loss_value = 0

    layers = []

    def __init__(self, epochs=100, layers=[Dense(16 * 16, 26)]) -> None:
        self.epochs = epochs
        self.layers = layers
        self.learning_log = LearningLog()

    def predict(self, x):
        for layer in self.layers:
            x = layer.f(x)
        return x

    def teach(self, xs: List[np.array], ys: List[np.array]):
        print('Shapes for teaching are', xs[0].shape, ys[0].shape)
        if len(xs) != len(ys):
            raise ValueError("x y should have same size")

        print(F'Teaching network on {len(xs)} samples')

        for i in range(0, self.epochs):
            self.loss_value = self.total_loss(xs, ys)
            self.learning_log.loss.append(np.sum(self.loss_value))
            self.learning_log.mpe.append(self.total_mpe(xs, ys))
            self.learning_log.mae.append(self.total_mae(xs, ys))
            print(f'Epoch {i}, loss {np.sum(self.loss_value)} {self.loss_value}', flush=True)
            for x, y in sorted(zip(xs, ys), key=lambda _: random()):
                self.teach_sample(x, y)

    def total_mpe(self, xs, ys):
        mpes = []
        for x, y in zip(xs, ys):
            pred = self.predict(x)
            percentage_error = abs(y - pred) / y
            mpes.append(percentage_error)
        return np.median(mpes)

    def total_mae(self, xs, ys):
        maes = []
        for x, y in zip(xs, ys):
            pred = self.predict(x)
            abs_error = abs(y - pred)
            maes.append(abs_error)
        return np.median(maes)


    def plot_loss(self, from_epoch=10):
        plt.figure()
        print(f'Loss: {self.learning_log.loss}')
        plt.title('Total Loss')
        plt.plot(self.learning_log.loss[from_epoch:])
        plt.show()

    def plot_mpes(self, from_epoch=10):
        plt.figure()
        plt.title('Median Percentage Error')
        print(f'MPES: {self.learning_log.mpe}')
        plt.plot(self.learning_log.mpe[from_epoch:])
        plt.show()

    def plot_mae(self, from_epoch=10):
        plt.figure()
        plt.title('Median Absolute Error')
        print(f'MAES: {self.learning_log.mae}')
        plt.plot(self.learning_log.mae[from_epoch:])
        plt.show()

    def total_loss(self, xs: List[np.array], ys: List[np.array]):
        error = 0
        for x, y in zip(xs, ys):
            pred = self.predict(x)
            error += self.loss(pred, y)
        return error / len(xs)

    def teach_sample(self, x: np.array, y: np.array):
        """
        SGD
        """
        pred = self.predict(x)
        # print(pred)
        # print(f'Predict shape {pred.shape}')
        err = self.loss(pred, y)
        # https://neerc.ifmo.ru/wiki/index.php?title=%D0%A1%D1%82%D0%BE%D1%85%D0%B0%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9_%D1%81%D0%BF%D1%83%D1%81%D0%BA
        self.loss_value = err / self.forget_speed + (1 - 1 / self.forget_speed) * self.loss_value
        local_grad = self.loss_derivative(pred, y)
        for layer in reversed(self.layers):
            local_grad = layer.teach(x, local_grad)

    def loss(self, pred: np.array, y: np.array):
        if pred.shape != y.shape:
            raise ValueError("Prediction shape " + str(pred.shape) + " doesn't mach y shape " + str(y.shape))
        return 0.5 * (pred - y) ** 2

    def loss_derivative(self, prediction, y):
        return prediction - y


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return x if x > 0 else 0
