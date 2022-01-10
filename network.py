import sys
from random import random
from typing import List

import numpy as np

input = np.zeros((16, 16))

speed = 0.01


class Network:
    EPOCHS = 200

    forget_speed = 10

    loss_value = 0

    layers = []

    def __init__(self) -> None:
        self.layers = [
            Dense(16 * 16, 27),
        ]

    def predict(self, x):
        for layer in self.layers:
            x = layer.f(x)
        return x

    def teach(self, xs: List[np.array], ys: List[np.array]):
        print('Shapes for teaching are', xs[0].shape, ys[0].shape)
        if len(xs) != len(ys):
            raise ValueError("x y should have same size")

        print(F'Teaching network on {len(xs)} samples')

        for i in range(0, self.EPOCHS):
            self.loss_value = self.total_loss(xs, ys)
            print(f'Epoch {i}, loss {self.loss_value}', flush=True)
            for x, y in sorted(zip(xs, ys), key=lambda _: random()):
                self.teach_sample(x, y)

    def total_loss(self, xs: List[np.array], ys: List[np.array]):
        error = 0
        for x, y in zip(xs, ys):
            error += self.loss(x, y)
        return error / len(xs)

    def teach_sample(self, x: np.array, y: np.array):
        """
        SGD
        """
        pred = self.predict(x)
        print(f'Prediceed shape {pred.shape}')
        err = self.loss_derivative(pred, y)
        self.loss_value = self.forget_speed * err + (1 - self.forget_speed) * self.loss_value
        local_grad = err
        for layer in reversed(self.layers):
            local_grad = layer.teach(local_grad)

    def loss(self, x: np.array, y: np.array):
        return (self.predict(x) - y) ** 2

    def loss_derivative(self, prediction, y):
        return 2 * (prediction - y)


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return x if x > 0 else 0


class Dense:
    w: np.array

    def __init__(self, in_shape, out_shape) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.w = np.empty((out_shape, in_shape))
        print('Init Dense layer with shape ' + str(self.w.shape))

    def f(self, x: np.array) -> np.array:
        signal = self.w.dot(x)
        activation = np.vectorize(relu)
        return np.transpose(activation(signal))

    def teach(self, local_grad: np.array) -> np.array:
        activation_grad_f = np.vectorize(relu_derivative)
        activation_grad = activation_grad_f(local_grad)
        w_grad = activation_grad * self.w
        self.w -= speed * w_grad
        return w_grad
