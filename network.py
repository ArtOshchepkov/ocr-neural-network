import sys
from random import random
from typing import List
import matplotlib.pyplot as plt

import numpy as np

input = np.zeros((16, 16))

learn_speed = 0.00001


class LearningLog:
    loss = []


class Network:

    forget_speed = 15

    loss_value = 0

    layers = []

    def __init__(self, epochs=100) -> None:
        self.epochs = epochs
        self.layers = [
            Dense(16 * 16, 26),
        ]
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
            print(f'Epoch {i}, loss {np.sum(self.loss_value)} {self.loss_value}', flush=True)
            for x, y in sorted(zip(xs, ys), key=lambda _: random()):
                self.teach_sample(x, y)

    def printLog(self):
        plt.figure()
        print(self.learning_log.loss)
        plt.plot(self.learning_log.loss[10:])
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
        # print(f'Predict shape {pred.shape}')
        err = self.loss(pred, y)
        self.loss_value = self.forget_speed * err + (1 - self.forget_speed) * self.loss_value
        local_grad = self.loss_derivative(pred, y)
        for layer in reversed(self.layers):
            local_grad = layer.teach(local_grad)

    def loss(self, pred: np.array, y: np.array):
        return 0.5 * (pred - y) ** 2

    def loss_derivative(self, prediction, y):
        return prediction - y


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return x if x > 1 else 0


class Dense:
    w: np.array

    def __init__(self, in_shape, out_shape) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.w = np.random.rand(out_shape, in_shape) - 0.5
        # print(self.w)
        print('Init Dense layer with shape ' + str(self.w.shape))

    def f(self, x: np.array) -> np.array:
        signal = self.w.dot(x)
        activation = np.vectorize(relu)
        return activation(signal)

    def teach(self, local_grad: np.array) -> np.array:
        activation_grad_f = np.vectorize(relu_derivative)
        activation_grad = activation_grad_f(local_grad)
        w_grad = activation_grad * self.w.transpose()  # TODO: transpose correct?
        self.w -= (learn_speed * w_grad).transpose()  # TODO: transpose correct?
        # print(self.w)
        # print(f'Input/Output local grads shapes: {local_grad.shape} {w_grad.shape}')
        return w_grad
