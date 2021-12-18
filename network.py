import numpy as np

input = np.zeros((16, 16))


class Network:
    layers = []

    def __init__(self) -> None:
        self.layers = [
            Dense(16 * 16, 1),
        ]

    def predict(self, x):
        for layer in self.layers:
            x = layer.f(x)
        return x

    def teach(self, x, y):
        pass



def relu(x):
    return max(0, x)


class Dense:
    w: np.array

    def __init__(self, in_shape, out_shape) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.w = np.array((in_shape, out_shape))

    def f(self, x: np.array) -> np.array:
        # print(x.shape)
        # if x.shape != self.in_shape:
        #     raise ValueError("In shape " + x.shape + " != " + x.shape)

        signal = self.w.dot(x)
        activation = np.vectorize(relu)
        return activation(signal)
