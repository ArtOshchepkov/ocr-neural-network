import numpy as np

from network import Network

x = np.zeros((16, 16))

x = x.reshape(16 * 16)

net = Network()

res = net.predict(x)

print(res)