import numpy as np


class NeuralNetwork(object):
    def __init__(self):
        # define hyper parameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        # weight
        self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, x):
        # propagate inputs though network
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # activation function
        return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    x = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    nn = NeuralNetwork()
    yHat = nn.forward(x)

    print(yHat)
