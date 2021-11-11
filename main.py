import numpy as np
from matplotlib import cm
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sigmoid(z):
    # activation function
    return 1 / (1 + np.exp(-z))


def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


class NeuralNetwork(object):
    def __init__(self):
        # define hyper parameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        # weight
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        # init
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.yHat = None

    def forward(self, X):
        # propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = sigmoid(self.z3)
        return yHat

    def costFunctionPrime(self, X, y):
        # compute derivative with respect to W1 and W2
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y - self.yHat), sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    def costFunction(self, X, y):
        # Compute cost for given X,y use weights already stored in class
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def getParams(self):
        # Get W1 and W2 Rolled into vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))

        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def computeNumericalGradient(self, X, y):
        # Get good numerical estimates of the derivative
        paramsInitial = self.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = e
            self.setParams(paramsInitial + perturb)
            loss2 = self.costFunction(X, y)

            self.setParams(paramsInitial - perturb)
            loss1 = self.costFunction(X, y)

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2 * e)
            perturb[p] = 0

        self.setParams(paramsInitial)

        return numgrad


class Trainer(object):
    def __init__(self, N):
        # Make local reference to Neural Network
        self.N = N

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def train(self, X, y):
        self.X = X
        self.y = y
        self.J = []

        params0 = self.N.getParams()
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0,
                                 jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res


if __name__ == '__main__':
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([0.75], [0.82], [0.93]), dtype=float)

    nn = NeuralNetwork()

    # test one
    # yHat = nn.forward(x)
    # print(yHat)

    # test two
    # cost1 = nn.costFunctionPrime(X, y)
    # dJdW1, dJdW2 = nn.costFunctionPrime(X, y)
    # print(dJdW1)
    # print(dJdW2)

    # test three
    # numgrad = nn.computeNumericalGradient(X, y)
    # grad = nn.computeGradients(X, y)
    # print(numgrad)
    # print(grad)

    # train the neural network
    T = Trainer(nn)
    T.train(X, y)

    # plt.plot(T.J)
    # plt.grid = 1
    # plt.ylabel('cost')
    # plt.xlabel('iterations')
    # plt.show()
    print(nn.forward(X))

    # test network for various combination of sleep /study
    hoursSleep = np.linspace(0, 10, 100)
    hoursStudy = np.linspace(0, 5, 100)

    hoursSleepNorm = hoursSleep / 10
    hoursStudyNorm = hoursStudy / 5

    a, b = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

    allInputs = np.zeros((a.size, 2))
    allInputs[:, 0] = a.ravel()
    allInputs[:, 1] = b.ravel()

    allOuputs = nn.forward(allInputs)

    yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1, 100)))
    xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1, 100))).T

    cs = plt.contour(xx, yy, 100 * allOuputs.reshape(100, 100))
    plt.clabel(cs, inline=1, fontsize=10)
    plt.xlabel('Hours Sleep')
    plt.ylabel('Hours Study')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, 100 * allOuputs.reshape(100, 100), cmap=cm.jet)
    ax.set_xlabel('Hours Sleep')
    ax.set_xlabel('Hours Study')
    ax.set_zlabel('Test Score')
    plt.show()