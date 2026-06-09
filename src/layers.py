import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dout):
        self.dweights = np.dot(self.x.T, dout)
        self.dbiases = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.weights.T)
