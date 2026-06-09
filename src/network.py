import numpy as np


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= lr * layer.dweights
                layer.biases -= lr * layer.dbiases

    def train(self, x, y, epochs, lr):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = -np.mean(np.sum(y * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7)), axis=1))
            grad = -y / np.clip(y_pred, 1e-7, 1 - 1e-7) / len(y)
            self.backward(grad)
            self.update(lr)
            if epoch % 10 == 0:
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                print(f"Epoch {epoch:3d} | loss: {loss:.4f} | acc: {acc:.4f}")