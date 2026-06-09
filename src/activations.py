import numpy as np

class ReLU:
    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, dout):
        return dout * (self.z > 0)
    
class Sigmoid:
    def forward(self, z):
        self.out = 1 / (1 + np.exp(-z))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

class Softmax:
    def forward(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.out = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        return dout