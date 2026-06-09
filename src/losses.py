import numpy as np

class CrossEntropyLoss:
    def forward(self,y_pred,y_true):
        self.y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        self.y_true = y_true
        return -np.mean(np.sum(y_true * np.log(self.y_pred), axis = 1))
    
    def backward(self):
        return -self.y_true / self.y_pred / len(self.y_pred)