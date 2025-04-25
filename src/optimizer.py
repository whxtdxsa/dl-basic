import numpy as np
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr, alpha=0.9):
        self.lr = lr
        self.alpha = alpha
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, param in params.items():
                self.v[key] = np.zeros_like(param)

        for key in params.keys():
            self.v[key] = self.alpha * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr):
        self.h = None
        self.lr = lr
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, param in params.items():
                self.h[key] = np.zeros_like(param) 

        for key in params.keys():
            self.h[key] = self.h[key] + grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

