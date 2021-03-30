import numpy as np

class SGD:
    def __init__(self,lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum_SGD:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params:
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val) + 1e-8

        for key in params:
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key])


class RMSProp:
    def __init__(self, lr=0.001, p=0.9, eps=1e-8):
        self.lr = lr
        self.p = p
        self.eps = eps
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params:
            self.h[key] = self.p * self.h[key] + (1-self.p) * (grads[key] * grads[key])
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)

class AdaDelta:
    def __init__(self, p=0.95, eps=1e-6):
        self.p = p
        self.eps = eps
        self.h = None
        self.s = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            self.s = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                self.s[key] = np.zeros_like(val)

        for key in params:
            self.h[key] = self.p * self.h[key] + (1-self.p) * (grads[key] * grads[key])
            deltaw = -np.sqrt(self.s[key]+self.eps)/np.sqrt(self.h[key]+self.eps) * grads[key]
            self.s[key] = self.p * self.h[key] + (1-self.p) * (deltaw * deltaw)
            params[key] += deltaw

class Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1-self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1-self.beta2) * (grads[key] * grads[key])
            _m_ = self.m[key] / (1 - self.beta1 ** self.iter)
            _v_ = self.v[key] / (1 - self.beta2 ** self.iter)
            params[key] -= self.alpha * _m_ / (np.sqrt(_v_) + self.eps)

