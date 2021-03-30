import numpy as np
from funcs import sigmoid, softmax, cross_entropy_error

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = sigmoid(x)
        self.out = out

        return out
    
    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class ReLU:
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        out = np.maximum(x, 0)
        self.out = out

        return out
    
    def backward(self,dout):
        dx = dout * (self.out>0)
        return dx



class FullyConnected:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self,x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxLoss:
    def __init__(self):
        self.loss = None      # loss
        self.y = None         # output of softmax
        self.y_one_hot = None # label(one-hot-vector)

    def forward(self, x, y_one_hot):
        self.y_one_hot = y_one_hot
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.y_one_hot)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.y_one_hot.shape[0]
        dx = (self.y - self.y_one_hot) * (dout / batch_size)

        return dx

