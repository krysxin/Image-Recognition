import numpy as np
from funcs import sigmoid, softmax, cross_entropy_error, im2col, col2im

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


class Dropout:
    def __init__(self, p=0.2):
        self.p = p
        self.mask = None
    
    def forward(self, x, training = False):
        t = 1 - self.p
        if training:
            self.mask = np.random.uniform(size=x.shape) > self.p
            t = self.mask
        out = x * t
        return out

    def backward(self, dout):
        return dout * self.mask


class FullyConnected:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        # (batch_size, c, h, w)に対応するため
        self.x_shape = None
        self.dW = None
        self.db = None
        
    def forward(self,x):
        self.x_shape = x.shape
        self.x = x.reshape(x.shape[0],-1) #追加
        out = np.dot(self.x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.x_shape) #追加
        return dx

class SoftmaxLoss:
    def __init__(self):
        self.loss = None      # loss
        self.y = None         # output of softmax
        self.y_one_hot = None # lable(one-hot-vector)

    def forward(self, x, y_one_hot):
        self.y_one_hot = y_one_hot
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.y_one_hot)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.y_one_hot.shape[0]
        dx = (self.y - self.y_one_hot) * (dout / batch_size)

        return dx

class BatchNorm:
    def __init__(self, gamma, beta, eps=10e-7, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.momentum = momentum

        self.running_mean = running_mean
        self.running_var = running_var
        self.dgamma = None
        self.dbeta = None
        self.cache = None
        self.x_shape = None

    def forward(self, x, training=True):
        self.x_shape = x.shape
        # if x.ndim == 1:
        #     x = x.reshape(1,-1)
        if x.ndim != 2: #4次元配列に対応
            n = x.shape[0]
            x = x.reshape(n,-1)

        if self.running_mean is None:
           N, D = x.shape
           self.running_mean = np.zeros(D, dtype=x.dtype) 
           self.running_var = np.zeros(D, dtype=x.dtype)

        if training:
           mean = x.mean(axis=0)
           var = x.var(axis=0) 
           std = np.sqrt(var + self.eps)
           x_centered = x - mean
           x_norm = x_centered / std
           out = self.gamma * x_norm + self.beta

           self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
           self.running_var = self.momentum * self.running_var + (1-self.momentum) * var

           self.cache = (std, x_centered, x_norm)

        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        
        out = out.reshape(*self.x_shape)

        return out

    def backward(self, dout):
        if dout.ndim != 2:
            n = dout.shape[0]
            dout = dout.reshape(n, -1)

        N = dout.shape[0]
        std, x_centered, x_norm = self.cache

        dgamma = np.sum(x_norm * dout, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx_norm = dout * self.gamma 
        dx_centered = dx_norm / std
        dvar = np.sum(dx_norm * x_centered, axis=0) * (-0.5) * std**(-3)
        dmean = -(np.sum(dx_centered, axis=0) + 2/N * dvar * np.sum(x_centered, axis=0)) 
        dx = dx_centered + (dvar * 2 * x_centered + dmean) / N
        dx = dx.reshape(*self.x_shape) #new

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Conv2D:
    def __init__(self,W,b,stride=1,padding=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.padding = padding

        self.dW = None
        self.db = None
        self.cache = None

    def forward(self,x):

        num_f, c, f_h, f_w = self.W.shape
        num_bt, c, h, w = x.shape
        new_h = int((h + 2 * self.padding - f_h)/ self.stride) + 1
        new_w = int((w + 2 * self.padding - f_w)/ self.stride) + 1
        
        x_col = im2col(x,f_h,f_w,self.stride,self.padding)
        x_col_W = self.W.reshape(num_f,-1).T

        out = np.dot(x_col, x_col_W) + self.b
        out = out.reshape(num_bt, new_h, new_w, -1).transpose(0, 3, 1, 2)

        self.cache = x, x_col, x_col_W
        return out
        
    def backward(self,dout):
        x, x_col, x_col_W = self.cache
        num_f, c, f_h, f_w = self.W.shape

        dout = dout.transpose(0,2,3,1).reshape(-1, num_f)

        self.db = np.sum(dout, axis=0)
        dW_col = np.dot(x_col.T, dout)
        self.dW = dW_col.transpose(1, 0).reshape(num_f, c, f_h, f_w)

        dx_col = np.dot(dout, x_col_W.T)
        dx = col2im(dx_col, x.shape, f_h, f_w, self.stride, self.padding)

        return dx


class MaxPool:
    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w  = pool_w
        self.stride = stride
        self.padding = padding
        self.cache = None

    def forward(self, x):

        num_bt, c, h, w = x.shape
        new_h = int(1 + (h - self.pool_h) / self.stride)
        new_w = int(1 + (w - self.pool_w) / self.stride)
        
        x_col = im2col(x, self.pool_h, self.pool_w, self.stride, self.padding)
        x_col = x_col.reshape(-1, self.pool_h*self.pool_w)
        arg_max = np.argmax(x_col, axis=1)
        max_pool = np.max(x_col, axis=1)
        max_pool = np.array(np.hsplit(max_pool, num_bt))
        max_pool = max_pool.reshape(num_bt, c, new_h, new_w)

        self.cache = x, arg_max
        return max_pool

    def backward(self, dout):
        x, arg_max = self.cache
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, x.shape, self.pool_h, self.pool_w, self.stride, self.padding)
        
        return dx

        





 
        
'''
class Conv2D:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class MaxPool:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

  '''  


