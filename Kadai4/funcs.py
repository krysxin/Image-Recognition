import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def softmax(x):
    row_max = np.max(x, axis = -1, keepdims=True)     
    x = x - row_max                         
    sum_expx = np.sum(np.exp(x), axis=-1, keepdims=True)
    y = np.exp(x) / sum_expx
    return y

def cross_entropy_error(y,y_one_hot):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        y_one_hot = y_one_hot.reshape(1,y_one_hot.size)
    
    batch_size = y.shape[0]
    return -np.sum(y_one_hot * np.log(y + delta)) / batch_size