import numpy as np
import matplotlib.pylab as plt


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

def im2col(img,f_h,f_w,stride=1,padding=0):
    num_bt, c, h, w = img.shape
    new_h = (h + 2 * padding - f_h) // stride + 1
    new_w = (w + 2 * padding - f_w) // stride + 1

    img_ = np.pad(img, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    im_col = np.zeros((num_bt, c, f_h, f_w, new_h, new_w))

    for y in range(f_h):
        y_max = y + stride * new_h
        for x in range(f_w):
            x_max = x + stride * new_w 
            im_col[:, :, y, x, :, :] = img_[:, :, y:y_max:stride, x:x_max:stride]
    
    im_col = im_col.transpose(0, 4, 5, 1, 2, 3) 
    im_col = im_col.reshape(num_bt * new_h * new_w, -1)

    return im_col

def col2im(col,img_shape,f_h,f_w,stride=1,padding=0):
    num_bt, c, h, w = img_shape
    new_h = (h + 2*padding - f_h) // stride + 1
    new_w = (w + 2*padding - f_w) // stride + 1
    col = col.reshape(num_bt, new_h, new_w, c, f_h, f_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((num_bt, c, h + 2*padding + stride - 1, w + 2*padding + stride - 1))
    for y in range(f_h):
        y_max = y + stride * new_h
        for x in range(f_w):
            x_max = x + stride * new_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:h + padding, padding:w + padding]

