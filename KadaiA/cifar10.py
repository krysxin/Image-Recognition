import numpy as np
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    X = np.array(dict[b'data'])
    X = X.reshape((X.shape[0],3,32,32))
    Y = np.array(dict[b'labels'])
    return X,Y


def load_CIFAE10(root):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'cifar-10-batches-py/data_batch_%d' % (b,))
        X, Y = unpickle(f)
        xs.append(X)         #全てのbatchを統合する
        ys.append(Y)
    X_train = np.concatenate(xs) # 行ベクトルに変換する, X_trainサイズ(50000,3,32,32)
    Y_train = np.concatenate(ys)
    del X,Y
    X_test, Y_test = unpickle(os.path.join(root, 'cifar-10-batches-py/test_batch'))
    return X_train, Y_train, X_test, Y_test

# root = os.path.dirname(os.path.abspath(__file__))
# Xtr, Ytr, Xt, Yt = load_CIFAE10(root)
# print(Xtr.shape, Ytr.shape, Xt.shape, Yt.shape) # (50000, 3, 32, 32) (50000,) (10000, 3, 32, 32) (10000,)