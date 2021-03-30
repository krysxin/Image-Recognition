import numpy as np
import os
import matplotlib.pylab as plt
from cifar10 import load_CIFAE10
from layers import Sigmoid, ReLU
from networkA5 import NetworkA5

root = os.path.dirname(os.path.abspath(__file__))
X_train, Y_train, X_test, Y_test = load_CIFAE10(root)
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

l0 = X_train.shape[2] * X_train.shape[3] #入力層(32*32)
l1 = 100  #中間層
l2 = len(np.unique(Y_train))  #出力層
net_sizes = np.array([l0,l1,l2]) #(32*32,100,10)

net = NetworkA5(net_sizes,batch_size=100, epoch_num=10, 
               use_trained_params=False, filename=None,
               img_dim=(3,32,32),
               conv_param={'filter_num':32, 'filter_size':3, 'padding':1, 'stride':1},
               optimizer='Adam', activation='ReLU',use_dropout=False,dropout_p=0.2,use_bn=True)
params = net.train(X_train, Y_train, X_test, Y_test, \
                    print_accuracy=False, show_loss_graph=True, show_accuracy_graph=False)
net.save_params(params)



