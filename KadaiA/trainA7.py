import numpy as np
import mnist
import matplotlib.pylab as plt
from layers import Sigmoid, ReLU
from networkA7v2 import NetworkA7v2

X_train = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y_train = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
X_test = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y_test = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")


l0 = X_train.shape[1] * X_train.shape[2] #入力層(28*28)
l1 = 100  #中間層
l2 = len(np.unique(Y_train))  #出力層
net_sizes = np.array([l0,l1,l2]) #(784,50,10)

net = NetworkA7v2(net_sizes,batch_size=100, epoch_num=10, 
               use_trained_params=False, filename=None,
               img_dim=(1,28,28),
               conv_param={'filter_num':32, 'filter_size':3, 'padding':1, 'stride':1},
               optimizer='Adam', activation='ReLU',use_dropout=False,dropout_p=0.2,use_bn=True)
params = net.train(X_train, Y_train, X_test, Y_test, \
                    print_accuracy=False, show_loss_graph=True, show_accuracy_graph=False)
net.save_params(params)



