import numpy as np
import mnist
import matplotlib.pylab as plt
from networkA7 import NetworkA7
from pylab import cm


X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")


# 入力獲得
flag = True
while flag:
        i = input("Enter a number between 0 and 9999: ")
        if i.isdigit():
                i = (int)(i)
                if i>=0 and i<=9999:
                        flag = False
                else:
                        print("Invalid Input!")
        else:
                print("Invalid Input!")

#　ネットワークサイズ指定
l0 = X.shape[1]* X.shape[2] 
l1 = 100  #中間層
l2 = len(np.unique(Y))  #出力層
sizes = np.array([l0,l1,l2])

#　ネットワーク構築
net = NetworkA7(sizes,batch_size=100, epoch_num=10, 
                use_trained_params=True,filename='params_mnist_v1.npz', 
                img_dim=(1,28,28),
               conv_param={'filter_num':32, 'filter_size':3, 'padding':1, 'stride':1},
               optimizer='Adam', activation='ReLU',use_dropout=False,dropout_p=0.2,use_bn=True)
result = net.get_result(X[i])
# x,y = net.normalize_vec(X,Y)
# acc = net.accuracy(x,y)
# print(acc)


print("Perdicted Result: {result}".format(result=result))
print("Expected Result: {right_answer} ".format(right_answer=Y[i]))
if(Y[i]==result):
    print("Correct Answer!")
else:
    print("Perdiction Failed...")

plt.imshow(X[i], cmap=cm.gray)
plt.show()