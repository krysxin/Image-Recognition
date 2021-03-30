import numpy as np
import mnist
import matplotlib.pylab as plt
from network3 import Network3
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
l1 = 50  #中間層ノード数
l2 = len(np.unique(Y))  
sizes = np.array([l0,l1,l2])

#　ネットワーク構築
net = Network3(sizes,batch_size=100, epoch_num=100, learning_rate =0.01,
               use_trained_params=True, filename='params_50n_100e.npz')
result = net.get_result(X[i])


print("Perdicted Result: {result}".format(result=result))
print("Expected Result: {right_answer} ".format(right_answer=Y[i]))
if(Y[i]==result):
    print("Correct Answer!")
else:
    print("Perdiction Failed...")

plt.title('Image{i}: {result}'.format(i=i,result=result))
plt.imshow(X[i], cmap=cm.gray)
plt.show()



