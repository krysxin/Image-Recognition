import numpy as np
import os
import matplotlib.pylab as plt
from networkA5 import NetworkA5
from pylab import cm
from cifar10 import load_CIFAE10



root = os.path.dirname(os.path.abspath(__file__))
X_train, Y_train, X_test, Y_test = load_CIFAE10(root)


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
l0 = X_test.shape[2]* X_test.shape[3] 
l1 = 100  #中間層
l2 = len(np.unique(Y_test))  #出力層
sizes = np.array([l0,l1,l2])

#　ネットワーク構築
net = NetworkA5(sizes,batch_size=100, epoch_num=10, 
                use_trained_params=True,filename='params_cifar10.npz', 
                img_dim=(3,32,32),
               conv_param={'filter_num':32, 'filter_size':3, 'padding':1, 'stride':1},
               optimizer='Adam', activation='ReLU',use_dropout=False,dropout_p=0.2,use_bn=True)
result = net.get_result(X_test[i])


print("Perdicted Result: {result}".format(result=result))
print("Expected Result: {right_answer} ".format(right_answer=Y_test[i]))
if(Y_test[i]==result):
    print("Correct Answer!")
else:
    print("Perdiction Failed...")

plt.title('image{i}'.format(i=i))
plt.imshow(X_test[i].transpose((1,2,0)))
plt.show()