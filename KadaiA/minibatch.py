import numpy as np  
import mnist

X_train = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y_train = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
X_test = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y_test = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")

  
def shuffle_batches(X,Y,batch_size):
    data_num = X.shape[0]
    mini_batches = []

    index = np.arange(data_num)
    np.random.shuffle(index)
    shuffled_X = X[index]
    shuffled_Y = Y[index]

    num_minibatches = int(data_num / batch_size)
    for k in range (0,num_minibatches):
        mini_batch_X = shuffled_X[(k * batch_size): ((k+1) * batch_size)]  
        mini_batch_Y = shuffled_Y[(k * batch_size): ((k+1) * batch_size)]
        mini_batch = (mini_batch_X, mini_batch_Y) #((100,784),(100,10))
        mini_batches.append(mini_batch)

    return mini_batches

# b = shuffle_batches(X_test,Y_test,100)
# print(b[0][0].reshape(100,1,28,28).shape)



