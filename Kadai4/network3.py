import numpy as np
import matplotlib.pylab as plt
from pylab import cm
import os
from funcs import sigmoid, softmax, cross_entropy_error
from layers import Sigmoid, ReLU, FullyConnected, SoftmaxLoss
from minibatch import shuffle_batches


class Network3:
    def __init__(self,sizes, batch_size, epoch_num, learning_rate, use_trained_params=False,filename=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate

        if use_trained_params:
            path = os.path.dirname(os.path.abspath(__file__))
            loaded_params = np.load(os.path.join(path,filename))
            self.W1 = loaded_params['W1']
            self.b1 = loaded_params['b1']
            self.W2 = loaded_params['W2']
            self.b2 = loaded_params['b2']
        else:
            np.random.seed(12)
            self.W1 = np.sqrt(1/sizes[0]) * np.random.randn(sizes[0],sizes[1]) #（784,50）
            self.b1 = np.sqrt(1/sizes[0]) * np.random.randn(sizes[1])
            self.W2 = np.sqrt(1/sizes[1]) * np.random.randn(sizes[1],sizes[2]) #（50,10)
            self.b2 = np.sqrt(1/sizes[1]) * np.random.randn(sizes[2])

        # layers of network
        self.layers = {}
        self.layers['FullyConnected1'] = FullyConnected(self.W1, self.b1)
        self.layers['Activation'] = Sigmoid()
        self.layers['FullyConnected2'] = FullyConnected(self.W2, self.b2)

        self.lastLayer = SoftmaxLoss()

    def normalize_vec(self,x,y):
        I = x.shape[0] #60000
        C = len(np.unique(y)) #10
        x = x.reshape(I,-1)
        x_norm = x.astype(np.float32)/255.0
        y_one_hot_v = np.eye(C)[y] #(60000,10)

        return x_norm, y_one_hot_v


    def feedforward(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, y_one_hot):
        y = self.feedforward(x)
        loss = self.lastLayer.forward(y,y_one_hot)
        return loss

    def accuracy(self, x, y_one_hot):
        y = self.feedforward(x)
        y = np.argmax(y, axis=1)
        y_one_hot = np.argmax(y_one_hot, axis=1)

        accurate_num = np.sum(y == y_one_hot) 
        return accurate_num


    def gradient(self, x, y_one_hot):
        # forward
        self.loss(x, y_one_hot)

        #backward
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in  layers:
            dout = layer.backward(dout)
        
        # gradient of weights,bias
        grads = {}
        grads['W1'] = self.layers['FullyConnected1'].dW
        grads['b1'] = self.layers['FullyConnected1'].db
        grads['W2'] = self.layers['FullyConnected2'].dW
        grads['b2'] = self.layers['FullyConnected2'].db

        return grads

    
    

    def train(self, X_train, Y_train, X_test, Y_test, print_accuracy=False, show_loss_graph=False, show_accuracy_graph=False):
        x_train, y_train = self.normalize_vec(X_train, Y_train)
        x_test, y_test = self.normalize_vec(X_test, Y_test)
        train_loss = []
        train_acc_histroy = []
        test_acc_history = []
        params = {}

        for i in range(self.epoch_num):
            loss_per_epoch = 0
            minibatches = shuffle_batches(x_train,y_train,self.batch_size)

            for j in range(len(minibatches)):
                x_batch = minibatches[j][0]
                y_batch = minibatches[j][1]

                grads = self.gradient(x_batch,y_batch)
            
                self.W1 -= self.learning_rate * grads['W1']
                self.b1 -= self.learning_rate * grads['b1']
                self.W2 -= self.learning_rate * grads['W2']
                self.b2 -= self.learning_rate * grads['b2']

                loss = self.loss(x_batch, y_batch)
                loss_per_epoch += loss

            ave_loss = loss_per_epoch / len(minibatches)
            train_loss.append(ave_loss)

            if print_accuracy:
                train_accuracy = self.accuracy(x_train,y_train) 
                test_accuracy = self.accuracy(x_test,y_test)
                train_acc_histroy.append(train_accuracy/X_train.shape[0])
                test_acc_history.append(test_accuracy/X_test.shape[0])

                print("Epoch {i}: Loss={ave_loss}  Accuracy_train={train_acc:.4f}  Accuracy_test={test_acc:.4f}"\
                    .format(i=i,ave_loss=ave_loss,train_acc=train_accuracy/X_train.shape[0],test_acc=test_accuracy/X_test.shape[0]))

            else:
                print("Epoch {i}: Loss={ave_loss}".format(i))

        print("Final test_accuracy: {acc}".format(acc=self.accuracy(x_test,y_test)/X_test.shape[0]))

        if show_loss_graph:
            x = np.linspace(0,self.epoch_num,self.epoch_num)
            plt.plot(x, train_loss, label='loss')
            plt.title('Average Loss of Each Epoch')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            # plt.savefig('fig_loss_{mid_node_num}n_{epoch_num}e.png'\
            #     .format(mid_node_num=self.sizes[1], epoch_num=self.epoch_num))
            plt.show()
        if show_accuracy_graph:
            x2 = np.linspace(0,len(test_acc_history),len(test_acc_history))
            plt.plot(x2, train_acc_histroy, label='train accuracy')
            plt.plot(x2, test_acc_history, label='test accuracy', linestyle='--')
            plt.title('Accuracy After Each Epoch')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.xlim(left=0)
            plt.ylim(0, 1.0)
            plt.legend(loc='lower right')
            # plt.savefig('fig_acc_{mid_node_num}n_{epoch_num}e.png'\
            #     .format(mid_node_num=self.sizes[1], epoch_num=self.epoch_num))
            plt.show()

        params['W1'], params['b1'], params['W2'], params['b2'] = self.W1, self.b1, self.W2, self.b2
        print("Training, Done!")

        return params


    def save_params(self, params):
        path = os.path.dirname(os.path.abspath(__file__))
        filename = "params_{mid_node_num}n_{epoch_num}e.npz".format(mid_node_num=self.sizes[1], epoch_num = self.epoch_num)
        np.savez( os.path.join(path, filename ),
                  W1=params['W1'], b1=params['b1'], W2=params['W2'], b2=params['b2'])
        print("Parameters saving, Done!")
    

    def get_result(self,X_test):
        """
        １枚の入力画像に対し、予測結果を返す
        """
        x_test = X_test.flatten() / 255.0
        out = self.feedforward(x_test)
        result = np.argmax(out,axis=-1)

        return result

    def get_wrong_img(self,X,Y,save_img=False):
        """
        予測の間違った画像を保存し、画像番号の記録した配列を返す
        """
        x,y = self.normalize_vec(X,Y)
        z = self.feedforward(x)
        z = np.argmax(z, axis=1)
        y = np.argmax(y, axis=1)
        wrong_img_index = np.asarray(np.where(y != z))

        if save_img:
            path='Kadai3/wrong_imgs'
            for i in range(wrong_img_index.shape[1]):
                index = wrong_img_index[0][i]
                plt.imshow(X[index], cmap=cm.gray)
                plt.savefig(os.path.join(path,'fig_{i}.png'.format(i=i))) 

        return wrong_img_index



