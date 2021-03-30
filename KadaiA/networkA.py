import numpy as np
import matplotlib.pylab as plt
import os, time
from funcs import sigmoid, softmax, cross_entropy_error
from layers import Sigmoid, ReLU, Dropout, FullyConnected, SoftmaxLoss, BatchNorm
from minibatch import shuffle_batches
from optimizers import SGD, Momentum_SGD, AdaGrad, RMSProp, AdaDelta, Adam

"""A1~A4の機能を統合したネットワーク"""
class NetworkA:
    def __init__(self,sizes, batch_size, epoch_num, 
    use_trained_params=False, filename=None,
    optimizer='SGD',activation='ReLU', use_dropout=True, dropout_p=0.2, use_bn=True):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.activation = activation
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.use_bn = use_bn
        self.opt = optimizer
        optimizers = {'SGD':SGD, 'Momentum_SGD':Momentum_SGD, 'AdaGrad':AdaGrad, 'RMSProp':RMSProp, 'AdaDelta':AdaDelta, 'Adam':Adam}
        self.optimizer = optimizers[self.opt]()
      

        if use_trained_params:
            path = os.path.dirname(os.path.abspath(__file__))
            loaded_params = np.load(os.path.join(path,filename))
            self.W1 = loaded_params['W1']
            self.b1 = loaded_params['b1']
            self.W2 = loaded_params['W2']
            self.b2 = loaded_params['b2']
            self.gamma = loaded_params['gamma']
            self.beta = loaded_params['beta']
            # Batch Normalizationを使う場合
            if self.use_bn:
                self.running_mean = loaded_params['running_mean']
                self.running_var = loaded_params['running_var']

        else:
            np.random.seed(12)
            self.W1 = np.sqrt(1/sizes[0]) * np.random.randn(sizes[0],sizes[1]) #（784,50）
            self.b1 = np.sqrt(1/sizes[0]) * np.random.randn(sizes[1])
            self.W2 = np.sqrt(1/sizes[1]) * np.random.randn(sizes[1],sizes[2]) #（50,10)
            self.b2 = np.sqrt(1/sizes[1]) * np.random.randn(sizes[2])
            self.gamma = np.ones(self.W1.shape[1])
            self.beta = np.zeros(self.W1.shape[1])

        # layers of network
        activation_function= {'Sigmoid':Sigmoid, 'ReLU':ReLU}
        self.layers = {}
        self.layers['FullyConnected1'] = FullyConnected(self.W1, self.b1)
        if self.use_bn:
            if use_trained_params:
                self.layers['BatchNorm'] = BatchNorm(self.gamma, self.beta, running_mean=self.running_mean,running_var=self.running_var)
            else:
                self.layers['BatchNorm'] = BatchNorm(self.gamma, self.beta)
        self.layers['Activation'] = activation_function[self.activation]()
        if self.use_dropout:
            self.layers['Dropout'] = Dropout(self.dropout_p)
        self.layers['FullyConnected2'] = FullyConnected(self.W2, self.b2)

        self.lastLayer = SoftmaxLoss()


    def normalize_vec(self,x,y):
        I = x.shape[0] #60000
        C = len(np.unique(y)) #10
        x = x.reshape(I,-1)
        x_norm = x.astype(np.float32)/255.0
        y_one_hot_v = np.eye(C)[y] #(60000,10)

        return x_norm, y_one_hot_v


    def feedforward(self,x,training=False):
        for key, layer in self.layers.items():
            if 'Dropout' in key or 'BatchNorm' in key:
                x = layer.forward(x,training)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, y_one_hot, training=False):
        y = self.feedforward(x, training)
        loss = self.lastLayer.forward(y,y_one_hot)
        return loss

    def accuracy(self, x, y_one_hot, training=False):
        y = self.feedforward(x, training)
        y = np.argmax(y, axis=1)
        y_one_hot = np.argmax(y_one_hot, axis=1)

        accurate_num = np.sum(y == y_one_hot) 
        return accurate_num


    def gradient(self, x, y_one_hot):
        # forward
        self.loss(x, y_one_hot, training=True)

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

        if self.use_bn:
                grads['gamma'] = self.layers['BatchNorm'].dgamma
                grads['beta' ] = self.layers['BatchNorm'].dbeta
        else:
            grads['gamma'] = np.zeros(self.W1.shape[1])
            grads['beta'] = np.zeros(self.W1.shape[1])


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
            accuracy_per_epoch = 0
            minibatches = shuffle_batches(x_train,y_train,self.batch_size)
            for j in range(len(minibatches)):
                x_batch = minibatches[j][0]
                y_batch = minibatches[j][1]

                grads = self.gradient(x_batch,y_batch)
                params = {'W1':self.W1, 'b1':self.b1, 'W2':self.W2, 'b2':self.b2, 'gamma':self.gamma, 'beta':self.beta}
                self.optimizer.update(params,grads)

                if self.use_bn:
                    self.running_mean=self.layers['BatchNorm'].running_mean
                    self.running_var=self.layers['BatchNorm'].running_var

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

            
        path = os.path.dirname(os.path.abspath(__file__))
        if show_loss_graph:
            x = np.linspace(0,self.epoch_num,self.epoch_num)
            plt.plot(x, train_loss, label='loss')
            plt.title('Average Loss of Each Epoch--{activation}'.format(activation=self.activation))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            # plt.savefig(os.path.join(path,'fig_loss_{act_f}_{mid_node_num}n_{epoch_num}e.png'\
            #     .format(act_f=self.activation,mid_node_num=self.sizes[1], epoch_num=self.epoch_num)))
            plt.show()
        if show_accuracy_graph:
            x2 = np.linspace(0,len(test_acc_history),len(test_acc_history))
            plt.plot(x2, train_acc_histroy, label='train accuracy')
            plt.plot(x2, test_acc_history, label='test accuracy', linestyle='--')
            plt.title('Accuracy After Each Epoch--{activation}'.format(activation=self.activation))
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.xlim(left=0)
            plt.ylim(0, 1.0)
            plt.legend(loc='lower right')
            # plt.savefig(os.path.join(path,'fig_acc_{act_f}_{mid_node_num}n_{epoch_num}e.png'\
            #     .format(act_f=self.activation,mid_node_num=self.sizes[1], epoch_num=self.epoch_num)))
            plt.show()
        if self.use_bn:
            params['W1'], params['b1'], params['W2'], params['b2'] ,params['gamma'], params['beta'], params['running_mean'], params['running_var']= \
            self.W1, self.b1, self.W2, self.b2, self.gamma, self.beta, self.running_mean,self.running_var
        else:
            params['W1'], params['b1'], params['W2'], params['b2'] ,params['gamma'], params['beta']= \
            self.W1, self.b1, self.W2, self.b2, self.gamma, self.beta
        print("Network Architecture:{a_f}-Dropout({use_dropout}({drop_p}))-BatchNorm({use_bn})"\
            .format(a_f=self.activation,use_dropout=self.use_dropout,drop_p=self.dropout_p, use_bn=self.use_bn))
        print("Training, Done!")

        return params


    def save_params(self, params):
        path = os.path.dirname(os.path.abspath(__file__))
        now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
        # filename = "params_{mid_node_num}n_{epoch_num}e.npz".format(mid_node_num=self.sizes[1], epoch_num = self.epoch_num)
        filename='params_'+now
        if self.use_bn:
            np.savez( os.path.join(path, filename ),
                    W1=params['W1'], b1=params['b1'], W2=params['W2'], b2=params['b2'], gamma = params['gamma'], beta=params['beta'],running_mean=params['running_mean'],running_var=params['running_var'])
        else:
            np.savez( os.path.join(path, filename ),
                    W1=params['W1'], b1=params['b1'], W2=params['W2'], b2=params['b2'], gamma = params['gamma'], beta=params['beta'])

        print("Parameters saving, Done!")



    def get_result(self,X_test):
        """
        １枚の入力画像に対し、予測結果を返す
        """
        x_test = X_test.flatten() / 255.0
        out = self.feedforward(x_test)
        result = np.argmax(out,axis=-1)

        return result


        return result

