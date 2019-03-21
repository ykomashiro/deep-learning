import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist_data_folder = r'/MNIST'
mnist = input_data.read_data_sets(mnist_data_folder, one_hot=False)
X = mnist.train.images[0:1000]
y = mnist.train.labels[0:1000]


class AutoEncoder(object):
    def __init__(self, x, hidden_dim=100, std=1e-2, reg=1e-2, dtype=np.float32):
        self.x = x.reshape(x.shape[0], -1)
        self.reg = reg
        self.params = {}
        self.dtype = dtype
        self.params['W1'] = np.random.normal(
            0, std, size=(x.shape[1], hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = np.random.normal(
            0, std, size=(hidden_dim, x.shape[1]))
        self.params['b2'] = np.zeros(x.shape[1])

        # convert the type of data
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def run(self, steps):
        """
        The main function of the autoencoder, we mainly run the loss
        function by step and show the 'loss'. 
            :param steps: the number of train times in once run.
        """
        for t in range(steps):
            mask = np.random.choice(self.x.shape[0], 200, replace=False)
            train_x = self.x[mask]
            hidden_x, sigmoid_x = self.encoder(train_x)
            hidden_x2, shift_x = self.decoder(sigmoid_x)
            loss = self.loss(shift_x, train_x, hidden_x, sigmoid_x, hidden_x2)
            if t % 100 == 0:
                print('>>> step: {0}\t loss: {1}'.format(t+1, loss))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        """
        Derivative of sigmoid function.
            :param x: result of sigmoid function.
        """
        return (self.sigmoid(x))*(1-self.sigmoid(x))

    def encoder(self, x):
        """
        Encode the input data to low variable. 
            :param x: the input data of shape (N, D)
        """
        hidden_x = x.dot(self.params['W1']) + self.params['b1']
        sigmoid_x = self.sigmoid(hidden_x)
        return hidden_x, sigmoid_x

    def decoder(self, x):
        """
        Decode the encoded data, restore the hidden data. 
            :param x: the hidden data of shape (N, H)
        """
        hidden_x = x.dot(self.params['W2']) + self.params['b2']
        shift_x = self.sigmoid(hidden_x)
        return hidden_x, shift_x

    def loss(self, out, y, hidden_x, sigmoid_x, hidden_x2):
        """
        NOTE: Calculate the loss and the gradient of all variables. There we use
        the MSE as the loss function.
            :param out: the output data of shape (N, D), it's the same as
                the original(input) data.
            :param y: the input data of shape (N, D)
            :param hidden_x: the intermediate data in encoder layer.
            :param sigmoid_x: the data in hidden layer through sigmoid function.
            :param hidden_x2: the intermediate data in decoder layer.
        """
        loss = np.sum((out-y)**2)/(2*y.shape[0])
        grads = {}
        # Calculate the gradient of parameters.
        dout = (out-y)/(y.shape[0])       # (N, D)
        dout = self.sigmoid_prime(hidden_x2)*(dout)

        grads['W2'] = sigmoid_x.reshape(
            sigmoid_x.shape[0], -1).T.dot(dout)
        grads['b2'] = np.sum(dout, axis=0)

        dsigmoid_x = dout.dot(self.params['W2'].T)  # (N, H)
        dhidden_x = self.sigmoid_prime(hidden_x)*(dsigmoid_x)  # (N, H)

        grads['W1'] = y.reshape(
            y.shape[0], -1).T.dot(dhidden_x)
        grads['b1'] = np.sum(dhidden_x, axis=0)

        for p, _ in self.params.items():
            self.params[p] -= grads[p]
        return loss


class Model(object):
    """
    It's a class for testing autoencoder. To train a model, you will first
    construct a Classification instance, passing the data(X) from encoder and
    corresponding categories(y) to the constructor. You will then call the 
    optimize() method to run the optimization procedure and train the model.

    After the optimizer() method run, you will run the predict() method to
    predict the labels of unknow data.
    """

    def __init__(self, X, y, num_classes=10, lr=1e-1):
        assert len(X.shape) == 2, "X must be a matrix"
        self.X = X
        self.y = y
        D = self.X.shape[1]
        self.W = np.random.normal(0, 1, (D, num_classes))
        self.lr = lr
        self.total_iterations = 0
        self.loss_history = []

    def optimize(self, steps=1000, batch=700, verbose=True):
        """
        Run optimization(softmax loss) to train the model. 
            :param steps: times of iteration.
            :param batch: how much data in each step.
            :param verbose: print some necessary message.
        """
        N = self.X.shape[0]
        for _ in range(steps):
            mask = np.random.choice(N, batch, replace=False)
            X_tr = self.X[mask]
            y_tr = self.y[mask]
            loss, dW = self.softmax_loss(X_tr, self.W, y_tr)
            self.W -= self.lr*dW
            self.loss_history.append(loss)
            self.total_iterations += 1
            if self.total_iterations % 100 == 0 and verbose:
                print('>>> step: {0}\t loss: {1}'.format(
                    self.total_iterations, loss))
        return self.loss_history

    def predict(self, X, y=None):
        """
        Check accuracy of the model on the provided data. 
            :param X: array of data, of shape (N, D)
            :param y: label of data, of shape (N,)
        """
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred

    def softmax_loss(self, X, W, y, reg=1e-2):
        """
        Softmax loss function. 
            :param X: numpy array of shape (N, D) containing a minibatch of data.
            :param W: numpy array of shape (D, C) containing weights.
            :param y: numpy array of shape (N,) containing training labels.
            :param reg: regularization strength
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(W)
        num_train = X.shape[0]
        scores = X.dot(W)
        correct_class_score = scores[np.arange(
            num_train), y].reshape(num_train, 1)
        exp_sum = np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
        loss += np.sum(np.log(exp_sum) - correct_class_score)

        margin = np.exp(scores) / exp_sum
        margin[np.arange(num_train), y] += -1
        dW = X.T.dot(margin)

        loss /= num_train
        dW /= num_train
        # Add regularization to the loss.
        loss += 0.5 * reg * np.sum(W * W)
        dW += reg * W

        return loss, dW


if __name__ == "__main__":
    model = AutoEncoder(X)
    model.run(3000)

    _,x= model.encoder(X)
    _,shift_x = model.decoder(x)
    images = [X[0],X[1],X[2],X[3],shift_x[0],shift_x[1],shift_x[2],shift_x[3]]
    fig, axes = plt.subplots(2,4)
    fig.subplots_adjust(hspace=0.3, wspace=2)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(28,28), cmap='binary')
    plt.show()

    
    _, x = model.encoder(X)
    softmax = Model(x, y)
    loss = softmax.optimize(1000)
    y_pred=softmax.predict(x)
    print('train acc: {}'.format(np.mean(y_pred==y)))

    # test dataset
    X_te = mnist.train.images[1000:2000]
    y_te = mnist.train.labels[1000:2000]
    # test accuracy
    _,xx = model.encoder(X_te)
    yy_pred=softmax.predict(xx)
    print('test acc: {}'.format(np.mean(yy_pred==y_te)))

