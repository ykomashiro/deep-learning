import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist_data_folder = r'/MNIST'
mnist = input_data.read_data_sets(mnist_data_folder, one_hot=False)
X_tr = mnist.train.images
y_tr = mnist.train.labels
X_te = mnist.test.images
y_te = mnist.test.labels


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
    softmax = Model(X_tr, y_tr)
    loss = softmax.optimize(1000)
    y_pred = softmax.predict(X_tr)
    print('train acc: {}'.format(np.mean(y_pred == y_tr)))
    # test accuracy
    yy_pred = softmax.predict(X_te)
    print('test acc: {}'.format(np.mean(yy_pred == y_te)))
