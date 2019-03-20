import numpy as np
from np_cnn import *


class CNN(object):
    """
    (conv->relu->max-pool) -> (fc->relu) -> (fc->softmax)
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dim=(1, 28, 28),
                 num_filters=32,
                 filter_size=3,
                 hidden_dim=100,
                 num_classes=10,
                 std=1e-3,
                 reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params['W1'] = np.random.normal(
            0, std, size=(num_filters, input_dim[0], filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)

        self.params['W2'] = np.random.normal(
            0, std, size=(num_filters * int(input_dim[1] / 2)**2, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.normal(
            0, std, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        conv_hidden, conv_cache = conv_relu_pool_forward(
            X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        fc_hidden, fc_relu_cache = fc_relu_forward(
            conv_hidden, self.params['W2'], self.params['b2'])
        out, fc_cache = fc_forward(fc_hidden, self.params['W3'],
                                   self.params['b3'])
        scores = out

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dout = softmax_loss(out, y)
        loss += 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])
        +0.5 * self.reg * (np.sum(self.params['W2'] * self.params['W2']))
        +0.5 * self.reg * (np.sum(self.params['W3'] * self.params['W3']))

        dfc_hidden, dw3, db3 = fc_backward(dout, fc_cache)
        grads['W3'] = dw3 + self.reg * self.params['W3']
        grads['b3'] = db3

        dconv_hidden, dw2, db2 = fc_relu_backward(dfc_hidden, fc_relu_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2

        dx, dw1, db1 = conv_relu_pool_backward(dconv_hidden, conv_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1

        return loss, grads


class Model(object):
    def __init__(self, model, data, **kwargs):

        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        self.batch_size = kwargs.pop('batch_size', 64)
        self.num_train_samples = kwargs.pop('num_iterations', 1000)
        self.num_test_samples = kwargs.pop('num_test_samples', None)
        self.print_every = kwargs.pop('print_every', 50)
        self.verbose = kwargs.pop('verbose', True)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]

            next_w = w - dw
            self.model.params[p] = next_w

    def check_accuracy(self, X, y, batch_size=100):
        # Maybe subsample the data
        N = X.shape[0]

        # Compute predictions in batches
        num_batches = N // batch_size

        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        if N % batch_size != 0:
            start = num_batches * batch_size
            scores = self.model.loss(X[start:-1])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_iterations = 10

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations,
                                                        self.loss_history[-1]))
                """
                if t % (5 * self.print_every) == 0:
                    print("accuracy %f" % (self.check_accuracy(
                        self.X_test, self.y_test)))
                """
