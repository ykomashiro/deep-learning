import numpy as np


########################################################################
# INFO: Compute the forward and backward of full-connected.            #
########################################################################
def fc_forward(x, w, b):
    """
    Compute the forward pass for full-connected layer.
    Inputs:
        - x: a numpy array contains input data with the shape of (N, D)
        - w: a numpy array of weight with the shape of (D, H)
        - b: a numpy array of bias with the shape of (H,)
    Returns:
        - out: the FC output with the shape of (N, H)
        - cache: the record of (x, w, b) by tuple
    """
    # initilize the parameter.
    out = None
    ########################################################################
    # TODO: Implement the affine forward pass and store the results. If    #
    # you input an image, it can reshape to a row vecter.                  #
    ########################################################################
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    return out, cache


def fc_backward(dout, cache):
    """
    Compute the backward pass for full-connected layer.
    Inputs:
        - dout: upstream derivative, of shape (N, H)
        - cache: with the patameter (x, w, b)
    Returns:
        - dx: gradient with respect to x, of shape (N, D)
        - dw: gradient with respect to w, of shape (D, H)
        - db: gradient with respect to b, of shape (H,)
    """
    # restore and initialize some paramters
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the affine backward pass.                            #
    ########################################################################
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    return dx, dw, db


def square_loss(out, y):
    loss = 0.5*np.sum((out-y).dot((out-y).T))
    dout = out-y
    return loss, dout


def softmax_loss(x, y):
    """
    Compute the loss and gradient for softmax classification.

    Inputs:
        - x: input data with the shape of (N, C) where x[i, j] is the score for
            the j^th class for the i^th input image.
        - y: one-hot matrix of labels with the shape of (N, C).
    Returns:
        - loss: softmax loss.
        - dx: gradient of the loss with respect to x
    Note: 
    Use shifted_x to avoid exp(x) become too big(like exp(1000)=INF), and loss
    is equal to sum(-log(exp(shifted_x) / sum(exp(shifted_x)))[y_max_idx]), dx
    is equal to exp(shifted_x)/sum(exp(shifted_x)) - y
    """
    # y_cls = np.argmax(y, axis=1)
    y_cls = y
    ########################################################################
    # TODO: Implement the softmax loss and cross entropy.                  #
    ########################################################################
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_x), axis=1, keepdims=True)
    log_probs = shifted_x - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -1 * np.sum(log_probs[np.arange(N), y_cls]) / N
    dx = probs.copy()
    dx[np.arange(N), y_cls] -= 1
    dx /= N
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    return loss, dx


class One_Layer_Network(object):
    def __init__(self, input_dim=(1, 28*28), hidden_dim=100, output_dim=None, std=1e-3, dtype=np.float32):
        self.params = {}
        self.dtype = dtype

        self.params['W1'] = np.random.normal(
            0, std, size=(input_dim[1], hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = np.random.normal(
            0, std, size=(hidden_dim, output_dim))
        self.params['b2'] = np.zeros(output_dim)

        # convert the type of data
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        scores = None
        fc_hidden, fc_cache_1 = fc_forward(
            X, self.params['W1'], self.params['b1'])
        out, fc_cache_2 = fc_forward(
            fc_hidden, self.params['W2'], self.params['b2'])

        scores = out
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dout = square_loss(out, y)
        dfc_hidden, dw2, db2 = fc_backward(dout, fc_cache_2)
        grads['W2'] = dw2
        grads['b2'] = db2

        dx, dw1, db1 = fc_backward(dfc_hidden, fc_cache_1)
        grads['W1'] = dw1
        grads['b1'] = db1

        return loss, grads


class Model(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        self.total_iterations = 0
        self.batch_size = kwargs.pop('batch_size', 64)
        self.print_every = kwargs.pop('print_every', 50)
        self.verbose = kwargs.pop('verbose', True)

    def _step(self):
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        # perform parameters update
        for p, w in self.model.params.items():
            dw = grads[p]
            next_w = w - dw
            self.model.params[p] = next_w

    def check_accuracy(self, X, y, batch_size=100):
        N = X.shape(0)
        num_batches = N//batch_size
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

    def train(self, num_iterations):
        for t in range(num_iterations):
            self._step()
            self.total_iterations += 1
            if self.total_iterations % self.print_every == 0:
                print('>>> iter: {0}')
