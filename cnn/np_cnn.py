import numpy as np


########################################################################
# NOTE: There are some explanations for the variable name and importnt #
#       numpy function.                                                #
# Paramters:                                                           #
#   - x: input data of images with the shape of (N, C, H, W)           #
#        or reshape to (N, C*H*W).                                     #
#   - w: a numpy array of weights.                                     #
#   - b: a numpy array of biases.                                      #
#   - dout: upstream derivative.                                       #
#   - cache: store some necessary intermediate variables               #
#   - N: number of data.                                               #
#   - C: channals of image or feature-map.                             #
#   - H: height of image or feature-map.                               #
#   - W: width of image or feature-map.                                #
# Function:                                                            #
########################################################################


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


########################################################################
# INFO: Compute the forward and backward of ReLU function.             #
########################################################################
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Inputs:
        - x: an input data of any shape
    Returns:
        - out: output with the same shape as x
        - cache: x
    """
    out = None
    ########################################################################
    # TODO: Implement the ReLU forward pass.                              #
    ########################################################################
    out = np.maximum(0, x)
    cache = x
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Inputs:
        - dout: upstream derivatives
        - cache: x
    Returns:
        - dx: Gradient with respect to x
    """
    ########################################################################
    # TODO: Implement the ReLU backward pass.                              #
    ########################################################################
    dx, x = None, cache
    dx = (x > 0) * dout
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    return dx


########################################################################
# INFO: Compute the forward and backward of convolution layer.         #
########################################################################
def conv_forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    Parameter: The input consists of N data points, each with C channels,
    height H and width W. And each input with F different filters, where
    each filter spans all C channels and has height HH and width HH.
    NOTE: 
    Inputs:
        - x: input data with the shape of (N, C, H, W)
        - w: fillter weights with the shape of (F, C, HH, WW)
        - b: biases with shape of (F,)
        - conv_param: a dictionary with following keys:
            - 'stride': the number of pixels for fillter move
            - 'pad': the number of pixels for input data use(zero-padding)
    Returns:
        -out: output of shape (N, F, H', W') where H' and W' are given by
                H' = 1 + (H + 2 * pad - HH) / stride
                W' = 1 + (W + 2 * pad - WW) / stride
                generally, H'=H and W'=W
        - cache: (x, w, b, conv_param)
    """
    out = None
    ########################################################################
    # TODO: Implement the convolutional forward pass.                      #
    ########################################################################
    # get the initial paramters
    stride, pad = conv_param['stride'], conv_param['pad'],
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    # calculate the shape of output
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_new, W_new))
    # zero-padding
    x_pad = np.pad(
        x, ((0, ), (0, ), (pad, ), (pad, )),
        mode='constant',
        constant_values=0)
    # execute the convolutional operation by cycling
    for i in range(H_new):
        for j in range(W_new):
            # stride
            x_padded_mask = x_pad[:, :, i * stride:i * stride +
                                  HH, j * stride:j * stride + WW]
            for k in range(F):
                # convolution
                out[:, k, i, j] = np.sum(
                    x_padded_mask * w[k, :, :, :], axis=(1, 2, 3))
    out += b[None, :, None, None]
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    Inputs:
        - dout: upstream derivatives
        - cache: (x, w, b, conv_param) from func-conv_forward
    Returns:
        - dx: gradient with respect to x
        - dw: gradient with respect to w
        - db: gradient with respect to b
    """
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the convolutional backward pass.                     #
    ########################################################################
    (x, w, b, conv_param) = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    # shape of new out
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    # zero-padding
    x_pad = np.pad(
        x, ((0, ), (0, ), (pad, ), (pad, )),
        mode='constant',
        constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))

    for i in range(H_new):
        for j in range(W_new):
            # stride
            x_padded_mask = x_pad[:, :, i * stride:i * stride +
                                  HH, j * stride:j * stride + WW]
            # calculate
            for k in range(F):
                dw[k, :, :, :] += np.sum(
                    (dout[:, k, i, j])[:, None, None, None] * x_padded_mask,
                    axis=0)
            for n in range(N):
                dx_pad[n, :, i * stride:i * stride +
                       HH, j * stride:j * stride + WW] += np.sum(
                           (dout[n, :, i, j])[:, None, None, None] * w, axis=0)
    # crop
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    return dx, dw, db


#########################################################################
# INFO: Compute the forward and backward of convolution layer by im2col #
#########################################################################
def conv_forward_im2col(x, w, b, conv_param):
    pass


def conv_backward_im2col(dout, cache):
    pass


########################################################################
# INFO: Compute the forward and backward of max pooling.               #
########################################################################
def maxpool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for max pooling layer.

    Inputs:
        - x: input data with shape of (N, C, H, W)
        - pool_param: a dictionary with the following keys:
            - 'pool_height': the height of each pooling region
            - 'pool_width': the width of each pooling region
            - 'stride': the distance between adjacent pooling regions
    Returns:
    - out: output data
    - cache: (x, pool_param)
    """
    out = None
    ########################################################################
    # TODO: Implement the max pooling forward pass.                        #
    ########################################################################
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param[
        'pool_width'], pool_param['stride'],
    N, C, H, W = x.shape
    # calculate the size of (H, W) after pooling, name (H_new, W_new)
    H_new = int(1 + (H - pool_height) / stride)
    W_new = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_new, W_new))

    for i in range(H_new):
        for j in range(W_new):
            # get box by stride
            x_padded_mask = x[:, :, i * stride:i * stride +
                              pool_height, j * stride:j * stride + pool_width]
            # find max element
            out[:, :, i, j] = np.max(x_padded_mask, axis=(2, 3))
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    cache = (x, pool_param)
    return out, cache


def maxpool_backward(dout, cache):
    """
    A naive implementation of the backward pass for max pooling layer.

    Inputs:
    - dout: upstream derivatives.
    - cache: (x, pool_param) from forward pass.
    Returns:
    - dx: gradient with respect to x.
    """
    dx = None
    ########################################################################
    # TODO: Implement the max pooling forward pass.                        #
    ########################################################################
    (x, pool_param) = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param[
        'pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    # shape of new out
    H_new = int(1 + (H - pool_height) / stride)
    W_new = int(1 + (W - pool_width) / stride)
    dx = np.zeros((N, C, H, W))

    for i in range(H_new):
        for j in range(W_new):
            # stride
            x_padded_mask = x[:, :, i * stride:i * stride +
                              pool_height, j * stride:j * stride + pool_width]
            # find max element
            max_mask = np.max(x_padded_mask, axis=(2, 3))
            # record the position of max element
            temp_binary_mask = (x_padded_mask == (max_mask)[:, :, None, None])
            dx[:, :, i * stride:i * stride + pool_height, j *
               stride:j * stride + pool_width] += temp_binary_mask * (
                   dout[:, :, i, j])[:, :, None, None]
    ########################################################################
    # END:                    END OF THE CODE                              #
    ########################################################################
    return dx


########################################################################
# INFO: Compute the loss with softmax or svm.                          #
########################################################################
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
    NOTE: 
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


def svm_loss(x, y):
    pass


########################################################################
# INFO: Union the layer units.                                         #
########################################################################
def fc_relu_forward(x, w, b):
    temp, fc_cache = fc_forward(x, w, b)
    out, relu_cache = relu_forward(temp)
    cache = (fc_cache, relu_cache)
    return out, cache


def fc_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    dtemp = relu_backward(dout, relu_cache)
    dx, dw, db = fc_backward(dtemp, fc_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    temp, conv_cache = conv_forward(x, w, b, conv_param)
    out, relu_cache = relu_forward(temp)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    dtemp = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward(dtemp, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    A, conv_cache = conv_forward(x, w, b, conv_param)
    B, relu_cache = relu_forward(A)
    out, pool_cache = maxpool_forward(B, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    conv_cache, relu_cache, pool_cache = cache
    dB = maxpool_backward(dout, pool_cache)
    dA = relu_backward(dB, relu_cache)
    dx, dw, db = conv_backward(dA, conv_cache)
    return dx, dw, db