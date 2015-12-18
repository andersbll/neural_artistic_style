import numpy as np
import scipy.io
import deeppy as dp


def conv_layer(weights, bias, border_mode):
    return dp.Convolution(
        n_filters=weights.shape[0],
        filter_shape=weights.shape[2:],
        border_mode=border_mode,
        weights=weights,
        bias=bias,
    )


def pool_layer(pool_method, border_mode):
    return dp.Pool(
        win_shape=(3, 3),
        strides=(2, 2),
        method=pool_method,
        border_mode=border_mode,
    )


def vgg_net(path, pool_method='max', border_mode='same'):
    matconvnet = scipy.io.loadmat(path)
    img_mean = matconvnet['meta'][0][0][1][0][0][0][0][0]
    vgg_layers = matconvnet['layers'][0]
    layers = []
    for layer in vgg_layers:
        layer = layer[0][0]
        layer_type = layer[1][0]
        if layer_type == 'conv':
            params = layer[2][0]
            weights = params[0]
            bias = params[1]
            weights = np.transpose(weights, (3, 2, 0, 1)).astype(dp.float_)
            bias = np.reshape(bias, (1, bias.size, 1, 1)).astype(dp.float_)
            layers.append(conv_layer(weights, bias, border_mode))
        elif layer_type == 'pool':
            layers.append(pool_layer(pool_method, border_mode))
        elif layer_type == 'relu':
            layers.append(dp.ReLU())
        elif layer_type == 'softmax':
            pass
        else:
            raise ValueError('invalid layer type: %s' % layer_type)
    return layers, img_mean
