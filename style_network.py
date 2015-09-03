import numpy as np
import cudarray as ca
import deeppy as dp
from deeppy.base import Model
from deeppy.parameter import Parameter


class Convolution(dp.Convolution):
    """ Convolution layer wrapper

    This layer does not propagate gradients to filters. Also, it reduces
    memory consumption as it does not store fprop() input for bprop().
    """
    def __init__(self, layer):
        self.layer = layer

    def fprop(self, x):
        y = self.conv_op.fprop(x, self.weights.array)
        y += self.bias.array
        return y

    def bprop(self, y_grad):
        # Backprop to input image only
        _, x_grad = self.layer.conv_op.bprop(
            imgs=None, filters=self.weights.array, convout_d=y_grad,
            to_imgs=True, to_filters=False
        )
        return x_grad

    # Wrap layer methods
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.layer, attr)


def gram_matrix(img_bc01):
    n_channels = img_bc01.shape[1]
    feats = ca.reshape(img_bc01, (n_channels, -1))
    gram = ca.dot(feats, feats.T)
    return gram


class StyleNetwork(Model):
    """ Artistic style network

    Implementation of [1].

    Differences:
    - The gradients for both subject and style are normalized. The original
      gradient summatino scheme seemed sensitive to changes in image size.

    References:
    [1]: A Neural Algorithm of Artistic Style; Leon A. Gatys, Alexander S.
         Ecker, Matthias Bethge; arXiv:1508.06576; 08/2015
    """

    def __init__(self, layers, subject_img, style_img, subject_weights,
                 style_weights):

        # Map weights (in convolution indices) to layer indices
        self.subject_weights = np.zeros(len(layers))
        self.style_weights = np.zeros(len(layers))
        layers_len = 0
        conv_idx = 0
        for l, layer in enumerate(layers):
            if isinstance(layer, dp.Activation):
                self.subject_weights[l] = subject_weights[conv_idx]
                self.style_weights[l] = style_weights[conv_idx]
                if subject_weights[conv_idx] > 0 or \
                   style_weights[conv_idx] > 0:
                    layers_len = l+1
                conv_idx += 1

        # Discard unused layers
        layers = layers[:layers_len]

        # Wrap convolution layers for better performance
        self.layers = [Convolution(l) if isinstance(l, dp.Convolution) else l
                       for l in layers]

        # Setup network
        x_shape = subject_img.shape
        self.x = Parameter(subject_img)
        self.x._setup(x_shape)
        for layer in self.layers:
            layer._setup(x_shape)
            x_shape = layer.y_shape(x_shape)

        # Precompute subject features and style Gram matrices
        self.subject_feats = [None]*len(self.layers)
        self.style_grams = [None]*len(self.layers)
        next_subject = ca.array(subject_img)
        next_style = ca.array(style_img)
        for l, layer in enumerate(self.layers):
            next_subject = layer.fprop(next_subject)
            next_style = layer.fprop(next_style)
            if self.subject_weights[l] > 0:
                self.subject_feats[l] = next_subject
            if self.style_weights[l] > 0:
                gram = gram_matrix(next_style)
                # Scale gram matrix to compensate for different image sizes
                n_pixels_subject = np.prod(next_subject.shape[2:])
                n_pixels_style = np.prod(next_style.shape[2:])
                scale = (n_pixels_subject / float(n_pixels_style))
                self.style_grams[l] = gram * scale

    @property
    def image(self):
        return np.array(self.x.array)

    @property
    def _params(self):
        return [self.x]

    def _update(self):
        # Forward propagation
        next_x = self.x.array
        x_feats = [None]*len(self.layers)
        x_grams = [None]*len(self.layers)
        for l, layer in enumerate(self.layers):
            next_x = layer.fprop(next_x)
            if self.subject_weights[l] > 0:
                x_feats[l] = next_x
            if self.style_weights[l] > 0:
                x_feats[l] = next_x
                x_grams[l] = gram_matrix(next_x)

        # Backward propagation
        grad = ca.zeros_like(next_x)
        loss = ca.zeros(1)
        for l, layer in reversed(list(enumerate(self.layers))):
            if self.subject_weights[l] > 0:
                diff = x_feats[l] - self.subject_feats[l]
                norm = ca.sum(ca.fabs(diff)) + 1e-8
                weight = float(self.subject_weights[l]) / norm
                grad += diff * weight
                loss += 0.5*weight*ca.sum(diff**2)
            if self.style_weights[l] > 0:
                diff = x_grams[l] - self.style_grams[l]
                n_channels = diff.shape[0]
                x_feat = ca.reshape(x_feats[l], (n_channels, -1))
                style_grad = ca.reshape(ca.dot(diff, x_feat), x_feats[l].shape)
                norm = ca.sum(ca.fabs(style_grad))
                weight = float(self.style_weights[l]) / norm
                style_grad *= weight
                grad += style_grad
                loss += 0.25*weight*ca.sum(diff**2)
            grad = layer.bprop(grad)
        ca.copyto(self.x.grad_array, grad)
        return loss
