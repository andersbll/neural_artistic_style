#!/usr/bin/env python

import os
import argparse
import numpy as np
import scipy.misc
import deeppy as dp

from matconvnet import vgg19_net
from style_network import StyleNetwork


def weight_tuple(s):
    try:
        conv_idx, weight = map(int, s.split(','))
        return conv_idx, weight
    except:
        raise argparse.ArgumentTypeError('weights must by "conv_idx,weight"')


def weight_array(weights):
    array = np.zeros(19)
    for idx, weight in weights:
        array[idx] = weight
    norm = np.sum(array)
    if norm > 0:
        array /= norm
    return array


def imread(path):
    return scipy.misc.imread(path).astype(dp.float_)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    return scipy.misc.imsave(path, img)


def to_bc01(img):
    return np.transpose(img, (2, 0, 1))[np.newaxis, ...]


def to_rgb(img):
    return np.transpose(img[0], (1, 2, 0))


def run():
    parser = argparse.ArgumentParser(
        description='Neural artistic style.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--subject', required=True, type=str,
                        help='subject image')
    parser.add_argument('--style', required=True, type=str,
                        help='style image')
    parser.add_argument('--output', default='out.png', type=str,
                        help='output image')
    parser.add_argument('--animation', default='animation', type=str,
                        help='output animation directory')
    parser.add_argument('--iterations', default=500, type=int,
                        help='Number of iterations')
    parser.add_argument('--learn-rate', default=5.0, type=float,
                        help='Learning rate')
    parser.add_argument('--subject-weights', nargs='*', type=weight_tuple,
                        default=[(9, 1)],
                        help='list of subject weights (conv_idx,weight)')
    parser.add_argument('--style-weights', nargs='*', type=weight_tuple,
                        default=[(0, 1), (2, 1), (4, 1), (8, 1), (12, 1)],
                        help='list of style weights (conv_idx,weight)')
    parser.add_argument('--subject-ratio', type=float, default=2e-2,
                        help='weight of subject relative to style')
    parser.add_argument('--vgg19', default='imagenet-vgg-verydeep-19.mat',
                        type=str, help='VGG-19 .mat file')
    args = parser.parse_args()

    layers, img_mean = vgg19_net(args.vgg19, pool_method='avg')

    # Inputs
    pixel_mean = np.mean(img_mean, axis=(0, 1))
    style_img = imread(args.style)
    subject_img = imread(args.subject)
    style_img -= pixel_mean
    subject_img -= pixel_mean

    # Setup network
    subject_weights = weight_array(args.subject_weights) * args.subject_ratio
    style_weights = weight_array(args.style_weights)
    net = StyleNetwork(layers, to_bc01(subject_img), to_bc01(style_img),
                       subject_weights, style_weights)

    # Repaint image
    def net_img():
        return to_rgb(net.image) + pixel_mean

    if not os.path.exists(args.animation):
        os.mkdir(args.animation)

    params = net._params
    learn_rule = dp.Adam(learn_rate=args.learn_rate)
    learn_rule_states = [learn_rule.init_state(p) for p in params]
    for i in range(args.iterations):
        imsave(os.path.join(args.animation, '%.4d.png' % i), net_img())
        cost = np.mean(net._update())
        for param, state in zip(params, learn_rule_states):
            learn_rule.step(param, state)
        print('Iteration: %i, cost: %.4f' % (i, cost))
    imsave(args.output, net_img())


if __name__ == "__main__":
    run()
