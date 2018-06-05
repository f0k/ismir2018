#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network architecture definition for Singing Voice Detection experiment.

Author: Jan Schlüter
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, MaxPool2DLayer,
                            DenseLayer, ExpressionLayer, dropout, batch_norm)
batch_norm_vanilla = batch_norm
try:
    from lasagne.layers.dnn import batch_norm_dnn as batch_norm
except ImportError:
    pass


class MelBankLayer(lasagne.layers.Layer):
    """
    Creates a mel filterbank layer of `num_bands` triangular filters, with
    the first filter initialized to start at `min_freq` and the last one
    to stop at `max_freq`. Expects to process magnitude spectra created
    from samples at a sample rate of `sample_rate` with a window length of
    `frame_len` samples. Learns a vector of `num_bands + 2` values, with
    the first value giving `min_freq` in mel, and remaining values giving
    the distance to the respective next peak in mel.
    """
    def __init__(self, incoming, sample_rate, frame_len, num_bands, min_freq,
                 max_freq, trainable=True, **kwargs):
        super(MelBankLayer, self).__init__(incoming, **kwargs)
        # mel-spaced peak frequencies
        min_mel = 1127 * np.log1p(min_freq / 700.0)
        max_mel = 1127 * np.log1p(max_freq / 700.0)
        spacing = (max_mel - min_mel) / (num_bands + 1)
        spaces = np.ones(num_bands + 2) * spacing
        spaces[0] = min_mel
        spaces = theano.shared(lasagne.utils.floatX(spaces))  # learned param
        peaks_mel = spaces.cumsum()

        # create parameter as a vector of real-valued peak bins
        peaks_hz = 700 * (T.expm1(peaks_mel / 1127))
        peaks_bin = peaks_hz * frame_len / sample_rate
        self.peaks = self.add_param(peaks_bin,
                shape=(num_bands + 2,), name='peaks', trainable=trainable,
                regularizable=False)

        # store what else is needed
        self.num_bands = num_bands

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.num_bands,)

    def get_output_for(self, input, **kwargs):
        num_bins = self.input_shape[-1] or input.shape[-1]
        x = T.arange(num_bins, dtype=input.dtype).dimshuffle(0, 'x')
        peaks = self.peaks
        l, c, r = peaks[0:-2], peaks[1:-1], peaks[2:]
        # triangles are the minimum of two linear functions f(x) = a*x + b
        # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
        tri_left = (x - l) / (c - l)
        # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
        tri_right = (x - r) / (c - r)
        # combine by taking the minimum of the left and right sides
        tri = T.minimum(tri_left, tri_right)
        # and clip to only keep positive values
        bank = T.maximum(0, tri)

        # the dot product of the input with this filter bank is the output
        return T.dot(input, bank)


class PowLayer(lasagne.layers.Layer):
    def __init__(self, incoming, exponent=lasagne.init.Constant(0), **kwargs):
        super(PowLayer, self).__init__(incoming, **kwargs)
        self.exponent = self.add_param(exponent, shape=(), name='exponent', regularizable=False)
    def get_output_for(self, input, **kwargs):
        return input ** self.exponent


class TimeDiffLayer(lasagne.layers.Layer):
    def __init__(self, incoming, delta=1, **kwargs):
        super(TimeDiffLayer, self).__init__(incoming, **kwargs)
        self.delta = delta
    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        if input_shape[2] is not None:
            output_shape[2] -= self.delta
        return tuple(output_shape)
    def get_output_for(self, input, **kwargs):
        return input[:, :, self.delta:] - input[:, :, :-self.delta]


class PCENLayer(lasagne.layers.Layer):
    def __init__(self, incoming,
                 log_s=lasagne.init.Constant(np.log(0.025)),
                 log_alpha=lasagne.init.Constant(0),
                 log_delta=lasagne.init.Constant(0),
                 log_r=lasagne.init.Constant(0),
                 eps=1e-6, **kwargs):
        super(PCENLayer, self).__init__(incoming, **kwargs)
        num_bands = self.input_shape[-1]
        self.log_s = self.add_param(log_s, shape=(num_bands,),
                                    name='log_s', regularizable=False)
        self.log_alpha = self.add_param(log_alpha, shape=(num_bands,),
                                        name='log_alpha', regularizable=False)
        self.log_delta = self.add_param(log_delta, shape=(num_bands,),
                                        name='log_delta', regularizable=False)
        self.log_r = self.add_param(log_r, shape=(num_bands,),
                                        name='log_r', regularizable=False)
        self.eps = eps
    def get_output_for(self, input, **kwargs):
        def smooth_step(current_in, previous_out, s):
            one = T.constant(1)
            return [(one - s) * previous_out + s * current_in]
        init = input[:, :, 0]  # start smoother from first frame
        s = T.exp(self.log_s).dimshuffle('x', 'x', 0)
        smoother = theano.scan(fn=smooth_step,
                               sequences=[input.transpose(2, 0, 1, 3)],
                               non_sequences=[s],
                               outputs_info=[init],
                               strict=True)[0].transpose(1, 2, 0, 3)
        alpha = T.exp(self.log_alpha)
        delta = T.exp(self.log_delta)
        r = T.exp(self.log_r)
        return (input / (self.eps + smoother)**alpha + delta)**r - delta**r


def architecture(input_var, input_shape, cfg):
    layer = InputLayer(input_shape, input_var)

    # filterbank, if any
    if cfg['filterbank'] == 'mel':
        import audio
        filterbank = audio.create_mel_filterbank(
                cfg['sample_rate'], cfg['frame_len'], cfg['mel_bands'],
                cfg['mel_min'], cfg['mel_max'])
        filterbank = filterbank[:input_shape[3]].astype(theano.config.floatX)
        layer = DenseLayer(layer, num_units=cfg['mel_bands'],
                num_leading_axes=-1, W=T.constant(filterbank), b=None,
                nonlinearity=None)
    elif cfg['filterbank'] == 'mel_learn':
        layer = MelBankLayer(layer, cfg['sample_rate'], cfg['frame_len'],
                cfg['mel_bands'], cfg['mel_min'], cfg['mel_max'])
    elif cfg['filterbank'] != 'none':
        raise ValueError("Unknown filterbank=%s" % cfg['filterbank'])

    # magnitude transformation, if any
    if cfg['magscale'] == 'log':
        layer = ExpressionLayer(layer, lambda x: T.log(T.maximum(1e-7, x)))
    elif cfg['magscale'] == 'log1p':
        layer = ExpressionLayer(layer, T.log1p)
    elif cfg['magscale'].startswith('log1p_learn'):
        # learnable log(1 + 10^a * x), with given initial a (or default 0)
        a = float(cfg['magscale'][len('log1p_learn'):] or 0)
        a = T.exp(theano.shared(lasagne.utils.floatX(a)))
        layer = lasagne.layers.ScaleLayer(layer, scales=a,
                                          shared_axes=(0, 1, 2, 3))
        layer = ExpressionLayer(layer, T.log1p)
    elif cfg['magscale'].startswith('pow_learn'):
        # learnable x^sigmoid(a), with given initial a (or default 0)
        a = float(cfg['magscale'][len('pow_learn'):] or 0)
        a = T.nnet.sigmoid(theano.shared(lasagne.utils.floatX(a)))
        layer = PowLayer(layer, exponent=a)
    elif cfg['magscale'] == 'pcen':
        layer = PCENLayer(layer)
        if cfg.get('pcen_fix_alpha'):
            layer.params[layer.log_alpha].remove("trainable")
    elif cfg['magscale'] == 'loudness_only':
        # cut away half a block length on the left and right
        layer = lasagne.layers.SliceLayer(
                layer, slice(cfg['blocklen']//2, -(cfg['blocklen']//2)), axis=2)
        # average over the frequencies and channels
        layer = lasagne.layers.ExpressionLayer(
                layer, lambda X: X.mean(axis=(1, 3), keepdims=True),
                lambda shp: (shp[0], 1, shp[2], 1))
    elif cfg['magscale'] != 'none':
        raise ValueError("Unknown magscale=%s" % cfg['magscale'])

    # temporal difference, if any
    if cfg['arch.timediff']:
        layer = TimeDiffLayer(layer, delta=cfg['arch.timediff'])

    # standardization per frequency band
    if cfg.get('input_norm', 'batch') == 'batch':
        layer = batch_norm_vanilla(layer, axes=(0, 2), beta=None, gamma=None)
    elif cfg['input_norm'] == 'instance':
        layer = lasagne.layers.StandardizationLayer(layer, axes=2)
    elif cfg['input_norm'] == 'none':
        pass
    else:
        raise ValueError("Unknown input_norm=%s" % cfg['input_norm'])

    # convolutional neural network
    kwargs = dict(nonlinearity=lasagne.nonlinearities.leaky_rectify,
                  W=lasagne.init.Orthogonal())
    maybe_batch_norm = batch_norm if cfg['arch.batch_norm'] else lambda x: x
    if cfg['arch.convdrop'] == 'independent':
        maybe_dropout = lambda x: dropout(x, 0.1)
    elif cfg['arch.convdrop'] == 'channels':
        maybe_dropout = lambda x: dropout(x, 0.1, shared_axes=(2, 3))
    elif cfg['arch.convdrop'] == 'bands':
        maybe_dropout = lambda x: dropout(x, 0.1, shared_axes=(1, 2))
    elif cfg['arch.convdrop'] == 'none':
        maybe_dropout = lambda x: x
    else:
        raise ValueError("Unknown arch.convdrop=%s" % cfg['arch.convdrop'])
    if cfg['arch'] == 'dense:16':
        layer = DenseLayer(layer, 16, **kwargs)
        layer = DenseLayer(layer, 1,
                           nonlinearity=lasagne.nonlinearities.sigmoid,
                           W=lasagne.init.Orthogonal())
        return layer
    convmore = cfg['arch.convmore']
    layer = Conv2DLayer(layer, int(64 * convmore), 3, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = maybe_dropout(layer)
    layer = Conv2DLayer(layer, int(32 * convmore), 3, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = MaxPool2DLayer(layer, 3)
    layer = maybe_dropout(layer)
    layer = Conv2DLayer(layer, int(128 * convmore), 3, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = maybe_dropout(layer)
    layer = Conv2DLayer(layer, int(64 * convmore), 3, **kwargs)
    layer = maybe_batch_norm(layer)
    if cfg['arch'] == 'ismir2015':
        layer = MaxPool2DLayer(layer, 3)
    elif cfg['arch'] == 'ismir2016':
        layer = maybe_dropout(layer)
        layer = Conv2DLayer(layer, int(128 * convmore), (3, layer.output_shape[3] - 3), **kwargs)
        layer = maybe_batch_norm(layer)
        layer = MaxPool2DLayer(layer, (1, 4))
    else:
        raise ValueError('Unknown arch=%s' % cfg['arch'])
    layer = DenseLayer(dropout(layer, 0.5), 256, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = DenseLayer(dropout(layer, 0.5), 64, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = DenseLayer(dropout(layer, 0.5), 1,
                       nonlinearity=lasagne.nonlinearities.sigmoid,
                       W=lasagne.init.Orthogonal())
    return layer
