#!/usr/bin/env python
# encoding: utf-8
"""
   All graphs used in tensorflow will be defined in this module,
   will be imported as differend classes.
"""

import tensorflow as tf
from . import config

def max_out(inputs, num_units=None, axis=None):
    num_units = num_units if num_units else config.NUM_UNIT
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(
                             num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs
