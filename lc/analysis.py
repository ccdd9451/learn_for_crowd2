#!/usr/bin/env python
# encoding: utf-8


import tensorflow as tf
from . import config

def L2_loss(weight, name):
    scale = tf.convert_to_tensor(config.L2_LAMBDA)
    l2_loss = scale * tf.reduce_sum(tf.square(scale))
    return tf.identity(l2_loss, name=name)



def add_L2_loss():
    with tf.name_scope("analysis"):
        weight_keys = tf.get_collection("weights")
        losses = [l2_loss(w, "L2_loss") for w in weight_keys]
        tf.add_to_collection("losses", losses)

def fin_loss():
    with tf.name_scope("analysis"):
        return tf.identity(
            sum(tf.get_collection("losses")),
            name="fin_loss")


