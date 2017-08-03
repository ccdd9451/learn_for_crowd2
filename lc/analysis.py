#!/usr/bin/env python
# encoding: utf-8


import tensorflow as tf
from . import config

__all__ = ["add_L2_loss", "add_RMSE_loss", "fin_loss"]

with tf.name_scope("analysis") as analysis: pass

def L2_loss(weight, name):
    scale = tf.convert_to_tensor(float(config.L2_LAMBDA))
    l2_loss = scale * tf.reduce_sum(tf.square(weight))
    return tf.identity(l2_loss, name=name)


def add_L2_loss():
    with tf.name_scope(analysis):
        weight_keys = [key
            for key in tf.get_collection("trainable_variables")
            if key.name.endswith("weights:0")]
        losses = [L2_loss(w, "L2_loss") for w in weight_keys]
        for loss in losses:
            tf.add_to_collection("losses", loss)

def add_RMSE_loss(y, ref_y, suffix):
    with tf.name_scope(analysis):
        loss = tf.sqrt(tf.reduce_mean(tf.square(y-ref_y)))
        accuracy = tf.identity(loss, name="accuracy_"+suffix)
        if suffix == "train":
            tf.add_to_collection("losses", accuracy)


def fin_loss():
    with tf.name_scope(analysis):
        fin_loss = tf.identity(
            sum(tf.get_collection("losses")),
            name="fin_loss")
        return fin_loss


