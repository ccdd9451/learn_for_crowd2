#!/usr/bin/env python
# encoding: utf-8


from tensorflow.contrib.layers import summarize_collection
import tensorflow as tf
from . import config

__all__ = ["add_L2_loss", "add_RMSE_loss", "fin_loss"]

with tf.name_scope("analysis") as analysis: pass
with tf.name_scope("visuals") as visuals: pass
with tf.name_scope("mean") as mean_ns: pass
with tf.name_scope("std") as std_ns: pass
with tf.name_scope("loss") as loss_ns: pass

def L2_loss(weight, name):
    with tf.name_scope(loss_ns):
        scale = tf.convert_to_tensor(float(config.L2_LAMBDA))
        l2_loss = scale * tf.reduce_sum(tf.square(weight))
        loss =  tf.identity(l2_loss, name=name)
        tf.add_to_collection("losses", loss)

def variance(tensor, name):
    with tf.name_scope(visuals):
        with tf.name_scope(mean_ns):
            mean = tf.reduce_mean(tensor, name="mean_"+name)
            tf.add_to_collection("visuals", mean)
        with tf.name_scope(std_ns):
            std_n = tf.sqrt(tf.reduce_mean(tf.square(tensor-mean)))
            std = tf.identity(std_n, name="std_"+name)
            tf.add_to_collection("visuals", std)
    return std

def add_L2_loss():
    weight_keys = [key
            for key in tf.get_collection("trainable_variables")
            if key.name.endswith("weights:0")]
    [L2_loss(w, "L2_loss") for w in weight_keys]
    [variance(w, "weight") for w in weight_keys]


def add_RMSE_loss(y, ref_y, suffix):
    with tf.name_scope(analysis):
        loss = tf.sqrt(tf.reduce_mean(tf.square(y-ref_y)))
        accuracy = tf.identity(loss, name="accuracy_"+suffix)
    with tf.name_scope(visuals):
        if suffix == "train":
            tf.add_to_collection("losses", accuracy)
        else:
            tf.add_to_collection("visuals", accuracy)


def fin_loss():
    with tf.name_scope(analysis):
        fin_loss = tf.identity(
            sum(tf.get_collection("losses")),
            name="fin_loss")
        return fin_loss

def collect_summaries(collections):
    for collect in collections:
        summarize_collection(collect)


