#!/usr/bin/env python
# encoding: utf-8


placeholder = """\
with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, feature_size], name="x")
    y = tf.placeholder(tf.float32, [None, output_size], name="y")

"""
