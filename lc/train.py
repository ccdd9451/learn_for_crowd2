#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import time
from .config import INFOMESSAGE
from contexlib import contextmanager
from pathlib import Path


tools = type("Tools", (), {})()

def epoch_train():
    sess = tools.sess
    optimizer = tools.optimizer

    try:
        while True: sess.run(optimizer)
    except tf.errors.OutOfRangeError:
        infos = tf.get_collection("infos")
        infos = sess.run(tools.infos)
        print(INFOMESSAGE(infos))
        tools.reporter()


@contextmanager
def training(graphs, merge_key=tf.GraphKeys.SUMMARIES):
    with tf.Session() as sess:

        path = Path(time.strftime("%m-%d-%y_%H:%M"))
        g = tf.Variable(0, "global_step", False)
        e = tf.Variable(0, "epoch_step", False)
        e_add = tf.assign(e, e+1)

        fin_loss = graphs[0].get_tensor_by_name("fin_loss")
        accur = graphs[0].get_collection("loss")[0]
        val_accur = graphs[1].get_collection("loss")[0]

        writer = tf.summary.Filewriter(path/"summary")
        summary = tf.summary.merge_all(merge_key)

        tools.sess = sess
        tools.graphs = graphs
        tools.saver = tf.Saver(sess, path/"checkpoints", g)
        tools.infos = [e, e_add, fin_loss, accur, val_accur]
        import types
        def reporter(self):
            writer.add_summary(summary)
            writer.flush()
        tools.reporter = types.MethodType(reporter, tools)

        yield tools

def simple_train(graphs, epoch_steps):
    with training(graphs) as tools:
        for i in range(epoch_steps):
            batch_init = tf.get_collection("batch_init")
            tools.sess.run(batch_init)
            epoch_train()
