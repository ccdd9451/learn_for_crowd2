#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import time
import sys

from . import analysis
from . import config
from . import supervisor
from contextlib import contextmanager
from xilio import dump, write, append

tools = type("Tools", (), {})()

__all__ = ["simple_train", "training"]


def epoch_train(tools, **kwargs):
    """
    Do epoch train for one times.

    input: tools

    """
    sess = tools.sess
    optimizer = tools.optimizer

    feed_dict = kwargs.get("feed_dict", {})

    infos, summary, e, _ = sess.run(tools.infos, feed_dict=feed_dict)
    if config.VERBOSE_EACH:
        if not int(e) % config.VERBOSE_EACH:
            print(config.INFOMESSAGE(infos))
            sys.stdout.flush()
    else:
        print(config.INFOMESSAGE(infos))
        sys.stdout.flush()

    tools.reporter(summary, e)

    try:
        if not feed_dict:
            while True:
                sess.run(optimizer)
        else:
            while True:
                sess.run(optimizer, feed_dict=feed_dict)
    except tf.errors.OutOfRangeError:
        pass
    return infos


@contextmanager
def training(restore_form=None, merge_key=tf.GraphKeys.SUMMARIES):
    with tf.Session() as sess:
        graph = tf.get_default_graph()

        path = config.DATANAME + "/" + time.strftime("%m-%d-%y_%H:%M")
        g = tf.Variable(0, name="global_step", trainable=False)
        with tf.name_scope("epoch_step"):
            e = tf.Variable(0, name="epoch_step", trainable=False)
            e_add = tf.assign(e, e + 1)

        fin_loss = analysis.fin_loss()
        with tf.name_scope("train"):
            learning_rate = tf.train.exponential_decay(
                float(config.LEARNING_RATE), e,
                float(config.DECAY_STEP), float(config.DECAY_RATE))
            tf.summary.scalar("learning_rate", learning_rate)
            tools.learning_rate = learning_rate

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(fin_loss, global_step=g)
            # gradients = optimizer.compute_gradients(fin_loss)
            # capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            # train_op = optimizer.apply_gradients(capped_gradients, global_step=g)

        accur = graph.get_tensor_by_name("analysis/accuracy_train:0")
        val_accur = graph.get_tensor_by_name("analysis/accuracy_test:0")
        infos = [e, fin_loss, accur, val_accur]
        updates = [e_add, train_op]

        writer = tf.summary.FileWriter(path + "/summary", graph)
        summary = tf.summary.merge_all(merge_key)
        saver = tf.train.Saver(tf.get_collection("trainable_variables"))

        tools.path = path
        tools.sess = sess
        tools.graph = graph
        tools.saver = saver
        tools.infos = [infos, summary, e, updates]
        tools.optimizer = train_op
        import types

        def reporter(self, summary, e):
            writer.add_summary(summary, e)
            writer.flush()
            if not int(e) % config.VERBOSE_EACH:
                saver.save(sess, path + "/chkpnt", e)

        tools.reporter = types.MethodType(reporter, tools)

        tf.global_variables_initializer().run(None, sess)
        tf.local_variables_initializer().run(None, sess)
        if restore_form:
            ckpt = tf.train.latest_checkpoint(restore_form)
            print("restoring file from: " + ckpt)
            if ckpt:
                saver.restore(sess, ckpt)
        graph.finalize()
        yield tools


def simple_train(epoch_steps):
    infos = []
    start_time = time.time()
    restore_form = getattr(config, "RESTORE_FROM", None)
    with training(restore_form) as tools:
        write(tools.path + "/description", config.details() + "\n")
        try:
            for i in range(epoch_steps):
                batch_init = tf.get_collection("batch_init")
                tools.sess.run(batch_init)
                infos.append(epoch_train(tools))
        except KeyboardInterrupt:
            pass
        finally:
            dump(tools.path + "/trace", infos)
            duration = time.time() - start_time
            append(tools.path + "/description", "Time usage: " + time.strftime(
                "%M minutes, %S seconds", time.gmtime(duration)) + "\n")

        return str(tools.path), infos[-1]


def adaptive_train(max_epoch_steps):
    learning_rate = config.LEARNING_RATE
    loss_hist = []
    accur_hist = []
    infos = []
    start_time = time.time()
    restore_form = getattr(config, "RESTORE_FROM", None)
    with training(restore_form) as tools:
        write(tools.path + "/description", config.details() + "\n")
        try:
            for i in range(max_epoch_steps):
                batch_init = tf.get_collection("batch_init")
                tools.sess.run(batch_init)
                info = epoch_train(
                    tools,
                    feed_dict={
                        tools.learning_rate: learning_rate,
                    }, )
                infos.append(info)
                loss_hist.append(float(info[1]))
                accur_hist.append(float(info[3]))
                if not i % int(config.DECAY_STEP / 3):
                    learning_rate = supervisor.adaptive_learning_rate(
                        learning_rate, loss_hist, accur_hist)
                if not i % 100 and supervisor.early_stop(
                        learning_rate, loss_hist):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            dump(tools.path + "/trace", infos)
            duration = time.time() - start_time
            append(tools.path + "/description", "Time usage: " + time.strftime(
                "%M minutes, %S seconds", time.gmtime(duration)) + "\n")
        return str(tools.path), infos[-1]
