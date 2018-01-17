#!/usr/bin/env python
# encoding: utf-8

import pickle
import numpy as np
import tensorflow as tf

from tensorflow.python.data import Dataset
from . import config


class Loader():
    """
    A loader will be created represents set of data.

    """

    def __init__(self, d, cut=[0.7, 0.85], size=None, test=False):
        """
        cut: Data boundary between train, valid, test set.
        size: Specific boundary, int, as train data
        """
        if not test:
            with open(config.DATAFILE, "rb") as f:
                self.__dict__.update(pickle.load(f))
        else:
            self.X = np.random.randint(0, 200, [5000, 50])
            self.Y = np.random.randint(0, 200, [5000, 1])
            self.info = "Test cases"

        assert self.X.shape[0] == self.Y.shape[0]
        self.data_size = self.X.shape[0]
        self.cut = int(self.data_size * cut[0])
        self.cut1 = int(self.data_size * cut[1])
        self.shape = (self.X.shape[1], self.Y.shape[1])

        config.DATANAME = d["name"]
        config.DISCRIPTION = ",".join([self.info, d["discription"]])
        np.random.seed(
            hash(config.DATANAME) % 100000000)  # Random 8 digits hash
        self.train_choices = np.random.choice(self.cut1, self.cut + 1, False)
        if size:
            self.train_choices = self.train_choices[:size]
        self.valid_choices = np.array(
            list(set(range(self.cut1)) - set(self.train_choices)))

    def train(self, Datasize=None):
        with tf.name_scope("Dataset"):
            fin = int(min(self.cut, Datasize) if Datasize else self.cut)
            choices = self.train_choices[:fin]
            dat = (Dataset.from_tensors(
                tuple(d[choices, :].astype(np.float32) for d in
                      [self.X, self.Y])).make_initializable_iterator())
            tf.add_to_collection("batch_init", dat.initializer)
            return dat.get_next("bs_dat")

    def validation(self, Datasize=None):
        with tf.name_scope("Dataset"):
            fin = int(
                min(self.cut1 - self.cut, Datasize)
                if Datasize else self.cut1 - self.cut)
            choices = self.valid_choices[:fin]
            mod = lambda data, name: (tf.convert_to_tensor(
                data[choices,:].astype(np.float32),
                name = name
            ))
            return mod(self.X, "valid_x"), mod(self.Y, "valid_y")

    def test(self):
        with tf.name_scope("Dataset"):
            fin = self.cut1
            mod = lambda data, name: (tf.convert_to_tensor(
                data[fin:,:].astype(np.float32),
                name = name
            ))
            return mod(self.X, "test_x"), mod(self.Y, "test_y")

    def shuffle_batch(self, shuffle_buffer=None, batch_size=100,
                      Datasize=None):
        with tf.name_scope("Dataset"):
            fin = int(min(self.cut, Datasize) if Datasize else self.cut)
            shuffle_buffer = shuffle_buffer if shuffle_buffer else self.cut
            choices = self.train_choices[:fin]
            dat = (Dataset.from_tensor_slices(
                tuple(d[choices, :].astype(np.float32)
                      for d in [self.X, self.Y])).shuffle(shuffle_buffer)
                   .batch(batch_size).make_initializable_iterator())
            tf.add_to_collection("batch_init", dat.initializer)
            return dat.get_next("bs_dat")

    def train_init(self, sess):
        initializer = tf.get_collection("batch_init")
        sess.run(initializer)
