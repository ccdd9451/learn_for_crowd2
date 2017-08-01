#!/usr/bin/env python
# encoding: utf-8

import pickle
import numpy as np
import tensorflow as tf

from tensorflow.contrib.data import Dataset
from . import config




class Loader():
    """
    A loader will be created represents set of data.
    """
    def __init__(self, cut=0.8, size=None):
        """
        cut: Data boundary between train and test set.
        """
        with open(config.DATAFILE, "rb") as f:
            self.__dict__.update(pickle.load(f))

        assert self.X.shape[0] == self.Y.shape[0]
        self.data_size = self.X.shape[0]
        self.feature_size = self.X.shape[1]
        self.output_size = self.Y.shape[1]
        self.cut = size if size else self.data_size // cut

    def train(self, Datasize=None):
        fin = min(self.cut, Datasize) if Datasize else self.cut
        mod = lambda data: (Dataset.from_tensor_slices(
                                data[:fin,:].astype(np.float32)))
        return mod(self.X), mod(self.Y)

    def test(self):
        fin = self.cut
        mod = lambda data, name: (tf.convert_to_tensor(
            data[fin:,:].astype(np.float32),
            name = name
        ))
        return mod(self.X, "test_x"), mod(self.Y, "test_y")

    def shuffle_batch(self,
                      shuffle_buffer=None,
                      batch_size=100,
                      Datasize=None):
        shuffle_buffer = shuffle_buffer if shuffle_buffer else self.cut
        dat = lambda data: (data
                            .shuffle(shuffle_buffer)
                            .batch(batch_size)
                            .make_initializable_iterator())
        x, y = self.train(Datasize)
        x_d, y_d = dat(x), dat(y)
        tf.add_to_collection("batch_init",
                             [x_d.initializer,
                              y_d.initializer])
        return x.get_next("x"), y.get_next("y")

