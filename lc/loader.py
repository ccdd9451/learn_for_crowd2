#!/usr/bin/env python
# encoding: utf-8

import pickle

from tensorflow.contrib.data import Dataset
from . import config



INF = float("inf")

class Loader():
    """
    A loader will be created represents set of data.
    """
    def __init__(self, cut=0.8, size=-1):
        """
        cut: Data boundary between train and test set.
        """
        with open(config.DATAFILE, "rb") as f:
            self.__dict__.update(pickle.load(f))

        assert self.X.shape[0] == self.Y.shape[0]
        self.data_size = self.X.shape[0]
        self.feature_size = self.X.shape[1]
        self.output_size = self.Y.shape[1]
        self.cut = self.size // cut

    def train(self, Datasize=INF):
        fin = int(min(self.cut, max(Datasize, 1)))
        return [ Dataset.from_tensor_slices(x[:fin,:])
                for x in (self.X, self.Y) ]

    def test(self, Datasize=INF):
        fin = int(min(self.cut, max(Datasize, 1)))
        return [ Dataset.from_tensor_slices(x[fin:,:])
                for x in (self.X, self.Y) ]



