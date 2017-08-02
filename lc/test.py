#!/usr/bin/env python
# encoding: utf-8

import unittest
import tensorflow as tf
import numpy as np

from . import Loader

class Test_loader(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.data = Loader(self, test=True)
        print("Loader Hashing Value: ", hash(self)%1_0000_0000)

    def test_load_train(self):
        x, y = self.data.train()
        xval, yval = self.sess.run([x,y])
        for i in range(10):
            xval1, yval1 = self.sess.run([x,y])
            self.assertTrue(np.array_equal(xval, xval1))
            self.assertTrue(np.array_equal(yval, yval1))



    def test_load_batch(self):
        x, y = self.data.shuffle_batch()
        self.data.shuffle_batch_init(self.sess)
        xval, yval = self.sess.run([x,y])
        with self.assertRaises(tf.errors.OutOfRangeError):
            while(True):
                xval1, yval1 = self.sess.run([x,y])
                self.assertFalse(np.array_equal(xval, xval1))
                self.assertFalse(np.array_equal(yval, yval1))
        self.data.shuffle_batch_init(self.sess)
        xval1, yval1 = self.sess.run([x,y])
        self.assertFalse(np.array_equal(xval, xval1))
        self.assertFalse(np.array_equal(yval, yval1))
