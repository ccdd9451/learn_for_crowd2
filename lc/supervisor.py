#!/usr/bin/env python
# encoding: utf-8

from . import config
from numpy import subtract


def adaptive_learning_rate(lRate, loss_history):

    decay_ref = (config.LEARNING_RATE * config.DECAY_RATE**
                 (len(loss_history) / config.DECAY_STEP))
    if (diff_test(loss_history, 20, 10)
            and loss_history[-11] > loss_history[-1] and lRate > decay_ref):
        new_lRate = lRate * config.DECAY_RATE
        print("           Current learning rate %.4e" % new_lRate)
        return new_lRate
    return lRate


def early_stop(lRate, loss_history):
    if (diff_test(loss_history, 200, 10, 10)
            and lRate < config.LEARNING_RATE * 10**-3):
        return True
    return False


def diff_test(array, compare_num, threshold, step=1, desire_diff=0):
    if len(array) > compare_num + 1:
        lslide1 = array[-compare_num - 1:-1:step]
        lslide2 = array[-compare_num::step]
        diff = subtract(lslide1, lslide2)
        uphills = sum(diff > desire_diff)
        if uphills > threshold:
            return True
    return False
