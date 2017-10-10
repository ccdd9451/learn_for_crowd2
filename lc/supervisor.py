#!/usr/bin/env python
# encoding: utf-8

from . import config
from numpy import subtract, mean, std

Neural_Tuning = None


def adaptive_learning_rate(lRate, loss_history, laccur):

    global Neural_Tuning
    if Neural_Tuning:
        return lRate

    decay_ref = (config.LEARNING_RATE * config.DECAY_RATE**
                 (len(loss_history) / config.DECAY_STEP))
    if (diff_test(loss_history, 100, 0.5, 5) and lRate > decay_ref):
        new_lRate = lRate * config.DECAY_RATE
        print("           Current learning rate %.4e" % new_lRate)
        if (std(laccur[-100:]) / mean(laccur[-100:]) < 0.001):
            Neural_Tuning = len(loss_history)
            print("Network tuning started")
        return new_lRate

    return lRate


def early_stop(lRate, lhist):
    global Neural_Tuning

    if (diff_test(lhist, 1000, 0.5, 25)
            and lRate < config.LEARNING_RATE * config.STOP_THRESHOLD):
        return True

    if (Neural_Tuning):
        nt = Neural_Tuning
        sl = int(min(max(500, nt * 0.2), 200))
        if (len(lhist) > nt + 3 * sl
                and mean(lhist[nt:nt + sl]) < mean(lhist[-sl:])):
            return True
    return False


def diff_test(array, compare_num, thr_cut, step=1, desire_diff=0):
    if len(array) > compare_num + step:
        lslide1 = array[-compare_num - step:-step:step]
        lslide2 = array[-compare_num::step]
        diff = subtract(lslide1, lslide2)
        uphills = sum(diff < desire_diff)
        if uphills > compare_num * thr_cut / step:
            return True
    return False
