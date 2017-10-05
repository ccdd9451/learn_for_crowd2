#!/usr/bin/env python
# encoding: utf-8

from numpy import substract

def adaptive_learning_rate(error_rate, loss_history):

    if diff_test(loss_history, 10, 4):
        return error_rate / 2
    return error_rate

def early_stop(loss_history):
    if diff_test(loss_history, 100, 40):
        return True
    return False

def diff_test(array, compare_num, threshold):
    if len(array) > compare_num + 1:
        lslide1 = array[-compare_num-1:-1]
        lslide2 = array[-compare_num:]
        diff = substract(lslide1, lslide2)
        uphills = sum(diff > 0)
        if uphills > threshold:
            return True
    return False

