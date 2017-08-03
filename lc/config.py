#!/usr/bin/env python
# encoding: utf-8




def INFOMESSAGE(info):
    info = [float(x) for x in info if x is not None]
    return _INFOMESSAGE.format(*info)
_INFOMESSAGE = "Epoch loop {0:.0f}: loss {1:.2f}, train accuracy {2:.2f}, cross validation accuracy {3:.2f}"


DATAFILE="df.dat"

LEARNING_RATE = 1
DECAY_STEP = 1
DECAY_RATE = 0.97
L2_LAMBDA = 0.1

STOP_THRESHOLD = 0.003
