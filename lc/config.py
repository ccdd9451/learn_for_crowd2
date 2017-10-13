#!/usr/bin/env python
# encoding: utf-8


def INFOMESSAGE(info):
    info = [float(x) for x in info if x is not None]
    return _INFOMESSAGE.format(*info)


_INFOMESSAGE = "El {0:.0f}: loss {1:.4f}, tacc {2:.4f}, cvacc {3:.4f}"

DATAFILE = "df.dat"

LEARNING_RATE = 0.001
DECAY_STEP = 50
DECAY_RATE = 0.90
L2_LAMBDA = 0.05
STOP_THRESHOLD = -1
KEEP_PROB = 0.5
VERBOSE_EACH = None


def details():
    import yaml
    return yaml.dump(
        {
            key: value
            for key, value in globals().items()
            if key[0].isupper() and key != "INFOMESSAGE"
        },
        default_flow_style=False)
