#!/usr/bin/env python
# encoding: utf-8




def INFOMESSAGE(info):
    info = [float(x) for x in info if x is not None]
    return _INFOMESSAGE.format(info)
_INFOMESSAGE = "Epoch loop {0}: loss {1}, train accuracy{2}, cross validation accuracy{3}"


DATAFILE="df.dat"
