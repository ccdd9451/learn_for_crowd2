#!env python
# coding: utf-8

from xilio import load
from pathlib import Path
import numpy as np

data = [load(x) for x in Path(".").glob("*_avg")]

for d in data:
    print(d["info"])
    print(d["X"].shape)
    print(np.mean(d["Y"]))
    print(np.std(d["Y"], ddof=1))
    print()
