#!/usr/bin/env python
"""A quick driver to examine the behavior of noggin.random_index and
noggin.adaptive_index.

2018-0529 ML Rilee, RSTLLC, mike@rilee.net.
"""

import math
import os
import Krige
import Krige.DataField as df

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


x0 = 0.0
y0 = 0.0

x  = np.arange(-20,20,0.1)
y  = np.arange(-20,20,0.1)

beta=10.0**np.arange(-4,4,0.2)

for i in beta:
    idx = np.full(x.shape,False)
    idx = noggin.random_index(x0,y0,x,y,i)
    size = idx[idx].size
    print 'i,len(idx): '+str(i)+', '+str(size)


# For adaptive, the idea is to start beta large, and then reduce. At
# least for the normal distribution.

