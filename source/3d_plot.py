#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:39:40 2019
    3D plots use matplotlib 

@author: weibo
"""
# %% Load libraries
import pandas as pd
import os
import sys
import numpy as np
from glob import glob
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plt2pandas as p2p
import matplotlib.ticker as ticker
from data_post import DataPost
from planar_field import PlanarField as pf

# %% data path settings
path = "/media/weibo/VID1/BFS_M1.7L/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + 'video/'
pathD = path + 'DMD/'

## figures properties settings
plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 13
numsize = 10

# %% 3D curve plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(0, 8, 1)
y = np.arange(1, 100, 0.2)
z = np.sin(0.2*y)
Y, Z = np.meshgrid(y, z)


for i in range(np.size(x)):
    X = x[i] * np.ones(np.shape(y))
    ax.plot(np.log10(y), X, z, zdir='z', linewidth=1.5)

ax.set_xscale('symlog')
ax.set_zticks([-1, 0.0, 1.0])
ax.set_xlabel(r'$f$', fontsize=textsize)
ax.set_ylabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_zlabel(r'$\mathrm{PSD}$', fontsize=textsize)
ax.view_init(elev=20, azim=-35)
plt.rcParams['grid.color'] = 'gray'
plt.grid(linestyle='dotted')
plt.show()
plt.savefig("test.svg", bbox_inches="tight")


# %%
