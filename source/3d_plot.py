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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib
import plt2pandas as p2p
import matplotlib.ticker as ticker
from data_post import DataPost
from matplotlib.ticker import ScalarFormatter
from planar_field import PlanarField as pf


# %% data path settings
path = "/media/weibo/IM1/BFS_M1.7Tur/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + 'video/'
pathD = path + 'DMD/'
pathSL = path + 'Slice/'

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

# %% 3D PSD
# load data
var = 'p'
xval = np.loadtxt(pathSL + 'FWPSD_x.dat', delimiter=' ')
freq = np.loadtxt(pathSL + 'FWPSD_freq.dat', delimiter=' ')
FPSD = np.loadtxt(pathSL + var + '_FWPSD_psd.dat', delimiter=' ')
freq = freq[1:]
FPSD = FPSD[1:, :]
newx = [-10.0, 2.0, 3.0, 5.0, 9.0, 10.0]

fig = plt.figure(figsize=(7.0, 4.0))
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = 'dotted'
plt.tick_params(labelsize=numsize)
ax = fig.add_subplot(111, projection='3d')
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.8, 1.2, 1.0, 1.0]))
for i in range(np.size(newx)):
    ind = np.where(xval[:]==newx[i])[0][0]
    xloc = newx[i] * np.ones(np.shape(freq))
    ax.plot(freq, xloc, FPSD[:, ind], zdir='z', linewidth=1.5)
    
    
ax.ticklabel_format(axis="z", style="sci", scilimits=(-2, 2))
ax.zaxis.offsetText.set_fontsize(numsize)
# ax.w_xaxis.set_xscale('log')
ax.set_xscale('symlog')
# ax.set_xticks([-2, -1, 0, 0.3])
ax.set_xlabel(r'$f$', fontsize=textsize)
ax.set_ylabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_zlabel(r'$\mathcal{P}$', fontsize=textsize)
#ax.xaxis._axinfo['label']['space_factor'] = 0.1
#ax.zaxis._axinfo["grid"]['linewidth'] = 1.0
#ax.zaxis._axinfo["grid"]['color'] = "gray"
#ax.zaxis._axinfo['grid']['linstyle'] = ':'
ax.tick_params(axis='x', direction='in')
ax.tick_params(axis='y', direction='in')
ax.tick_params(axis='z', direction='in')
ax.view_init(elev=50, azim=-20)
ax.axes.xaxis.labelpad=1
ax.axes.yaxis.labelpad=6
ax.axes.zaxis.labelpad=0.1
ax.tick_params(axis='x', which='major', pad=1)
ax.tick_params(axis='y', which='major', pad=0.1)
ax.tick_params(axis='z', which='major', pad=0.1)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "3dFWPSD.svg")
plt.show()

#plt.rcParams['xtick.direction'] = 'in'

# %% 3d contour plot
# load data
var = 'p'
xval = np.loadtxt(pathSL + 'FWPSD_x.dat', delimiter=' ')
freq = np.loadtxt(pathSL + 'FWPSD_freq.dat', delimiter=' ')
FPSD = np.loadtxt(pathSL + var + '_FWPSD_psd.dat', delimiter=' ')
freq = freq[1:]
FPSD = FPSD[1:, :]


Yxval, Xfreq = np.meshgrid(xval, freq)

fig = plt.figure(figsize=(7.0, 4.0))
ax = fig.add_subplot(111, projection='3d')
# ax = fig.gca(projection='3d')
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = 'dotted'
plt.tick_params(labelsize=numsize)

ax.plot_surface(np.log10(Xfreq), Yxval, FPSD, rstride=1, cstride=1, cmap=cm.bwr)
xticks = [1e-2, 1e-1, 1e0]
ax.set_xticks(np.log10(xticks))
# ax.set_xticklabels([r'$10^{-2}$', r'$10^{-2}$', r'$10^{-2}$'])

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.8, 1.2, 1.0, 1.0]))
ax.set_xlabel(r'$f$', fontsize=textsize)
ax.set_ylabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_zlabel(r'$\mathcal{P}$', fontsize=textsize)
# ax.set_powerlimits((-2, 2))
# ax.w_zaxis.set_major_formatter()
# ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="z", style="sci", scilimits=(-2, 2))
ax.view_init(elev=50, azim=-20)
ax.axes.xaxis.labelpad=1
ax.axes.yaxis.labelpad=6
ax.axes.zaxis.labelpad=0.1
ax.tick_params(axis='x', which='major', pad=1)
ax.tick_params(axis='y', which='major', pad=0.1)
ax.tick_params(axis='z', which='major', pad=0.1)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
#plt.savefig(pathF + "3dContourFWPSD.svg")
plt.show()

