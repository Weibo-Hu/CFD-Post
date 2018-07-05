#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:22:31 2018
    This code for plotting line/curve figures

@author: weibo
"""
#%% Load necessary module
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter
from scipy.interpolate import griddata
from scipy.interpolate import spline
from scipy import integrate
import sys
from DataPost import DataPost
import FlowVar as fv

plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',
    #'color' : 'k',
    'weight': 'normal',
}

path = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

#%% Load Data
VarName  = ['x', 'y', 'u', 'v', 'w', \
            'rho', 'p', 'T', 'uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'Qcrit']
MeanFlow = DataPost()
MeanFlow.UserData(VarName, path2+'Meanflow.dat', 1, Sep='\t')
MeanFlow.AddWallDist(3.0)

#%% Plot BL profile along streamwise
xcoord = np.array([-40, -20, 0, 5, 10, 15, 20, 30, 40])
num = np.size(xcoord)
xtick = np.zeros(num+1)
xtick[-1] = 1.0
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(np.arange(num+1), np.zeros(num+1), 'w-')
ax.set_xlim([0, num+0.5])
ax.set_ylim([0, 4])
ax.set_xticks(np.arange(num+1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = xtick
#ax.set_xticklabels(labels, fontdict=font)
ax.set_xticklabels(["$%d$"%f for f in xtick])
for i in range(num):
    y0, q0 = MeanFlow.BLProfile('x', xcoord[i], 'u')
    ax.plot(q0+i, y0, 'k-')
    ax.text(i+0.75, 3.0, r'$x/\delta_0={}$'.format(xcoord[i]), rotation=90, fontdict=font)
matplotlib.rc('font', size=12)
plt.tick_params(labelsize=12)
ax.set_xlabel(r'$u/u_{\infty}$', fontsize=14)
ax.set_ylabel(r'$y/\delta_0$', fontsize=14)
ax.grid (b=True, which = 'both', linestyle = ':')
plt.show()
plt.savefig(path2 + 'StreamwiseBLprofile.svg', bbox_inches='tight', pad_inches=0.1)
