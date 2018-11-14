#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:22:31 2018
    This code for plotting line/curve figures

@author: weibo
"""
# %% Load necessary module
import numpy as np
import pandas as pd
import plt2pandas as p2p
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, splev, splrep
from DataPost import DataPost
import FlowVar as fv
from timer import timer
import sys
import os

plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
font1 = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "medium",
}
path = "/media/weibo/Data1/BFS_M1.7L_0505/7/2/TP_data_01044880/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/temp/"
path3 = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
path4 = "/media/weibo/Data1/BFS_M1.7L_0505/MeanFlow/"

matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 18
numsize = 15
matplotlib.rc("font", size=textsize)

VarName = [
    "x",
    "y",
    "z",
    "u",
    "v",
    "w",
    "rho",
    "p",
    "T",
    "Q-criterion",
    "L2-criterion",
]
orig, sol = p2p.NewReadINCAResults(240, path, VarName)
mean = pd.read_hdf(path3 + 'TimeAve.h5')

# %% Plot BL profile along streamwise direction
# load data
loc = ['z', 'y']
val = [0.0, -2.99704]
pert = fv.PertAtLoc(orig, mean, 'u', loc, val)
fig, ax = plt.subplots(figsize=(10, 3.5))
matplotlib.rc('font', size=14)
ax.set_ylabel(r"$u^{\prime}/u_{\infty}$", fontsize=textsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.plot(pert.x, pert.u, 'k')
ax.set_xlim([0.0, 30])
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    path2 + "PerturProfileX.svg", bbox_inches="tight", pad_inches=0.1
)


# %% Plot BL profile along wall-normal direction
# load data
loc = ['x', 'z']
val = [10.0, 0.0]
pert = fv.PertAtLoc(orig, mean, 'u', loc, val)
fig, ax = plt.subplots(figsize=(10, 3.5))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$u^{\prime}/u_{\infty}$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.plot(pert.u, pert.y)
ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    path2 + "PerturProfileY.svg", bbox_inches="tight", pad_inches=0.1
)


# %% Plot BL profile along spanwise
# load data
loc = ['x', 'y']
valarr = [[0.0078125, -0.00781],
          [2.5625, -0.43750],
          [7.625, -1.4375],
          [11.3125, -1.6875],
          [15.0, -1.6875]]
fig, ax = plt.subplots(1, 5, figsize=(10, 3.5))
fig.subplots_adjust(hspace=0.5, wspace=0.05)
ax = ax.ravel()
matplotlib.rc('font', size=14)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
for i in range(np.shape(valarr)[0]):
    val = valarr[i]
    pert = fv.PertAtLoc(orig, mean, 'u', loc, val)
    frame1 = orig.loc[orig[loc[0]] == val[0]]
    pert1 = frame1.loc[np.around(frame1[loc[1]], 5) == val[1]]
    ax[i].plot(pert.u, pert.z, 'k')
    ax[i].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax[i].grid(b=True, which="both", linestyle=":")
    ax[i].set_xlim([-0.45, 0.45])
    ax[i].set_title(title[i], fontsize=numsize)
    ax[i].set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax[i].set_xticklabels('')
    ax[i].set_yticklabels('')
ax[2].set_xticklabels([r'$-0.4$', '',  r'$0.0$', '', r'$0.4$'])
ax[2].set_xlabel(r'$u^{\prime}/u_{\infty}$', fontsize=textsize)
ax[0].set_yticks([-2.0, 0.0, 2.0])
ax[0].set_yticklabels([r'$-2.0$', r'$0.0$', r'$2.0$'])      
ax[0].set_ylabel(r"$z/\delta_0$", fontsize=textsize)
#ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.show()
plt.savefig(
    path2 + "PerturProfileZ.svg", bbox_inches="tight", pad_inches=0.1
)


# %% Plot amplitude of fluctuations from temporal data 
# load data
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots/"
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
var = 'u'    
fa = 1.0 #1.7*1.7*1.4
timepoints = np.arange(650.0, 899.5 + 0.5, 0.5)
Snapshots = DataFrame[['x', 'y', 'z', var]]
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(InFolder + dirs[i+1])
        Frame2 = TempFrame[['x', 'y', 'z', var]]
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        Snapshots = pd.concat([Snapshots, Frame2])
m, n = np.shape(Snapshots)
# %%
xcor = np.arange(0.0, 30.0 + 0.125, 0.125)
amplit = np.zeros(np.size(xcor))
for i in range(np.size(xcor)):
    xyz = [xcor[i], -2.99704, 0.0]
    amplit[i] = fv.Amplit(Snapshots, xyz, 'u')
    
fig, ax = plt.subplots(figsize=(10, 3.5))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$A_{u^{\prime}}$", fontsize=textsize)
ax.plot(xcor, amplit)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    path2 + "AmplitX.svg", bbox_inches="tight", pad_inches=0.1
)

# %% 
grow = fv.GrowthRate(xcor, amplit)
fig, ax = plt.subplots(figsize=(10, 3.5))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\alpha_i$", fontsize=textsize)
ax.plot(xcor, grow)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    path2 + "AmplitX.svg", bbox_inches="tight", pad_inches=0.1
)
