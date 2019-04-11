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
from scipy import signal
from DataPost import DataPost
import FlowVar as fv
from timer import timer
import sys
import os
from planarfield import PlanarField as pf

plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}

path = "/media/weibo/Data3/BFS_M1.7L_0505/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"

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

# fluc = p2p.ReadAllINCAResults(240, path+'TP_stat', 
#                               FoldPath2=pathM, OutFile='TP_stat')
fluc = pd.read_hdf(pathM+'TP_stat.h5')
# %% Plot BL profile along streamwise direction
# load data
loc = ['z', 'y']
val = [0.0, -2.99704]
pert = fv.PertAtLoc(fluc, '<u`u`>', loc, val)
fig, ax = plt.subplots(figsize=(10, 3.5))
matplotlib.rc('font', size=14)
ax.set_ylabel(r"$\sqrt{u^{\prime 2}}/u_{\infty}$", fontsize=textsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.plot(pert['x'], np.sqrt(pert['<u`u`>']), 'k')
ax.set_xlim([0.0, 30])
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "PerturProfileX.svg", bbox_inches="tight", pad_inches=0.1
)

# %% Plot BL profile along wall-normal direction
# load data
loc = ['x', 'z']
val = [10.0, 0.0]
pert = fv.PertAtLoc(fluc, '<u`u`>', loc, val)
fig, ax = plt.subplots(figsize=(10, 3.5))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$u^{\prime}/u_{\infty}$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.plot(pert['<u`u`>'], pert['y'])
ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "PerturProfileY.svg", bbox_inches="tight", pad_inches=0.1
)


# %% Plot BL profile along spanwise
# load data
loc = ['x', 'y']
valarr = [[0.0078125, -0.00781],
          [2.5625, -0.43750],
          [5.5625, -1.1875],
          [7.625, -1.4375],
          [11.3125, -1.6875]]
#          [15.0, -1.6875]]
#plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
fig, ax = plt.subplots(1, 5, figsize=(10, 3.5))
fig.subplots_adjust(hspace=0.5, wspace=0.15)
matplotlib.rc('font', size=14)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
# a
val = valarr[0]
pert = fv.PertAtLoc(fluc, '<u`u`>', loc, val)
ax[0].plot(np.sqrt(pert['<u`u`>']), pert['z'], 'k-')
ax[0].set_xlim([0.0, 2e-3])
ax[0].ticklabel_format(axis="x", style="sci", scilimits=(-1, 1))
ax[0].set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])      
ax[0].set_ylabel(r"$z/\delta_0$", fontsize=textsize)
ax[0].tick_params(labelsize=numsize)
ax[0].set_title(title[0], fontsize=numsize)
ax[0].grid(b=True, which="both", linestyle=":")
# b 
val = valarr[1]
pert = fv.PertAtLoc(fluc, '<u`u`>', loc, val)
ax[1].plot(np.sqrt(pert['<u`u`>']), pert['z'], 'k-')
ax[1].set_xlim([0.0, 2.0e-2])
ax[1].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[1].set_yticklabels('')
ax[1].tick_params(labelsize=numsize)
ax[1].set_title(title[1], fontsize=numsize)
ax[1].grid(b=True, which="both", linestyle=":")
# c
val = valarr[2]
pert = fv.PertAtLoc(fluc, '<u`u`>', loc, val)
ax[2].plot(np.sqrt(pert['<u`u`>']), pert['z'], 'k-')
ax[2].set_xlim([0.025, 0.20])
ax[2].set_xlabel(r"$\sqrt{u^{\prime 2}}/u_{\infty}$",
                 fontsize=textsize, labelpad=18.0)
ax[2].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[2].set_yticklabels('')
ax[2].tick_params(labelsize=numsize)
ax[2].set_title(title[2], fontsize=numsize)
ax[2].grid(b=True, which="both", linestyle=":")
# d
val = valarr[3]
pert = fv.PertAtLoc(fluc, '<u`u`>', loc, val)
ax[3].plot(np.sqrt(pert['<u`u`>']), pert['z'], 'k-')
ax[3].set_xlim([0.025, 0.20])
ax[3].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[3].set_yticklabels('')
ax[3].tick_params(labelsize=numsize)
ax[3].set_title(title[3], fontsize=numsize)
ax[3].grid(b=True, which="both", linestyle=":")
# e
val = valarr[4]
pert = fv.PertAtLoc(fluc, '<u`u`>', loc, val)
ax[4].plot(np.sqrt(pert['<u`u`>']), pert['z'], 'k-')
ax[4].set_xlim([0.025, 0.20])
ax[4].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[4].set_yticklabels('')
ax[4].tick_params(labelsize=numsize)
ax[4].set_title(title[4], fontsize=numsize)
ax[4].grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "PerturProfileZ.svg", bbox_inches="tight", pad_inches=0.1
)


# %% RMS along dividing line
# extract data
# pert1 = fluc.loc[fluc['z']==0.0]
dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)[:-2, :]
#x0 = np.arange(0.0, 5.0, 4*0.03125)
#y0 = np.arange(0.0625, 1.3125, 0.03125)*(-1)
#x1 = np.arange(5.0, 11.25, 4*0.0625)
#y1 = np.arange(1.3125, 2.875, 0.0625)*(-1)
x2 = np.arange(11.1875, 50.0+0.0625, 0.0625)
y2 = np.ones(np.size(x2))*(-2.99342)
x3 = np.concatenate((dividing[:,0], x2), axis=0)
y3 = np.concatenate((dividing[:,1], y2), axis=0)
xx = np.zeros(np.size(x3))
yy = np.zeros(np.size(y3))
frame1 = fluc.loc[fluc['z'] == 0, ['x', 'y']]
frame2 = frame1.query("x>=0.0 & y<=3.0")
xmesh = frame2['x'].values
ymesh = frame2['y'].values
for i in range(np.size(x3)):
    variance = np.sqrt(np.square(xmesh - x3[i]) + np.square(ymesh - y3[i]))
    idx = np.where(variance == np.min(variance))[0][0]
    xx[i] = xmesh[idx]
    yy[i] = ymesh[idx]
grouped = fluc.groupby(['x', 'y'])
pert1 = grouped.mean().reset_index()
xmesh, ymesh = np.meshgrid(dividing[:, 0], dividing[:, 1])
var0 = griddata((pert1.x, pert1.y), pert1['<u`u`>'], (xmesh, ymesh))
    
# %% Plot Max RMS from BL profile 
# compute
grouped = fluc.groupby(['x', 'y'])
pert1 = grouped.mean().reset_index()
xnew = np.arange(0.0, 50.0+0.0625, 0.0625)
zz = np.zeros(np.size(xnew))
var1 = np.zeros(np.size(xnew))
ynew = np.zeros(np.size(xnew))
for i in range(np.size(xnew)):
    df = fv.MaxPertAlongY(pert1, '<u`u`>', [xnew[i], zz[i]])
    var1[i] = df['<u`u`>']
    ynew[i] = df['y']
#loc = ['z', 'y']
#val = [0.0, -2.875]
#pert = fv.PertAtLoc(fluc, '<u`u`>', loc, val)

#%% draw maximum value curve
fig, ax = plt.subplots(figsize=(10, 3.0))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.plot(xnew, ynew, 'k')
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "MaxPertLoc.svg", bbox_inches="tight", pad_inches=0.1
)

# %% Plot amplitude of fluctuations from temporal data 
# load data
InFolder = path + "Snapshots/"
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
var = 'u'    
fa = 1.0 #1.7*1.7*1.4
timepoints = np.arange(800, 1149.5 + 0.5, 0.5)
Snapshots = DataFrame[['x', 'y', 'z', var]]
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(InFolder + dirs[i+1])
        Frame2 = TempFrame[['x', 'y', 'z', var]]
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        Snapshots = pd.concat([Snapshots, Frame2])

m, n = np.shape(Snapshots)

# %% compute
xval = xnew # xx
yval = ynew # yy
amplit = np.zeros(np.size(xval))
for i in range(np.size(xval)):
    xyz = [xval[i], yval[i], 0.0]
    amplit[i] = fv.Amplit(Snapshots, xyz, 'u')

# %% draw amplitude along streamwise
fig, ax = plt.subplots(figsize=(10, 3))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$A_{u^{\prime}}$", fontsize=textsize)
ax.plot(xval, amplit)
b, a = signal.butter(3, 0.15, btype='lowpass', analog=False)
amplit1 = signal.filtfilt(b, a, amplit)
ax.plot(xval, amplit1, 'r--')
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "AmplitX.svg", bbox_inches="tight", pad_inches=0.1
)


# %% draw RMS along streamwise
fig, ax = plt.subplots(figsize=(10, 3.0))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\sqrt{u^{\prime 2}}/u_{\infty}$", fontsize=textsize)
ax.plot(xval, np.sqrt(var1), 'k')
ax.set_xlim([-0.5, 20.5])
#ax.plot(pert['x'], np.sqrt(pert['<u`u`>']))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax.grid(b=True, which="both", linestyle=":")
# plt.show()
# plt.savefig(
#    path2 + "StreamwisePert.svg", bbox_inches="tight", pad_inches=0.1
# )
# % draw growth rate along streamwise
# grow = fv.GrowthRate(xx, amplit1)
# fig, ax2 = plt.subplots(figsize=(10, 3.0))
# matplotlib.rc('font', size=14)
ax2 = ax.twinx()
ax2.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax2.set_ylabel(r"$A/A_0$", fontsize=textsize)
ax2.set_yscale('log')
ax2.plot(xval[1:], amplit1[1:]/amplit1[0], 'k--')
ax2.set_xlim([-0.5, 20.5])
ax2.set_ylim([0.8, 30])
ax2.axvline(x=0.6, linewidth=1.0, linestyle='--', color='gray')
ax2.axvline(x=3.2, linewidth=1.0, linestyle='--', color='gray')
ax2.axvline(x=6.5, linewidth=1.0, linestyle='--', color='gray')
ax2.axvline(x=9.7, linewidth=1.0, linestyle='--', color='gray')
ax2.axvline(x=12., linewidth=1.0, linestyle='--', color='gray')
# ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax2.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "GrowRateX.svg", bbox_inches="tight", pad_inches=0.1
)


# %% Plot WSPD map along the streamwise
# load data
InFolder = path + 'Snapshots/'
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
grouped = DataFrame.groupby(['x', 'y', 'z'])
DataFrame = grouped.mean().reset_index()
var = 'p'    
fa = 1.7*1.7*1.4
timepoints = np.arange(800, 1149.5 + 0.5, 0.5)
Snapshots = DataFrame[['x', 'y', 'z', var]]
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(InFolder + dirs[i+1])
        grouped = TempFrame.groupby(['x', 'y', 'z'])
        Frame1 = grouped.mean().reset_index()
        Frame2 = Frame1[['x', 'y', 'z', var]]
        if np.shape(Frame1)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        Snapshots = pd.concat([Snapshots, Frame2])

Snapshots[var] = Snapshots[var] * fa
m, n = np.shape(Snapshots)
dt = 0.5
freq_samp = 2.0

# %% compute RMS
x1 = xnew
y1 = ynew
rms = np.zeros(np.size(x1))
for i in range(np.size(x1)):
    xyz = [x1[i], y1[i], 0.0]
    rms[i] = fv.RMS_map(Snapshots, xyz, var)
    
# %% Plot RMS along the streamwise (dividing line or max perturbation location)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
fig, ax = plt.subplots(figsize=(10, 4))
# ax.yaxis.major.formatter.set_powerlimits((-2, 3))
ax.plot(x1, rms, 'k-')
ax.set_xlim([0.0, 30.0])
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\mathrm{RMS}(p^{\prime}/p_{\infty})$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
ax.axvline(x=10.9, linewidth=1.0, linestyle='--', color='k')
plt.savefig(pathF + "RMSMap.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
    
# %% compute
x0 = xx
y0 = yy
#x0 = xnew
#y0 = ynew
# FPSD = np.zeros((np.ceil(np.size(timepoints)/freq_samp), np.size(x0)))
FPSD = np.zeros((int(np.size(timepoints)/freq_samp), np.size(x0)))
for i in range(np.size(x0)):
    xyz = [x0[i], y0[i], 0.0]
    freq, FPSD[:, i] = fv.FW_PSD_Map(Snapshots, xyz, var, 0.5, 2.0, opt=1)

# %% Plot WSPD map along the streamwise
SumFPSD = np.sum(FPSD, axis=0)
FPSD1 = np.log(FPSD/SumFPSD)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
fig, ax = plt.subplots(figsize=(8, 3.5))
# ax.yaxis.major.formatter.set_powerlimits((-2, 3))
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
print(np.max(FPSD1))
print(np.min(FPSD1))
lev = np.linspace(-10, -2, 41)
# lev = np.linspace(0.0, 0.1, 41)
cbar = ax.contourf(x0, freq, FPSD1, cmap='gray_r', levels=lev)
# ax.axvline(x=10.9, color="k", linestyle="--", linewidth=1.0)
ax.axvline(x=0.6, linewidth=1.0, linestyle='--', color='k')
ax.axvline(x=3.2, linewidth=1.0, linestyle='--', color='k')
ax.axvline(x=6.5, linewidth=1.0, linestyle='--', color='k')
ax.axvline(x=9.7, linewidth=1.0, linestyle='--', color='k')
ax.axvline(x=12., linewidth=1.0, linestyle='--', color='k')
ax.set_yscale('log')
ax.set_xlim([0.0, 30.0])
rg = np.linspace(-10, -2, 5)
# rg = np.linspace(0.0, 0.1, 5)
cbar = plt.colorbar(cbar, ticks=rg)
cbar.ax.xaxis.offsetText.set_fontsize(numsize)
cbar.ax.tick_params(labelsize=numsize)
cbar.update_ticks()
barlabel = r'$\log_{10} [f\cdot\mathcal{P}(f)/\int \mathcal{P}(f) \mathrm{d}f]$'
cbar.set_label(barlabel, rotation=90, fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "FWPSDMap00.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()


# %% Plot BL profile along streamwise
# load data
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
    "uu",
    "uv",
    "uw",
    "vv",
    "vw",
    "ww",
    "Q-criterion",
    "L2-criterion",
    "gradp",
]
MeanFlow = DataPost()
# MeanFlow.UserData(VarName, pathM + "MeanFlow.dat", 1, Sep="\t")
MeanFlow.UserDataBin(pathM + "MeanFlow.h5")
MeanFlow.AddWallDist(3.0)
# %% plot BL profile
fig, ax = plt.subplots(1, 7, figsize=(10, 3.5))
fig.subplots_adjust(hspace=0.5, wspace=0.15)
matplotlib.rc('font', size=14)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
xcoord = np.array([-40, 0, 5, 10, 15, 20, 30])
for i in range(np.size(xcoord)):
    y0, q0 = MeanFlow.BLProfile('x', xcoord[i], 'u')
    ax[i].plot(q0, y0, "k-")
    ax[i].set_ylim([0, 3])
    if i != 0:
        ax[i].set_yticklabels('')
    ax[i].set_xticks([0, 0.5, 1], minor=True) 
    ax[i].set_title(r'$x/\delta_0={}$'.format(xcoord[i]), fontsize=numsize-2)
    ax[i].grid(b=True, which="both", linestyle=":")
ax[0].set_ylabel(r"$\Delta y/\delta_0$", fontsize=textsize)
ax[3].set_xlabel(r'$u/u_\infty$', fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLProfile.svg", bbox_inches="tight", pad_inches=0.1
)

# %% plot BL fluctuations profile
TempFlow = DataPost()
InFolder = '/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/7/'
TempFlow.UserDataBin(InFolder+'SolTime793.00.h5', Uniq=True)
TempFlow.AddWallDist(3.0)

fig, ax = plt.subplots(1, 7, figsize=(10, 3.5))
fig.subplots_adjust(hspace=0.5, wspace=0.15)
matplotlib.rc('font', size=14)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
xcoord = np.array([-40, 0, 5, 10, 15, 20, 30])
loc = ['z', 'x']
for i in range(np.size(xcoord)):
    # y0, q1 = TempFlow.BLProfile('x', xcoord[i], 'u')
    # y0, q0 = MeanFlow.BLProfile('x', xcoord[i], 'u')
    # ax[i].plot(q1-q0, y0, "k-")
    pert = fv.PertAtLoc(fluc, '<u`u`>', loc, [0.0, xcoord[i]])
    if xcoord[i] > 0.0:
        ax[i].plot(np.sqrt(pert['<u`u`>']), pert['y']+3.0, 'k-')
    else:
        ax[i].plot(np.sqrt(pert['<u`u`>']), pert['y'], 'k-')
    ax[i].set_ylim([0, 4])
    if i != 0:
        ax[i].set_yticklabels('')
    # ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    ax[i].set_title(r'$x/\delta_0={}$'.format(xcoord[i]), 
                    fontsize=numsize-2, y=0.88)
    ax[i].grid(b=True, which="both", linestyle=":")
ax[0].set_ylabel(r"$\Delta y/\delta_0$", fontsize=textsize)
ax[3].set_xlabel(r'$u^{\prime}/u_\infty$', fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    path2 + "BLFlucProfile.pdf", bbox_inches="tight", pad_inches=0.1
)
