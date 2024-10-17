# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:25:17 2024
    post-process fluctuation data for ramp cases

@author: Weibo
"""
# %% load necessary modules
import os
from natsort import natsorted
import imageio as iio
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import variable_analysis as va
from planar_field import PlanarField as pf
import pytecio as pytec
import plt2pandas as p2p
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import splprep, splev, interp1d
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from line_field import LineField as lf
from IPython import get_ipython
from scipy import fft
# get_ipython().run_line_magic("matplotlib", "qt")

# %% set path and basic parameters
path = "F:/AAS/ramp_st14_2nd/"
# path = 'E:/cases/wavy_1009/'
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + "Vortex/"
pathSL = path + "Slice/"
pathSN = path + "snapshots/"
tsize = 13
nsize = 10
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
matplotlib.rc("font", size=tsize)
plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
cm2in = 1 / 2.54
# %% flow with fluctuations
fluc = pf()
fluc.load_data(pathI, FileList='TP_fluc_2d.h5', ExtName='h5')

# %% mean flow contour in x-y plane
MeanFlow = pf()
MeanFlow.load_meanflow(path)
MeanFlow.copy_meanval()
ind0 = (MeanFlow.PlanarData.y == 0.0)
MeanFlow.PlanarData['walldist'][ind0] = 0.0
# %% BL profile along streamwise
fig, ax = plt.subplots(1, 9, figsize=(13 * cm2in, 4 * cm2in), dpi=500)
fig.subplots_adjust(hspace=0.5, wspace=0.25)
matplotlib.rc("font", size=nsize)
title = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$"]
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
xcoord = np.array([-120, -80, -40, -20, 0, 20, 40, 60, 80])
for i in range(np.size(xcoord)):
    df = fluc.yprofile("x", xcoord[i])
    y0 = df["walldist"]
    q0 = df["p`"]
    if xcoord[i] == 0.0:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind]/np.max(q0[ind]), y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    else:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind]/np.max(q0[ind]), y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    if i != 0:
        ax[i].set_yticklabels("")
        ax[i].set_title(r"${}$".format(xcoord[i]), fontsize=nsize - 2)
    ax[i].set_xticks([0, 1], minor=True)
    ax[i].tick_params(axis="both", which="major", labelsize=nsize)
    ax[i].grid(visible=True, which="both", linestyle=":")
ax[0].set_title(r"$x={}$".format(xcoord[0]), fontsize=nsize - 2)
ax[0].set_ylabel(r"$\Delta y$", fontsize=tsize)
ax[4].set_xlabel(r"$u /u_\infty$", fontsize=tsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "BLProfileP_fluc.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Draw impose mode
sloc = 'N100'
inmode = pd.read_csv(path+"UnstableMode" + sloc + ".inp", skiprows=5,
                     sep=' ', index_col=False)
fig, ax = plt.subplots(figsize=(7*cm2in, 6.5*cm2in))
matplotlib.rc('font', size=nsize)
xlab = r"$|q^{\prime}|/|u^\prime|_{\max}$"
ax.set_xlabel(xlab, fontsize=tsize)
ax.set_ylabel(r"$y/l_f$", fontsize=tsize)
ax.plot(np.sqrt(inmode['u_r']**2+inmode['u_i']**2), inmode['y'], 'k')
ax.plot(np.sqrt(inmode['v_r']**2+inmode['v_i']**2), inmode['y'], 'r')
ax.plot(np.sqrt(inmode['w_r']**2+inmode['w_i']**2), inmode['y'], 'g')
ax.plot(np.sqrt(inmode['p_r']**2+inmode['p_i']**2), inmode['y'], 'b')
ax.plot(np.sqrt(inmode['t_r']**2+inmode['t_i']**2), inmode['y'], 'c')
# ax.set_xlim([-100, 20])
ax.set_ylim([0.0, 8.0])
# ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.legend(['u', 'v', 'w', 'p', 'T'])
ax.grid(visible=True, which="both", linestyle=":")
ax.tick_params(labelsize=nsize)

plt.show()
plt.savefig(
    pathF + "ModeProf" + sloc + ".svg", bbox_inches="tight", pad_inches=0.1
)

# %% compare profiles
fig, ax = plt.subplots(figsize=(7.5*cm2in, 7*cm2in))
matplotlib.rc("font", size=nsize)
title = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$"]
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
xloc = -100
df = MeanFlow.yprofile("x", xloc)
y0 = df["walldist"]
ua = np.sqrt(df["<u`u`>"])
pa = np.sqrt(df["<p`p`>"])
Ta = np.sqrt(df["<T`T`>"])
vref = np.max(ua)
xlab = r"$|q^{\prime}|/|u^\prime|_{\max}$"
ax.plot(np.sqrt(inmode['u_r']**2+inmode['u_i']**2), inmode['y'], 'k')
ax.plot(np.sqrt(inmode['p_r']**2+inmode['p_i']**2), inmode['y'], 'b')
ax.plot(np.sqrt(inmode['t_r']**2+inmode['t_i']**2), inmode['y'], 'r')

ax.scatter(ua/vref, y0, s=8, marker='o', facecolors='none', edgecolors='k')
ax.scatter(pa/vref, y0, s=8, marker='o', facecolors='none', edgecolors='b')
ax.scatter(Ta/vref, y0, s=8, marker='o', facecolors='none', edgecolors='r')

ax.legend([r'$u$', r'$p$', r'$T$'])
ax.set_xlabel(xlab, fontsize=tsize)
ax.set_ylabel(r"$y/l_f$", fontsize=tsize)
ax.set_ylim([0.0, 6.0])
ax.grid(visible=True, which="both", linestyle=":")
ax.tick_params(labelsize=nsize)
plt.show()
plt.savefig(
    pathF + "ModeComp" + sloc + ".svg", bbox_inches="tight", pad_inches=0.1
)