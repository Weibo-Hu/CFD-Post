#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 27 10:39:40 2023
    post-process data for cooling/heating cases

@author: weibo
"""

# %% Load necessary module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import plt2pandas as p2p
import matplotlib.ticker as ticker
import variable_analysis as va
from timer import timer
import warnings
from glob import glob
from scipy import signal
import sys

from line_field import LineField as lf
from planar_field import PlanarField as pf
from triaxial_field import TriField as tf


path0 = "/media/weibo/Weibo_data/2023cases/flat/"
path0F, path0P, path0M, path0S, path0T, path0I = p2p.create_folder(path0)
path1 = "/media/weibo/Weibo_data/2023cases/heating1/"
path1F, path1P, path1M, path1S, path1T, path1I = p2p.create_folder(path1)
path2 = "/media/weibo/Weibo_data/2023cases/heating2/"
path2F, path2P, path2M, path2S, path2T, path2I = p2p.create_folder(path2)
path3 = "/media/weibo/Weibo_data/2023cases/cooling1/"
path3F, path3P, path3M, path3S, path3T, path3I = p2p.create_folder(path3)
path4 = "/media/weibo/Weibo_data/2023cases/cooling2/"
path4F, path4P, path4M, path4S, path4T, path4I = p2p.create_folder(path4)
pathC = path0 + 'Comparison/'

plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman", 
    "weight": "normal",
    "size": "large",
}
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
tsize = 16
nsize = 11

# %% calculate
MeanFlow0 = pf()
MeanFlow0.load_meanflow(path0)
MeanFlow0.copy_meanval()
WallFlow0 = MeanFlow0.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow1 = pf()
MeanFlow1.load_meanflow(path1)
MeanFlow1.copy_meanval()
WallFlow1 = MeanFlow1.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow2 = pf()
MeanFlow2.load_meanflow(path2)
MeanFlow2.copy_meanval()
WallFlow2 = MeanFlow2.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow3 = pf()
MeanFlow3.load_meanflow(path3)
MeanFlow3.copy_meanval()
WallFlow3 = MeanFlow3.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow4 = pf()
MeanFlow4.load_meanflow(path4)
MeanFlow4.copy_meanval()
WallFlow4 = MeanFlow4.PlanarData.groupby("x", as_index=False).nth(1)

# if there is duplicate points
if np.size(np.unique(WallFlow0["y"])) > 2:  
    maxy = np.max(WallFlow0["y"])
    WallFlow0 = WallFlow0.drop(WallFlow0[WallFlow0["y"] == maxy].index)
    WallFlow1 = WallFlow1.drop(WallFlow1[WallFlow1["y"] == maxy].index)
    WallFlow2 = WallFlow2.drop(WallFlow2[WallFlow2["y"] == maxy].index)
    WallFlow3 = WallFlow3.drop(WallFlow3[WallFlow3["y"] == maxy].index)
    WallFlow4 = WallFlow4.drop(WallFlow4[WallFlow4["y"] == maxy].index)

xwall = WallFlow0["x"].values
mu0 = va.viscosity(10000, WallFlow0["T"], T_inf=45, law="SL", )
Cf0 = va.skinfriction(mu0, WallFlow0["u"], WallFlow0["walldist"])

mu1 = va.viscosity(10000, WallFlow1["T"], T_inf=45, law="SL", )
Cf1 = va.skinfriction(mu1, WallFlow1["u"], WallFlow1["walldist"])

mu2 = va.viscosity(10000, WallFlow2["T"], T_inf=45, law="SL", )
Cf2 = va.skinfriction(mu2, WallFlow2["u"], WallFlow2["walldist"])

mu3 = va.viscosity(10000, WallFlow3["T"], T_inf=45, law="SL", )
Cf3 = va.skinfriction(mu3, WallFlow3["u"], WallFlow3["walldist"])

mu4 = va.viscosity(10000, WallFlow4["T"], T_inf=45, law="SL", )
Cf4 = va.skinfriction(mu4, WallFlow4["u"], WallFlow4["walldist"])

# %% Cf
b, a = signal.butter(6, Wn=1/10, fs=1/0.125)
fig2, ax2 = plt.subplots(figsize=(8, 2.5), dpi=500)
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)

# ax2.scatter(xwall[0::8], Cf0[0::8], s=10, marker='o',
#             facecolors='y', edgecolors='C7', linewidths=0.8)
ax2.scatter(xwall[0::8], signal.filtfilt(b, a, Cf0)[0::8], s=10, 
            marker='o', facecolors='w', edgecolors='C7', linewidths=0.8)
# ax2.plot(xwall, Cf1, "k:", linewidth=1.5)
# ax2.plot(xwall, Cf2, "k-", linewidth=1.5)
# ax2.plot(xwall, Cf3, "g:", linewidth=1.5)
# ax2.plot(xwall, Cf4, "g-", linewidth=1.5)
ax2.plot(xwall, signal.filtfilt(b, a, Cf1), "r:", linewidth=1.5)
ax2.plot(xwall, signal.filtfilt(b, a, Cf2), "r-", linewidth=1.5)
ax2.plot(xwall, signal.filtfilt(b, a, Cf3), "b:", linewidth=1.5)
ax2.plot(xwall, signal.filtfilt(b, a, Cf4), "b-", linewidth=1.5)
ax2.legend(labels=['FP', 'H1', 'H2', 'C1', 'C2'], ncols=2)
ax2.set_xlabel(r"$x$", fontsize=tsize)
ax2.set_xlim([0, 350])
ax2.set_xticks(np.arange(0, 351, 50))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=75, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cf.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Cp
fig1, ax1 = plt.subplots(figsize=(8, 2.5), dpi=500)
Ma = 6.0
fa = 1.0  # Ma * Ma *1.4
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
xwall = WallFlow0["x"].values
# ax1.scatter(xwall[0::8], WallFlow0['p'][0::8], s=10, marker='o',
#             facecolors='y', edgecolors='C7', linewidths=0.8)
ax1.scatter(xwall[0::8], signal.filtfilt(b, a, WallFlow0.p)[0::8], s=10,
            marker='o', facecolors='w', edgecolors='C7', linewidths=0.8)
# ax1.plot(xwall, WallFlow1['p'], "k:", linewidth=1.5)
# ax1.plot(xwall, WallFlow2['p'], "k-", linewidth=1.5)
# ax1.plot(xwall, WallFlow3['p'], "g:", linewidth=1.5)
# ax1.plot(xwall, WallFlow4['p'], "g-", linewidth=1.5)
ax1.plot(xwall, signal.filtfilt(b, a, WallFlow1.p), "r:", linewidth=1.5)
ax1.plot(xwall, signal.filtfilt(b, a, WallFlow2.p), "r-", linewidth=1.5)
ax1.plot(xwall, signal.filtfilt(b, a, WallFlow3.p), "b:", linewidth=1.5)
ax1.plot(xwall, signal.filtfilt(b, a, WallFlow4.p), "b-", linewidth=1.5)
ax1.legend(labels=['FP', 'H1', 'H2', 'C1', 'C2'], ncols=2)
ax1.set_xlabel(r"$x$", fontsize=tsize)
ax1.set_ylabel(r"$\langle C_p \rangle$", fontsize=tsize)
ax1.set_xlim([0, 350])
ax1.set_xticks(np.arange(0, 351, 50))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.axvline(x=75, color="gray", linestyle=":", linewidth=1.5)
ax1.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax1.grid(visible=True, which="both", linestyle=":")
ax1.yaxis.offsetText.set_fontsize(nsize)
ax1.annotate("(b)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cp.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% fluctuations of probe
"""
#1 coordinates where the fluctuations is max
"""
stat = MeanFlow1.PlanarData
var = 'RMS_p'
if var == 'RMS_u':
    varnm = '<u`u`>'
    savenm = "MaxRMS_u"
elif var =='RMS_p':
    varnm = '<p`p`>'
    savenm = "MaxRMS_p"
xloc = np.arange(0, 360.0+0.25, 0.25)
zloc = np.zeros(np.size(xloc))
varv = np.zeros(np.size(xloc))
yloc = np.zeros(np.size(xloc))
for i in range(np.size(xloc)):
    df = va.max_pert_along_y(stat, varnm, [xloc[i], zloc[i]])
    varv[i] = df[varnm]
    yloc[i] = df['y']
data = np.vstack((xloc, yloc, varv))
df = pd.DataFrame(data.T, columns=['x', 'y', varnm])
# df = df.drop_duplicates(keep='last')
df.to_csv(path1M + savenm + '.dat', sep=' ',
          float_format='%1.8e', index=False)

fig, ax = plt.subplots(figsize=(8, 2.5))
matplotlib.rc('font', size=14)
ax.set_ylabel(r"$y$", fontsize=tsize)
ax.set_xlabel(r"$x$", fontsize=tsize)
ax.plot(df['x'], df['y'], 'b')
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    path1F + "loc_" + savenm+ ".svg", bbox_inches="tight", pad_inches=0.1
)
# %%
"""
#2 the growth of the fluctuations
"""
df = pd.read_csv(path1M + savenm + '.dat', sep=' ')
fig, ax = plt.subplots(figsize=(8, 2.5))
matplotlib.rc('font', size=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r'$q^\prime_\mathrm{max}$', fontsize=tsize)
ax.plot(df['x'], np.sqrt(df[varnm]), 'k') # , label=r'$q^\prime_\mathrm{max}$')
# ax.plot(xx, yy, 'k--', label='bubble')
legend = ax.legend(loc='upper right', shadow=False, fontsize=nsize)
ax.set_xlim([0, 350])
ax.set_xticks(np.arange(0, 351, 50))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    path1F + var + ".svg", bbox_inches="tight", pad_inches=0.1
)