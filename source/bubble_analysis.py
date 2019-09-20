#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:57:06 2019
    Analysis instability within the separation bubble
@author: weibo
"""

# %% Load necessary module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, splev, splrep
from data_post import DataPost
import variable_analysis as fv
from timer import timer
import os
from planar_field import PlanarField as pf
from line_field import LineField as lf
from scipy.interpolate import griddata

# %% data path settings
path = "/media/weibo/VID2/BFS_M1.7TS/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"

# %% figures properties settings
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

# %%############################################################################
"""
    Examination of the computational mesh
"""
# %% PSD analysis
xloc = -10.0
yloc = 0.0
zloc = 0.0
fa = 1.7 * 1.7 * 1.4
var = 'u'
timezone = [600, 900]
#filenm = pathP + 'timeline_' + str(xloc) + '.dat'
#upstream = pd.read_csv(filenm, sep=' ', skiprows=0,
#                       index_col=False, skipinitialspace=True)
probe = lf()
probe.load_probe(pathP, (-0.01032, 0.01358, zloc)) # (4.64466, -1.26135e+00, zloc)) 
probe.extract_series(timezone)
downstream = probe.ProbeSignal
probe.load_probe(pathP, (-10.0, 0.0, zloc))
probe.extract_series(timezone)
upstream = probe.ProbeSignal

# %%
varnm = 'u'
if varnm == 'u':
    ylab = r"$u^\prime / u_\infty$"
    ylab_sub = r"$_{u^\prime}$"
else:
    ylab = r"$p^\prime/(\rho_\infty u_\infty ^2)$"
    ylab_sub = r"$_{p^\prime}$"
fig = plt.figure(figsize=(6.0, 2.8))
matplotlib.rc("font", size=textsize)
matplotlib.rc("font", size=textsize)
ax1 = fig.add_subplot(211)
ax1.plot(upstream['time'], upstream[var]-np.mean(upstream[var]), 
         'k-', linewidth=1.5)
ax1.set_ylabel(ylab, fontsize=textsize)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.grid(b=True, which="both", linestyle=":")
ax1.yaxis.offsetText.set_fontsize(numsize)
ax1.tick_params(labelsize=numsize)

ax2 = fig.add_subplot(212)
ax2.plot(downstream['time'], downstream[var]-np.mean(downstream[var]),
         'k--', linewidth=1.5)
ax2.set_ylabel(ylab, fontsize=textsize)
ax2.set_xlabel(r"$t u_\infty / \delta_0$", fontsize=textsize)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
ax2.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + varnm + "_time2.svg", dpi=300)
plt.show()

# %% 
dt = 0.014857
Freq_samp = 4.0
fig = plt.figure(figsize=(6.4, 3.2))
matplotlib.rc("font", size=textsize)
ax1 = fig.add_subplot(121)
Fre, FPSD = fv.psd(upstream[var], dt, Freq_samp, opt=1, seg=8, overlap=2)
# Fre, FPSD = fv.psd(upstream[var], dt, Freq_samp, opt=2)
ax1.semilogx(Fre, FPSD, "k", linewidth=1.0)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax1.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax1.grid(b=True, which="both", linestyle=":")
ax1.yaxis.offsetText.set_fontsize(numsize)
ax1.tick_params(labelsize=numsize)

ax2 = fig.add_subplot(122)
Fre2, FPSD2 = fv.psd(downstream[var], dt, Freq_samp, opt=1, seg=8, overlap=4)
ax2.semilogx(Fre2, FPSD2, "k", linewidth=1.0)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax2.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "TSFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %%############################################################################
"""
    Examination of the computational mesh
"""
# %% Maximum shear line
filenm = pathM + 'ShearLine.dat'
xcoord = np.unique(TimeAve['x'])
grouped = TimeAve.groupby(['x'])
shear = '||shear(<velocity>)||'
idx = grouped[shear].transform(max) == TimeAve[shear]
shear_max = TimeAve[idx]
shear_max.to_csv(filenm, sep=' ', index=False, float_format='%1.8e')

# %% Plot rms contour of the mean flow field
MeanFlow = DataPost()
# MeanFlow.UserData(VarName, pathM + "MeanFlow.dat", 1, Sep="\t")
TimeAve = pd.read_hdf(pathM + "MeanFlow.h5")
grouped = TimeAve.groupby(['x', 'y'])
MeanFlow = grouped.mean().reset_index()
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
corner = (x < 0.0) & (y < 0.0)
var = "<u`u`>"
uu = griddata((MeanFlow.x, MeanFlow.y), getattr(MeanFlow, var), (x, y))
print("uu_max=", np.max(np.sqrt(np.abs(getattr(MeanFlow, var)))))
print("uu_min=", np.min(np.sqrt(np.abs(getattr(MeanFlow, var)))))
corner = (x < 0.0) & (y < 0.0)
uu[corner] = np.nan
fig, ax = plt.subplots(figsize=(10, 4))
matplotlib.rc("font", size=textsize)
rg1 = np.linspace(0.0, 0.22, 21)
cbar = ax.contourf(
    x, y, np.sqrt(np.abs(uu)), cmap="Spectral_r", levels=rg1
)  # rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/\delta_0$", fontdict=font)
ax.set_ylabel(r"$y/\delta_0$", fontdict=font)
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar box
rg2 = np.linspace(0.0, 0.22, 3)
cbbox = fig.add_axes([0.15, 0.57, 0.22, 0.23], alpha=0.9)
[cbbox.spines[k].set_visible(False) for k in cbbox.spines]
cbbox.tick_params(
    axis="both",
    left=False,
    right=False,
    top=False,
    bottom=False,
    labelleft=False,
    labeltop=False,
    labelright=False,
    labelbottom=False,
)
cbbox.set_facecolor([1, 1, 1, 0.7])
# Add colorbar
cbaxes = fig.add_axes(
    [0.17, 0.70, 0.18, 0.07], frameon=True
)  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar.set_label(
    r"$\sqrt{|\langle v^\prime v^\prime \rangle|}$",
    rotation=0,
    fontsize=textsize,
)
# Add shock wave
shock = np.loadtxt(pathM + "ShockLineFit.dat", skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], "w", linewidth=1.5)
# Add sonic line
sonic = np.loadtxt(pathM + "SonicLine.dat", skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], "w--", linewidth=1.5)
# Add boundary layer
boundary = np.loadtxt(pathM + "BoundaryEdge.dat", skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], "k", linewidth=1.5)
# Add dividing line(separation line)
dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], "k--", linewidth=1.5)
# Add maximum shear line
# shear = np.loadtxt(pathM + "ShearLine.dat", skiprows=1)
ax.plot(shear_max['x'], shear_max['y'], "w:", linewidth=1.5)

plt.savefig(pathF + "MeanFlowRMSUU.svg", bbox_inches="tight")
plt.show()