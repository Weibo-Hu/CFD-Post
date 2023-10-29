#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 27 10:39:40 2023
    post-process data for cooling/heating cases

@author: weibo
"""

# %% load necessary modules
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import variable_analysis as va
from planar_field import PlanarField as pf
import plt2pandas as p2p
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

# %% set path and basic parameters
path = "/media/weibo/Weibo_data/2023cases/heating2/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + "Vortex/"
pathSL = path + "Slice/"
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
plt.close("All")
plt.rc("text", usetex=True)
tsize = 16
nsize = 10
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}

# df = pd.read_hdf(path + 'MeanFlow/MeanFlow_02000.00.h5')
# df['<u`u`>'] = MeanFlow.PlanarData['<u`u`>']
# df.to_hdf(path + "MeanFlow/MeanFlow_02001.00.h5", "w", format="fixed")

# %% mean flow contour in x-y plane
MeanFlow = pf()
MeanFlow.load_meanflow(path)
MeanFlow.copy_meanval()
va.wall_line(MeanFlow.PlanarData, pathM)
points = np.array([[0.0], [2.0]])
va.streamline(pathM, MeanFlow.PlanarData, points, opt="up")
# va.dividing_line(MeanFlow.PlanarData, pathM, show=True)
va.sonic_line(MeanFlow.PlanarData, pathM, option="velocity", Ma_inf=6.0)
va.boundary_edge(MeanFlow.PlanarData, pathM, shock=False, mask=False)

# %% Plot rho contour of the mean flow field
lh = 1.0
MeanFlow.rescale(lh)
MeanFlow.copy_meanval()
df = MeanFlow.PlanarData
var = "T"
x, y = np.meshgrid(np.unique(df.x), np.unique(df.y))
varval = griddata((df.x, df.y), df[var], (x, y))
walldist = griddata((df.x, MeanFlow.y), df.walldist, (x, y))
print("rho_max=", np.max(varval))
print("rho_min=", np.min(varval))
cval1 = 1.0  # 1.0 # 0.1
cval2 = 6.7  # 7.0  # 1.0
fig, ax = plt.subplots(figsize=(8, 2.5))  # 7.3  2.3
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(cval1, cval2, 21)
cbar = ax.contourf(x, y, varval, cmap="rainbow", levels=rg1, extend="both")
# ax.contour(x, y, u, levels=[0.0], linestyles="dotted")  # rainbow_r
ax.set_xlim(20, 360)
ax.set_ylim(np.min(y), 10.0)
# ax.set_xticks(np.linspace(20, 360, 8))
ax.set_yticks(np.linspace(np.min(y), 10.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x$", fontsize=tsize)
ax.set_ylabel(r"$y$", fontsize=tsize)
# Add colorbar
cbbox = fig.add_axes([0.14, 0.54, 0.22, 0.32])
[cbbox.spines[k].set_visible(False) for k in cbbox.spines]
cbbox.tick_params(
    axis="both",
    left=False,
    labelleft=False,
    bottom=False,
    labelbottom=False,
)
cbbox.set_facecolor([1, 1, 1, 0.8])
rg2 = np.linspace(cval1, cval2, 3)
cbaxes = fig.add_axes([0.17, 0.76, 0.16, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(
    cbar, cax=cbaxes, extendrect="False", orientation="horizontal", ticks=rg2
)
if var == "rho":
    lab = r"$\langle \rho \rangle/\rho_{\infty}$"
if var == "u":
    lab = r"$\langle u \rangle/u_{\infty}$"
if var == "T":
    lab = r"$\langle T \rangle/T_{\infty}$"
cbar.set_label(lab, rotation=0, fontsize=tsize - 1)
# Add boundary layer
boundary = pd.read_csv(pathM + "BoundaryEdge.dat", skipinitialspace=True)
ax.plot(boundary.x / lh, boundary.y / lh, "k", linewidth=1.5)
# Add sonic line
sonic = pd.read_csv(pathM + "SonicLine.dat", skipinitialspace=True)
ax.plot(sonic.x / lh, sonic.y / lh, "w--", linewidth=1.5)
# plt.gca().set_aspect("equal", adjustable="box")
plt.savefig(pathF + "MeanFlow_" + var + ".svg", bbox_inches="tight")
plt.show()

# %% root mean square
df = MeanFlow.PlanarData
var = 'rms_w'
if var == 'rms_u':
    varnm = '<u`u`>'
    lab = r"$\sqrt{\langle u^\prime u^\prime \rangle}$"
if var == 'rms_v':
    varnm = '<v`v`>'
    lab = r"$\sqrt{\langle v^\prime v^\prime \rangle}$"
if var == 'rms_w':
    varnm = '<w`w`>'
    lab = r"$\sqrt{\langle w^\prime w^\prime \rangle}$"
if var == 'rms_p':
    varnm = '<p`p`>'
    lab = r"$\sqrt{\langle p^\prime p^\prime \rangle}$"
var_val = getattr(df, varnm)
uu = griddata((df.x, MeanFlow.y), var_val, (x, y))
print("uu_max=", np.max(np.sqrt(np.abs(var_val))))
print("uu_min=", np.min(np.sqrt(np.abs(var_val))))
fig, ax = plt.subplots(figsize=(8, 2.5))
matplotlib.rc("font", size=tsize)
cb1 = 0.0
cb2 = 0.1
rg1 = np.linspace(cb1, cb2, 21)  # 21)
cbar = ax.contourf(
    x / lh, y / lh, np.sqrt(np.abs(uu)), cmap="Spectral_r", levels=rg1, extend="both"
)  # rainbow_r # jet # Spectral_r
ax.set_xlim(20, 360.0)
ax.set_ylim(np.min(y), 10.0)
ax.set_yticks(np.linspace(0.0, 10.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x$", fontdict=font)
ax.set_ylabel(r"$y$", fontdict=font)
# Add colorbar box
rg2 = np.linspace(cb1, cb2, 3)
cbbox = fig.add_axes([0.14, 0.53, 0.22, 0.30])
[cbbox.spines[k].set_visible(False) for k in cbbox.spines]
cbbox.tick_params(
    axis="both",
    left=False,
    bottom=False,
    labelleft=False,
    labelbottom=False,
)
cbbox.set_facecolor([1, 1, 1, 0.7])
# Add colorbar
cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.056])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(
    cbar, cax=cbaxes, orientation="horizontal", extendrect="False", ticks=rg2
)
cbar.set_label(
    lab,
    rotation=0,
    fontsize=tsize,
)
# Add boundary layer
boundary = pd.read_csv(pathM + "BoundaryEdge.dat", skipinitialspace=True)
ax.plot(boundary.x / lh, boundary.y / lh, "k", linewidth=1.0)
# Add sonic line
sonic = pd.read_csv(pathM + "SonicLine.dat", skipinitialspace=True)
ax.plot(sonic.x / lh, sonic.y / lh, "w--", linewidth=1.5)
# plt.gca().set_aspect("equal", adjustable="box")
plt.savefig(pathF + "MeanFlow_" + var + ".svg", bbox_inches="tight")
plt.show()


# %% BL profile along streamwise
fig, ax = plt.subplots(1, 9, figsize=(8, 2.5), dpi=500)
fig.subplots_adjust(hspace=0.5, wspace=0.18)
matplotlib.rc("font", size=nsize)
title = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$"]
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
xcoord = np.array([20, 34, 53, 71.5, 82, 95, 200, 280, 340])
for i in range(np.size(xcoord)):
    df = MeanFlow.yprofile("x", xcoord[i])
    y0 = df["walldist"]
    q0 = df["u"]
    if xcoord[i] == 0.0:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind], y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    else:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind], y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    if i != 0:
        ax[i].set_yticklabels("")
        ax[i].set_title(r"${}$".format(xcoord[i]), fontsize=nsize - 2)
    ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].tick_params(axis="both", which="major", labelsize=nsize)
    ax[i].grid(visible=True, which="both", linestyle=":")
ax[0].set_title(r"$x={}$".format(xcoord[0]), fontsize=nsize - 2)
ax[0].set_ylabel(r"$y$", fontsize=tsize)
ax[4].set_xlabel(r"$u /u_\infty$", fontsize=tsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "BLProfile.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Cf and Cp curves
WallFlow = MeanFlow.PlanarData.groupby("x", as_index=False).nth(1)
# if there is duplicate points
if np.size(np.unique(WallFlow["y"])) > 2:  
    maxy = np.max(WallFlow["y"])
    WallFlow = WallFlow.drop(WallFlow[WallFlow["y"] == maxy].index)
mu = va.viscosity(10000, WallFlow["T"], T_inf=45, law="Suther")
Cf = va.skinfriction(mu, WallFlow["u"].values, WallFlow["walldist"].values)
# low-pass filter 
b, a = signal.butter(6, Wn=1/12, fs=1/0.125)
Cf_fit = signal.filtfilt(b, a, Cf)
# ind = np.where(Cf[:] < 0.008)
fig2, ax2 = plt.subplots(figsize=(8, 2.5), dpi=500)
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
xwall = WallFlow["x"].values
ax2.plot(xwall, Cf, "b-", linewidth=1.5)
ax2.plot(xwall, Cf_fit, "k", linewidth=1.5)
ax2.set_xlabel(r"$x$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=tsize)
ax2.set_xlim([0, 360])
ax2.set_xticks(np.arange(0, 361, 40))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=75, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cf.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Cp
Ma = 6.0
fa = 1.0 # Ma * Ma * 1.4
p_fit = signal.filtfilt(b, a, WallFlow["p"] * fa)
fig3, ax3 = plt.subplots(figsize=(8, 2.5), dpi=500)
# ax3 = fig.add_subplot(212)
ax3.plot(WallFlow["x"], WallFlow["p"] * fa, "b-", linewidth=1.5)
ax3.plot(WallFlow["x"], p_fit, "k", linewidth=1.5)
ax3.set_xlabel(r"$x$", fontsize=tsize)
# ylab = r"$\langle p_w \rangle/\rho_{\infty} u_{\infty}^2$"
ax3.set_ylabel(r"$\langle C_p \rangle$", fontsize=tsize)
ax3.set_xlim([0, 360])
ax3.set_xticks(np.arange(0, 361, 40))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=75, color="gray", linestyle=":", linewidth=1.5)
ax3.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax3.grid(visible=True, which="both", linestyle=":")
ax3.annotate("(b)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cp.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()


# %% Stanton number 
Tt = va.stat2tot(Ma=6.0, Ts=45, opt='t') / 45
Tw = 6.667
mu = va.viscosity(10000, WallFlow["T"], T_inf=45, law="Suther")
kt = va.thermal(mu, 0.72)
Cs = va.Stanton(kt, WallFlow["T"].values, WallFlow["walldist"].values, Tt, Tw)
# low-pass filter 
b, a = signal.butter(6, Wn=1/12, fs=1/0.125)
Cs_fit = signal.filtfilt(b, a, Cs)
# ind = np.where(Cf[:] < 0.008)
fig2, ax2 = plt.subplots(figsize=(8, 2.5), dpi=500)
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
xwall = WallFlow["x"].values
ax2.plot(xwall, Cs, "b-", linewidth=1.5)
ax2.plot(xwall, Cs_fit, "k", linewidth=1.5)
ax2.set_xlabel(r"$x$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_s \rangle$", fontsize=tsize)
ax2.set_xlim([0, 360])
ax2.set_xticks(np.arange(0, 361, 40))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=75, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cs.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% animation for vortex structure
pathF = path + "Figures/"
pathall = glob(path + "TP_data_*/")
tp.session.connect(port=7600)
for i in range(np.size(pathall)):
    dirs = glob(pathall[i] + "*.szplt")
    # tp.session.stop()
    print("process " + pathall[i])
    dataset = tp.data.load_tecplot_szl(dirs, read_data_option=2)
    soltime = np.round(dataset.solution_times[0], 2)
    # with tp.session.suspend():
    # tp.session.suspend_enter()
    frame = tp.active_frame()
    frame.width = 12.8
    frame.height = 7.8
    frame.position = (-1.0, 0.5)
    # tp.macro.execute_file('/path/to/macro_file.mcr')
    frame.load_stylesheet(path + "vortex_ani.sty")
    tp.macro.execute_command("$!Interface ZoneBoundingBoxMode = Off")
    tp.macro.execute_command("$!FrameLayout ShowBorder = No")
    tp.export.save_png(path + str(soltime) + ".png", width=2048)
# %% create videos
import imageio as iio
from natsort import natsorted

pathani = path + "ani/"
prenm = "heating2"
dirs = glob(pathani + "*.png")
dirs = natsorted(dirs, key=lambda y: y.lower())
flnm = pathani + prenm + "_Anima.mp4"
recyc = np.size(dirs)
with iio.get_writer(flnm, mode="I", fps=6, macro_block_size=None) as writer:
    for ii in range(recyc * 6):
        ind = ii % recyc  # mod, get reminder
        image = iio.v3.imread(dirs[ind])
        writer.append_data(image)
    writer.close()
