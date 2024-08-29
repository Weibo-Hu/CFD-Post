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
from line_field import LineField as lf
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "qt")

# %% set path and basic parameters
path = "/media/weibo/VID21/wavy_018_small/"
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
# matplotlib.use("QtAgg")
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
# %%
wall = va.wall_line(MeanFlow.PlanarData, pathM)
wall = pd.read_csv(pathM + "WallBoundary.dat", skipinitialspace=True)
ind1 = wall.index[wall["y"] > 0.0]
ind2 = wall.index[wall['x'] < 20]
ind = np.union1d(ind1.values, ind2.values)
wall.loc[ind, 'y'] = 0.0
wall.to_csv(pathM + "WallBoundary.dat", index=False, float_format="%9.8e")
points = np.array([[0.0], [2.0]])
va.sonic_line(MeanFlow.PlanarData, pathM, option="velocity", Ma_inf=6.0)
va.boundary_edge(MeanFlow.PlanarData, pathM, shock=False, mask=False)
# va.streamline(pathM, MeanFlow.PlanarData, points, opt="up")
# va.dividing_line(MeanFlow.PlanarData, pathM, show=True)
# %% smooth curves
curve = pd.read_csv(pathM + "BoundaryEdge.dat", skipinitialspace=True)
b, a = signal.butter(6, Wn=1 / 10, fs=1 / 0.25)
fig2, ax2 = plt.subplots(figsize=(8, 2.5), dpi=300)
matplotlib.rc("font", size=nsize)
ax2.scatter(
    curve.x[0::4],
    curve.y[0::4],
    s=10,
    marker="o",
    facecolors="w",
    edgecolors="C7",
    linewidths=0.8,
)
curve_x = curve.x
curve_y = signal.filtfilt(b, a, curve.y)
header = "x, y"
xycor = np.transpose(np.vstack((curve_x, curve_y)))
np.savetxt(
    pathM + "BoundaryEdgeFit.dat",
    xycor,
    fmt="%9.6f",
    delimiter=", ",
    comments="",
    header=header,
)
ax2.plot(curve_x, curve_y, "r:", linewidth=1.5)
plt.show()
plt.savefig(pathF + "test.svg")
# %% save wavy wall data
x1 = np.linspace(-1, 26, 660, endpoint=True)
x2 = np.linspace(26, 108.811328, 2000, endpoint=True)
x3 = np.linspace(108.811328, 401, 6000, endpoint=True)
alpha = 0.758735
lambd = 2 * np.pi / alpha
amplit = 0.1 * 2.6  # 0.3, 0.2, 0.1
y1 = np.zeros(np.size(x1))
y2 = amplit * np.sin(alpha * (x2 - 26) + np.pi / 2) - amplit
y3 = np.zeros(np.size(x3))
y3 = np.zeros(np.size(x3))
xx = np.concatenate((x1[:-1], x2, x3[1:]))
yy = np.concatenate((y1[:-1], y2, y3[1:]))
arr = np.vstack((xx, yy)).T

# %%  ramp 
x1 = np.linspace(-200, 0, 801, endpoint=True)
x2 = np.linspace(0, 90, 361, endpoint=True)
angle = 15
y1 = np.zeros(np.size(x1))
y2 = x2 * np.tan(15/180*np.pi)
xx = np.concatenate((x1[:-1], x2))
yy = np.concatenate((y1[:-1], y2))
arr = np.vstack((xx, yy)).T
np.savetxt(
    pathM + "wavy.dat",
    arr,
    header="x, y",
    comments="",
    delimiter=",",
    fmt="%9.6f",
)

# %% plot wavy wall
xy = pd.read_csv(pathM + "WallBoundary.dat", skipinitialspace=True)
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(xy["x"], xy["y"], "b-", label="DNS")
ax.plot(xx, yy, "r:", label="Real")
ax.set_xlabel(r"$x$", size=tsize)
ax.set_ylabel(r"$y$", size=tsize)
ax.set_xlim(10, 110)
ax.grid(which="both")
plt.legend()
plt.savefig(pathF + "wavy_wall.png")
plt.show()

# %% Plot rho contour of the mean flow field
lh = 1.0
MeanFlow.rescale(lh)
MeanFlow.copy_meanval()
df = MeanFlow.PlanarData
var = "T"
x, y = np.meshgrid(np.unique(df.x), np.unique(df.y))
varval = griddata((df.x, df.y), df[var], (x, y))
walldist = griddata((df.x, MeanFlow.y), df.walldist, (x, y))
cover = walldist < 0.0
print("rho_max=", np.max(varval))
print("rho_min=", np.min(varval))
cval1 = 1.0  # 1.0 # 0.1
cval2 = 6.6  # 7.0  # 1.0
rg1 = np.linspace(cval1, cval2, 21)
varval[cover] = np.nan
fig, ax = plt.subplots(figsize=(8, 2.8))  # 7.3  2.3
matplotlib.rc("font", size=tsize)
cbar = ax.contourf(x, y, varval, cmap="rainbow", levels=rg1, extend="both")
ax.set_xlim(20, 360)
ax.set_ylim(-2.0, 10.0)
# ax.set_xticks(np.linspace(20, 360, 8))
ax.set_yticks(np.linspace(np.min(y), 10.0, 7))
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
boundary = pd.read_csv(pathM + "BoundaryEdgeFit.dat", skipinitialspace=True)
ax.plot(boundary.x / lh, boundary.y / lh, "k", linewidth=1.5)
# Add sonic line
sonic = pd.read_csv(pathM + "SonicLine.dat", skipinitialspace=True)
ax.plot(sonic.x / lh, sonic.y / lh, "w--", linewidth=1.5)
# plt.gca().set_aspect("equal", adjustable="box")
plt.savefig(pathF + "MeanFlow_" + var + ".svg", bbox_inches="tight")
plt.show()

# %% root mean square
df = MeanFlow.PlanarData
var = "rms_u"
if var == "rms_u":
    varnm = "<u`u`>"
    lab = r"$\sqrt{\langle u^\prime u^\prime \rangle}$"
if var == "rms_v":
    varnm = "<v`v`>"
    lab = r"$\sqrt{\langle v^\prime v^\prime \rangle}$"
if var == "rms_w":
    varnm = "<w`w`>"
    lab = r"$\sqrt{\langle w^\prime w^\prime \rangle}$"
if var == "rms_p":
    varnm = "<p`p`>"
    lab = r"$\sqrt{\langle p^\prime p^\prime \rangle}$"
var_val = getattr(df, varnm)
uu = griddata((df.x, MeanFlow.y), var_val, (x, y))
print("uu_max=", np.max(np.sqrt(np.abs(var_val))))
print("uu_min=", np.min(np.sqrt(np.abs(var_val))))
walldist = griddata((df.x, MeanFlow.y), df.walldist, (x, y))
cover = walldist < 0.0
uu[cover] = np.nan
fig, ax = plt.subplots(figsize=(8, 2.8))
matplotlib.rc("font", size=tsize)
cb1 = 0.0
cb2 = 0.1
rg1 = np.linspace(cb1, cb2, 21)  # 21)
cbar = ax.contourf(
    x / lh,
    y / lh,
    np.sqrt(np.abs(uu)),
    cmap="Spectral_r",
    levels=rg1,
    extend="both",
)  # rainbow_r # jet # Spectral_r
ax.set_xlim(20, 360.0)
ax.set_ylim(-2, 10.0)
ax.set_yticks(np.linspace(-2.0, 10.0, 7))
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
boundary = pd.read_csv(pathM + "BoundaryEdgeFit.dat", skipinitialspace=True)
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
        ax[i].set_ylim([-2, 6])
    else:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind], y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    if i != 0:
        ax[i].set_yticklabels("")
        ax[i].set_title(r"${}$".format(xcoord[i]), fontsize=nsize - 2)
    # ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].tick_params(axis="both", which="major", labelsize=nsize)
    ax[i].grid(visible=True, which="both", linestyle=":")
ax[0].set_title(r"$x={}$".format(xcoord[0]), fontsize=nsize - 2)
ax[0].set_ylabel(r"$\Delta y$", fontsize=tsize)
ax[4].set_xlabel(r"$u /u_\infty$", fontsize=tsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "BLProfileU.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()


# %%
# corner = MeanFlow.PlanarData.walldist < 0.0
firstval = 0.03125
xy_val = va.wall_line(MeanFlow.PlanarData, path, val=firstval)  # mask=corner)
# onelev = pd.DataFrame(data=xy_val, columns=["x", "y"])
# onelev.drop_duplicates(subset='x', keep='first', inplace=True)
FirstLev = pd.DataFrame(data=xy_val, columns=["x", "y"])
ind = FirstLev.index[FirstLev['x'] < 20]
FirstLev.loc[ind, 'y'] = firstval
xx = np.arange(0.0, 400.0, 0.25)
FirstLev = FirstLev[FirstLev["x"].isin(xx)]
FirstLev.to_csv(pathM + "FirstLev.dat", index=False, float_format="%9.6f")
# %%
firstval = 0.03125
T_inf = 45
Re = 10000
df = MeanFlow.PlanarData
wavy = pd.read_csv(pathM + "FirstLev.dat", skipinitialspace=True)
wavy = va.add_variable(df, wavy)
xwall, Cf = va.skinfric_wavy(pathM, wavy, Re, T_inf, firstval)
# %% skin friction
fig2, ax2 = plt.subplots(figsize=(8, 2.5), dpi=500)
matplotlib.rc("font", size=nsize)
ax2.plot(xwall, Cf, "b-", linewidth=1.5)
ax2.set_xlabel(r"$x$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=tsize)
ax2.set_xlim([0, 360])
ax2.set_xticks(np.arange(0, 361, 40))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=26, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=108.8, color="gray", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cf.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% methond 1
"""
wavy = pd.read_csv(pathM + "FirstLev.dat", skipinitialspace=True)
xwall = wavy.x
nms = ["p", "T", "u", "v"]
for i in range(np.size(nms)):
    wavy[nms[i]] = griddata(
        (df.x, df.y), df[nms[i]],
        (wavy.x, wavy.y),
        method="cubic",
    )
mu = va.viscosity(10000, wavy["T"])
ddx = np.diff(wavy["x"])
ddx = np.insert(ddx, 1, ddx[0])
ddy = np.diff(wavy["y"])
ddy = np.insert(ddy, 1, ddy[0])
tang = np.transpose(np.vstack((ddx, ddy)))
xposi = np.array([1, 0])
modules = np.linalg.norm(tang, axis=1) * np.linalg.norm(xposi)
cos_val = np.dot(tang, xposi) / modules
sin_val = np.cross(tang, xposi) / modules
delta_y = 0.03125 * np.abs(cos_val)
delta_x = 0.03125 * np.abs(sin_val)
Cf = (
    va.skinfriction(mu.values, wavy["u"], delta_y) * cos_val
    + va.skinfriction(mu.values, wavy["u"], delta_x) * sin_val
)
"""
# %% Cp
Ma = 6.0
fa = 1.0  # Ma * Ma * 1.4
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.125)
p_fit = signal.filtfilt(b, a, wavy["p"] * fa)
fig3, ax3 = plt.subplots(figsize=(8, 2.5), dpi=500)
# ax3 = fig.add_subplot(212)
ax3.plot(wavy['x'], wavy["p"] * fa, "b-", linewidth=1.5)
# ax3.plot(wavy["x"], p_fit, "k", linewidth=1.5)
ax3.set_xlabel(r"$x$", fontsize=tsize)
# ylab = r"$\langle p_w \rangle/\rho_{\infty} u_{\infty}^2$"
ax3.set_ylabel(r"$\langle C_p \rangle$", fontsize=tsize)
ax3.set_xlim([0, 360])
ax3.set_xticks(np.arange(0, 361, 40))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=26, color="gray", linestyle=":", linewidth=1.5)
ax3.axvline(x=108.8, color="blue", linestyle=":", linewidth=1.5)
ax3.grid(visible=True, which="both", linestyle=":")
ax3.annotate("(b)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cp.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% Stanton number
firstval = 0.03125
T_wall = 6.66
xwall, Cs = va.Stanton_wavy(pathM, wavy, 10000, T_inf, T_wall, firstval)
# %%
"""
Tt = va.stat2tot(Ma=6.0, Ts=45, opt="t") / 45
mu = va.viscosity(10000, wavy["T"], T_inf=45, law="Suther")
kt = va.thermal(mu, 0.72)
Cs = va.Stanton(
    kt,
    wavy["T"].values,
    6.66,
    wall_val,
    Tt,
)
"""
#%% low-pass filter
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.125)
Cs_fit = signal.filtfilt(b, a, Cs)
# ind = np.where(Cf[:] < 0.008)
fig2, ax2 = plt.subplots(figsize=(8, 2.5), dpi=500)
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
ax2.plot(xwall, np.abs(Cs), "b-", linewidth=1.5)
# ax2.plot(xwall, Cs_fit, "k", linewidth=1.5)
ax2.set_xlabel(r"$x$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_s \rangle$", fontsize=tsize)
ax2.set_xlim([0, 360])
ax2.set_xticks(np.arange(0, 361, 40))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=26, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=108.8, color="gray", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(c)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cs.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %%############################################################################
"""
    temporal evolution of signals along an axis
"""
# %% Time evolution of a specific variable at several streamwise locations
probe = lf()
lh = 1.0
fa = 1.0
var = "u"
ylab = r"$u^\prime/u_\infty$"
if var == "p":
    fa = 6.0 * 6.0 * 1.4
    ylab = r"$p^\prime/p_\infty$"
timezone = np.arange(700, 1999 + 0.125, 0.125)
xloc = [20, 60, 90, 160, 348]  # [-50.0, -30.0, -20.0, -10.0, 4.0]
yloc = [0, 0, 0, 0, 0]
zloc = 0.0
fig, ax = plt.subplots(np.size(xloc), 1, figsize=(6.4, 5.6))
fig.subplots_adjust(hspace=0.6, wspace=0.15)
matplotlib.rc("font", size=nsize)
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
for i in range(np.size(xloc)):
    probe.load_probe(pathP, (xloc[i], yloc[i], zloc))
    probe.rescale(lh)
    probe.extract_series([timezone[0], timezone[-1]])
    temp = (getattr(probe, var) - np.mean(getattr(probe, var))) * fa
    ax[i].plot(probe.time, temp, "k")
    ax[i].ticklabel_format(
        axis="y", style="sci", useOffset=False, scilimits=(-2, 2)
    )
    ax[i].set_xlim([timezone[0], timezone[-1]])
    # ax[i].set_ylim([-0.001, 0.001])
    if i != np.size(xloc) - 1:
        ax[i].set_xticklabels("")
    ax[i].set_ylabel(ylab, fontsize=tsize)
    ax[i].grid(visible=True, which="both", linestyle=":")
    ax[i].set_title(r"$x/\delta_0={}$".format(xloc[i]), fontsize=nsize - 1)
ax[-1].set_xlabel(r"$t u_\infty/\delta_0$", fontsize=tsize)

plt.savefig(
    pathF + var + "_TimeEvolveX" + str(zloc) + ".svg", bbox_inches="tight"
)
plt.show()

# %%############################################################################
"""
    streamwise evolution of signals along an axis
"""
# %% Streamwise evolution of a specific variable
df = pd.read_hdf(path + "TP_fluc_00643627_01892.h5")
wall = df.groupby("x", as_index=False).nth(1)
fa = 1.0  # 1.7 * 1.7 * 1.4
var = "p`"
meanval = np.mean(wall[var])
fig, ax = plt.subplots(figsize=(6.4, 2.8))
matplotlib.rc("font", size=nsize)
ax.plot(wall.x, wall[var] - meanval, "k")
ax.set_xlim([0, 360.0])
# ax.set_ylim([-0.001, 0.001])
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$2p^\prime/\rho_\infty u_\infty^2$", fontsize=tsize)
ax.ticklabel_format(axis="y", style="sci", useOffset=False, scilimits=(-2, 2))
ax.grid(visible="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + var + "_streamwise.svg", bbox_inches="tight", pad_inches=0.1
)

# %% animation for vortex structure
pathF = path + "Figures/"/
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
