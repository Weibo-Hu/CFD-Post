#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:39:31 2018
    Plot for time-averaged flow (3D meanflow)
@author: Weibo Hu
"""
# %%
# %matplotlib ipympl
import numpy as np
import plt2pandas as p2p
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.signal import savgol_filter
get_ipython().run_line_magic("matplotlib", "qt")
# import modin.pandas as pd

import variable_analysis as va
from line_field import LineField as lf
from planar_field import PlanarField as pf
from triaxial_field import TriField as tf
from data_post import DataPost
from glob import glob
from scipy.interpolate import griddata
from numpy import NaN, Inf, arange, isscalar, asarray, array
from timer import timer
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os
from scipy.interpolate import splrep, splprep, splev, interp1d

plt.close("All")
plt.rc("text", usetex=True)
font = {"family": "Times New Roman", "color": "k", "weight": "normal"}
cm2in = 1 / 2.54
# %%
path = "/media/weibo/VID21/ramp_st14/"
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
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
tsize = 13
nsize = 10
matplotlib.rc("font", size=tsize)

# %% Load Data for time- spanwise-averaged results
# filter files if necessary
FileId = pd.read_csv(path + "StatList.dat", sep="\t")
filelist = FileId["name"].to_list()
pltlist = [os.path.join(path + "TP_stat/", name) for name in filelist]
MeanFlow = pf()
MeanFlow.load_meanflow(path, FileList=pltlist)
# %% Load Data for time- spanwise-averaged results
MeanFlow = pf()
MeanFlow.load_meanflow(path)
# %%
MeanFlow.merge_stat(pathM)
# %% rescaled if necessary
lh = 1.0
MeanFlow.rescale(lh)
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
walldist = griddata((MeanFlow.x, MeanFlow.y),
                    MeanFlow.walldist,
                    (x, y),
                    method="cubic")
corner1 = (x < 25) & (y < 0.0)
corner2 = (x > 110) & (y < 0.0)
corner = (walldist < 0.0)  # corner1 | corner2

# %%############################################################################
"""
    Examination of the computational mesh
"""
# %% wall buondary
va.wall_line(MeanFlow.PlanarData, pathM, mask=None) # corner)
# %% check mesh
temp = MeanFlow.PlanarData  # [["x", "y"]]
df = temp.query("x>=-100.0 & x<=90.0 & y>=0.0 & y<=30.0")
# df = temp.query("x>=-5.0 & x<=10.0 & y>=-3.0 & y<=2.0")
ux = np.unique(df.x)[::2]
uy = np.unique(df.y)[::2]
fig, ax = plt.subplots(figsize=(14*cm2in, 6*cm2in))
matplotlib.rc("font", size=tsize)
for i in range(np.size(ux)):
    if i % 1 == 0:
        df_x = df.loc[df["x"] == ux[i]]
        ax.plot(df_x["x"], df_x["y"], color="gray", linestyle="-", linewidth=0.4)
for j in range(np.size(uy)):
    if j % 1 == 0:  # 4
        df_y = df.loc[df["y"] == uy[j]]
        ax.plot(df_y["x"], df_y["y"], color="gray", linestyle="-", linewidth=0.4)
# plt.gca().set_aspect("equal", adjustable="box")
ax.set_xlim(np.min(df.x), np.max(df.x))
ax.set_ylim(np.min(df.y), np.max(df.y))
ax.set_yticks(np.linspace(0.0, 30.0, 7))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x$", fontsize=tsize)
ax.set_ylabel(r"$y$", fontsize=tsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "Grid.svg", bbox_inches="tight")
plt.show()
# %%
# wavy = temp.loc[np.round(temp['walldist'], 3) == 0.001]
wavy0 = pd.read_csv(pathM + "wavy.dat", skipinitialspace=True)
wavy1 = pd.read_csv(pathM + "WallBoundary.dat", skipinitialspace=True)
fig, ax = plt.subplots(figsize=(14*cm2in, 6*cm2in))
ax.plot(wavy0["x"], wavy0["y"], "b--", linewidth=1.5)
ax.scatter(
    wavy1["x"][::4],
    wavy1["y"][::4],
    linewidth=0.8,
    s=18.0,
    facecolor="red",
    edgecolor="red",
)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "wall_comp.svg", bbox_inches="tight")
plt.show()

# %% dividing streamline
MeanFlow.copy_meanval()
points = np.array([[0.0], [2.0]])
xyzone = np.array([[-80.0, 0.0, 20.0], [0.0, 3.0, 6.0]])
va.streamline(pathM, MeanFlow.PlanarData, points, partition=xyzone, opt="up")
# %% Mean flow isolines
# dataframe = MeanFlow.PlanarData
# NewFrame = dataframe.query("x>=-70.0 & x<=0.0 & y<=10.0")
walldist = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.walldist, (x, y), method="cubic")
cover = (walldist < 0.0)  # | corner
NewFrame = MeanFlow.PlanarData
va.dividing_line(
    NewFrame, pathM, show=True, mask=cover
)  # (MeanFlow.PlanarData, pathM, loc=2.5)
# %% Save sonic line
va.sonic_line(MeanFlow.PlanarData, pathM, option="velocity", Ma_inf=6.0, mask=corner)
# %% enthalpy boundary layer
va.enthalpy_boundary_edge(MeanFlow.PlanarData, pathM, Ma_inf=6.0, crit=0.99)
# %% thermal boundary layer
va.thermal_boundary_edge(MeanFlow.PlanarData, pathM, T_wall=3.35)
# %% Save sonic line
va.shock_line(MeanFlow.PlanarData, pathM, var="|gradp|", val=0.05)
# %% Save shock line
va.shock_line_ffs(
    MeanFlow.PlanarData, pathM, val=[0.05], show=True
)  # 0.065 for TB, 0.06 for ZA
# %% Save boundary layer
# va.boundary_edge(MeanFlow.PlanarData, pathM, jump0=-25, jump1=-10,
#                 jump2=6.0, val1=0.811, val2=0.95)
va.boundary_edge(MeanFlow.PlanarData, pathM, shock=False, mask=False)
# %%
boundary = np.loadtxt(pathM + "BoundaryEdge.dat", skiprows=1)
dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
bsp = splrep(boundary[1200:1520, 0], boundary[1200:1520, 1], k=3, s=0.6, per=0.0)
x_new = np.arange(np.min(boundary[1200:1520, 0]), np.max(boundary[1200:1520, 0]), 0.125)
y_new = splev(x_new, bsp)
xy_fit = np.vstack((x_new, y_new))
np.savetxt(
    pathM + "BoundaryEdgeFit2.dat", xy_fit.T, fmt="%.8e", delimiter="  ", header="x, y"
)
# %%
xy_fit = np.loadtxt(pathM + "BoundaryEdgeFit.dat", skiprows=1)
fig, ax = plt.subplots(figsize=(6.6, 2.3))
ax.plot(xy_fit[:, 0], xy_fit[:, 1], "b-")
ax.plot(boundary[:, 0], boundary[:, 1], "r:")
ax.plot(dividing[:, 0], dividing[:, 1], "g--", linewidth=1.5)
bubble = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
ax.plot(bubble[:, 0], bubble[:, 1], "k", linewidth=1.5)
# shock = np.loadtxt(pathM + "ShockLineFit.dat", skiprows=1)
# ax.plot(shock[:, 0], shock[:, 1], "k", linewidth=1.5)
# shock = np.loadtxt(pathM + "ShockLine2.dat", skiprows=1)
# ax.plot(shock[:, 0], shock[:, 1], "k", linewidth=1.5)
ax.set_xlim(-40, 10.0)
ax.set_ylim(0, 5.0)
plt.show()
# %%
plt.close("All")
# %%############################################################################
"""
    mean flow field contouring by velocity
"""
MeanFlow.copy_meanval()
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
var = "u"
lh = 1.0
u = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (x, y))
walldist = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.walldist, (x, y), method="cubic")
# cover1 = walldist < 0.0  # np.where(walldist[:,:] < 0.0)
# cover = cover1 | corner
# u = np.ma.array(u, mask=cover)  # mask=cover
# u[cover] = np.ma.masked
print("u_max=", np.max(MeanFlow.u))
print("u_min=", np.min(MeanFlow.u))
# u[corner] = np.nan
cval1 = -0.1  # 0.3
cval2 = 1.0  # 1.4
fig, ax = plt.subplots(figsize=(7.3, 2.3))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(cval1, cval2, 41)
cbar = ax.contourf(x, y, u, cmap="rainbow", levels=rg1, extend="both")  # rainbow_r
# cbar1 = ax.contour(x, y, u, levels=[0.0])
ax.set_xlim(-200, 80)
ax.set_ylim(np.min(y), 40.0)
ax.set_xticks(np.linspace(-200, 80.0, 8))
ax.set_yticks(np.linspace(0.0, 40.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x$", fontsize=tsize)
ax.set_ylabel(r"$y$", fontsize=tsize)
plt.savefig(pathF + "MeanFlow_preview.svg", bbox_inches="tight")
# plt.gca().set_aspect("equal", adjustable="box")
plt.show()
# %%############################################################################
"""
    mean flow field contouring by density
"""
# %% Plot rho contour of the mean flow field
# MeanFlow.AddVariable('rho', 1.7**2*1.4*MeanFlow.p/MeanFlow.T)
MeanFlow.copy_meanval()
var = "u"      
lh = 1.0
rho = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.rho, (x, y))
u = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (x, y))
walldist = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.walldist, (x, y))
# cover = (walldist < 0.0) | corner
# rho = np.ma.array(rho, mask=cover)
# u = np.ma.array(u, mask=cover)
print("rho_max=", np.max(rho))
print("rho_min=", np.min(rho))
cval1 = 0.4 #1.0 # 
cval2 = 3.6 #7.0  # 
fig, ax = plt.subplots(figsize=(7.3, 2.3))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(cval1, cval2, 21)
cbar = ax.contourf(x, y, rho, cmap="rainbow", levels=rg1, extend="both")
ax.contour(x, y, u, levels=[0.0], linestyles="dotted")  # rainbow_r
ax.set_xlim(-200, 80)
ax.set_ylim(np.min(y), 36.0)
ax.set_xticks(np.linspace(-200, 80.0, 8))
ax.set_yticks(np.linspace(np.min(y), 36.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x$", fontsize=tsize)
ax.set_ylabel(r"$y$", fontsize=tsize)
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cval1, cval2, 3)
cbaxes = fig.add_axes([0.17, 0.76, 0.16, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(
    cbar, cax=cbaxes, extendrect="False", orientation="horizontal", ticks=rg2
)
lab = r"$\langle \rho \rangle/\rho_{\infty}$" #r"$\langle T \rangle/T_{\infty}$" #     
cbar.set_label(lab, rotation=0, fontsize=tsize)
# Add boundary layer
# boundary = pd.read_csv(pathM + "BoundaryEdge.dat", skipinitialspace=True)
# ax.plot(boundary.x / lh, boundary.y / lh, "k-", linewidth=1.5)
# Add shock wave
Enthalpy = pd.read_csv(pathM + "EnthalpyBoundaryEdge99.dat", skipinitialspace=True)
ax.plot(Enthalpy.x / lh, Enthalpy.y / lh, "k--", linewidth=1.5)
# thermal = pd.read_csv(pathM + "ThermalpyBoundaryEdge.dat", skipinitialspace=True)
# ax.plot(thermal.x / lh, thermal.y / lh, "w:", linewidth=1.5)
# shock1 = np.loadtxt(pathM + "ShockLine1.dat", skiprows=1)
# ax.plot(shock1[:, 0], shock1[:, 1], "w", linewidth=1.5)
# shock2 = pd.read_csv(pathM + "EnthalpyBoundaryEdge1.dat", skipinitialspace=True)
# ax.plot(shock2.x / lh, shock2.y / lh, "k:", linewidth=1.5)
# Add sonic line
sonic = pd.read_csv(pathM + "SonicLine.dat", skipinitialspace=True)
ax.plot(sonic.x / lh, sonic.y / lh, "w--", linewidth=1.5)
# Add dividing line(separation line)
dividing = pd.read_csv(pathM + "BubbleLine.dat", skipinitialspace=True)
ax.plot(dividing.x / lh, dividing.y / lh, "k:", linewidth=1.5)
# streamlines
x1 = np.linspace(0.0, 12.0, 120)
y1 = np.linspace(-3.0, -0.0, 100)
seeds = np.array(
    [
        [6.83, 5.93, 5.35, 3.85, 2.50, 1.42, 0.76],
        [-2.1, -1.8, -1.6, -1.14, -0.8, -0.54, -0.35],
    ]
)
xbox, ybox = np.meshgrid(x1, y1)
u = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (xbox, ybox))
v = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.v, (xbox, ybox))
# ax.streamplot(
#    xbox,
#    ybox,
#    u,
#    v,
#    color="w",
#    density=[3.0, 2.0],
#    arrowsize=0.7,
#    start_points=seeds.T,
#    maxlength=30.0,
#    linewidth=1.0,
# )
plt.savefig(pathF + "MeanFlow_rho.svg", bbox_inches="tight")
plt.show()
# %%############################################################################
"""
    instantaneous spanwise-average flow field
"""
# %% Plot contour of the instantaneous flow field with isolines
# MeanFlow.AddVariable('rho', 1.7**2*1.4*MeanFlow.p/MeanFlow.T)
# xx = np.arange(0.0, 20.0 + 0.25, 0.25)
# yy = np.arange(-3.0, 4.0 + 0.125, 0.125)
# x, y = np.meshgrid(xx, yy)
# corner = (x < 0.0) & (y < 0.0)
vtm = 700
vtm = np.linspace(600, 700, 401)
for i in range(np.size(vtm)):
    stm = "%08.2f" % vtm[i]
    InstFlow = pd.read_hdf(pathSL + "video/TP_2D_Z_003_" + stm + ".h5")
    var = "u"
    var1 = "|gradp|"
    u = griddata((InstFlow.x, InstFlow.y), InstFlow[var], (x, y))
    gradp = griddata((InstFlow.x, InstFlow.y), InstFlow[var1], (x, y))
    cover = (walldist < 0.0) | corner
    u = np.ma.array(u, mask=cover)
    print("u=", np.max(u))
    print("u=", np.min(u))
    cval1 = -0.4
    cval2 = 1.1
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    matplotlib.rc("font", size=tsize)
    rg1 = np.linspace(cval1, cval2, 21)
    cbar = ax.contourf(x, y, u, cmap="bwr", levels=rg1, extend="both")  # rainbow_r
    ax.set_xlim(-20.0, 5.0)
    ax.set_ylim(0, 8.0)
    ax.tick_params(labelsize=nsize)
    ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
    ax.set_ylabel(r"$y/\delta_0$", fontsize=tsize)
    ax.set_title(
        r"$t u_\infty/\delta_0={:.2f}$".format(vtm[i]), fontsize=tsize - 1, pad=0.1
    )
    ax.grid(b=True, which="both", linestyle=":")
    plt.gca().set_aspect("equal", adjustable="box")
    # Add colorbar
    rg2 = np.linspace(cval1, cval2, 3)
    cbaxes = fig.add_axes([0.16, 0.73, 0.20, 0.07])  # x, y, width, height
    cbaxes.tick_params(labelsize=nsize)
    cbar = plt.colorbar(
        cbar, cax=cbaxes, extendrect="False", orientation="horizontal", ticks=rg2
    )
    cbar.set_label(r"$u/u_{\infty}$", rotation=0, fontsize=tsize)

    # Add dividing line
    cbar = ax.contour(x, y, u, levels=[0.0], colors="k", linewidths=1.0)
    # Add shock wave, gradp = 0.1, alpha=0.8
    cbar = ax.contour(
        x,
        y,
        gradp,
        levels=[0.1],
        colors="w",
        alpha=0.8,
        linewidths=1.0,
        linestyles="--",
    )
    # add mean bubble line
    # dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
    # ax.plot(dividing[:, 0], dividing[:, 1], "gray", linewidth=1.5)
    # add mean shock line
    # shock = np.loadtxt(pathM + "ShockLineFit.dat", skiprows=1)
    # ax.plot(shock[:, 0], shock[:, 1], "gray", linewidth=1.5)

    plt.savefig(
        pathF + "video/InstFlow_" + str(i) + ".jpg",
        bbox_inches="tight",
        dpi=600,
        width=2048,
        supersample=3,
    )
    plt.show()
    plt.close()

# %%############################################################################
"""
    instantaneous flow field
"""
# %% Plot contour of the instantaneous flow field with isolines
# MeanFlow.AddVariable('rho', 1.7**2*1.4*MeanFlow.p/MeanFlow.T)
vtm = 700
stm = "%08.2f" % vtm
InstFlow = pd.read_hdf(pathSL + "Z_003/TP_2D_Z_003_" + stm + ".h5")
var = "u"
var1 = "vorticity_3"
u = griddata((InstFlow.x, InstFlow.y), InstFlow[var], (x, y))
gradp = griddata((InstFlow.x, InstFlow.y), InstFlow[var1], (x, y))
print("u=", np.max(InstFlow[var]))
print("u=", np.min(InstFlow[var]))
cover = (walldist < 0.0) | corner
u = np.ma.array(u, mask=cover)
gradp = np.ma.array(gradp, mask=cover)
cval1 = -0.4
cval2 = 1.1
fig, ax = plt.subplots(figsize=(5.0, 2.5))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(cval1, cval2, 41)
cbar = ax.contourf(x, y, u, cmap="bwr", levels=rg1, extend="both")  # rainbow_r
ax.set_xlim(0.0, 4.0)
ax.set_ylim(-2.0, 0.0)
ax.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0])
ax.tick_params(labelsize=nsize + 1)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize + 1)
ax.set_ylabel(r"$y/\delta_0$", fontsize=tsize + 1)
ax.set_title(r"$t u_\infty /\delta_0=1295$", fontsize=textsize - 1, pad=0.1)
# ax.grid(b=True, which="both", linestyle=":")
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cval1, cval2, 3)
cbar = plt.colorbar(cbar, ticks=rg2, extendrect=True, fraction=0.025, pad=0.05)
cbar.ax.tick_params(labelsize=nsize)
cbar.set_label(r"$u/u_{\infty}$", rotation=0, fontsize=tsize, labelpad=-30, y=1.1)

# Add isolines
# cbar = ax.contour(x, y, gradp, levels=[-3.6], colors='k',
#                  alpha=1.0, linewidths=1.0, linestyles='--')

# streamlines
x1 = np.linspace(0.0, 6.0, 140)
y1 = np.linspace(-3.0, -0.0, 90)
xbox, ybox = np.meshgrid(x1, y1)
u = griddata((InstFlow.x, InstFlow.y), InstFlow.u, (xbox, ybox))
v = griddata((InstFlow.x, InstFlow.y), InstFlow.v, (xbox, ybox))
ax.streamplot(
    xbox,
    ybox,
    u,
    v,
    color="k",
    density=[3.0, 3.0],
    arrowsize=0.7,
    maxlength=30.0,
    linewidth=0.6,
)

plt.savefig(pathF + "Shedding2.svg", bbox_inches="tight")
plt.show()

# %%############################################################################
#
# Root Mean Square of velocity from the statistical flow
#
# %% Plot rms contour of the mean flow field
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
var = "<u`u`>"
if var == "<u`u`>":
    lab = r"$\sqrt{\langle u^\prime u^\prime \rangle}$"
    nm = 'RMSUU'
elif var == "<v`v`>":
    lab = r"$\sqrt{\langle v^\prime v^\prime \rangle}$"
    nm = 'RMSVV'
elif var == "<w`w`>":
    lab = r"$\sqrt{\langle w^\prime w^\prime \rangle}$"
    nm = 'RMSWW'
var_val = getattr(MeanFlow.PlanarData, var)
uu = griddata((MeanFlow.x, MeanFlow.y), var_val, (x, y))
u = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (x, y))
print("uu_max=", np.max(np.sqrt(np.abs(var_val))))
print("uu_min=", np.min(np.sqrt(np.abs(var_val))))
# cover = (walldist < 0.0) | corner
# uu = np.ma.array(uu, mask=cover)
# u = np.ma.array(u, mask=cover)
fig, ax = plt.subplots(figsize=(7.3, 2.3))
matplotlib.rc("font", size=tsize)
cb1 = 0.0
cb2 = 0.2
rg1 = np.linspace(cb1, cb2, 21)  # 21)
cbar = ax.contourf(
    x / lh, y / lh, np.sqrt(np.abs(uu)), cmap="coolwarm", levels=rg1, extend="both"
)  # rainbow_r # jet # Spectral_r
ax.set_xlim(-200, 80.0)
ax.set_ylim(np.min(y), 36.0)
ax.set_xticks(np.linspace(-200.0, 80.0, 8))
ax.set_yticks(np.linspace(0.0, 36.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x$", fontdict=font)
ax.set_ylabel(r"$y$", fontdict=font)
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar box
rg2 = np.linspace(cb1, cb2, 3)
cbbox = fig.add_axes([0.14, 0.52, 0.22, 0.31], alpha=0.9)
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
cbaxes = fig.add_axes([0.16, 0.75, 0.18, 0.058])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(
    cbar, cax=cbaxes, orientation="horizontal", extendrect="False", ticks=rg2
)

cbar.set_label(
    lab, rotation=0, fontsize=tsize,
)
# Add boundary layer
boundary = pd.read_csv(pathM + "EnthalpyBoundaryEdge99.dat", skipinitialspace=True)
ax.plot(boundary.x / lh, boundary.y / lh, "k--", linewidth=1.0)
# Add bubble line
bubble = pd.read_csv(pathM + "BubbleLine.dat", skipinitialspace=True)
ax.plot(bubble.x / lh, bubble.y / lh, "k:", linewidth=1.0)
# Add sonic line
sonic = pd.read_csv(pathM + "SonicLine.dat", skipinitialspace=True)
ax.plot(sonic.x / lh, sonic.y / lh, "w--", linewidth=1.0)
# cbar = ax.contour(
#     x, y, u, levels=[0.0], colors="w", linewidths=1.0, linestyles="dashed" 
#)
# dividing = pd.read_csv(pathM + "DividingLine.dat", skipinitialspace=True)
# ax.plot(dividing.x / lh, dividing.y / lh, "k--", linewidth=1.2)

plt.savefig(pathF + "MeanFlow" + nm + ".svg", bbox_inches="tight")
plt.show()

# %%############################################################################
###
# Vorticity contour
###
# %% Load Data for time-averaged results
path4 = "/media/weibo/Data1/BFS_M1.7L/Slice/"
MeanFlow = DataPost()
MeanFlow.UserDataBin(pathF + "SolTime995.00.h5")
# %% Preprocess the data in the xy plane (z=0.0)
MeanFlowZ0 = MeanFlow.DataTab.loc[MeanFlow.DataTab["z"] == 0.0]
MeanFlowZ1 = MeanFlow.DataTab.loc[MeanFlow.DataTab["z"] == 0.5]
MeanFlowZ2 = MeanFlow.DataTab.loc[MeanFlow.DataTab["z"] == -0.5]
MeanFlowZ3 = pd.concat((MeanFlowZ1, MeanFlowZ2))
MeanFlowZ4 = MeanFlowZ3.groupby(["x", "y"]).mean().reset_index()
MeanFlowZ5 = MeanFlowZ4.loc[MeanFlowZ4["y"] > 3.0]
MeanFlowZ0 = pd.concat((MeanFlowZ0, MeanFlowZ5), sort=False)
MeanFlowZ0.to_hdf(pathF + "MeanFlowZ0.h5", "w", format="fixed")

# %% Load data
fluc_flow = tf()
fluc_flow.load_3data(path, FileList=path + "/ZFluctuation_900.0.h5", NameList="h5")
# %% Plot contour of the fluctuation flow field in the xy plane
zslice = fluc_flow.TriData.loc[fluc_flow.TriData["z"] == 0.0]
x, y = np.meshgrid(np.unique(zslice.x), np.unique(zslice.y))
omegaz = griddata((zslice.x, zslice.y), zslice.vorticity_3, (x, y))
corner = (x < 0.0) & (y < 0.0)
omegaz[corner] = np.nan
fig, ax = plt.subplots(figsize=(6.4, 3.2))
matplotlib.rc("font", size=tsize)
plt.tick_params(labelsize=nsize)
val1 = -0.2
val2 = 0.2
lev1 = np.linspace(val1, val2, 21)
cbar = ax.contourf(x, y, omegaz, cmap="RdBu_r", levels=lev1, extend="both")
ax.set_xlim(-5.0, 10.0)
ax.set_ylim(-3.0, 3.0)
cbar.cmap.set_under("#053061")
cbar.cmap.set_over("#67001f")
ax.set_xlabel(r"$x/\delta_0$", fontdict=font)
ax.set_ylabel(r"$y/\delta_0$", fontdict=font)
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(val1, val2, 3)
cbaxes = fig.add_axes([0.17, 0.72, 0.18, 0.05])  # x, y, width, height
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar.set_label(r"$\omega_z$", rotation=0, fontdict=font)

box = patches.Rectangle(
    (1.0, -1.5), 4.0, 1.3, linewidth=1, linestyle="--", edgecolor="k", facecolor="none",
)
# ax.add_patch(box)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Vorticity3_z=0.svg", bbox_inches="tight")
plt.show()

# %% Zoom box for streamline
fig, ax = plt.subplots(figsize=(7, 3))
cbar = ax.contourf(x, y, omegaz, cmap="rainbow", levels=lev1, extend="both")
ax.set_xlim(1.0, 5.0)
ax.set_ylim(-1.5, -0.2)
ax.set_xticks([])
ax.set_yticks([])
x1 = np.linspace(1.0, 5.0, 50)
y1 = np.linspace(-1.5, -0.2, 50)
xbox, ybox = np.meshgrid(x1, y1)
u = griddata((MeanFlowZ0.x, MeanFlowZ0.y), MeanFlowZ0.u, (xbox, ybox))
v = griddata((MeanFlowZ0.x, MeanFlowZ0.y), MeanFlowZ0.v, (xbox, ybox))
ax.streamplot(
    xbox,
    ybox,
    u,
    v,
    density=[2.0, 1.5],
    color="w",
    linewidth=1.0,
    integration_direction="both",
)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(pathF + "V3ZoomBox.svg", bbox_inches="tight", pad_inches=0.0)
plt.show()

# %% Preprocess the data in the xz plane (y=-0.5)
MeanFlowY = MeanFlow.DataTab.loc[MeanFlow.DataTab["y"] == -1.5].reset_index(drop=True)
MeanFlowY.to_hdf(path + "MeanFlowYN3.h5", "w", format="fixed")
MeanFlowY = DataPost()
MeanFlowY.UserDataBin(path + "MeanFlowYN3.h5")
# %% Plot contour of the mean flow field in the xz plane
x, z = np.meshgrid(np.unique(MeanFlowY.x), np.unique(MeanFlowY.z))
omegaz = griddata((MeanFlowY.x, MeanFlowY.z), MeanFlowY.vorticity_1, (x, z))
fig, ax = plt.subplots(figsize=(5, 3))
matplotlib.rc("font", size=tsize)
plt.tick_params(labelsize=nsize)
lev1 = np.linspace(-0.1, 0.1, 2)
lev1 = [-0.1, 0.1]
# lev1 = [-0.1, 0.0, 0.1]
cbar1 = ax.contourf(
    x, z, omegaz, cmap="RdBu_r", levels=lev1, extend="both"
)  # rainbow_r
ax.set_xlim(2.0, 10.0)
ax.set_ylim(-2.5, 2.5)
cbar.cmap.set_over("#ff5c33")  # ('#053061') #Red
cbar.cmap.set_under("#33adff")  # ('#67001f') #Blue
ax.set_xlabel(r"$x/\delta_0$")
ax.set_ylabel(r"$z/\delta_0$")
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
# rg2 = np.linspace(-0.1, -0.1, 3)
# ax1_divider = make_axes_locatable(ax)
# cax1 = ax1_divider.append_axes("top", size="7%", pad="12%")
# cbar = plt.colorbar(cbar1, cax=cax1, orientation="horizontal", ticks=rg2)
# plt.tick_params(labelsize=nsize)
# cax1.xaxis.set_ticks_position("top")
# cbar.ax.tick_params(labelsize=nsize)
# ax.set_aspect('auto')
# cbar.set_label(r'$\omega_x$', rotation=0, fontdict=font)
# Add isolines of Lambda2 criterion
# L2 = griddata((MeanFlowY.x, MeanFlowY.z), MeanFlowY.L2crit, (x, z))
# ax.contour(x, z, L2, levels=-0.001, \
#           linewidths=1.2, colors='k',linestyles='solid')
plt.show()
plt.savefig(pathF + "Vorticity1XZ.svg", bbox_inches="tight")

# %% Plot contour of the mean flow field in the zy plane
xslice = fluc_flow.TriData.loc[fluc_flow.TriData["x"] == -0.5]
z, y = np.meshgrid(np.unique(xslice.z), np.unique(xslice.y))
omega1 = griddata((xslice.z, xslice.y), xslice.vorticity_3, (z, y))
fig, ax = plt.subplots(figsize=(3.2, 2.2))
matplotlib.rc("font", size=tsize)
plt.tick_params(labelsize=nsize)
val1 = -0.2
val2 = 0.2
lev1 = np.linspace(val1, val2, 21)
cbar1 = ax.contourf(
    z, y, omega1, cmap="RdBu_r", levels=lev1, extend="both"
)  # rainbow_r # RdYlBu
ax.set_xlim(-8.0, 8.0)
ax.set_xticks(np.linspace(-8.0, 8.0, 5))
ax.set_ylim(0.0, 1.0)
cbar1.cmap.set_under("#053061")
cbar1.cmap.set_over("#67001f")
ax.set_xlabel(r"$z/\delta_0$", fontdict=font)
ax.set_ylabel(r"$y/\delta_0$", fontdict=font)
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(val1, val2, 3)
# cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
ax1_divider = make_axes_locatable(ax)
cax1 = ax1_divider.append_axes("top", size="7%", pad="13%")
cbar = plt.colorbar(cbar1, cax=cax1, orientation="horizontal", ticks=rg2)
plt.tick_params(labelsize=nsize)
cax1.xaxis.set_ticks_position("top")
cbar.set_label(r"$\omega_z$", rotation=0, fontsize=tsize)
# Add streamlines
# zz = np.linspace(-8.0, 8.0, 20)
# yy = np.linspace(0.0, 1.0, 10)
# zbox, ybox = np.meshgrid(zz, yy)
# v = griddata((xslice.z, xslice.y), xslice.v, (zbox, ybox))
# w = griddata((xslice.z, xslice.y), xslice.w, (zbox, ybox))
# ax.set_xlim(-8.0, 8.0)
# ax.set_ylim(0.0, 1.0)
# ax.streamplot(
#    zbox,
#    ybox,
#    w,
#    v,
#    density=[2.5, 1.5],
#    color="w",
#    linewidth=1.0,
#    integration_direction="both",
# )
plt.show()
plt.savefig(pathF + "vorticity3_x=0.1.svg", bbox_inches="tight", pad_inches=0.1)

# %%############################################################################
###
# velocity contour and streamlines
###
# %% Plot contour and streamline of spanwise-averaged flow (Zoom in)
path = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/7/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
# MeanFlowZ = DataPost()
# MeanFlowZ.UserDataBin(path+'SolTime772.50.h5')
time = "790.50"
MeanFlowZ = pd.read_hdf(path + "SolTime" + time + ".h5")
xval = np.linspace(0.0, 10, 100)
yval = np.linspace(-3.0, 0.0, 30)
x, y = np.meshgrid(xval, yval)
u = griddata((MeanFlowZ.x, MeanFlowZ.y), MeanFlowZ.u, (x, y))
# Contour
fig, ax = plt.subplots(figsize=(5, 3))
matplotlib.rc("font", size=tsize)
plt.tick_params(labelsize=nsize)
lev1 = np.linspace(-0.4, 1.2, 30)
cbar = ax.contourf(x, y, u, cmap="rainbow", levels=lev1)  # rainbow_r
ax.set_xlim(2.0, 10.0)
ax.set_ylim(-3.0, 0.0)
# cbar.cmap.set_under('b')
# cbar.cmap.set_over('r')
ax.set_xlabel(r"$x/\delta_0$", fontdict=font)
ax.set_ylabel(r"$y/\delta_0$", fontdict=font)
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar box
rg2 = np.linspace(0.0, 0.20, 3)
cbbox = fig.add_axes([0.46, 0.54, 0.435, 0.19], alpha=0.2)
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
rg2 = np.linspace(-0.4, 1.2, 3)
cax = fig.add_axes([0.6, 0.65, 0.25, 0.05])  # x, y, width, height
cbar = plt.colorbar(cbar, cax=cax, orientation="horizontal", ticks=rg2)
# cbar.ax.set_title(r'$u/u_\infty$', fontsize=tsize)
cbar.set_label(r"$u/u_\infty$", rotation=0, x=-0.28, labelpad=-28, fontsize=tsize)
plt.tick_params(labelsize=nsize)
# Streamline
v = griddata((MeanFlowZ.x, MeanFlowZ.y), MeanFlowZ.v, (x, y))
# x, y must be equal spaced
ax.streamplot(
    x,
    y,
    u,
    v,
    density=[4, 2.5],
    color="w",
    arrowsize=0.5,
    linewidth=0.6,
    integration_direction="both",
)
plt.savefig(pathF + "StreamVortex" + time + ".svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %%############################################################################
#
# vorticity enstrophy along streamwise
#
# %% Load data and calculate enstrophy in every direction
dirs = glob(pathV + "Enstrophy_*dat")
tab_new = pd.read_csv(dirs[0], sep=" ", index_col=False, skipinitialspace=True)
for j in range(np.size(dirs) - 1):
    tab_dat = pd.read_csv(dirs[j + 1], sep=" ", index_col=False, skipinitialspace=True)
    tab_new = tab_new.add(tab_dat, fill_value=0)
# %%
x1 = 4.0
x2 = 7.5
enstro = tab_new / np.size(dirs)
# enstro = enstro.iloc[1:,:]
fx = np.arange(0.2, 25 + 0.5, 0.5)
f3 = savgol_filter(enstro["enstrophy"], 41, 4)
f1 = interp1d(enstro["x"], enstro["enstrophy"], kind="cubic", fill_value="extrapolate")
f2 = interp1d(
    enstro["x"], enstro["enstrophy_x"], kind="cubic", fill_value="extrapolate"
)
me = np.max(enstro["enstrophy"])
fig3, ax3 = plt.subplots(figsize=(6.4, 2.6))
ax3.plot(enstro["x"], enstro["enstrophy"] / me, "k:", linewidth=1.2)
ax3.plot(enstro["x"], f3 / me, "b--", linewidth=1.2)
ax3.plot(enstro["x"], enstro["enstrophy_x"] / me, "r", linewidth=1.2)
ax3.plot(enstro["x"], enstro["enstrophy_y"] / me, "g", linewidth=1.2)
ax3.plot(enstro["x"], enstro["enstrophy_z"] / me, "b", linewidth=1.2)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax3.set_ylabel(r"$\mathcal{E}/\mathcal{E}_\mathrm{max}$", fontsize=tsize)
ax3.set_xlim([0.0, 25.0])
ax3.set_ylim([0.0, 1.2])
ax3.axvline(x=x1, color="gray", linestyle="--", linewidth=1.0)
ax3.axvline(x=x2, color="gray", linestyle="--", linewidth=1.0)
lab = [r"$\mathcal{E}$", r"$\mathcal{E}_x$", r"$\mathcal{E}_y$", r"$\mathcal{E}_z$"]
ax3.legend(lab, ncol=2, fontsize=nsize)  # loc="lower right"
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=nsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
# plt.savefig(pathF + "enstrophy.pdf", dpi=300)
plt.show()

# %%############################################################################
#
# vorticity transportation equation analysis
#
# %% vortex dynamics in every direction
dirs = glob(pathV + "vortex_z_*dat")
tab_new = pd.read_csv(dirs[0], sep=" ", index_col=False, skipinitialspace=True)
for j in range(np.size(dirs) - 1):
    tab_dat = pd.read_csv(dirs[j + 1], sep=" ", index_col=False, skipinitialspace=True)
    tab_new = tab_new.add(tab_dat, fill_value=0)

if dirs[0].find("_x_") > 0:
    lab = [r"$T_{xy}$", r"$T_{xz}$", r"$S_{xx}$", r"$D$"]
if dirs[0].find("_y_") > 0:
    lab = [r"$T_{yx}$", r"$T_{yz}$", r"$S_{yy}$", r"$D$"]
if dirs[0].find("_z_") > 0:
    lab = [r"$T_{zx}$", r"$T_{zy}$", r"$S_{zz}$", r"$D$"]
vortex3 = tab_new / np.size(dirs)
# vortex3 = vortex3.iloc[1:,:]


def full_term(df, ax):
    sc = np.amax(np.abs(df.iloc[:, 1:].values))
    ax.plot(df["x"], df["tilt1_p"] / sc, "k", linewidth=1.2)
    ax.plot(df["x"], df["tilt2_p"] / sc, "b", linewidth=1.2)
    ax.plot(df["x"], df["stretch_p"] / sc, "r", linewidth=1.2)
    ax.plot(df["x"], df["dilate_p"] / sc, "g", linewidth=1.2)
    ax.plot(df["x"], df["tilt1_n"] / sc, "k-", linewidth=1.2)
    ax.plot(df["x"], df["tilt2_n"] / sc, "b-", linewidth=1.2)
    ax.plot(df["x"], df["stretch_n"] / sc, "r-", linewidth=1.2)
    ax.plot(df["x"], df["dilate_n"] / sc, "g-", linewidth=1.2)


def partial_term(df, ax, filt=False):
    tilt1 = df["tilt1_p"] + df["tilt1_n"]
    tilt2 = df["tilt2_p"] + df["tilt1_n"]
    stret = df["stretch_p"] + df["stretch_n"]
    dilat = df["dilate_p"] + df["dilate_n"]
    torqu = df["torque_p"] + df["torque_n"]
    sc = np.max([tilt1, tilt2, stret, dilat, torqu])
    if filt == True:
        ws = 17
        order = 3
        tilt1 = savgol_filter(tilt1, ws, order)
        tilt2 = savgol_filter(tilt2, ws, order)
        stret = savgol_filter(stret, ws, order)
        dilat = savgol_filter(dilat, ws, order)
        torqu = savgol_filter(torqu, ws, order)
    ax3.plot(vortex3["x"], tilt1 / sc, "k", linewidth=1.2)
    ax3.plot(vortex3["x"], tilt2 / sc, "b", linewidth=1.2)
    ax3.plot(vortex3["x"], stret / sc, "r", linewidth=1.2)
    ax3.plot(vortex3["x"], dilat / sc, "g", linewidth=1.2)
    # ax3.plot(vortex3['x'], torqu/sc, "C7:", linewidth=1.5)


# %% plot streamwise vorticity balance
fig3, ax3 = plt.subplots(figsize=(6.4, 2.6))
partial_term(vortex3, ax3, filt=True)
# partial_term(vortex3, ax3)
ax3.legend(lab, ncol=2, loc="upper right", fontsize=nsize)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax3.set_ylabel("Term", fontdict=font, fontsize=tsize)
ax3.set_xlim([0.0, 25.0])
# ax3.set_ylim([0.0, 1.2])
ax3.axvline(x=x1, color="gray", linestyle="--", linewidth=1.0)
ax3.axvline(x=x2, color="gray", linestyle="--", linewidth=1.0)
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=nsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "vortex_dynamics_z.pdf", dpi=300)
plt.show()

# %% Control volume analysis
xmin = 8.25
xmax = 12.0
xloc = np.arange(xmin, xmax + 0.125, 0.125)
vortex3 = pd.read_csv(
    pathV + "vortex_z_0950.0.dat", sep=" ", index_col=False, skipinitialspace=True
)
cvol = vortex3.loc[(vortex3["x"] <= xmax) & (vortex3["x"] >= xmin)]
print("x range: ", np.min(cvol["x"]), np.max(cvol["x"]))
barloc = np.arange(5)
barhet1 = np.zeros(5)
barhet2 = np.zeros(5)
barhet1[0] = np.trapz(cvol["tilt1_p"], cvol["x"])
barhet1[1] = np.trapz(cvol["tilt2_p"], cvol["x"])
barhet1[2] = np.trapz(cvol["stretch_p"], cvol["x"])
barhet1[3] = np.trapz(cvol["dilate_p"], cvol["x"])
barhet1[4] = np.trapz(cvol["torque_p"], cvol["x"])
barhet2[0] = np.trapz(cvol["tilt1_n"], cvol["x"])
barhet2[1] = np.trapz(cvol["tilt2_n"], cvol["x"])
barhet2[2] = np.trapz(cvol["stretch_n"], cvol["x"])
barhet2[3] = np.trapz(cvol["dilate_n"], cvol["x"])
barhet2[4] = np.trapz(cvol["torque_n"], cvol["x"])
# barsum = np.sum(np.abs(barhet1)) + np.sum(np.abs(barhet2))
# barhet1 = barhet1/barsum * 100
barsum = np.sum(np.abs(barhet1 + barhet2))
barhet1 = (barhet1 + barhet2) / barsum * 100
barhet2 = barhet2 / barsum * 100
# plot figure of vorticity contribution percent
vlab = [r"$T_y$", r"$T_z$", r"$S$", r"$D$", r"$B$"]
fig, ax = plt.subplots(figsize=(3.2, 2.8))
matplotlib.rc("font", size=tsize)
width = 0.4
ax.bar(barloc, barhet1, width, color="C0", alpha=1.0)
# ax.bar(barloc, barhet2, width, color='C2', alpha=1.0)
ax.set_xticks(barloc)
ax.set_xticklabels(vlab, fontsize=tsize)
ax.set_ylabel(r"$\mathcal{E}_t(\%)$")
ax.grid(b=True, which="both", linestyle=":", linewidth=0.6)
ax.axhline(y=0.0, color="k", linestyle="-", linewidth=1.0)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "vortex_term_zt4_0950.svg", bbox_inches="tight")
plt.show()

# %%############################################################################
#
# Numerical schiliren (gradient of density)
#
# %% Numerical schiliren in X-Y plane
flow = pf()
file = path + "snapshots/" + "TP_2D_Z_001.szplt"  # 'TP_2D_Z_03_01000.00.h5'  #
flow.load_data(path, FileList=file, NameList='tecio')
plane = flow.PlanarData
# plane.to_hdf(path+'snapshots/TP_2D_Z_03_1000.h5', 'w', format='fixed')
x, y = np.meshgrid(np.unique(plane.x), np.unique(plane.y))
# corner = (x < 0.0) & (y < 0.0)
var = "|grad(rho)|"
schlieren = griddata((plane.x, plane.y), plane[var], (x, y))
print("rho_max=", np.max(plane[var]))
print("rho_min=", np.min(plane[var]))
va.sonic_line(plane, pathI, option="velocity", Ma_inf=6.0, mask=None)
va.enthalpy_boundary_edge(plane, pathI, Ma_inf=6.0, crit=0.99)
va.dividing_line(plane, pathI, show=True, mask=None) 
# schlieren[corner] = np.nan
# %%
fig, ax = plt.subplots(figsize=(7.3, 2.3))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(0, 3.0, 21)
cbar = ax.contourf(
    x, y, schlieren, cmap="binary", levels=rg1, extend="both"
)  # binary #rainbow_r# bwr
ax.set_xlim(-200, 80.0)
ax.set_ylim(np.min(y), 36.0)
ax.set_xticks(np.linspace(-200, 80.0, 8))
ax.set_yticks(np.linspace(0.0, 36.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=tsize)
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(0, 3.0, 3)
cbbox = fig.add_axes([0.16, 0.50, 0.20, 0.30], alpha=0.9)
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
cbaxes = fig.add_axes([0.17, 0.71, 0.18, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(
    cbar, cax=cbaxes, orientation="horizontal", extendrect="False", ticks=rg2
)
cbar.set_label(
    r"$|\nabla \rho|$", rotation=0, fontsize=tsize - 1,
)
# Add boundary layer
boundary = pd.read_csv(pathI + "EnthalpyBoundaryEdge99.dat", skipinitialspace=True)
# ax.plot(boundary.x, boundary.y, "g--", linewidth=1.0)
# Add dividing line(separation line)
dividing = pd.read_csv(pathI + "BubbleLine.dat", skipinitialspace=True)
# ax.plot(dividing.x, dividing.y, "g:", linewidth=1.0)
# Add sonic line
sonic = pd.read_csv(pathI + "SonicLine.dat", skipinitialspace=True)
# ax.plot(sonic.x, sonic.y, "b--", linewidth=1.0)
# ax.grid(b=True, which="both", linestyle=":")
plt.savefig(pathF + "SchlierenXY.svg", bbox_inches="tight")
plt.show()
# %% Numerical schiliren in X-Z plane
flow = pf()
file = path + "iso_u0p1.h5"
flow.load_data(path, FileList=file, NameList="h5")
df = flow.PlanarData.query("x<=30.0")
zz = np.linspace(-8.0, 8.0, 257)
xx = np.linspace(-10.0, 30.0, 801)
x, z = np.meshgrid(xx, zz)
var = "schlieren"
rho = griddata((df.x, df.z), df[var], (x, z))
print("rho_max=", np.max(df[var]))
print("rho_min=", np.min(df[var]))
# rho[corner] = np.nan
# %%
fig, ax = plt.subplots(figsize=(6.4, 2.4))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(0, 2.0, 21)
cbar = ax.contourf(x, z, rho, cmap="bwr", levels=rg1, extend="both")  # rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-8.0, 8.0)
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$z/\delta_0$", fontsize=tsize)
ax.set_yticks(np.linspace(-8.0, 8.0, 5))
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar box
rg2 = np.linspace(0, 2.0, 3)
cbbox = fig.add_axes([0.16, 0.16, 0.07, 0.68], alpha=0.9)
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
cbaxes = fig.add_axes([0.17, 0.18, 0.02, 0.55])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(
    cbar, cax=cbaxes, orientation="vertical", extendrect="False", ticks=rg2
)
cbar.set_label(r"$S_c$", rotation=0, fontsize=tsize - 1, labelpad=-16, y=1.15)
plt.savefig(pathF + "SchlierenXZ.svg", bbox_inches="tight")
plt.show()
