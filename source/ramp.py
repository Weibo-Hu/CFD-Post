#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 27 10:39:40 2023
    post-process data for cooling/heating cases

@author: weibo
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

# get_ipython().run_line_magic("matplotlib", "qt")

# %% set path and basic parameters
path = "/media/weibo/VID2/ramp_st14/"
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

# %% mean flow contour in x-y plane
MeanFlow = pf()
MeanFlow.load_meanflow(path)
MeanFlow.copy_meanval()
ind0 = (MeanFlow.PlanarData.y == 0.0)
MeanFlow.PlanarData['walldist'][ind0] = 0.0
# %% merge mean flow
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

# %% wall buondary
va.wall_line(MeanFlow.PlanarData, pathM, mask=None)  # corner)
# %%
walldist = griddata((MeanFlow.x, MeanFlow.y),
                    MeanFlow.walldist, (x, y), method="cubic")
cover = (walldist < 0.0)  # | corner
NewFrame = MeanFlow.PlanarData
va.dividing_line(
    NewFrame, pathM, show=True, mask=cover
)  # (MeanFlow.PlanarData, pathM, loc=2.5)
# %% Save sonic line
va.sonic_line(MeanFlow.PlanarData, pathM,
              option="velocity", Ma_inf=6.0, mask=corner)
# %% enthalpy boundary layer
va.enthalpy_boundary_edge(MeanFlow.PlanarData, pathM, Ma_inf=6.0, crit=0.989)
# %% thermal boundary layer
va.thermal_boundary_edge(MeanFlow.PlanarData, pathM, T_wall=3.35)
# %% Save shock line
va.shock_line(MeanFlow.PlanarData, pathM, var="|gradp|", val=0.005, mask=True)
# %% boundary layer
va.boundary_edge(MeanFlow.PlanarData, pathM, shock=False, mask=False)

# %% check mesh
"""
    Examination of the computational mesh
"""
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
        ax.plot(df_x["x"], df_x["y"], color="gray",
                linestyle="-", linewidth=0.4)
for j in range(np.size(uy)):
    if j % 1 == 0:  # 4
        df_y = df.loc[df["y"] == uy[j]]
        ax.plot(df_y["x"], df_y["y"], color="gray",
                linestyle="-", linewidth=0.4)
# plt.gca().set_aspect("equal", adjustable="box")
ax.set_xlim(np.min(df.x), np.max(df.x))
ax.set_ylim(np.min(df.y), np.max(df.y))
ax.set_yticks(np.linspace(0.0, 30.0, 7))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.set_ylabel(r"$y/l_r$", fontsize=tsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "Grid.svg", bbox_inches="tight")
plt.show()


# %% Plot rho contour of the mean flow field
"""
    mean flow field contouring by density
"""
var = "u"
lh = 1.0
rho = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.rho, (x, y))
print("rho_max=", np.max(MeanFlow.rho))
print("rho_min=", np.min(MeanFlow.rho))
u = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (x, y))
walldist = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.walldist, (x, y))
corner = (walldist < 0.0)
rho[corner] = np.nan
u[corner] = np.nan
cval1 = 0.4  # 1.0 #
cval2 = 3.6  # 7.0  #
fig, ax = plt.subplots(figsize=(15*cm2in, 5*cm2in))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(cval1, cval2, 21)
cbar = ax.contourf(x, y, rho, cmap="rainbow", levels=rg1, extend="both")
ax.contour(x, y, u, levels=[0.0], linestyles="dotted")  # rainbow_r
ax.set_xlim(-200, 80)
ax.set_ylim(np.min(y), 36.0)
ax.set_xticks(np.linspace(-200, 80.0, 8))
ax.set_yticks(np.linspace(np.min(y), 36.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.set_ylabel(r"$y/l_r$", fontsize=tsize)
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cval1, cval2, 3)
cbaxes = fig.add_axes([0.17, 0.76, 0.16, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(
    cbar, cax=cbaxes, extendrect="False", orientation="horizontal", ticks=rg2
)
lab = r"$\langle \rho \rangle/\rho_{\infty}$"
cbar.set_label(lab, rotation=0, fontsize=tsize)
# Add boundary layer
Enthalpy = pd.read_csv(
    pathM + "EnthalpyBoundaryEdge99.dat", skipinitialspace=True)
ax.plot(Enthalpy.x / lh, Enthalpy.y / lh, "k--", linewidth=1.5)
# Add sonic
sonic = pd.read_csv(pathM + "SonicLine.dat", skipinitialspace=True)
ax.plot(sonic.x / lh, sonic.y / lh, "w--", linewidth=1.5)
# Add shock line
shock = pd.read_csv(pathM + "ShockLine.dat", skipinitialspace=True)
ax.plot(shock.x / lh, shock.y / lh, "w-", linewidth=1.5)
# Add dividing line(separation line)
# dividing = pd.read_csv(pathM + "BubbleLine.dat", skipinitialspace=True)
# ax.plot(dividing.x / lh, dividing.y / lh, "k:", linewidth=1.5)
plt.savefig(pathF + "MeanFlow_rho.svg", bbox_inches="tight")
plt.show()

# %% Plot rms contour of the mean flow field
"""
    Root Mean Square of velocity from the statistical flow
"""
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
var = "<w`w`>"
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
uu[corner] = np.nan
print("uu_max=", np.max(np.sqrt(np.abs(var_val))))
print("uu_min=", np.min(np.sqrt(np.abs(var_val))))

fig, ax = plt.subplots(figsize=(15*cm2in, 5*cm2in))
matplotlib.rc("font", size=tsize)
cb1 = 0.0
cb2 = 0.05
rg1 = np.linspace(cb1, cb2, 21)  # 21)
cbar = ax.contourf(
    x / lh, y / lh, np.sqrt(np.abs(uu)), cmap="coolwarm",
    levels=rg1, extend="both"
)  # rainbow_r # jet # Spectral_r
ax.set_xlim(-200, 80.0)
ax.set_ylim(np.min(y), 36.0)
ax.set_xticks(np.linspace(-200.0, 80.0, 8))
ax.set_yticks(np.linspace(0.0, 36.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/l_r$", fontdict=font)
ax.set_ylabel(r"$y/l_r$", fontdict=font)
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar box
rg2 = np.linspace(cb1, cb2, 3)
cbbox = fig.add_axes([0.14, 0.53, 0.22, 0.31], alpha=0.9)
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
cbaxes = fig.add_axes([0.16, 0.78, 0.18, 0.058])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(
    cbar, cax=cbaxes, orientation="horizontal", extendrect="False", ticks=rg2
)

cbar.set_label(
    lab, rotation=0, fontsize=tsize,
)
# Add boundary layer
boundary = pd.read_csv(
    pathM + "EnthalpyBoundaryEdge99.dat", skipinitialspace=True)
ax.plot(boundary.x / lh, boundary.y / lh, "k--", linewidth=1.0)
# Add bubble line
bubble = pd.read_csv(pathM + "BubbleLine.dat", skipinitialspace=True)
ax.plot(bubble.x / lh, bubble.y / lh, "k:", linewidth=1.0)
# Add shock line
shock = pd.read_csv(pathM + "ShockLine.dat", skipinitialspace=True)
ax.plot(shock.x / lh, shock.y / lh, "w-", linewidth=1.5)
# Add sonic line
sonic = pd.read_csv(pathM + "SonicLine.dat", skipinitialspace=True)
ax.plot(sonic.x / lh, sonic.y / lh, "w--", linewidth=1.0)
plt.savefig(pathF + "MeanFlow" + nm + ".svg", bbox_inches="tight")
plt.show()


# %% Numerical schiliren in X-Y plane
"""
    Numerical schiliren (gradient of density)
"""
flow = pf()
file = path + "snapshots/" + "TP_2D_Z_001.szplt"
flow.load_data(path, FileList=file, NameList='tecio')
plane = flow.PlanarData
# plane.to_hdf(path+'snapshots/TP_2D_Z_03_1000.h5', 'w', format='fixed')
x, y = np.meshgrid(np.unique(plane.x), np.unique(plane.y))
var = "|grad(rho)|"
schlieren = griddata((plane.x, plane.y), plane[var], (x, y))
walldist = griddata((plane.x, plane.y), plane['walldist'], (x, y))
print("rho_max=", np.max(plane[var]))
print("rho_min=", np.min(plane[var]))
va.sonic_line(plane, pathI, option="velocity", Ma_inf=6.0, mask=None)
va.enthalpy_boundary_edge(plane, pathI, Ma_inf=6.0, crit=0.98)
va.dividing_line(plane, pathI, show=True, mask=None)
va.shock_line(plane, pathI, var="|grad(rho)|", val=0.005, mask=True)
# schlieren[corner] = np.nan
# %%
fig, ax = plt.subplots(figsize=(15*cm2in, 5*cm2in))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(0, 3.0, 21)
corner = (walldist < 0.0)
schlieren[corner] = np.nan
cbar = ax.contourf(
    x, y, schlieren, cmap="binary", levels=rg1, extend="both"
)  # binary #rainbow_r# bwr
ax.set_xlim(-200, 80.0)
ax.set_ylim(np.min(y), 36.0)
ax.set_xticks(np.linspace(-200, 80.0, 8))
ax.set_yticks(np.linspace(0.0, 36.0, 5))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.set_ylabel(r"$y/l_r$", fontsize=tsize)
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
boundary = pd.read_csv(
    pathI + "EnthalpyBoundaryEdge.dat", skipinitialspace=True)
# ax.plot(boundary.x, boundary.y, "g--", linewidth=1.0)
# Add dividing line(separation line)
dividing = pd.read_csv(pathI + "BubbleLine.dat", skipinitialspace=True)
# ax.plot(dividing.x, dividing.y, "g:", linewidth=1.0)
# Add sonic line
sonic = pd.read_csv(pathI + "SonicLine.dat", skipinitialspace=True)
# ax.plot(sonic.x, sonic.y, "b--", linewidth=1.0)
plt.savefig(pathF + "SchlierenXY.svg", bbox_inches="tight")
plt.show()
# %% BL profile along streamwise
fig, ax = plt.subplots(1, 9, figsize=(13 * cm2in, 4 * cm2in), dpi=500)
fig.subplots_adjust(hspace=0.5, wspace=0.25)
matplotlib.rc("font", size=nsize)
title = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$"]
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
xcoord = np.array([-120, -80, -40, -20, 0, 20, 40, 60, 80])
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
    ax[i].set_xticks([0, 1], minor=True)
    ax[i].tick_params(axis="both", which="major", labelsize=nsize)
    ax[i].grid(visible=True, which="both", linestyle=":")
ax[0].set_title(r"$x={}$".format(xcoord[0]), fontsize=nsize - 2)
ax[0].set_ylabel(r"$\Delta y$", fontsize=tsize)
ax[4].set_xlabel(r"$u /u_\infty$", fontsize=tsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "BLProfileU.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% dataframe of the first level points
# corner = MeanFlow.PlanarData.walldist < 0.0
firstval = 0.015625
ind0 = (MeanFlow.PlanarData.y == 0.0)
MeanFlow.PlanarData['walldist'][ind0] = 0.0
xy_val = va.wall_line(MeanFlow.PlanarData, path, val=firstval)  # mask=corner)
# onelev = pd.DataFrame(data=xy_val, columns=["x", "y"])
# onelev.drop_duplicates(subset='x', keep='first', inplace=True)
FirstLev = pd.DataFrame(data=xy_val, columns=["x", "y"])
# ind = FirstLev.index[FirstLev['x'] < 0]
# FirstLev.loc[ind, 'y'] = firstval
xx = np.arange(-200.0, 80.0, 0.25)
FirstLev = FirstLev[FirstLev["x"].isin(xx)]
FirstLev.to_csv(pathM + "FirstLev.dat", index=False, float_format="%9.6f")
# %%
firstval = 0.015625
T_inf = 86.6
Re = 7736
df = MeanFlow.PlanarData
ramp_wall = pd.read_csv(pathM + "FirstLev.dat", skipinitialspace=True)
ramp_wall = va.add_variable(df, ramp_wall)
ramp1 = ramp_wall.loc[ramp_wall['x'] <= 0.0]
ramp2 = ramp_wall.loc[ramp_wall['x'] > 0.0]
# for flat plate
mu = va.viscosity(Re, ramp1["T"])
xwall1 = ramp1['x']
Cf1 = va.skinfriction(mu, ramp1['u'], firstval, factor=1)
# for ramp
angle = 15 / 180 * np.pi
mu = va.viscosity(Re, ramp2["T"])
ramp2_u = ramp2['u'] * np.cos(angle) + ramp2['v'] * np.sin(angle)
xwall2 = ramp2['x']
Cf2 = va.skinfriction(mu, ramp2_u, firstval*np.cos(angle), factor=1)
ind = np.argmin(np.abs(Cf1[:-20]))
xline1 = xwall1[ind]
xline2 = np.round(np.interp(0.0, Cf2, xwall2), 2)
yline = np.max(Cf2) * 0.88
# xwall, Cf = va.skinfric_wavy(pathM, wavy, Re, T_inf, firstval)
# %% skin friction
# xw1_fit, Cf1_fit = va.curve_fit(xwall1, Cf1, [-72.0, -46.0, -2.5], deg=[3, 4, 5, 4])
# xw2_fit, Cf2_fit = va.curve_fit(xwall2, Cf2, [20.0, 60.0], deg=[3, 4, 5])
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.25)
Cf1_fit = signal.filtfilt(b, a, Cf1)
Cf2_fit = signal.filtfilt(b, a, Cf2)
fig2, ax2 = plt.subplots(figsize=(15*cm2in, 5*cm2in), dpi=500)
matplotlib.rc("font", size=nsize)
# ax2.plot(xwall1, Cf1, "b--", linewidth=1.5)
ax2.plot(xwall1, Cf1_fit, "k", linewidth=1.5)
# ax2.plot(xwall2, Cf2, "b--", linewidth=1.5)
ax2.plot(xwall2, Cf2_fit, "k", linewidth=1.5)
ax2.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=tsize)
ax2.set_xlim([-200.0, 80.0])
ax2.set_xticks(np.linspace(-200.0, 80.0, 8))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
ax2.annotate(r"$x_s={}$".format(xline1), (xline1-16, yline))
ax2.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
ax2.annotate(r"$x_r={}$".format(xline2), (xline2-16, yline))
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cf.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Cp
Ma = 6.0
fa = 1.0  # Ma * Ma * 1.4
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.25)
Cp_fit = signal.filtfilt(b, a, ramp_wall["p"] * fa)
fig3, ax3 = plt.subplots(figsize=(15*cm2in, 5*cm2in), dpi=500)
# ax3 = fig.add_subplot(212)
# ax3.plot(ramp_wall['x'], ramp_wall["p"] * fa, "b--", linewidth=1.5)
ax3.plot(ramp_wall["x"], Cp_fit * fa, "k", linewidth=1.5)
ax3.set_xlabel(r"$x/l_r$", fontsize=tsize)
# ylab = r"$\langle p_w \rangle/\rho_{\infty} u_{\infty}^2$"
ax3.set_ylabel(r"$\langle C_p \rangle$", fontsize=tsize)
ax3.set_xlim([-200.0, 80])
ax3.set_xticks(np.arange(-200.0, 80.0, 40))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
# ax3.annotate(r"$x_s={}$".format(xline1), (xline1-16, yline))
ax3.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
# ax3.annotate(r"$x_r={}$".format(xline2), (xline2-16, yline))
ax3.grid(visible=True, which="both", linestyle=":")
ax3.annotate("(b)", xy=(-0.13, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cp.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% Stanton number
firstval = 0.015625
T_wall = 3.35
mu = va.viscosity(Re, ramp_wall["T"], T_inf=T_inf, law="Suther")
kt = va.thermal(mu, 0.72)
Tt = va.stat2tot(Ma=6.0, Ts=1.0, opt="t")
Cs = va.Stanton(ramp_wall["T"], T_wall, firstval,
                Re, Ma, T_inf=1.0, factor=1)

# %% low-pass filter
xwall = ramp_wall['x']
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.25)
Cs_fit = signal.filtfilt(b, a, Cs)
# ind = np.where(Cf[:] < 0.008)
fig2, ax2 = plt.subplots(figsize=(15*cm2in, 5*cm2in), dpi=500)
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
# ax2.plot(xwall, np.abs(Cs), "b--", linewidth=1.5)
ax2.plot(xwall, np.abs(Cs_fit), "k", linewidth=1.5)
ax2.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_s \rangle$", fontsize=tsize)
ax2.set_xlim([-200, 80])
ax2.set_xticks(np.arange(-200.0, 80.0, 40))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(c)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "Cs.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Time evolution of a specific variable at several streamwise locations
"""
    temporal evolution of signals along an axis
"""
probe = lf()
lh = 1.0
fa = 1.0
var = "u"
ylab = r"$u^\prime/u_\infty$"
if var == "p":
    fa = 6.0 * 6.0 * 1.4
    ylab = r"$p^\prime/p_\infty$"
timezone = np.arange(600.0, 900.0 + 0.125, 0.125)
xloc = [-160.0, -80.0, -40.0, -20.0, 0.0]  # [-50.0, -30.0, -20.0, -10.0, 4.0]
yloc = [0, 0, 0, 0, 0]
zloc = 0.0
fig, ax = plt.subplots(np.size(xloc), 1, figsize=(15*cm2in, 10*cm2in))
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
    ax[i].set_title(r"$x/l_r={}$".format(xloc[i]), fontsize=nsize - 1)
ax[-1].set_xlabel(r"$t u_\infty/\delta_0$", fontsize=tsize)

plt.savefig(
    pathF + var + "_TimeEvolveX" + str(zloc) + ".svg", bbox_inches="tight"
)
plt.show()

# %%############################################################################


# %% save and load data
"""
    streamwise evolution of signals along an axis
"""
df, SolTime = pytec.ReadINCAResults(path + 'TP_fluc_3d/',
                                    SavePath=pathI,
                                    SpanAve=False,
                                    OutFile='TP_fluc_2d')

# %%
MeanFlow = pf()
MeanFlow.merge_stat(pathI)
# %% Streamwise evolution of a specific variable
fa = 1.0  # 1.7 * 1.7 * 1.4
var = "p`"
df = pd.read_hdf(pathI + "TP_fluc_2d.h5")
ramp_wall = pd.read_csv(pathM + "FirstLev.dat", skipinitialspace=True)
ramp_wall = va.add_variable(df, ramp_wall)
ramp_wall[var] = griddata((df.x, df.y), df[var],
                          (ramp_wall.x, ramp_wall.y),
                          method="linear")

meanval = np.mean(ramp_wall[var])
fig, ax = plt.subplots(figsize=(15*cm2in, 5*cm2in))
matplotlib.rc("font", size=nsize)
ax.plot(ramp_wall.x, ramp_wall[var] - meanval, "k")
ax.set_xlim([-200.0, 80.0])
ax.set_xticks(np.linspace(-200.0, 80.0, 8))
# ax.set_ylim([-0.001, 0.001])
ax.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
ax.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.set_ylabel(r"$2p^\prime/\rho_\infty u_\infty^2$", fontsize=tsize)
ax.ticklabel_format(axis="y", style="sci", useOffset=False, scilimits=(-2, 2))
ax.grid(visible="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + var + "_streamwise.svg", bbox_inches="tight", pad_inches=0.1
)

# %% Streamwise evolution of a specific variable
fa = 1.0  # 1.7 * 1.7 * 1.4
var = "u`"
ramp_wall[var] = griddata((df.x, df.y), df[var],
                          (ramp_wall.x, ramp_wall.y),
                          method="linear")
meanval = np.mean(ramp_wall[var])
fig, ax = plt.subplots(figsize=(15*cm2in, 5*cm2in))
matplotlib.rc("font", size=nsize)
ax.plot(ramp_wall.x, ramp_wall[var] - meanval, "k")
ax.set_xlim([-200.0, 80.0])
ax.set_xticks(np.linspace(-200.0, 80.0, 8))
# ax.set_ylim([-0.001, 0.001])
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
# ax.set_ylabel(r"$p^\prime/\rho_\infty u_\infty^2$", fontsize=tsize)
ax.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
ax.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
ax.set_ylabel(r"$u^\prime/u_\infty$", fontsize=tsize)
ax.ticklabel_format(axis="y", style="sci", useOffset=False, scilimits=(-2, 2))
ax.grid(visible="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + var + "_streamwise.svg", bbox_inches="tight", pad_inches=0.1
)

# %% the locations with most energetic fluctuations
"""
    streamwise evolution of the most energetic signals
"""
FlucFlow = pd.read_hdf(pathM + "MeanFlow.h5")
varn = '<p`p`>'
if varn == '<u`u`>':
    savenm = "MaxRMS_u"
    ylab = r"$\sqrt{u^{\prime 2}_\mathrm{max}}/u_{\infty}$"
elif varn == '<p`p`>':
    savenm = "MaxRMS_p"
    ylab = r"$\sqrt{p^{\prime 2}_\mathrm{max}}/\rho_{\infty} u_{\infty}^2$"
xnew = np.arange(-200.0, 80.0, 0.25)
znew = np.zeros(np.size(xnew))
varv = np.zeros(np.size(xnew))
ynew = np.zeros(np.size(xnew))
for i in range(np.size(xnew)):
    df = va.max_pert_along_y(FlucFlow, varn, [xnew[i], znew[i]])
    varv[i] = df[varn]
    ynew[i] = df['y']
data = np.vstack((xnew, ynew, varv))
df = pd.DataFrame(data.T, columns=['x', 'y', varn])
# df = df.drop_duplicates(keep='last')
df.to_csv(pathM + savenm + ".dat", sep=' ',
          float_format='%1.8e', index=False)

# %% draw maximum value curve
fig, ax = plt.subplots(figsize=(15*cm2in, 5*cm2in))
matplotlib.rc('font', size=nsize)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.set_ylabel(r"$y/l_r$", fontsize=tsize)
ax.plot(xnew, ynew, 'k', label=r'$q^\prime_\mathrm{max}$')
ax.set_xticks(np.linspace(-200.0, 80.0, 8))
# ax.plot(xx, yy, 'k--', label='bubble')
legend = ax.legend(loc='upper left', shadow=False, fontsize=nsize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(visible=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "MaxPertLoc" + savenm + ".svg", bbox_inches="tight", pad_inches=0.1
)

# %% draw maximum value curve
ramp_max = pd.read_csv(pathM + savenm + ".dat", sep=' ', skipinitialspace=True)
fig, ax = plt.subplots(figsize=(15*cm2in, 5.0*cm2in))
matplotlib.rc('font', size=nsize)
ax.set_ylabel(ylab, fontsize=tsize)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.plot(ramp_max['x'], ramp_max[varn], 'k')
ax.set_xlim([-200.0, 80])
# ax.set_ylim([1e-7, 1e-1])
ax.set_xticks(np.linspace(-200.0, 80.0, 8))
ax.set_yscale('log')
# ax.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
# ax.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
ax.grid(visible=True, which="both", linestyle=":")
ax.tick_params(labelsize=nsize)
plt.show()
plt.savefig(
    pathF + "MaxFluc" + savenm + ".svg", bbox_inches="tight", pad_inches=0.1
)
# %% test section
thick = pd.read_csv(pathM + 'fit_thickness.dat', sep=' ')
shape_fa = thick['displace'] / thick['momentum']
xynew = pd.read_csv(pathM + 'MaxRMS_u.dat', sep=' ')
fig, ax = plt.subplots(figsize=(6.4, 2.2))
matplotlib.rc('font', size=nsize)
ylab = r"$\sqrt{u^{\prime 2}_\mathrm{max}}/u_{\infty}$"
ax.set_ylabel(ylab, fontsize=tsize)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.plot(xynew['x'], xynew['<u`u`>'], 'k')
ax.set_yscale('log')
ax.set_xlim([-100, 20])
# ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.grid(b=True, which="both", linestyle=":")
ax.tick_params(labelsize=nsize)
ax.yaxis.offsetText.set_fontsize(nsize-1)

ax2 = ax.twinx()
ax2.plot(thick['x'], shape_fa, 'k--')
ax2.set_ylabel(r"$H$", fontsize=tsize)

plt.show()
plt.savefig(
    pathF + "MaxRMS_x.svg", bbox_inches="tight", pad_inches=0.1
)


# %% Draw impose mode
inmode = pd.read_csv(path+"UnstableMode.inp", skiprows=5,
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
    pathF + "ModeProf.svg", bbox_inches="tight", pad_inches=0.1
)

# %% szplt to h5
dirs = sorted(os.listdir(pathSN))
var_list = ['x', 'y', 'z', 'rho', 'u', 'v', 'w', 'p', 'T',
            'walldist', '|grad(rho)|', 'Q-criterion']
stype = 'TP_2D_Z_001'
pathZ = path + stype + '/'
for i in range(np.size(dirs)):
    finm = pathSN + dirs[i] + '/' + stype + '.szplt'
    df, s_time = pytec.ReadSinglePlt(finm, var_list)
    # df, s_time = p2p.ReadINCAResults(pathSN,  var_list, FileName=finm)
    outfile = stype + "_%07.2f" % s_time
    df.to_hdf(pathZ + outfile + ".h5", "w", format='fixed')

# %% collect snapshots: method1 (bad results)
dirs = sorted(os.listdir(pathZ))
# preprocess
data = pd.read_hdf(pathZ + dirs[0])
data.reset_index(drop=True, inplace=True)
data['ind'] = data.index
xx = np.arange(-200.0, 80.0, 0.25) + 0.125
data1 = data[data["x"].isin(xx)]
data2 = data1.groupby("x", as_index=False, sort=True).nth(0)
data2.sort_values(by='x', inplace=True)
ind = data2.index
xval = data.iloc[ind]['x'].values
yval = data.iloc[ind]['y'].values
snapshots = data.iloc[ind]['p']
for i in range(np.size(dirs)):
    df = pd.read_hdf(pathZ + dirs[i])
    df.reset_index(drop=True, inplace=True)
    df_temp = df.iloc[ind]['p']
    if i != 0:
        snapshots = np.vstack((snapshots, df_temp))
xyval = pd.DataFrame(data=np.vstack((xval, yval)).T, columns=["x", "y"])
xyval.to_csv(pathM + "snapshots_p1xy.dat", index=False, float_format="%9.6f")
np.save(path + 'snapshots_p1', snapshots)
# %% collect snapshots: method2 (best results, low performance)
timez = np.arange(600.0, 900.0 + 0.25, 0.25)
dirs = sorted(os.listdir(pathZ))
xx = np.arange(-200.0, 80.0, 0.25)
firstval = 0.015625
ramp_wall = pd.read_csv(pathM + "FirstLev.dat", skipinitialspace=True)
xval = ramp_wall.x
data = pd.read_hdf(pathZ + dirs[0])[['x', 'y', 'p', 'walldist']]
data = data.query("walldist < 1.0")
snapshots = va.add_variable(data, ramp_wall, nms=['p'])['p']
for i in range(np.size(dirs)):
    df = pd.read_hdf(pathZ + dirs[i])[['x', 'y', 'p', 'walldist']]
    df = df.query("walldist < 1.0")
    df_temp = va.add_variable(df, ramp_wall, nms=['p'])['p']
    if i != 0:
        snapshots = np.vstack((snapshots, df_temp))
np.save(path + 'snapshots_p2', snapshots)
# %% collect snapshots: method3
data = pd.read_hdf(pathZ + dirs[0])[['x', 'y', 'p', 'walldist']]
ind0 = (data.y == 0.0)
data['walldist'][ind0] = 0.0
data.reset_index(drop=True, inplace=True)
data0 = data.query("walldist < 0.5 & walldist >= 0")
data1 = data0.sort_values(by=['x', 'walldist'], inplace=False)
data2 = data1.groupby(["x"], as_index=False, sort=True).nth(0)
ind = data2.index
xval = data.iloc[ind]['x'].values
yval = data.iloc[ind]['y'].values
snapshots = data2['p']  # va.add_variable(data, ramp_wall, nms=['p'])['p']
for i in range(np.size(dirs)):
    df = pd.read_hdf(pathZ + dirs[i])['p']
    df.reset_index(drop=True, inplace=True)
    df_temp = df.iloc[ind]
    if i != 0:
        snapshots = np.vstack((snapshots, df_temp))
xyval = pd.DataFrame(data=np.vstack((xval, yval)).T, columns=["x", "y"])
xyval.to_csv(pathM + "snapshots_p3xy.dat", index=False, float_format="%9.6f")
np.save(path + 'snapshots_p3', snapshots)
# %%
xyval = pd.read_csv(pathM + "FirstLev.dat", skipinitialspace=True)
xval = xyval.x
snapshots = np.load(path + 'snapshots_p2.npy')
timez = np.linspace(600.0, 900.0, 1201)
press_in = snapshots[:, 0]
# xval = ramp_wall.x
# yval = ramp_wall.y
intermit = np.zeros(np.size(xval))
for i in range(np.size(xval)):
    press_wa = snapshots[:, i]
    intermit[i] = va.intermittency(press_in, press_wa, timez)
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.25)
inter_fit = signal.filtfilt(b, a, intermit)
# %%
b, a = signal.butter(2, Wn=1/6, fs=1/0.25)
inter_fit = signal.filtfilt(b, a, intermit)
fig, ax = plt.subplots(figsize=(15*cm2in, 5*cm2in), dpi=300)
matplotlib.rc("font", size=nsize)
# ax.plot(xval, yval, "b--", linewidth=1.5)
ax.plot(xval, intermit, "b:", linewidth=1.5)
# ax.plot(xval, inter_fit, "k--", linewidth=1.5)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.set_ylabel(r"$\gamma$", fontsize=tsize)
ax.set_xlim([-200, 80])
ax.set_xticks(np.arange(-200.0, 80.0, 40))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
# ax.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
ax.grid(visible=True, which="both", linestyle=":")
ax.yaxis.offsetText.set_fontsize(nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "gamma.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Fourier transform of variables
from scipy import fft
t_samp = 0.25
Nt = 1201
i1 = np.where(np.round(xval, 2)==-180)[0][0]  # 20.0
i2 = np.where(np.round(xval, 2)==-80.0)[0][0]  # 40.25
i3 = np.where(np.round(xval, 2)==-40.0)[0][0]  # 60.0
i4 = np.where(np.round(xval, 2)==0.0)[0][0]    # 79.75

p_fft1 = fft.fft(snapshots[:, i1]-np.mean(snapshots[:, i1]))
p_fft2 = fft.fft(snapshots[:, i2]-np.mean(snapshots[:, i2]))
p_fft3 = fft.fft(snapshots[:, i3]-np.mean(snapshots[:, i3]))
p_fft4 = fft.fft(snapshots[:, i4]-np.mean(snapshots[:, i4]))

# p_freq = fft.fftfreq(Nt, t_samp)
p_freq = np.linspace(1/t_samp/Nt, 1/t_samp/2, Nt//2)
fig, ax = plt.subplots(figsize=(14*cm2in, 4.5*cm2in), dpi=300)
matplotlib.rc("font", size=nsize)
# ax.vlines(p_fre[1:Nt//2], [0], np.abs(p_fft)[1:Nt//2])
ax.plot(p_freq[:Nt//2], np.abs(p_fft1)[1:Nt//2+1], "k--", linewidth=1.2)
ax.plot(p_freq[:Nt//2], np.abs(p_fft2)[1:Nt//2+1], "r--", linewidth=1.2)
ax.plot(p_freq[:Nt//2], np.abs(p_fft3)[1:Nt//2+1], "g--", linewidth=1.2)
ax.plot(p_freq[:Nt//2], np.abs(p_fft4)[1:Nt//2+1], "b--", linewidth=1.2)

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend([r'$-180$', r'$-80$', r'$-40$', r'$0$'],
          loc='upper right', fontsize=nsize-2,
          ncols=2, columnspacing=0.6, framealpha=0.4)
# ax.scatter(p_fre[1:Nt//2], np.abs(p_fft)[1:Nt//2], marker='o', facecolor=None, edgecolors='gray', s=15)
# ax.plot(xval, inter_fit, "k--", linewidth=1.5)
ax.set_xlabel(r"$f$", fontsize=tsize)
ax.set_ylabel(r"$A_p$", fontsize=tsize)
ax.set_xlim([3e-3, 3])
# ax.set_xticks(np.arange(-200.0, 80.0, 40))
# ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
# ax.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
ax.grid(visible=True, which="both", linestyle=":")
ax.yaxis.offsetText.set_fontsize(nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathF + "p_fft_upstream.svg", bbox_inches="tight", pad_inches=0.1)
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
    frame.position = (-1.4, 0.25)
    # tp.macro.execute_file('/path/to/macro_file.mcr')
    frame.load_stylesheet(path + "L2_2021.sty")
    tp.macro.execute_command("$!Interface ZoneBoundingBoxMode = Off")
    tp.macro.execute_command("$!FrameLayout ShowBorder = No")
    tp.export.save_png(pathF + str(soltime) + ".png", width=2048)
# %% create videos

pathani = path + "ani/"
prenm = "ramp"
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
