#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:39:31 2018
    Plot for time-averaged flow (3D meanflow)
@author: Weibo Hu
"""
# %%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import matplotlib
import pandas as pd
import plt2pandas as p2p
import variable_analysis as fv
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

plt.close("All")
plt.rc("text", usetex=True)
font = {"family": "Times New Roman", "color": "k", "weight": "normal"}

path = "/media/weibo/VID2/BFS_M1.7TS/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + "Vortex/"
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
textsize = 13
numsize = 10
matplotlib.rc("font", size=textsize)

# %% Load Data for time- spanwise-averaged results
# filter files
FileId = pd.read_csv(path + "StatList.dat", sep='\t')
filelist = FileId['name'].to_list()
pltlist = [os.path.join(path + 'TP_stat/', name) for name in filelist]
MeanFlow = pf()
MeanFlow.load_meanflow(path, FileList=pltlist)
    
# %% Load Data for time- spanwise-averaged results
MeanFlow = pf()
MeanFlow.load_meanflow(path)

# %%############################################################################
"""
    Examination of the computational mesh
"""
# %% check mesh 
temp = MeanFlow.PlanarData[['x', 'y']]
df = temp.query("x>=-5.0 & x<=5.0 & y>=-3.0 & y<=1.0")
# df = temp.query("x>=-5.0 & x<=10.0 & y>=-3.0 & y<=2.0")
ux = np.unique(df.x)
uy = np.unique(df.y)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
matplotlib.rc("font", size=textsize)
for i in range(np.size(ux)):
    if i % 1 == 0: # 2
        df_x = df.loc[df['x']==ux[i]]
        ax.plot(df_x['x'], df_x['y'], 'k-', linewidth=0.4)
for j in range(np.size(uy)):
    if j % 1 == 0: # 4
        df_y = df.loc[df['y']==uy[j]]
        ax.plot(df_y['x'], df_y['y'], 'k-', linewidth=0.4)
plt.gca().set_aspect("equal", adjustable="box")
ax.set_xlim(-5.0, 10.0)
ax.set_ylim(-3.0, 2.0)
ax.set_xticks(np.linspace(-5.0, 5.0, 5))
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "Grid.svg", bbox_inches="tight")
plt.show()
    

# %%
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
corner = (x < 0.0) & (y < 0.0)

# %% Mean flow isolines
fv.dividing_line(MeanFlow.PlanarData, pathM)
# %% Save sonic line
fv.sonic_line(MeanFlow.PlanarData, pathM, option='velocity', Ma_inf=1.7)
# %% Save boundary layer
fv.boundary_edge(MeanFlow.PlanarData, pathM)
# %% Save shock line
fv.shock_line(MeanFlow.PlanarData, pathM)
# %% 
plt.close("All")

# %%############################################################################
"""
    mean flow field contouring by density
"""
# %% Plot rho contour of the mean flow field
# MeanFlow.AddVariable('rho', 1.7**2*1.4*MeanFlow.p/MeanFlow.T)
MeanFlow.copy_meanval()
var = 'rho'
rho = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.rho, (x, y))
print("rho_max=", np.max(MeanFlow.rho_m))
print("rho_min=", np.min(MeanFlow.rho_m))
rho[corner] = np.nan
fig, ax = plt.subplots(figsize=(6.4, 2.2))
matplotlib.rc("font", size=textsize)
rg1 = np.linspace(0.33, 1.03, 41)
cbar = ax.contourf(x, y, rho, cmap="rainbow", levels=rg1)  # rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(0.33, 1.03, 3)
cbaxes = fig.add_axes([0.17, 0.75, 0.18, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar.set_label(
    r"$\langle \rho \rangle/\rho_{\infty}$", rotation=0, fontdict=font
)
# Add shock wave
shock = np.loadtxt(pathM + "ShockLineFit.dat", skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], "w", linewidth=1.5)
# shock1 = np.loadtxt(pathM + "ShockLine1.dat", skiprows=1)
# ax.plot(shock1[:, 0], shock1[:, 1], "w", linewidth=1.5)
# shock2 = np.loadtxt(pathM + "ShockLine2.dat", skiprows=1)
# ax.plot(shock2[:, 0], shock2[:, 1], "w", linewidth=1.5)
# Add sonic line
sonic = np.loadtxt(pathM + "SonicLine.dat", skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], "w--", linewidth=1.5)
# Add boundary layer
boundary = np.loadtxt(pathM + "BoundaryEdge.dat", skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], "k", linewidth=1.5)
# Add dividing line(separation line)
dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], "k--", linewidth=1.5)
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
ax.streamplot(
    xbox,
    ybox,
    u,
    v,
    color="w",
    density=[3.0, 2.0],
    arrowsize=0.7,
    start_points=seeds.T,
    maxlength=30.0,
    linewidth=1.0,
)

plt.savefig(pathF + "MeanFlow.svg", bbox_inches="tight")
plt.show()

# %%############################################################################
"""
    Root Mean Square of velocity from the statistical flow
"""
# %% Plot rms contour of the mean flow field
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
corner = (x < 0.0) & (y < 0.0)
var = "R22"
uu = griddata((MeanFlow.x, MeanFlow.y), getattr(MeanFlow, var), (x, y))
print("uu_max=", np.max(np.sqrt(np.abs(getattr(MeanFlow, var)))))
print("uu_min=", np.min(np.sqrt(np.abs(getattr(MeanFlow, var)))))
corner = (x < 0.0) & (y < 0.0)
uu[corner] = np.nan
fig, ax = plt.subplots(figsize=(6.4, 2.2))
matplotlib.rc("font", size=textsize)
cb1 = 0.0
cb2 = 0.2
rg1 = np.linspace(cb1, cb2, 21)
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
rg2 = np.linspace(cb1, cb2, 3)
cbbox = fig.add_axes([0.14, 0.52, 0.24, 0.32], alpha=0.9)
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
cbaxes = fig.add_axes([0.17, 0.75, 0.18, 0.07])  # x, y, width, height
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

plt.savefig(pathF + "MeanFlowRMSVV.svg", bbox_inches="tight")
plt.show()

# %%############################################################################
"""
    Vorticity contour
"""
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

#%% Load data
fluc_flow = tf()
fluc_flow.load_3data(
        path, FileList=path+"/ZFluctuation_900.0.h5", NameList='h5'
)
# %% Plot contour of the fluctuation flow field in the xy plane
zslice = fluc_flow.TriData.loc[fluc_flow.TriData['z']==0.0]
x, y = np.meshgrid(np.unique(zslice.x), np.unique(zslice.y))
omegaz = griddata((zslice.x, zslice.y), zslice.vorticity_3, (x, y))
corner = (x < 0.0) & (y < 0.0)
omegaz[corner] = np.nan
fig, ax = plt.subplots(figsize=(6.4, 3.2))
matplotlib.rc("font", size=textsize)
plt.tick_params(labelsize=numsize)
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
    (1.0, -1.5),
    4.0,
    1.3,
    linewidth=1,
    linestyle="--",
    edgecolor="k",
    facecolor="none",
)
# ax.add_patch(box)
plt.tick_params(labelsize=numsize)
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
MeanFlowY = MeanFlow.DataTab.loc[MeanFlow.DataTab["y"] == -1.5].reset_index(
    drop=True
)
MeanFlowY.to_hdf(path + "MeanFlowYN3.h5", "w", format="fixed")
MeanFlowY = DataPost()
MeanFlowY.UserDataBin(path + "MeanFlowYN3.h5")
# %% Plot contour of the mean flow field in the xz plane
x, z = np.meshgrid(np.unique(MeanFlowY.x), np.unique(MeanFlowY.z))
omegaz = griddata((MeanFlowY.x, MeanFlowY.z), MeanFlowY.vorticity_1, (x, z))
fig, ax = plt.subplots(figsize=(5, 3))
matplotlib.rc("font", size=textsize)
plt.tick_params(labelsize=numsize)
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
# plt.tick_params(labelsize=numsize)
# cax1.xaxis.set_ticks_position("top")
# cbar.ax.tick_params(labelsize=numsize)
# ax.set_aspect('auto')
# cbar.set_label(r'$\omega_x$', rotation=0, fontdict=font)
# Add isolines of Lambda2 criterion
# L2 = griddata((MeanFlowY.x, MeanFlowY.z), MeanFlowY.L2crit, (x, z))
# ax.contour(x, z, L2, levels=-0.001, \
#           linewidths=1.2, colors='k',linestyles='solid')
plt.show()
plt.savefig(pathF + "Vorticity1XZ.svg", bbox_inches="tight")

# %% Plot contour of the mean flow field in the zy plane
xslice = fluc_flow.TriData.loc[fluc_flow.TriData['x']==-0.5]
z, y = np.meshgrid(np.unique(xslice.z), np.unique(xslice.y))
omega1 = griddata((xslice.z, xslice.y), xslice.vorticity_3, (z, y))
fig, ax = plt.subplots(figsize=(3.2, 2.2))
matplotlib.rc("font", size=textsize)
plt.tick_params(labelsize=numsize)
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
plt.tick_params(labelsize=numsize)
cax1.xaxis.set_ticks_position("top")
cbar.set_label(r"$\omega_z$", rotation=0, fontsize=textsize)
# Add streamlines
#zz = np.linspace(-8.0, 8.0, 20)
#yy = np.linspace(0.0, 1.0, 10)
#zbox, ybox = np.meshgrid(zz, yy)
#v = griddata((xslice.z, xslice.y), xslice.v, (zbox, ybox))
#w = griddata((xslice.z, xslice.y), xslice.w, (zbox, ybox))
#ax.set_xlim(-8.0, 8.0)
#ax.set_ylim(0.0, 1.0)
#ax.streamplot(
#    zbox,
#    ybox,
#    w,
#    v,
#    density=[2.5, 1.5],
#    color="w",
#    linewidth=1.0,
#    integration_direction="both",
#)
plt.show()
plt.savefig(pathF + "vorticity3_x=0.1.svg", bbox_inches="tight", pad_inches=0.1)

# %%############################################################################
"""
    velocity contour and streamlines
"""
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
matplotlib.rc("font", size=textsize)
plt.tick_params(labelsize=numsize)
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
# cbar.ax.set_title(r'$u/u_\infty$', fontsize=textsize)
cbar.set_label(
    r"$u/u_\infty$", rotation=0, x=-0.28, labelpad=-28, fontsize=textsize
)
plt.tick_params(labelsize=numsize)
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
plt.savefig(
    pathF + "StreamVortex" + time + ".svg", bbox_inches="tight", pad_inches=0.1
)
plt.show()

# %%############################################################################
"""
    vorticity enstrophy along streamwise
"""
#%% Load data and calculate enstrophy in every direction
xloc = np.arange(0.125, 30.0 + 0.125, 0.125)
tp_time = np.arange(900, 1000 + 5.0, 5.0)
y = np.linspace(-3.0, 0.0, 151)
z = np.linspace(-8.0, 8.0, 161)
ens, ens1, ens2, ens3 = np.zeros((4, np.size(xloc)))
nms = ['x', 'enstrophy', 'enstrophy_x', 'enstrophy_y', 'enstrophy_z']
flow = tf()
dirs = glob(pathV + '*.h5')
for j in range(np.size(dirs)):
    flow.load_3data(pathV, FileList=dirs[j], NameList='h5')
    file = os.path.basename(dirs[j])
    file = os.path.splitext(file)[0]
    # flow.copy_meanval()
    for i in range(np.size(xloc)):
        df = flow.TriData
        xslc = df.loc[df['x']==xloc[i]]
        ens[i] = fv.enstrophy(xslc, type='x', mode=None, rg1=y, rg2=z, opt=2)
        ens1[i] = fv.enstrophy(xslc, type='x', mode='x', rg1=y, rg2=z, opt=2)
        ens2[i] = fv.enstrophy(xslc, type='x', mode='y', rg1=y, rg2=z, opt=2)
        ens3[i] = fv.enstrophy(xslc, type='x', mode='z', rg1=y, rg2=z, opt=2)
    print("finish " + dirs[j])
    res = np.vstack((xloc, ens, ens1, ens2, ens3))
    enstrophy = pd.DataFrame(data=res.T, columns=nms)
    enstrophy.to_csv(pathV + 'Enstrophy_z' + file + '.dat',
                         index=False, float_format='%1.8e', sep=' ') 

# %% plot enstrophy along streamwise
dirs = glob(pathV + 'Enstrophy_*dat')
tab_new = pd.read_csv(dirs[0], sep=' ', index_col=False, skipinitialspace=True)
for j in range(np.size(dirs)-1):
    tab_dat = pd.read_csv(dirs[j+1], sep=' ',
                          index_col=False, skipinitialspace=True)
    tab_new = tab_new.add(tab_dat, fill_value=0)
#%%
enstro = tab_new/np.size(dirs)
# enstro = enstro.iloc[1:,:]
me = np.max(enstro['enstrophy'])
fig3, ax3 = plt.subplots(figsize=(6.4, 2.8))
ax3.plot(enstro['x'], enstro['enstrophy']/me, "k", linewidth=1.5)
ax3.plot(enstro['x'], enstro['enstrophy_x']/me, "r", linewidth=1.5)
ax3.plot(enstro['x'], enstro['enstrophy_y']/me, "k:", linewidth=1.5)
ax3.plot(enstro['x'], enstro['enstrophy_z']/me, "b", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$\epsilon/\epsilon_{max}$", fontsize=textsize)
ax3.set_xlim([0.0, 30.0])
ax3.set_ylim([0.0, 1.2])
ax3.axvline(x=2.5, color="gray", linestyle="--", linewidth=1.2)
ax3.axvline(x=5.5, color="gray", linestyle="--", linewidth=1.2)
ax3.axvline(x=8.25, color="gray", linestyle="--", linewidth=1.2)
ax3.axvline(x=12.0, color="gray", linestyle="--", linewidth=1.2)
lab = [r"$\epsilon$", r"$\epsilon_x$", r"$\epsilon_y$", r"$\epsilon_z$"]
ax3.legend(lab, ncol=2, fontsize=numsize)  # loc="lower right", 
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "enstrophy.svg", dpi=300)
plt.show()

# %%############################################################################
"""
    vorticity transportation equation analysis
"""
# %% vortex dynamics
# z-direction
files = path+"/TP_912.h5" # glob(pathT + '*h5') 
flow = tf()
flow.load_3data(
        path, FileList=files, NameList='h5'
)
xloc = np.arange(0.0, 30.0 + 0.125, 0.125)
y = np.linspace(-3.0, 0.0, 151)
z = np.linspace(-8.0, 8.0, 161)
tilt1, tilt2, stretch, dilate, torque = np.zeros((5, np.size(xloc)))

#%% Load data and calculate vorticity term in every direction
xloc = np.arange(0.125, 30.0 + 0.125, 0.125)
tp_time = np.arange(900, 1000 + 5.0, 5.0)
y = np.linspace(-3.0, 0.0, 151)
z = np.linspace(-8.0, 8.0, 161)
tilt1, tilt2, stret, dilate, torque = np.zeros((5, np.size(xloc), 2))
nms = ['x', 'tilt1_p', 'tilt1_n', 'tilt2_p', 'tilt2_n', 'stretch_p', \
       'stretch_n', 'dilate_p', 'dilate_n', 'torque_p', 'torque_n']
flow = tf()
dirs = glob(pathV + '*.h5')
for j in range(np.size(dirs)):
    flow.load_3data(pathV, FileList=dirs[j], NameList='h5')
    file = os.path.basename(dirs[j])
    file = os.path.splitext(file)[0]
    for i in range(np.size(xloc)):
        df = flow.TriData
        df1 = df.loc[df['y']==xloc[i]]
        xslc = fv.vortex_dyna(df1, type='y', opt=2)
        tilt1[i, :] = fv.integral_db(xslc['y'], xslc['z'], xslc['tilt1'],
                                     range1=y, range2=z, opt=3)
        tilt2[i, :] = fv.integral_db(xslc['y'], xslc['z'], xslc['tilt2'],
                                     range1=y, range2=z, opt=3)
        stret[i, :] = fv.integral_db(xslc['y'], xslc['z'], xslc['stretch'],
                                       range1=y, range2=z, opt=3)
        dilate[i, :] = fv.integral_db(xslc['y'], xslc['z'], xslc['dilate'],
                                      range1=y, range2=z, opt=3)
        torque[i, :] = fv.integral_db(xslc['y'], xslc['z'], xslc['bar_tor'],
                                      range1=y, range2=z, opt=3)
    print("finish " + dirs[j])
    res = np.hstack((xloc.reshape(-1, 1), tilt1, tilt2, stret, dilate, torque))
    ens_z = pd.DataFrame(data=res, columns=nms)
    ens_z.to_csv(pathV + 'vortex_y' + file + '.dat',
                     index=False, float_format='%1.8e', sep=' ')
# %% plot vortex dynamics components
dirs = glob(pathV + 'vortex_x_*dat')
tab_new = pd.read_csv(dirs[0], sep=' ', index_col=False, skipinitialspace=True)
for j in range(np.size(dirs)-1):
    tab_dat = pd.read_csv(dirs[j+1], sep=' ',
                          index_col=False, skipinitialspace=True)
    tab_new = tab_new.add(tab_dat, fill_value=0)
# %%
vortex3 = tab_new/np.size(dirs)
# vortex3 = vortex3.iloc[1:,:]
fig3, ax3 = plt.subplots(figsize=(6.4, 2.8))
tilt1 = vortex3['tilt1_p']+vortex3['tilt1_n']
tilt2 = vortex3['tilt2_p']+vortex3['tilt1_n']
stret = vortex3['stretch_p']+vortex3['stretch_n']
dilat = vortex3['dilate_p']+vortex3['dilate_n']
torqu = vortex3['torque_p']+vortex3['torque_n']
sc = np.max([tilt1, tilt2, stret, dilat, torqu])  # 1
ax3.plot(vortex3['x'], tilt1/sc, "k", linewidth=1.5)
ax3.plot(vortex3['x'], tilt2/sc, "b", linewidth=1.5)
ax3.plot(vortex3['x'], stret/sc, "r", linewidth=1.5)
ax3.plot(vortex3['x'], dilat/sc, "g", linewidth=1.5)
ax3.plot(vortex3['x'], torqu/sc, "C7:", linewidth=1.5)
#ax3.plot(vortex3['x'], vortex3['tilt1_n']/sc, "k:", linewidth=1.5)
#ax3.plot(vortex3['x'], vortex3['tilt2_n']/sc, "b:", linewidth=1.5)
#ax3.plot(vortex3['x'], vortex3['stretch_n']/sc, "r:", linewidth=1.5)
#ax3.plot(vortex3['x'], vortex3['dilate_n']/sc, "g:", linewidth=1.5)
#ax3.plot(vortex3['x'], vortex3['torque_n']/sc, "C7:", linewidth=1.5)
lab = [r"$T_y$", r"$T_z$", r"$S$", r"$D$", r"$B$"]
ax3.legend(lab, ncol=2, loc="upper right", fontsize=numsize)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel("Term", fontsize=textsize)
ax3.set_xlim([0.0, 30.0])
# ax3.set_ylim([0.0, 1.2])
# ax3.axvline(x=5.7, color="gray", linestyle="--", linewidth=1.2)
# ax3.axvline(x=11.2, color="gray", linestyle="--", linewidth=1.2)
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "vortex_dynamics_x.svg", dpi=300)
plt.show()

# %% Control volume analysis
xmin = 8.25
xmax = 12.0
xloc = np.arange(xmin, xmax + 0.125, 0.125)
vortex3 = pd.read_csv(pathV + 'vortex_z_0950.0.dat', sep=' ',
                      index_col=False, skipinitialspace=True)
cvol = vortex3.loc[(vortex3['x'] <= xmax) & (vortex3['x'] >= xmin)]
print('x range: ', np.min(cvol['x']), np.max(cvol['x']))
barloc = np.arange(5)
barhet1 = np.zeros(5)
barhet2 = np.zeros(5)
barhet1[0] = np.trapz(cvol['tilt1_p'], cvol['x'])
barhet1[1] = np.trapz(cvol['tilt2_p'], cvol['x'])
barhet1[2] = np.trapz(cvol['stretch_p'], cvol['x'])
barhet1[3] = np.trapz(cvol['dilate_p'], cvol['x'])
barhet1[4] = np.trapz(cvol['torque_p'], cvol['x'])
barhet2[0] = np.trapz(cvol['tilt1_n'], cvol['x'])
barhet2[1] = np.trapz(cvol['tilt2_n'], cvol['x'])
barhet2[2] = np.trapz(cvol['stretch_n'], cvol['x'])
barhet2[3] = np.trapz(cvol['dilate_n'], cvol['x'])
barhet2[4] = np.trapz(cvol['torque_n'], cvol['x'])
# barsum = np.sum(np.abs(barhet1)) + np.sum(np.abs(barhet2))
# barhet1 = barhet1/barsum * 100
barsum = np.sum(np.abs(barhet1 + barhet2))
barhet1 = (barhet1+barhet2)/barsum * 100
barhet2 = barhet2/barsum * 100
# plot figure of vorticity contribution percent
vlab = [r'$T_y$', r'$T_z$', r'$S$', r'$D$', r'$B$']
fig, ax = plt.subplots(figsize=(3.2, 2.8))
matplotlib.rc("font", size=textsize)
width = 0.4
ax.bar(barloc, barhet1, width, color='C0', alpha=1.0)
# ax.bar(barloc, barhet2, width, color='C2', alpha=1.0)
ax.set_xticks(barloc)
ax.set_xticklabels(vlab, fontsize=textsize)
ax.set_ylabel(r'$\mathcal{E}_t(\%)$')
ax.grid(b=True, which="both", linestyle=":", linewidth=0.6)
ax.axhline(y=0.0, color="k", linestyle="-", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "vortex_term_zt4_0950.svg", bbox_inches="tight")
plt.show()

# %%############################################################################
"""
    Numerical schiliren (gradient of density)
"""
# %% Numerical schiliren in X-Y plane
flow = pf()
file = path + 'iso_z0.h5'
flow.load_data(path, FileList=file, NameList='h5')
df = flow.PlanarData.query("x<=30.0 & y<=10.0")
x, y = np.meshgrid(np.unique(df.x), np.unique(df.y))
corner = (x < 0.0) & (y < 0.0)
var = 'schlieren'
rho = griddata((df.x, df.y), df[var], (x, y))
print("rho_max=", np.max(df[var]))
print("rho_min=", np.min(df[var]))
rho[corner] = np.nan
# %%
fig, ax = plt.subplots(figsize=(6.4, 2.0))
matplotlib.rc("font", size=textsize)
rg1 = np.linspace(0, 2.0, 21)
cbar = ax.contourf(x, y, rho, cmap="bwr", levels=rg1, extend='both')  # rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(0, 2.0, 3)
cbbox = fig.add_axes([0.16, 0.56, 0.20, 0.30], alpha=0.9)
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
cbaxes = fig.add_axes([0.17, 0.77, 0.18, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal",
                    extendrect='False', ticks=rg2)
cbar.set_label(
    r"$S_c$",
    rotation=0,
    fontsize=textsize-1,
)
plt.savefig(pathF + "SchlierenXY.svg", bbox_inches="tight")
plt.show()
# %% Numerical schiliren in X-Z plane
flow = pf()
file = path + 'iso_u0p1.h5'
flow.load_data(path, FileList=file, NameList='h5')
df = flow.PlanarData.query("x<=30.0")
zz = np.linspace(-8.0, 8.0, 257)
xx = np.linspace(-10.0, 30.0, 801)
x, z = np.meshgrid(xx, zz)
var = 'schlieren'
rho = griddata((df.x, df.z), df[var], (x, z))
print("rho_max=", np.max(df[var]))
print("rho_min=", np.min(df[var]))
# rho[corner] = np.nan
# %%
fig, ax = plt.subplots(figsize=(6.4, 2.4))
matplotlib.rc("font", size=textsize)
rg1 = np.linspace(0, 2.0, 21)
cbar = ax.contourf(x, z, rho, cmap="bwr", levels=rg1, extend='both')  # rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-8.0, 8.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$z/\delta_0$", fontsize=textsize)
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
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="vertical",
                    extendrect='False', ticks=rg2)
cbar.set_label(
    r"$S_c$",
    rotation=0,
    fontsize=textsize-1,
    labelpad=-16,
    y=1.15
)
plt.savefig(pathF + "SchlierenXZ.svg", bbox_inches="tight")
plt.show()