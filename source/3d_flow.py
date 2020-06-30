#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:39:31 2018
    Plot for time-averaged flow (3D meanflow)
@author: Weibo Hu
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.signal import savgol_filter
# import modin.pandas as pd
import plt2pandas as p2p
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

path = "/media/weibo/IM2/FFS_M1.7TB/"
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
# filter files
FileId = pd.read_csv(path + "StatList.dat", sep='\t')
filelist = FileId['name'].to_list()
pltlist = [os.path.join(path + 'TP_stat/', name) for name in filelist]
MeanFlow = pf()
MeanFlow.load_meanflow(path, FileList=pltlist)
    
# %% Load Data for time- spanwise-averaged results
MeanFlow = pf()
MeanFlow.load_meanflow(path)
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
corner = (x > 0.0) & (y < 3.0)

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
matplotlib.rc("font", size=tsize)
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
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=tsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "Grid.svg", bbox_inches="tight")
plt.show()
    
# %% dividing streamline
MeanFlow.copy_meanval()
points = np.array([[0.0], [3.0]])
xyzone = np.array([[-100.0, 0.0, 40.0], [0.0, 3.0, 6.0]])
va.streamline(pathM, MeanFlow.PlanarData, points, partition=xyzone, opt='up')
# %% Mean flow isolines
va.dividing_line(MeanFlow.PlanarData, pathM, loc=2.5)
# %% Save sonic line
va.sonic_line(MeanFlow.PlanarData, pathM, option='velocity', Ma_inf=1.7)
# %% Save shock line
va.shock_line_ffs(MeanFlow.PlanarData, pathM, val=[0.065])
# %% Save boundary layer
va.boundary_edge(MeanFlow.PlanarData, pathM, jump0=-18, jump1=-15.0,
                 jump2=6.0, val1=0.81, val2=0.973)
# %%
boundary = np.loadtxt(pathM + "BoundaryEdge.dat", skiprows=1)
bsp = splrep(boundary[:, 0], boundary[:, 1], s=5.0, per=1)
y_new = splev(boundary[:, 0], bsp)
xy_fit = np.vstack((boundary[:, 0], y_new))
np.savetxt(
    pathM + "BoundaryEdgeFit.dat",
    xy_fit.T,
    fmt="%.8e",
    delimiter="  ",
    header="x, y"
)
fig, ax = plt.subplots(figsize=(6.6, 2.3))
ax.plot(xy_fit[0, :], xy_fit[1, :])
plt.show()
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
cval1 = 0.3 # 0.3
cval2 = 1.5 # 1.4
fig, ax = plt.subplots(figsize=(6.6, 2.3))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(cval1, cval2, 41)
cbar = ax.contourf(x, y, rho, cmap="rainbow", levels=rg1, extend='both')  # rainbow_r
ax.set_xlim(-25.0, 15.0)
ax.set_ylim(0.0, 12.0)
ax.set_yticks(np.linspace(0.0, 12.0, 4))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=tsize)
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cval1, cval2, 3)
cbaxes = fig.add_axes([0.17, 0.74, 0.16, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(cbar, cax=cbaxes, extendrect='False',
                    orientation="horizontal", ticks=rg2)
cbar.set_label(
    r"$\langle \rho \rangle/\rho_{\infty}$", rotation=0, fontsize=tsize
)
# Add boundary layer
# boundary = np.loadtxt(pathM + "BoundaryEdge.dat", skiprows=1)
boundary = np.loadtxt(pathM + "BoundaryEdgeFit.dat", skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], "k", linewidth=1.5)
# Add shock wave
shock = np.loadtxt(pathM + "ShockLineFit.dat", skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], "w", linewidth=1.5)
shock = np.loadtxt(pathM + "ShockLine2.dat", skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], "w", linewidth=1.5)
# shock1 = np.loadtxt(pathM + "ShockLine1.dat", skiprows=1)
# ax.plot(shock1[:, 0], shock1[:, 1], "w", linewidth=1.5)
# shock2 = np.loadtxt(pathM + "ShockLine2.dat", skiprows=1)
# ax.plot(shock2[:, 0], shock2[:, 1], "w", linewidth=1.5)
# Add sonic line
sonic = np.loadtxt(pathM + "SonicLine.dat", skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], "w--", linewidth=1.5)
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
#ax.streamplot(
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
#)
plt.savefig(pathF + "MeanFlow.svg", bbox_inches="tight")
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
stm = "%08.2f" % vtm
InstFlow = pd.read_hdf(pathSL + 'Z_003/TP_2D_Z_003_' + stm + '.h5')
var = 'u'
var1 = '|gradp|'
u = griddata((InstFlow.x, InstFlow.y), InstFlow[var], (x, y))
gradp = griddata((InstFlow.x, InstFlow.y), InstFlow[var1], (x, y))
print("u=", np.max(InstFlow.u))
print("u=", np.min(InstFlow.u))
u[corner] = np.nan
cval1 = -0.4
cval2 = 1.1
fig, ax = plt.subplots(figsize=(6.0, 2.3))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(cval1, cval2, 21)
cbar = ax.contourf(x, y, u, cmap="bwr", levels=rg1, extend='both')  # rainbow_r
ax.set_xlim(-20.0, 5.0)
ax.set_ylim(0, 8.0)
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=tsize)
ax.set_title(r"$t u_\infty/\delta_0={}$".format(vtm), fontsize=tsize-1, pad=0.1)
ax.grid(b=True, which="both", linestyle=":")
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cval1, cval2, 3)
cbaxes = fig.add_axes([0.16, 0.73, 0.20, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(cbar, cax=cbaxes, extendrect='False',
                    orientation="horizontal", ticks=rg2)
cbar.set_label(
    r"$u/u_{\infty}$", rotation=0, fontsize=tsize
)

# Add dividing line
cbar = ax.contour(x, y, u, levels=[0.0], colors='k', linewidths=1.0)
# Add shock wave, gradp = 0.1, alpha=0.8
cbar = ax.contour(x, y, gradp, levels=[0.1], colors='w',
                  alpha=0.8, linewidths=1.0, linestyles='--')
# add mean bubble line
# dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
# ax.plot(dividing[:, 0], dividing[:, 1], "gray", linewidth=1.5)
# add mean shock line
# shock = np.loadtxt(pathM + "ShockLineFit.dat", skiprows=1)
# ax.plot(shock[:, 0], shock[:, 1], "gray", linewidth=1.5)

plt.savefig(pathF + "InstFlow_" + str(vtm) + ".svg", bbox_inches="tight")
plt.show()

# %%############################################################################
"""
    instantaneous flow field
"""
# %% Plot contour of the instantaneous flow field with isolines
# MeanFlow.AddVariable('rho', 1.7**2*1.4*MeanFlow.p/MeanFlow.T)
vtm = 700
stm = "%08.2f" % vtm
InstFlow = pd.read_hdf(pathSL + 'Z_003/TP_2D_Z_003_' + stm + '.h5')
var = 'u'
var1 = 'vorticity_3'
u = griddata((InstFlow.x, InstFlow.y), InstFlow[var], (x, y))
gradp = griddata((InstFlow.x, InstFlow.y), InstFlow[var1], (x, y))
print("u=", np.max(InstFlow[var]))
print("u=", np.min(InstFlow[var]))
u[corner] = np.nan
gradp[corner] = np.nan
cval1 = -0.4
cval2 = 1.1
fig, ax = plt.subplots(figsize=(3.6, 2.0))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(cval1, cval2, 41)
cbar = ax.contourf(x, y, u, cmap="bwr", levels=rg1, extend='both')  # rainbow_r
ax.set_xlim(-17.0, -11.0)
ax.set_ylim(0.0, 2.0)
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=tsize)
stime = str(500)
ax.set_title(r"$t u_\infty /\delta_0={}$".format(vtm), fontsize=tsize-1, pad=0.1)
ax.grid(b=True, which="both", linestyle=":")
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cval1, cval2, 3)
cbar = plt.colorbar(cbar, ticks=rg2, extendrect=True, fraction=0.025, pad=0.05)
cbar.ax.tick_params(labelsize=nsize)
cbar.set_label(
    r"$u/u_{\infty}$", rotation=0, fontsize=tsize-1, labelpad=-28, y=1.13
)

# Add isolines
cbar = ax.contour(x, y, gradp, levels=[-1.5], colors='k',
                  alpha=1.0, linewidths=1.0, linestyles='--')

plt.savefig(pathF + "Shedding_" + str(vtm) + ".svg", bbox_inches="tight")
plt.show()

# %%############################################################################
# 
# Root Mean Square of velocity from the statistical flow
# 
# %% Plot rms contour of the mean flow field
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
corner = (x > 0.0) & (y < 3.0)
var = "<v`v`>"
var_val = getattr(MeanFlow.PlanarData, var)
uu = griddata((MeanFlow.x, MeanFlow.y), var_val, (x, y))
print("uu_max=", np.max(np.sqrt(np.abs(var_val))))
print("uu_min=", np.min(np.sqrt(np.abs(var_val))))
corner = (x > 0.0) & (y < 3.0)
uu[corner] = np.nan
fig, ax = plt.subplots(figsize=(6.6, 2.3))
matplotlib.rc("font", size=tsize)
cb1 = 0.0
cb2 = 0.1
rg1 = np.linspace(cb1, cb2, 21) # 21)
cbar = ax.contourf(
    x, y, np.sqrt(np.abs(uu)), cmap="jet", levels=rg1, extend='both'
)  # rainbow_r # jet # Spectral_r
ax.set_xlim(-25.0, 15.0)
ax.set_ylim(0.0, 12.0)
ax.set_yticks(np.linspace(0.0, 12.0, 4))
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontdict=font)
ax.set_ylabel(r"$y/\delta_0$", fontdict=font)
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar box
rg2 = np.linspace(cb1, cb2, 3)
cbbox = fig.add_axes([0.14, 0.55, 0.24, 0.27], alpha=0.9)
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
cbaxes = fig.add_axes([0.17, 0.75, 0.18, 0.06])  # x, y, width, height
cbaxes.tick_params(labelsize=nsize)
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal",
                    extendrect='False', ticks=rg2)
cbar.set_label(
    r"$\sqrt{|\langle w^\prime w^\prime \rangle|}$",
    rotation=0,
    fontsize=tsize,
)
# Add boundary layer
boundary = np.loadtxt(pathM + "BoundaryEdgeFit.dat", skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], "k", linewidth=1.0)
# Add shock wave
shock = np.loadtxt(pathM + "ShockLineFit.dat", skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], "w", linewidth=1.0)
shock = np.loadtxt(pathM + "ShockLine2.dat", skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], "w", linewidth=1.5)
# Add sonic line
sonic = np.loadtxt(pathM + "SonicLine.dat", skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], "w--", linewidth=1.2)
# Add dividing line(separation line)
dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], "k--", linewidth=1.2)

plt.savefig(pathF + "MeanFlowRMSVV.svg", bbox_inches="tight")
plt.show()

# %%############################################################################
###
### Vorticity contour
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
    (1.0, -1.5),
    4.0,
    1.3,
    linewidth=1,
    linestyle="--",
    edgecolor="k",
    facecolor="none",
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
xslice = fluc_flow.TriData.loc[fluc_flow.TriData['x']==-0.5]
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
###
### velocity contour and streamlines
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
cbar.set_label(
    r"$u/u_\infty$", rotation=0, x=-0.28, labelpad=-28, fontsize=tsize
)
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
plt.savefig(
    pathF + "StreamVortex" + time + ".svg", bbox_inches="tight", pad_inches=0.1
)
plt.show()

# %%############################################################################
#
# vorticity enstrophy along streamwise
#
# %% Load data and calculate enstrophy in every direction
dirs = glob(pathV + 'Enstrophy_*dat')
tab_new = pd.read_csv(dirs[0], sep=' ', index_col=False, skipinitialspace=True)
for j in range(np.size(dirs)-1):
    tab_dat = pd.read_csv(dirs[j+1], sep=' ',
                          index_col=False, skipinitialspace=True)
    tab_new = tab_new.add(tab_dat, fill_value=0)
# %%
x1 = 4.0
x2 = 7.5
enstro = tab_new/np.size(dirs)
# enstro = enstro.iloc[1:,:]
fx = np.arange(0.2, 25 + 0.5, 0.5)
f3 = savgol_filter(enstro['enstrophy'], 41, 4)
f1 = interp1d(enstro['x'], enstro['enstrophy'], kind='cubic', fill_value='extrapolate')
f2 = interp1d(enstro['x'], enstro['enstrophy_x'], kind='cubic', fill_value='extrapolate')
me = np.max(enstro['enstrophy'])
fig3, ax3 = plt.subplots(figsize=(6.4, 2.6))
ax3.plot(enstro['x'], enstro['enstrophy']/me, "k:", linewidth=1.2)
ax3.plot(enstro['x'], f3/me, "b--", linewidth=1.2)
ax3.plot(enstro['x'], enstro['enstrophy_x']/me, "r", linewidth=1.2)
ax3.plot(enstro['x'], enstro['enstrophy_y']/me, "g", linewidth=1.2)
ax3.plot(enstro['x'], enstro['enstrophy_z']/me, "b", linewidth=1.2)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax3.set_ylabel(r"$\mathcal{E}/\mathcal{E}_\mathrm{max}$", fontsize=tsize)
ax3.set_xlim([0.0, 25.0])
ax3.set_ylim([0.0, 1.2])
ax3.axvline(x=x1, color="gray", linestyle="--", linewidth=1.0)
ax3.axvline(x=x2, color="gray", linestyle="--", linewidth=1.0)
lab = [r"$\mathcal{E}$", r"$\mathcal{E}_x$",
       r"$\mathcal{E}_y$", r"$\mathcal{E}_z$"]
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
dirs = glob(pathV + 'vortex_z_*dat')
tab_new = pd.read_csv(dirs[0], sep=' ', index_col=False, skipinitialspace=True)
for j in range(np.size(dirs)-1):
    tab_dat = pd.read_csv(dirs[j+1], sep=' ',
                          index_col=False, skipinitialspace=True)
    tab_new = tab_new.add(tab_dat, fill_value=0)

if (dirs[0].find('_x_') > 0):
    lab = [r"$T_{xy}$", r"$T_{xz}$", r"$S_{xx}$", r"$D$"]
if (dirs[0].find('_y_') > 0):
    lab = [r"$T_{yx}$", r"$T_{yz}$", r"$S_{yy}$", r"$D$"]
if (dirs[0].find('_z_') > 0):
    lab = [r"$T_{zx}$", r"$T_{zy}$", r"$S_{zz}$", r"$D$"]
vortex3 = tab_new / np.size(dirs)
# vortex3 = vortex3.iloc[1:,:]
def full_term(df, ax):
    sc = np.amax(np.abs(df.iloc[:, 1:].values))
    ax.plot(df['x'], df['tilt1_p']/sc, "k", linewidth=1.2)
    ax.plot(df['x'], df['tilt2_p']/sc, "b", linewidth=1.2)
    ax.plot(df['x'], df['stretch_p']/sc, "r", linewidth=1.2)
    ax.plot(df['x'], df['dilate_p']/sc, "g", linewidth=1.2)
    ax.plot(df['x'], df['tilt1_n']/sc, "k-", linewidth=1.2)
    ax.plot(df['x'], df['tilt2_n']/sc, "b-", linewidth=1.2)
    ax.plot(df['x'], df['stretch_n']/sc, "r-", linewidth=1.2)
    ax.plot(df['x'], df['dilate_n'] / sc, "g-", linewidth=1.2)


def partial_term(df, ax, filt=False):
    tilt1 = df['tilt1_p']+df['tilt1_n']
    tilt2 = df['tilt2_p']+df['tilt1_n']
    stret = df['stretch_p']+df['stretch_n']
    dilat = df['dilate_p']+df['dilate_n']
    torqu = df['torque_p']+df['torque_n']
    sc = np.max([tilt1, tilt2, stret, dilat, torqu])
    if filt == True:
        ws = 17
        order = 3
        tilt1 = savgol_filter(tilt1, ws, order)
        tilt2 = savgol_filter(tilt2, ws, order)
        stret = savgol_filter(stret, ws, order)
        dilat = savgol_filter(dilat, ws, order)
        torqu = savgol_filter(torqu, ws, order)
    ax3.plot(vortex3['x'], tilt1/sc, "k", linewidth=1.2)
    ax3.plot(vortex3['x'], tilt2/sc, "b", linewidth=1.2)
    ax3.plot(vortex3['x'], stret/sc, "r", linewidth=1.2)
    ax3.plot(vortex3['x'], dilat/sc, "g", linewidth=1.2)
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
matplotlib.rc("font", size=tsize)
width = 0.4
ax.bar(barloc, barhet1, width, color='C0', alpha=1.0)
# ax.bar(barloc, barhet2, width, color='C2', alpha=1.0)
ax.set_xticks(barloc)
ax.set_xticklabels(vlab, fontsize=tsize)
ax.set_ylabel(r'$\mathcal{E}_t(\%)$')
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
file = path + 'snapshots/' + 'TP_2D_Z_03_996.szplt' # 'TP_2D_Z_03_01000.00.h5'  #  
flow.load_data(path, FileList=file, NameList=None)
plane = flow.PlanarData
# plane.to_hdf(path+'snapshots/TP_2D_Z_03_1000.h5', 'w', format='fixed')
x, y = np.meshgrid(np.unique(plane.x), np.unique(plane.y))
corner = (x < 0.0) & (y < 0.0)
var = '|grad(rho)|'
schlieren = griddata((plane.x, plane.y), plane[var], (x, y))
print("rho_max=", np.max(plane[var]))
print("rho_min=", np.min(plane[var]))
schlieren[corner] = np.nan
# %%
fig, ax = plt.subplots(figsize=(6.4, 2.2))
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(0, 3.0, 21)
cbar = ax.contourf(x, y, schlieren, cmap="binary", levels=rg1, extend='both')  #binary #rainbow_r# bwr
ax.set_xlim(-35.0, 25.0)
ax.set_ylim(-3.0, 15.0)
ax.tick_params(labelsize=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=tsize)
plt.gca().set_aspect("equal", adjustable="box")
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
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal",
                    extendrect='False', ticks=rg2)
cbar.set_label(
    r"$|\nabla \rho|$",
    rotation=0,
    fontsize=tsize-1,
)
# Add boundary layer
boundary = np.loadtxt(pathM + "BoundaryEdge.dat", skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], "b", linewidth=1.0)
# Add dividing line(separation line)
dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], "b--", linewidth=1.0)
# ax.grid(b=True, which="both", linestyle=":")
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
matplotlib.rc("font", size=tsize)
rg1 = np.linspace(0, 2.0, 21)
cbar = ax.contourf(x, z, rho, cmap="bwr", levels=rg1, extend='both')  # rainbow_r
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
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="vertical",
                    extendrect='False', ticks=rg2)
cbar.set_label(
    r"$S_c$",
    rotation=0,
    fontsize=tsize-1,
    labelpad=-16,
    y=1.15
)
plt.savefig(pathF + "SchlierenXZ.svg", bbox_inches="tight")
plt.show()
