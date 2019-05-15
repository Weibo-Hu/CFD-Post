#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:50:40 2019

    Plot backup/extra figures

@author: weibo
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import matplotlib
import pandas as pd
import plt2pandas as p2p
import variable_analysis as fv
from planar_field import PlanarField as pf
from data_post import DataPost
from scipy.interpolate import griddata
from numpy import NaN, Inf, arange, isscalar, asarray, array
from timer import timer
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.close("All")
plt.rc("text", usetex=True)
font = {"family": "Times New Roman", "color": "k", "weight": "normal"}

path = "/media/weibo/Data2/BFS_M1.7L/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
textsize = 18
numsize = 14
matplotlib.rc("font", size=textsize)

# %% Plot schematic of the computational domain
pd = p2p.ReadAllINCAResults(240, pathI, FileName='ZSliceSolTime1000.plt')
x, y = np.meshgrid(np.unique(pd.x), np.unique(pd.y))
corner = (x < 0.0) & (y < 0.0)
rho_grad = griddata((pd.x, pd.y), pd['|grad(rho)|'], (x, y))
print("rho_grad max = ", np.max(pd['|grad(rho)|']))
print("rho_grad min = ", np.min(pd['|grad(rho)|']))
rho_grad[corner] = np.nan

fig, ax = plt.subplots(figsize=(10, 4))
matplotlib.rc("font", size=textsize)
rg = np.linspace(0.1, 1.8, 18)
cbar = ax.contourf(x, y, rho_grad, cmap='gray_r', levels=rg, extend='max')
cbar.cmap.set_over('#000000')
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.set_xlabel(r"$x/\delta_0$", fontdict=font)
ax.set_ylabel(r"$y/\delta_0$", fontdict=font)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Schematic.svg", bbox_inches="tight")
plt.show()

# %% Plot schematic of the mean flow field
fig, ax = plt.subplots(figsize=(10, 4))
matplotlib.rc("font", size=textsize)
rg1 = np.linspace(0.33, 1.03, 41)
# cbar = ax.contourf(x, y, rho, cmap='rainbow', levels=rg1) #rainbow_r
ax.set_xlim(-5.0, 25.0)
ax.set_ylim(-3.0, 7.5)
ax.set_yticks(np.arange(-2.5, 7.6, 2.5))
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/\delta_0$", fontdict=font)
ax.set_ylabel(r"$y/\delta_0$", fontdict=font)
plt.gca().set_aspect("equal", adjustable="box")
# Add shock wave
shock = np.loadtxt(pathM + "Shock.dat", skiprows=1)
ax.scatter(shock[0::3, 0], shock[0::3, 1], marker="o", s=8.0, c="k")
# Add boundary layer
# boundary = np.loadtxt(path3+'BoundaryLayer.dat', skiprows=1)
# ax.plot(boundary[:, 0], boundary[:, 1], 'k', linewidth=1.5)
# Add dividing line(separation line)
dividing = np.loadtxt(pathM + "DividingLine.dat", skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], "k--", linewidth=1.5)
# Add reference line
u = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (x, y))
u[corner] = np.nan
# cs = ax.contour(x, y, u, levels=0.8, colors='k', linestyles='--')
# ax.clabel(cs, fmt='%2.1f', colors='k', fontsize=numsize)

# ax.hlines(-1.875, 0, 30, colors='k', linestyles=':')
# ax.grid(b=True, which='both', linestyle=':')
plt.savefig(pathF + "Schematic.svg", bbox_inches="tight")
plt.show()

# %% Isosurface of vorticity1 criterion
Isosurf = MeanFlow._DataTab.query("vorticity_1 <= 0.101 & vorticity_1 >= 0.099")
xx, yy = np.mgrid[-0.0:30.0:50j, -3.0:0.0:30j]
zz = griddata((Isosurf.x, Isosurf.y), Isosurf.z, (xx, yy))
fig = plt.figure(figsize=(5, 3))
ax = fig.gca(projection="3d")
surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, linewidth=0)
plt.show()

#%% Isosurface of lambda2 criterion
MeanFlow = DataPost()
MeanFlow.UserDataBin(path+'MeanFlow2.h5')

Isosurf = MeanFlow._DataTab.loc[np.round(MeanFlow._DataTab['L2-criterion'], 5) == -0.005]
xx, yy  = np.mgrid[-10.0:30.0:300j, -3.0:10.0:100j]
zz      = griddata((Isosurf.x, Isosurf.y), Isosurf['z'], (xx, yy))
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(0.0, -90.0)
surf = ax.plot_surface(xx, yy, zz, cmap="rainbow", antialiased=False)
cbar = plt.colorbar(surf)
ax.set_xlabel(r'$x/\delta_0$', fontdict = font3)
ax.set_ylabel(r'$y/\delta_0$', fontdict = font3)
plt.gca().set_aspect('equal', adjustable='box')
#ax.contourf()
plt.show()

# %% Plot BL profile along streamwise
xcoord = np.array([-40, -5, 0, 5.0, 10, 15, 20, 30, 40])
num = np.size(xcoord)
xtick = np.zeros(num + 1)
xtick[-1] = 1.0
fig, ax = plt.subplots(figsize=(6.4, 2.2))
# fig.set_size_inches(3.2, 1.0)
matplotlib.rc('font', size=textsize)
ax.plot(np.arange(num + 1), np.zeros(num + 1), "w-")
ax.set_xlim([0, num + 0.5])
ax.set_ylim([0, 4])
ax.set_xticks(np.arange(num + 1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = xtick
var = '<u>'
# ax.set_xticklabels(labels, fontdict=font)
ax.set_xticklabels(["$%d$" % f for f in xtick])
for i in range(num):
    df = MeanFlow.yprofile("x", xcoord[i])
    y0 = df['y']
    q0 = df[var]
    ax.plot(q0 + i, y0, "k-")
    ax.text(
        i + 0.75,
        3.0,
        r"$x/\delta_0={}$".format(xcoord[i]),
        rotation=90,
        fontsize=numsize,
    )
plt.tick_params(labelsize=numsize)
ax.set_xlabel(r"$u/u_{\infty}$", fontsize=textsize)
ax.set_ylabel(r"$\Delta y/\delta_0$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "StreamwiseBLProfile.svg", bbox_inches="tight", pad_inches=0.1
)