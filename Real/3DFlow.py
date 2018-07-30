#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:38:00 2018
    Plot mean flow or a slice of a 3D flow
@author: Weibo Hu
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import FlowVar as fv
import copy
from DataPost import DataPost
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from scipy.interpolate import griddata
from scipy.interpolate import spline
import scipy.optimize
from numpy import NaN, Inf, arange, isscalar, asarray, array
import plt2pandas as p2p
from timer import timer
import sys

plt.close ("All")
plt.rc('text', usetex=True)
font1 = {'family' : 'Times New Roman',
         #'color' : 'k',
         'weight' : 'normal',
         'size' : 12,}

font2 = {'family' : 'Times New Roman',
         #'color' : 'k',
         'weight' : 'normal',
         'size' : 14,
        }

font3 = {'family' : 'Times New Roman',
         #'color' : 'k',
         'weight' : 'normal',
         'size' : 18,
}

path = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
# path3 = "D:/ownCloud/0509/Data/"
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('font', **font1)

# %% Import Data
MeanFlow = DataPost()
# VarName  = ['x', 'y', 'z', 'u', 'v', 'w', \
#            'rho', 'p', 'Q_crit', 'Mach', 'T']
# MeanFlow.UserData(VarName, path+'t401.txt', 1, Sep = '\t')
# MeanFlow.SpanAve(path+'MeanSlice260.dat')
# MeanFlow.LoadData(path+'MeanSlice260.dat', Sep = '\t')
# %%
VarName  = ['x', 'y', 'u', 'v', 'w', 'rho', 'p', 'T', 'uu', \
            'uv', 'uw', 'vv', 'vw', 'ww', 'Q-criterion', 'L2-criterion']
MeanFlow.UserData(VarName, path+'Meanflow.dat', 1, Sep = '\t')
# MeanFlow.UserDataBin(VarName, path+'Meanflow.h5')
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
rho  = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.rho, (x, y))
# %% Plot contour of the mean flow field
corner = (x<0.0) & (y<0.0)
rho[corner] = np.nan
fig, ax = plt.subplots(figsize=(12, 4))
rg1 = np.linspace(0.25, 1.06, 20)
cbar = ax.contourf(x, y, rho, cmap = 'rainbow', levels = rg1) #rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.set_xlabel(r'$x/\delta_0$', fontdict = font3)
ax.set_ylabel(r'$y/\delta_0$', fontdict = font3)
# ax.grid (b=True, which = 'both', linestyle = ':')
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
rg2 = np.linspace(0.25, 1.06, 4)
cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
cbar = plt.colorbar(cbar, cax = cbaxes, orientation="horizontal", ticks=rg2)
cbar.set_label(r'$\rho/\rho_{\infty}$', rotation=0, fontdict=font3)
# Add iosline for Mach number
MeanFlow.AddMach()
ax.tricontour(MeanFlow.x, MeanFlow.y, MeanFlow.Mach,
              levels = 1.0, linewidths=1.2, colors='w')
# Add isoline for boudary layer edge
u  = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (x, y))
umax = u[-1,:]
# umax = np.amax(u, axis = 0)
rg2  = (x[1,:]<16.0) # in front of the shock wave
umax[rg2] = 1.0
rg1  = (x[1,:]>=16.0)
umax[rg1] = 0.95
u  = u/(np.transpose(umax))
u[corner] = np.nan # mask the corner
rg1 = (y>0.3*np.max(y)) # remove the upper part
u[rg1]     = np.nan
ax.contour(x, y, u, levels = 0.99, linewidths = 1.2, colors='k')
ax.contour(x, y, u, levels = 0.0, \
        linewidths = 1.2, linestyles = 'dashed', colors='k')
# plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
# ax.tick_params(axis='both', which='major', labelsize=10)
# fig = matplotlib.pyplot.gcf()
plt.savefig(path+'MeanFlow.svg', bbox_inches='tight')
plt.show()

# %% Plot streamwise skin friction
MeanFlow.AddWallDist(3.0)
WallFlow = MeanFlow.DataTab.groupby("x", as_index=False).nth(1)
mu = fv.Viscosity(13718, WallFlow['T'])
Cf = fv.SkinFriction(mu, WallFlow['u'], WallFlow['walldist'])
fig2, ax2 = plt.subplots()
ax2.plot(WallFlow['x'], Cf, 'k', linewidth = 1.5)
ax2.set_xlabel (r'$x/\delta_0$', fontdict = font3)
ax2.set_ylabel (r'$C_f$', fontdict = font3)
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2, 2))
ax2.axvline(x=12.7, color='gray', linestyle='--', linewidth=1.0)
ax2.grid(b=True, which = 'both', linestyle = ':')
plt.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.3)
fig2.set_size_inches(6, 5, forward=True)
plt.savefig(path2+'Cf.pdf', dpi = 300)
plt.show()

# %% pressure coefficiency
fig3, ax3 = plt.subplots()
ax3.plot(WallFlow['x'], WallFlow['p'], 'k', linewidth = 1.5)
ax3.set_xlabel(r'$x/\delta_0$', fontdict = font3)
ax3.set_ylabel(r'$p/\rho_{\infty} U_{\infty}^2$', fontdict = font3)
ax3.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2, 2))
ax3.axvline(x=12.7, color='gray', linestyle='--', linewidth=1.0)
ax3.grid(b=True, which = 'both', linestyle = ':')
plt.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.3)
fig2.set_size_inches(6, 5, forward=True)
plt.savefig(path2+'Cp.pdf', dpi = 300)
plt.show()

# %% Draw boundary layer profile along streamwise
minorLocator = MultipleLocator(10)
xlocations = np.array([-40, -20, 0, 10, 20, 50, 60])
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for j in range(np.size(xlocations)):
    wd, ux = MeanFlow.BLProfile('x', xlocations[j], 'u')
    ux = ux*10 + xlocations[j]
    ax1.plot(ux, wd, 'k')

ax1.text(xlocations[0], 6.1, 0.0, ha='center', va='center')
ax1.text(xlocations[0]+10.0, 6.1, 1.0, ha='center', va='center')
ax1.xaxis.set_minor_locator(minorLocator)
ax1.grid(b=True, which = 'both', linestyle = ':')
ax1.set_ylim ([0, 6])
ax1.set_yticks ([0, 2, 4, 6])
ax1.set_xlabel (r'$x/\delta_0$', fontdict = font3)
ax1.set_ylabel (r'$\Delta y/\delta_0$', fontdict = font3)

ax2 = ax1.twiny()
ax2.set_xticks([])
ax2.set_xticklabels([])
ax2.set_xlabel(r'$u/u_\infty$', fontdict = font3)
plt.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.3)
fig1.set_size_inches(10, 4, forward=True)
plt.savefig(path2+'StreamwiseBLProfile.pdf', dpi = 300)
plt.show()


