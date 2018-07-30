#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:39:31 2018
    Plot for time-averaged flow (3D meanflow)
@author: Weibo Hu
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
import sys, os
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from skimage import measure
#os.chdir('./')

plt.close ("All")
plt.rc('text', usetex=True)
font = {'family' : 'Times New Roman',
         #'color' : 'k',
         'weight' : 'normal',
         }

path = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
#path3 = "D:/ownCloud/0509/Data/"
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('font', **font)


#%% Preprocess the data in the xy plane (z=0.0)
#MeanFlow = DataPost()
#MeanFlow.UserDataBin(path+'MeanFlow2.h5')
#MeanFlowZ0 = MeanFlow.DataTab.loc[MeanFlow.DataTab['z'] == 0.0]
#MeanFlowZ1 = MeanFlow.DataTab.loc[MeanFlow.DataTab['z'] == 0.5]
#MeanFlowZ2 = MeanFlow.DataTab.loc[MeanFlow.DataTab['z'] == -0.5]
#MeanFlowZ3 = pd.concat((MeanFlowZ1, MeanFlowZ2))
#MeanFlowZ4 = MeanFlowZ3.groupby(['x', 'y']).mean().reset_index()
#MeanFlowZ5 = MeanFlowZ4.loc[MeanFlowZ4['y'] > 3.0]
#MeanFlowZ0 = pd.concat((MeanFlowZ0, MeanFlowZ5), sort=False)
#MeanFlowZ0.to_hdf(path2 + "MeanFlowZ0.h5", 'w', format='fixed')

#%% Load Data
MeanFlowZ0 = DataPost()
MeanFlowZ0.UserDataBin(path2+'MeanFlowZ0.h5')

x, y = np.meshgrid(np.unique(MeanFlowZ0.x), np.unique(MeanFlowZ0.y))

#%% Plot contour of the mean flow field in the xy plane
omegaz = griddata((MeanFlowZ0.x, MeanFlowZ0.y), MeanFlowZ0.vorticity_3, (x, y))
corner = (x<0.0) & (y<0.0)
omegaz[corner] = np.nan
textsize = 18
numsize  = 13
fig, ax = plt.subplots(figsize=(12,4))
matplotlib.rc('font', size=textsize)
plt.tick_params(labelsize=numsize)
lev1 = np.linspace(-5.0, 1.0, 30)
cbar = ax.contourf(x, y, omegaz, cmap = 'rainbow', levels = lev1, extend="both") #rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0,  10.0)
cbar.cmap.set_under('b')
cbar.cmap.set_over('r')
ax.set_xlabel(r'$x/\delta_0$', fontdict = font)
ax.set_ylabel(r'$y/\delta_0$', fontdict = font)
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
rg2 = np.linspace(-5.0, 1.0, 4)
cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07]) # x, y, width, height
cbar = plt.colorbar(cbar, cax = cbaxes, orientation="horizontal", ticks=rg2)
cbar.set_label(r'$\omega_z$', rotation=0,  fontdict = font)

box = patches.Rectangle((2.0, -2.0), 3.5, 1.5, linewidth=1, \
                        linestyle='--', edgecolor='k', facecolor='none')
ax.add_patch(box)
plt.tick_params(labelsize=numsize)
plt.savefig(path2+'Vorticity3.svg', bbox_inches='tight')
plt.show()

#%% Zoom box for streamline
fig, ax = plt.subplots(figsize=(7,3))
cbar = ax.contourf(x, y, omegaz, cmap = 'rainbow', levels = lev1, extend="both")
ax.set_xlim(2.0, 5.5)
ax.set_ylim(-2.0,  -0.5)
ax.set_xticks([])
ax.set_yticks([])
x1 = np.linspace(2.0, 5.5, 50)
y1 = np.linspace(-2.0, -0.5, 50)
xbox, ybox = np.meshgrid(x1, y1)
u  = griddata((MeanFlowZ0.x, MeanFlowZ0.y), MeanFlowZ0.u, (xbox, ybox))
v  = griddata((MeanFlowZ0.x, MeanFlowZ0.y), MeanFlowZ0.v, (xbox, ybox))
ax.streamplot(xbox, ybox, u, v, density=[1.5, 1.5], color='w', \
              linewidth=1.0, integration_direction='both')
plt.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.3)
plt.savefig(path2+'V3ZoomBox.svg', bbox_inches='tight', pad_inches=0)
plt.show()
#%% Preprocess the data in the xz plane (y=-0.5)
#MeanFlow = DataPost()
#MeanFlow.UserDataBin(path+'MeanFlow2.h5')
#MeanFlowY05 = MeanFlow.DataTab.loc[MeanFlow.DataTab['y'] == -0.5].reset_index(drop=True)
#MeanFlowY05.to_hdf(path2 + "MeanFlowY05.h5", 'w', format='fixed')

#%% Load Data
MeanFlowY05 = DataPost()
MeanFlowY05.UserDataBin(path2+'MeanFlowY05.h5')
#%% Plot contour of the mean flow field in the xz plane
x, z = np.meshgrid(np.unique(MeanFlowY05.x), np.unique(MeanFlowY05.z))
omegaz = griddata((MeanFlowY05.x, MeanFlowY05.z), MeanFlowY05.vorticity_3, (x, z))
fig, ax = plt.subplots(figsize=(8,4))
matplotlib.rc('font', size=textsize)
plt.tick_params(labelsize=numsize)
lev1 = np.linspace(-5.0, 1.0, 30)
cbar1 = ax.contourf(x, z, omegaz, cmap = 'rainbow', levels = lev1, extend="both") #rainbow_r
ax.set_xlim(0.0, 10.0)
ax.set_ylim(-2.5,  2.5)
cbar.cmap.set_under('b')
cbar.cmap.set_over('r')
ax.set_xlabel(r'$x/\delta_0$', fontdict = font)
ax.set_ylabel(r'$z/\delta_0$', fontdict = font)
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
rg2 = np.linspace(-5.0, 1.0, 4)
cbar = plt.colorbar(cbar1, ticks=rg2)
cbar.ax.tick_params(labelsize=numsize)
ax.set_aspect('auto')
cbar.set_label(r'$\omega_z$', rotation=0, fontdict = font)
# Add isolines of Lambda2 criterion
L2 = griddata((MeanFlowY05.x, MeanFlowY05.z), MeanFlowY05.L2_criterion, (x, z))
ax.contour(x, z, L2, levels = -0.001, \
           linewidths = 1.2, colors='k',linestyles='solid')
plt.show()
plt.savefig(path2+'Vorticity3XZ.svg', bbox_inches='tight')

#%% Preprocess the data in the xz plane (y=-0.5)
#MeanFlow = DataPost()
#MeanFlow.UserDataBin(path+'MeanFlow2.h5')
#MeanFlowX3 = MeanFlow.DataTab.loc[MeanFlow.DataTab['x'] == 3.0].reset_index(drop=True)
#MeanFlowX3.to_hdf(path2 + "MeanFlowX3.h5", 'w', format='fixed')
#%% Load Data
MeanFlowX3 = DataPost()
MeanFlowX3.UserDataBin(path2 + 'MeanFlowX3.h5')

#%% Plot contour of the mean flow field in the xy plane
z, y = np.meshgrid(np.unique(MeanFlowX3.z), np.unique(MeanFlowX3.y))
omegax = griddata((MeanFlowX3.z, MeanFlowX3.y), MeanFlowX3.vorticity_1, (z, y))
textsize = 18
numsize = 13
fig, ax = plt.subplots(figsize=(4, 3))
matplotlib.rc('font', size=textsize)
plt.tick_params(labelsize=numsize)
lev1 = np.linspace(-0.1, 0.1, 30)
cbar1 = ax.contourf(
    z, y, omegax, cmap='rainbow', levels=lev1, extend="both")  #rainbow_r
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-3.0, 0.0)
cbar1.cmap.set_under('b')
cbar1.cmap.set_over('r')
ax.set_xlabel(r'$z/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
rg2 = np.linspace(-0.1, 0.1, 5)
cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
ax1_divider = make_axes_locatable(ax)
cax1 = ax1_divider.append_axes("top", size="7%", pad="12%")
cbar = plt.colorbar(cbar1, cax=cax1, orientation="horizontal", ticks=rg2)
cax1.xaxis.set_ticks_position("top")
cbar.set_label(r'$\omega_z$', rotation=0, fontdict=font)
# Add streamlines
zbox = np.linspace(-2.5, 2.5, 50)
ybox = np.linspace(-3.0, 0.0, 30)
v = griddata((MeanFlowX3.z, MeanFlowX3.y), MeanFlowX3.v, (zbox, ybox))
w = griddata((MeanFlowX3.z, MeanFlowX3.y), MeanFlowX3.w, (zbox, ybox))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-3.0, 0.0)
ax.streamplot(zbox, ybox, w, v, density=[2.5, 1.5], color='w', \
              linewidth=1.0, integration_direction='both')
plt.show()
plt.savefig(path2 + 'vorticity1.svg', bbox_inches='tight', pad_inches=0)


#%% Isosurface of vorticity1 criterion
MeanFlow = DataPost()
MeanFlow.UserDataBin(path+'MeanFlow2.h5')
xx, yy, zz = np.mgrid[-10.0:30.0:100j, -3.0:10.0:52j, -2.5:2.5:20j]
vort1 = griddata((MeanFlow.x, MeanFlow.y, MeanFlow.z), \
                  MeanFlow.vorticity_1, (xx, yy, zz))
#coord, index, normals, values = \
#    measure.marching_cubes(vort1, -0.2, spacing=(1.0, 1.0, 1.0))
#fig = plt.figure(figsize=(12,4))
#ax = fig.add_subplot(111, projection='3d')
#ax.view_init(0.0, -90.0)
##surf = ax.plot_surface(xx, yy, zz, cmap="rainbow", antialiased=False)
#ax.plot_trisurf(coord[:,0], coord[:,1], coord[:,2], lw=0.2)
#ax.set_xlabel(r'$x/\delta_0$', fontdict = font)
#ax.set_ylabel(r'$y/\delta_0$', fontdict = font)
#plt.show()

#%%
"""
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
"""
