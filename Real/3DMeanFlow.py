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
#os.chdir('./')

plt.close ("All")
plt.rc('text', usetex=True)
font = {'family' : 'Times New Roman',
         'color' : 'k',
         'weight' : 'normal',
         }

path = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
textsize = 18
numsize = 15
matplotlib.rc('font', size=textsize)


# %% Load Data for time- spanwise-averaged results 
MeanFlow = DataPost()
VarName = ['x', 'y', 'u', 'v', 'w', 'rho', 'p', 'T', 'uu', 'uv',
           'uw', 'vv', 'vw', 'ww', 'Q-criterion', 'L2-criterion', 'gradp']
MeanFlow.UserData(VarName, path2+'Meanflow.dat', 1, Sep='\t')
# MeanFlow.UserDataBin(VarName, path+'Meanflow.h5')
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
# %% Plot rho contour of the mean flow field
rho  = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.rho, (x, y))
print("rho_max=", np.max(MeanFlow.rho))
print("rho_min=", np.min(MeanFlow.rho))
corner = (x<0.0) & (y<0.0)
rho[corner] = np.nan
fig, ax = plt.subplots(figsize=(10, 4))
matplotlib.rc('font', size=textsize)
rg1 = np.linspace(0.33, 1.03, 21)
cbar = ax.contourf(x, y, rho, cmap='rainbow', levels=rg1) #rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
rg2 = np.linspace(0.33, 1.03, 3)
cbaxes = fig.add_axes([0.17, 0.70, 0.18, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax = cbaxes, orientation="horizontal", ticks=rg2)
cbar.set_label(r'$\langle \rho \rangle/\rho_{\infty}$',
               rotation=0, fontdict=font)
# Add iosline for Mach number
MeanFlow.AddMach(1.7)
ax.tricontour(MeanFlow.x, MeanFlow.y, MeanFlow.Mach,
              levels=1.0, linestyles='--', linewidths=1.5, colors='gray')
# Add isoline for boudary layer edge
u = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (x, y))
umax = u[-1,:]
# umax = np.amax(u, axis = 0)
rg2  = (x[1,:]<10.375) # in front of the shock wave
umax[rg2] = 1.0
rg1  = (x[1,:]>=10.375)
umax[rg1] = 0.95
u  = u/(np.transpose(umax))
u[corner] = np.nan # mask the corner
rg1 = (y>0.3*np.max(y)) # remove the upper part
u[rg1] = np.nan
ax.contour(x, y, u, levels=0.99, linewidths=1.2, colors='k')
ax.contour(x, y, u, levels=0.0,
           linewidths=1.5, linestyles=':', colors='k')
#% Add isoline for grad(p)
"""
fig1, ax1 = plt.subplots(figsize=(10, 4))
matplotlib.rc('font', size=textsize)
gradp = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.DataTab['gradp'], (x, y))
corner = (x<0.0) & (y<0.0)
gradp[corner] = np.nan
cs = ax1.contour(x, y, gradp, levels=0.06, linewidths=1.2, colors='gray')
xcor = []
ycor = []
f = open("IsoGradP.dat", 'w')
f.write('gradp=0.06\t')
f.write('x, y\t')
f = open("IsoGradP.dat", 'a')
for isoline in cs.collections[0].get_paths():
    xcor = isoline.vertices[:, 0]
    ycor = isoline.vertices[:, 1]
    np.savetxt(f, xcor, ycor)
    ax1.plot(xcor, ycor, 'r:')
    
#isoline = cs.collections[0].get_paths()[1]
#xcor, ycor = isoline.vertices.T
ax1.set_xlim(-10.0, 30.0)
ax1.set_ylim(-3.0, 10.0)
"""
plt.savefig(path2+'MeanFlow.svg', bbox_inches='tight')
plt.show()

# %% Plot rms contour of the mean flow field
uu  = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.uu, (x, y))
print("uu_max=", np.max(np.sqrt(np.abs(MeanFlow.uu))))
print("uu_min=", np.min(np.sqrt(np.abs(MeanFlow.uu))))
corner = (x<0.0) & (y<0.0)
uu[corner] = np.nan
fig, ax = plt.subplots(figsize=(10, 4))
matplotlib.rc('font', size=textsize)
rg1 = np.linspace(0.0, 0.22, 21)
cbar = ax.contourf(x, y, np.sqrt(np.abs(uu)), cmap='rainbow', levels=rg1) #rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
rg2 = np.linspace(0.0, 0.22, 3)
cbaxes = fig.add_axes([0.17, 0.70, 0.18, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax = cbaxes, orientation="horizontal", ticks=rg2)
cbar.set_label(r'$\sqrt{|\langle u^\prime u^\prime \rangle|}$',
               rotation=0, fontsize=textsize)

# Add isoline for boudary layer edge
u = griddata((MeanFlow.x, MeanFlow.y), MeanFlow.u, (x, y))
u[corner] = np.nan # mask the corner
ax.contour(x, y, u, levels=0.0,
           linewidths=1.5, linestyles=':', colors='k')
# Add iosline for Mach number
MeanFlow.AddMach(1.7)
ax.tricontour(MeanFlow.x, MeanFlow.y, MeanFlow.Mach,
              levels=1.0, linestyles='--', linewidths=1.5, colors='gray')
plt.savefig(path2+'MeanFlowRMS.svg', bbox_inches='tight')
plt.show()

#%% Load Data for time-averaged results
MeanFlow = DataPost()
MeanFlow.UserDataBin(path+'MeanFlow.h5')
# %% Preprocess the data in the xy plane (z=0.0)
MeanFlowZ0 = MeanFlow.DataTab.loc[MeanFlow.DataTab['z'] == 0.0]
MeanFlowZ1 = MeanFlow.DataTab.loc[MeanFlow.DataTab['z'] == 0.5]
MeanFlowZ2 = MeanFlow.DataTab.loc[MeanFlow.DataTab['z'] == -0.5]
MeanFlowZ3 = pd.concat((MeanFlowZ1, MeanFlowZ2))
MeanFlowZ4 = MeanFlowZ3.groupby(['x', 'y']).mean().reset_index()
MeanFlowZ5 = MeanFlowZ4.loc[MeanFlowZ4['y'] > 3.0]
MeanFlowZ0 = pd.concat((MeanFlowZ0, MeanFlowZ5), sort=False)
MeanFlowZ0.to_hdf(path + "MeanFlowZ0.h5", 'w', format='fixed')

#%% Plot contour of the mean flow field in the xy plane
MeanFlowZ0 = DataPost()
MeanFlowZ0.UserDataBin(path+'MeanFlowZ0.h5')
x, y = np.meshgrid(np.unique(MeanFlowZ0.x), np.unique(MeanFlowZ0.y))
omegaz = griddata((MeanFlowZ0.x, MeanFlowZ0.y), MeanFlowZ0.vorticity_3, (x, y))
corner = (x<0.0) & (y<0.0)
omegaz[corner] = np.nan
textsize = 18
numsize  = 15
fig, ax = plt.subplots(figsize=(10, 4))
matplotlib.rc('font', size=textsize)
plt.tick_params(labelsize=numsize)
lev1 = np.linspace(-5.0, 1.0, 30)
cbar = ax.contourf(x, y, omegaz, cmap = 'rainbow', levels = lev1, extend="both") #rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 8.0)
cbar.cmap.set_under('b')
cbar.cmap.set_over('r')
ax.set_xlabel(r'$x/\delta_0$', fontdict = font)
ax.set_ylabel(r'$y/\delta_0$', fontdict = font)
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
rg2 = np.linspace(-5.0, 1.0, 4)
cbaxes = fig.add_axes([0.17, 0.66, 0.18, 0.07])  # x, y, width, height
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
MeanFlowY = MeanFlow.DataTab.loc[MeanFlow.DataTab['y'] == -3.0].reset_index(drop=True)
MeanFlowY.to_hdf(path + "MeanFlowYN3.h5", 'w', format='fixed')
MeanFlowY = DataPost()
MeanFlowY.UserDataBin(path+'MeanFlowYN3.h5')
#%% Plot contour of the mean flow field in the xz plane
x, z = np.meshgrid(np.unique(MeanFlowY.x), np.unique(MeanFlowY.z))
omegaz = griddata((MeanFlowY.x, MeanFlowY.z), MeanFlowY.vorticity_1, (x, z))
fig, ax = plt.subplots(figsize=(5, 3))
matplotlib.rc('font', size=textsize)
plt.tick_params(labelsize=numsize)
lev1 = np.linspace(-1.0, 1.0, 4)
#lev1 = [-0.1, 0.0, 0.1]
cbar1 = ax.contourf(x, z, omegaz, cmap='RdBu_r', levels=lev1, extend="both") #rainbow_r
ax.set_xlim(0.0, 10.0)
ax.set_ylim(-2.5,  2.5)
cbar.cmap.set_under('#053061') # 
cbar.cmap.set_over('#67001f') # 
ax.set_xlabel(r'$x/\delta_0$')
ax.set_ylabel(r'$z/\delta_0$')
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
rg2 = np.linspace(-1.0, 1.0, 3)
ax1_divider = make_axes_locatable(ax)
cax1 = ax1_divider.append_axes("top", size="7%", pad="12%")
cbar = plt.colorbar(cbar1, cax=cax1, orientation="horizontal", ticks=rg2)
plt.tick_params(labelsize=numsize)
cax1.xaxis.set_ticks_position("top")
cbar.ax.tick_params(labelsize=numsize)
ax.set_aspect('auto')
cbar.set_label(r'$\omega_z$', rotation=0, fontdict=font)
# Add isolines of Lambda2 criterion
#L2 = griddata((MeanFlowY.x, MeanFlowY.z), MeanFlowY.L2crit, (x, z))
#ax.contour(x, z, L2, levels=-0.001, \
#           linewidths=1.2, colors='k',linestyles='solid')
plt.show()
plt.savefig(path2+'Vorticity3XZ.svg', bbox_inches='tight')

#%% Preprocess the data in the xz plane (y=-0.5)
#MeanFlow = DataPost()
#MeanFlow.UserDataBin(path+'MeanFlow2.h5')
#MeanFlowX3 = MeanFlow.DataTab.loc[MeanFlow.DataTab['x'] == 3.0].reset_index(drop=True)
#MeanFlowX3.to_hdf(path2 + "MeanFlowX3.h5", 'w', format='fixed')

#%% Plot contour of the mean flow field in the zy plane
MeanFlowX = MeanFlow.DataTab.loc[MeanFlow.DataTab['x'] == 3.5].reset_index(drop=True)
MeanFlowX.to_hdf(path + "MeanFlowX3.h5", 'w', format='fixed')
MeanFlowX3 = DataPost()
MeanFlowX3.UserDataBin(path+'MeanFlowX3.h5')

# %% Plot contour of the mean flow field in the zy plane
z, y = np.meshgrid(np.unique(MeanFlowX3.z), np.unique(MeanFlowX3.y))
omegax = griddata((MeanFlowX3.z, MeanFlowX3.y), MeanFlowX3.vorticity_1, (z, y))
textsize = 18
numsize = 15
fig, ax = plt.subplots(figsize=(5, 3))
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
#cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
ax1_divider = make_axes_locatable(ax)
cax1 = ax1_divider.append_axes("top", size="7%", pad="12%")
cbar = plt.colorbar(cbar1, cax=cax1, orientation="horizontal", ticks=rg2)
plt.tick_params(labelsize=numsize)
cax1.xaxis.set_ticks_position("top")
cbar.set_label(r'$\omega_z$', rotation=0, fontsize=textsize)
# Add streamlines
zz = np.linspace(-2.5, 2.5, 50)
yy = np.linspace(-3.0, 0.0, 30)
zbox, ybox = np.meshgrid(zz, yy)
v = griddata((MeanFlowX3.z, MeanFlowX3.y), MeanFlowX3.v, (zbox, ybox))
w = griddata((MeanFlowX3.z, MeanFlowX3.y), MeanFlowX3.w, (zbox, ybox))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-3.0, 0.0)
ax.streamplot(zbox, ybox, w, v, density=[2.5, 1.5], color='k', \
              linewidth=0.5, integration_direction='both')
plt.show()
plt.savefig(path2 + 'vorticity1.svg', bbox_inches='tight', pad_inches=0)


#%% Isosurface of vorticity1 criterion
Isosurf = MeanFlow._DataTab.query("vorticity_1 <= 0.101 & vorticity_1 >= 0.099")
xx, yy = np.mgrid[-0.0:30.0:50j, -3.0:0.0:30j]
zz = griddata((Isosurf.x, Isosurf.y), Isosurf.z, (xx, yy))
fig = plt.figure(figsize=(5,3))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, 
                       linewidth=0)
plt.show()

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
