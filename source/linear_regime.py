#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:17:05 2019
    This code for plotting line/curve figures for the linear regime

@author: weibo
"""

# %% Load necessary module
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, splev, splrep, griddata
from data_post import DataPost
import variable_analysis as fv
from timer import timer
import os
from planar_field import PlanarField as pf
from triaxial_field import TriField as tf
from scipy.interpolate import griddata


# %% data path settings
path = "/media/weibo/VID2/BFS_M1.7TS/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathL = path + "Linear/"

# %% figures properties settings
plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 14
numsize = 11

# %% load data for clean flow
path1 = "/media/weibo/VID2/BFS_M1.7L/"
cflow = tf()
cflow.load_3data(
        path, FileList=path1+"/ZFluctuation_912.0.h5", NameList='h5'
)
zslice = cflow.TriData.loc[cflow.TriData['z']==0.0]
cflow.PlanarData = zslice
# %% load data for flow with oblique waves
oflow = tf()
oflow.load_3data(
        path, FileList=path+"/ZFluctuation_912.0.h5", NameList='h5'
)
zslice = oflow.TriData.loc[oflow.TriData['z']==0.0]
oflow.PlanarData = zslice
# %% compare BL profile for \omega_z
fig, ax = plt.subplots(figsize=(3.2, 3.2))
# fig.subplots_adjust(hspace=0.5, wspace=0.15)
matplotlib.rc('font', size=numsize)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
df = cflow.yprofile('x', 0.0)
ax.plot(df['u'], df['y'], 'k-', linewidth=1.2)
df1 = oflow.yprofile('x', 0.0)
ax.plot(df1['u'], df1['y'], 'k:', linewidth=1.2)
ax.set_ylim([0, 4])
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.set_xlabel(r'$\hat{u} / u_\infty$', fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLFluc2.svg", bbox_inches="tight", pad_inches=0.1
)

# %% compare BL profile for \omega_z
fig, ax = plt.subplots(figsize=(3.2, 3.2))
# fig.subplots_adjust(hspace=0.5, wspace=0.15)
matplotlib.rc('font', size=numsize)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
df = cflow.yprofile('x', 0.0)
ax.plot(df['vorticity_3'], df['y'], 'k-', linewidth=1.2)
df1 = oflow.yprofile('x', 0.0)
ax.plot(df1['vorticity_3'], df1['y'], 'k:', linewidth=1.2)
ax.set_xlim([-0.8, 0.2])
ax.set_ylim([0, 1.0])
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.set_xlabel(r'$\hat{\omega}_z \delta_0 /u_\infty$', fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLFluc2VorticityZ.pdf", bbox_inches="tight", pad_inches=0.1
)

# %% calculate amplitude in the streamwise
coord = pd.read_csv(pathM + "ShearLine.dat", sep=' ', skiprows=0,
                    index_col=False, skipinitialspace=True)
xarr = coord['x'].values[424:550]
yarr = coord['y'].values[424:550]
varnm = 'u'
fm = cflow.TriData
freq_x = np.zeros((1, np.size(xarr)))
amplt_x = np.zeros((1, np.size(xarr)))
phase_x = np.zeros((1, np.size(xarr)))
for i in range(np.size(xarr)):
    var = fm.loc[
        (fm['x'] == xarr[i]) & (fm['y'] == yarr[i]),
        ['z', 'u', 'p']
        ]
    num_samp = np.size(var['z'])
    freq_samp = (num_samp - 1 ) / 16
    freq = np.linspace(0.0, freq_samp / 2, math.ceil(num_samp / 2),
                       endpoint=False)
    var_fft = np.fft.rfft(var[varnm]-np.mean(var[varnm]))
    amplt = np.abs(var_fft)
    phase = np.angle(var_fft, deg=False)
    ind = np.argmax(amplt)
    freq_x[0, i] = freq[ind]
    amplt_x[0, i] = amplt[ind]
    phase_x[0, i] = phase[ind]
res_fft = np.concatenate((xarr.reshape(1,-1), freq_x, amplt_x, phase_x))
varlist = ['x', 'beta', 'amplt', 'phase']
df = pd.DataFrame(data=res_fft.T, columns=varlist)
df.to_csv(pathL + varnm + '_beta_linear.dat', sep=' ',
          index=False, float_format='%1.8e')

# %% growth rate
filenm = pathL + varnm + '_beta_linear.dat'
fft_x = pd.read_csv(filenm, sep=' ', skiprows=0,
                    index_col=False, skipinitialspace=True)

# %% amplitude along the streamwise direction
fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc('font', size=numsize)
ax.plot(fft_x.x[1:], fft_x.amplt[1:], 'k', linewidth=1.2)
ax.set_yscale('log')
# ax.set_ylim([0, 5])
ax.set_ylabel(r"$A(\tilde{u})$", fontsize=textsize)
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLAmplitLinear_" + varnm + ".svg", bbox_inches="tight", pad_inches=0.1
)


# %% streamline in the x-y plane
zslice = cflow.PlanarData
x, y = np.meshgrid(np.unique(zslice), np.unique(zslice.y))
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

# %% load 3d flow data
x1 = np.linspace(0.0, 5.0, 50)
y1 = np.linspace(-1.5, 0.0, 50)
xbox, ybox = np.meshgrid(x1, y1)
pflow1 = pf()
pflow1.load_data(
        path, FileList=path+"/TP_2D_Z_03_00912.00.plt"
)
pflow1.PlanarData.to_hdf(path1 + 'TP_2D_Z_03_00912.00.h5', 'w', format='fixed')

# %% Streamline plot
zslice = pflow1.PlanarData
fig, ax = plt.subplots(figsize=(6.4, 3.0))
matplotlib.rc("font", size=textsize)
plt.tick_params(labelsize=numsize)
shear = griddata((zslice.x, zslice.y), zslice.vorticity_3, (xbox, ybox))
u = griddata((zslice.x, zslice.y), zslice.u, (xbox, ybox))
v = griddata((zslice.x, zslice.y), zslice.v, (xbox, ybox))
lev1 = np.linspace(-5, 0, 6)
cbar = ax.contourf(xbox, ybox, shear, cmap="rainbow_r", extend='both', levels=lev1)
cbar.cmap.set_under('#ff0000')
cbar.cmap.set_over('#4169e1') # #0000ff
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.set_xlim(0.0, 4.0)
ax.set_ylim(-1.4, 0.0)
rg2 = np.linspace(-5, 0, 6)
# cbaxes = fig.add_axes([0.17, 0.72, 0.18, 0.05])  # x, y, width, height
cbar = plt.colorbar(cbar, orientation="vertical", ticks=rg2, 
                    pad=0.02, extendrect=True, shrink=0.9)
cbar.ax.tick_params(labelsize=numsize)
cbar.set_label(r"$\omega_z$", y=1.1, labelpad=-18, 
               rotation=0, fontsize=textsize)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.quiver(xbox[::2], ybox[::2], u[::2], v[::2], units='width', scale=1.0)
ax.streamplot(
    xbox,
    ybox,
    u,
    v,
    density=[3.0, 3.0],
    color="k",
    linewidth=0.6,
    arrowstyle='->',
    integration_direction="both"
)
plt.tick_params(labelsize=numsize)
# plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(pathF + "KH_Streamline_L.svg", bbox_inches="tight")
plt.show()

# %% Streamline plot
flow1 = tf()
flow1.load_3data(
        path1, FileList=path1+"TP_data_full_912.h5", NameList='h5'
)
#flow1.load_3data(path="/media/weibo/VID1/BFS_M1.7L/K-H/TP_data_01402108/")
yslice = flow1.TriData.loc[flow1.TriData['y']==-0.21875]
z, x = np.meshgrid(np.unique(yslice.z), np.unique(yslice.x))
u = griddata((yslice.z, yslice.x), yslice.u, (z, x))
vort3 = griddata((yslice.z, yslice.x), yslice.vorticity_3, (z, x))
lamb2 = griddata((yslice.z, yslice.x), yslice['L2-criterion'], (z, x))
# %% vorticity in z-x plane
fig, ax = plt.subplots(figsize=(6.4, 2.5))
matplotlib.rc("font", size=textsize)
plt.tick_params(labelsize=numsize)
cb1 = -0.2
cb2 = 1.2
lev1 = np.linspace(cb1, cb2, 21)
cbar = ax.contourf(z, x, u, cmap="rainbow_r", extend='neither', levels=lev1)
ax.set_title(r"$y/\delta_0=-0.21875$", pad=1.0, fontsize=textsize-2)
# cbar.cmap.set_under('#ff0000')
# cbar.cmap.set_over('#4169e1') # #0000ff
ax.set_xlabel(r"$z/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_xlim(-8.0, 8.0)
ax.set_ylim(-0.0, 4.0)
rg2 = np.linspace(cb1, cb2, 3)
cbar = plt.colorbar(cbar, orientation="vertical", ticks=rg2, 
                    pad=0.02, extendrect=True, shrink=0.9)
cbar.ax.tick_params(labelsize=numsize)
cbar.set_label(r"$u/u_\infty$", y=1.13, labelpad=-15, 
               rotation=0, fontsize=textsize)
cbar1 = ax.contour(z, x, lamb2, levels=[-0.002],
                   linewidths=1.2, colors='k',linestyles='solid')
ax.clabel(cbar1, inline=1, fontsize=numsize-1)
plt.tick_params(labelsize=numsize)
# plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(pathF + "ZXVorStretch_L.svg", bbox_inches="tight")
plt.show()

# %%
xslice = flow1.TriData.loc[flow1.TriData['x']==0.3125]
z, y = np.meshgrid(np.unique(xslice.z), np.unique(xslice.y))
u = griddata((xslice.z, xslice.y), xslice.u, (z, y))
v = griddata((xslice.z, xslice.y), xslice.v, (z, y))
vort3 = griddata((xslice.z, xslice.y), xslice.vorticity_3, (z, y))
lamb2 = griddata((xslice.z, xslice.y), xslice["L2-criterion"], (z, y))

# %% vorticity in z-y plane
fig, ax = plt.subplots(figsize=(6.4, 2.0))
matplotlib.rc("font", size=textsize)
plt.tick_params(labelsize=numsize)
cb1 = -0.2
cb2 = 1.2
lev1 = np.linspace(cb1, cb2, 15)
cbar = ax.contourf(z, y, u, cmap="rainbow_r", extend='neither', levels=lev1)
ax.set_title(r"$x/\delta_0=0.3125$", pad=1.0, fontsize=textsize-2)
# cbar.cmap.set_under('#ff0000')
# cbar.cmap.set_over('#4169e1') # #0000ff
ax.set_xlabel(r"$z/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.set_xlim(-8.0, 8.0)
ax.set_ylim(-1.0, 1.0)
rg2 = np.linspace(cb1, cb2, 3)
cbar = plt.colorbar(cbar, orientation="vertical", ticks=rg2, 
                    pad=0.02, extendrect=True, shrink=0.9)
cbar.ax.tick_params(labelsize=numsize)
cbar.set_label(r"$u/u_\infty$", y=1.15, labelpad=-20, 
               rotation=0, fontsize=textsize)
cbar1 = ax.contour(z, y, lamb2, levels=[-0.004],
                   linewidths=1.2, colors='k',linestyles='solid')
ax.clabel(cbar1, inline=1, fontsize=numsize-1)
plt.tick_params(labelsize=numsize)
# plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(pathF + "ZYVorStretch_L.svg", bbox_inches="tight")
plt.show()

# %% Streamline plot for vortex merge (load data)
flow3 = pf()
filepath = '/media/weibo/VID2/BFS_M1.7TS/Slice/TP_2D_Z_03/'
flow3.load_data(
    path, 
    FileList='/media/weibo/VID1/BFS_M1.7TS/Slice/TP_2D_Z_03/'+ 'TP_2D_Z_03_00927.75.plt',
    #FileList=filepath + 'TP_2D_Z_03_00927.75.h5', NameList='h5'
)
flow4 = pf()
flow4.load_data(
    path, 
    FileList='/media/weibo/VID1/BFS_M1.7TS/Slice/TP_2D_S_10/'+ 'TP_2D_S_10_00927.75.plt',
)
zslice = flow3.PlanarData.query(" x>0 & x<=4")
zslice1 = flow4.PlanarData.query(" x>0 & x<=4")
x1 = np.linspace(0.0, 4.0, 500)
y1 = np.linspace(-1.5, 0.0, 500)
xbox, ybox = np.meshgrid(x1, y1)
# %% Streamline plot for vortex merge (plot)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
matplotlib.rc("font", size=textsize)
plt.tick_params(labelsize=numsize)
omega3 = griddata((zslice.x, zslice.y), zslice.vorticity_3, (xbox, ybox))
u = griddata((zslice.x, zslice.y), zslice.u, (xbox, ybox))
v = griddata((zslice.x, zslice.y), zslice.v, (xbox, ybox))
cb1 = -0.5 # -4.0
cb2 = 0.5 # 0.0
lev1 = np.linspace(cb1, cb2, 21)
cbar = ax.contourf(xbox, ybox, omega3, cmap="RdYlBu_r", extend='both', levels=lev1)
# cbar.cmap.set_under('#ff0000')
# cbar.cmap.set_over('#4169e1') # #0000ff
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.set_title(r"$t u_\infty / \delta_0=927.75$", fontsize=textsize-2)
ax.set_xlim(0.5, 3) #(1.0, 4.0)
ax.set_ylim(-1, 0) # (-1.5, 0.0)
rg2 = np.linspace(cb1, cb2, 5)
# cbaxes = fig.add_axes([0.17, 0.72, 0.18, 0.05])  # x, y, width, height
cbar = plt.colorbar(cbar, orientation="vertical", ticks=rg2, 
                    pad=0.02, extendrect=True, shrink=0.9)
cbar.ax.tick_params(labelsize=numsize)
cbar.set_label(r"$\omega_z$", y=1.1, labelpad=-18, 
               rotation=0, fontsize=textsize)
ax.streamplot(
    xbox,
    ybox,
    u,
    v,
    density=[2.0, 1.5],
    color="k",
    linewidth=0.6,
    arrowstyle='->',
    integration_direction="both"
)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "VortexPairTS3.svg", bbox_inches="tight")
plt.show()