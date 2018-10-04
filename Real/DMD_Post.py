#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:59:58 2018

@author: weibo
"""

# %% Load necessary module
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timer import timer
from DMD import DMD
from scipy.interpolate import griddata
from sparse_dmd import dmd, sparse
import os
import sys
plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',  # 'color' : 'k',
    'weight': 'normal',
}

matplotlib.rc('font', **font)
textsize = 18
numsize = 15

# %% load data
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots/"
SaveFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Test"
path = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/DMD1/"
timepoints = np.arange(650, 899.50 + 0.5, 0.5)
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
DataFrame['walldist'] = DataFrame['y']
DataFrame.loc[DataFrame['x'] >= 0.0, 'walldist'] += 3.0
NewFrame = DataFrame.query("x>=-5.0 & x<=20.0 & walldist>0.0 & y<=5.0")
#NewFrame = DataFrame.query("x>=9.0 & x<=13.0 & y>=-3.0 & y<=5.0")
ind = NewFrame.index.values
xval = NewFrame['x']
yval = NewFrame['y']
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
x1 = -5.0
x2 = 20.0
y1 = -3.0
y2 = 5.0
#with timer("Load Data"):
#    Snapshots = np.vstack(
#        [pd.read_hdf(InFolder + dirs[i])['u'] for i in range(np.size(dirs))])
var = 'u'
fa = 1.0
Snapshots = DataFrame[var]
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(InFolder + dirs[i+1])
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        Snapshots = np.vstack((Snapshots, TempFrame[var]))
        DataFrame += TempFrame
Snapshots = Snapshots.T  
Snapshots = Snapshots[ind, :] 
Snapshots = Snapshots*fa
m, n = np.shape(Snapshots)
AveFlow = DataFrame/np.size(dirs)
meanflow = AveFlow.query("x>=-5.0 & x<=20.0 & y>=-3.0 & y<=5.0")

# %% DMD 
Snapshots1 = Snapshots[:, :-1]
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
dt = 0.5
bfs = dmd.DMD(Snapshots, dt=dt)
with timer("DMD computing"):
    bfs.compute()
print("The residuals of DMD is ", bfs.residuals)
eigval = bfs.eigval

# %% SPDMD
bfs1 = sparse.SparseDMD(Snapshots, bfs, dt=dt)
gamma = [450, 500]
with timer("SPDMD computing"):
    bfs1.compute_sparse(gamma)
print("The nonzero amplitudes of each gamma:", bfs1.sparse.Nz)
sp = 0
bfs1.sparse.Nz[sp]
bfs1.sparse.gamma[sp] 
r = np.size(eigval)
sp_ind = np.arange(r)[bfs1.sparse.nonzero[:, sp]]

# %% Eigvalue Spectrum
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(5, 4.5))
unit_circle = plt.Circle((0., 0.), 1., color='grey', linestyle='-', fill=False,
                         label='unit circle', linewidth=7.0, alpha=0.5)
ax1.add_artist(unit_circle)
ax1.scatter(eigval.real, eigval.imag, marker='o',
            facecolor='none', edgecolors='k', s=18)
sp_eigval = eigval[sp_ind]
ax1.scatter(sp_eigval.real, sp_eigval.imag, marker='o',
            facecolor='gray', edgecolors='gray', s=18)
limit = np.max(np.absolute(eigval))+0.1
ax1.set_xlim((-limit, limit))
ax1.set_ylim((-limit, limit))
ax1.tick_params(labelsize=numsize)
plt.xlabel(r'$\Re(\mu_i)$')
plt.ylabel(r'$\Im(\mu_i)$')
ax1.grid(b=True, which='both', linestyle=':')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(path + 'DMDEigSpectrum.svg', bbox_inches='tight')
plt.show()

# %% discard the bad DMD modes
#ind0 = np.where(np.abs(eigval) > 0.95)[0][1:]
phi = bfs.modes
freq = bfs.omega/2/np.pi
beta = bfs.beta
coeff = bfs.amplitudes

# %% Mode frequency specturm
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
phi1 = np.abs(coeff)/np.max(np.abs(coeff))
ind1 = freq > 0.0 
ax1.set_xscale("log")
ax1.vlines(freq[ind1], [0], phi1[ind1], color='k', linewidth=1.0)
ind3 = bfs1.sparse.nonzero[:, sp] & ind1
ax1.scatter(freq[ind3], phi1[ind3], marker='o',
            facecolor='gray', edgecolors='gray', s=15.0)
ax1.set_ylim(bottom=0.0)
ax1.tick_params(labelsize=numsize, pad=6)
plt.xlabel(r'$f \delta_0/u_\infty$')
plt.ylabel(r'$|\phi_i|$')
ax1.grid(b=True, which='both', linestyle=':')
plt.savefig(path + 'DMDFreqSpectrum.svg', bbox_inches='tight')
plt.show()

# %% specific mode in real space
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
ind = 6
num = sp_ind[ind-1] # ind from small to large->freq from low to high
modeflow = phi[:, num].real
print("The limit value: ", np.min(modeflow), np.max(modeflow))
u = griddata((xval, yval), modeflow, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(5, 2))
c1 = -0.026 #-0.024
c2 = 0.022  #0.018
lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, levels=lev1, cmap='RdBu_r') #, extend='both') 
#cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.grid(b=True, which='both', linestyle=':')
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.18, 0.76, 0.24, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(numsize)
cbar1.update_ticks()
cbar1.set_label(r'$\Re(\phi_{})$'.format(var), rotation=0, fontdict=font)
cbaxes.tick_params(labelsize=numsize)
# Add shock wave
shock = np.loadtxt(path+'ShockPosition.dat', skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], 'g', linewidth=1.0)
# Add sonic line
sonic = np.loadtxt(path+'SonicLine.dat', skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], 'g--', linewidth=1.0)
# Add boundary layer
boundary = np.loadtxt(path+'BoundaryLayer.dat', skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], 'k', linewidth=1.0)
# Add dividing line(separation line)
dividing = np.loadtxt(path+'DividingLine.dat', skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], 'k--', linewidth=1.0)

plt.savefig(path+var+'DMDMode'+str(ind)+'Real.svg', bbox_inches='tight')
plt.show()

# %% specific mode in imaginary space
imagflow = phi[:, num].imag
print("The limit value: ", np.min(imagflow), np.max(imagflow))
u = griddata((xval, yval), imagflow, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=18)
fig, ax = plt.subplots(figsize=(6, 2))
c1 = -0.024 #-0.024
c2 = 0.022  #0.018
lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, levels=lev1, cmap='RdBu_r') #, extend='both') 
#cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.grid(b=True, which='both', linestyle=':')
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.18, 0.76, 0.24, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar1.set_label(r'$\Im(\phi_{})$'.format(var), rotation=0, fontdict=font)
cbaxes.tick_params(labelsize=numsize)
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)

plt.savefig(path+'DMDMode'+str(ind)+'Imag.svg', bbox_inches='tight')
plt.show()

# %% growth rate of a specific mode
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(6, 4.5))
beta = predmd.beta
ind1 = freq > 0.0
ax1.set_xscale("log")
ax1.vlines(freq[ind1], [0], phi1[ind1], color='k', linewidth=1.0)
ind2 = ans.nonzero[:, sp] & ind1
ax1.scatter(freq[ind2], phi1[ind2], marker='o',
            facecolor='gray', edgecolors='gray', s=15.0)
ax1.set_ylim(bottom=0.0)
ax1.tick_params(labelsize=numsize)
plt.xlabel(r'$f \delta_0/u_\infty$')
plt.ylabel(r'$|\phi_i|$')
ax1.grid(b=True, which='both', linestyle=':')
plt.savefig(path + 'DMDGrowthRate.svg', bbox_inches='tight')
plt.show()