#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:59:58 2018

@author: weibo
"""

# %% Load necessary module
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from timer import timer
from DMD import DMD
from scipy.interpolate import griddata
from sparse_dmd import dmd, sparse
import os
import sys
import plt2pandas as p2p
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
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots1/"
SaveFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Test"
path = "/media/weibo/Data1/BFS_M1.7L_0505/temp/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/MeanFlow/"
timepoints = np.arange(650.0, 949.50 + 0.5, 0.5)
dirs = sorted(os.listdir(InFolder))
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
DataFrame = pd.read_hdf(InFolder + dirs[0])
DataFrame['walldist'] = DataFrame['y']
DataFrame.loc[DataFrame['x'] >= 0.0, 'walldist'] += 3.0
NewFrame = DataFrame.query("x>=-5.0 & x<=45.0 & walldist>=0.0 & y<=5.0")
#NewFrame = DataFrame.query("x>=9.0 & x<=13.0 & y>=-3.0 & y<=5.0")
ind = NewFrame.index.values
xval = DataFrame['x'][ind] # NewFrame['x']
yval = DataFrame['y'][ind] # NewFrame['y']
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
x1 = -5.0
x2 = 30.0
y1 = -3.0
y2 = 5.0
#with timer("Load Data"):
#    Snapshots = np.vstack(
#        [pd.read_hdf(InFolder + dirs[i])['u'] for i in range(np.size(dirs))])
var0 = 'u'
var1 = 'v'
var2 = 'p'
col = [var0, var1, var2]
fa = 1 #/(1.7*1.7*1.4)
FirstFrame = DataFrame[col].values
Snapshots = FirstFrame[ind].ravel(order='F')
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(InFolder + dirs[i+1])
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        NextFrame = TempFrame[col].values
        Snapshots = np.vstack((Snapshots, NextFrame[ind].ravel(order='F')))
        DataFrame += TempFrame
Snapshots = Snapshots.T  
# Snapshots = Snapshots[ind, :] 
Snapshots = Snapshots*fa
m, n = np.shape(Snapshots)
o = np.size(col)
if (m % o != 0):
    sys.exit("Dimensions of snapshots are wrong!!!")
m = int(m/o)
AveFlow = DataFrame/np.size(dirs)
meanflow = AveFlow.query("x>=-5.0 & x<=45.0 & y>=-3.0 & y<=5.0")

# %% DMD 
varset = { var0: [0, m],
           var1: [m, 2*m],
           var2: [2*m, 3*m]
        }
Snapshots1 = Snapshots[:, :-1]
dt = 0.5
bfs = dmd.DMD(Snapshots, dt=dt)
with timer("DMD computing"):
    bfs.compute()
print("The residuals of DMD is ", bfs.residuals)
eigval = bfs.eigval

# %% SPDMD
bfs1 = sparse.SparseDMD(Snapshots, bfs, dt=dt)
gamma = [700, 800, 850, 900]
with timer("SPDMD computing"):
    bfs1.compute_sparse(gamma)
print("The nonzero amplitudes of each gamma:", bfs1.sparse.Nz)

# %% 
sp = 0
bfs1.sparse.Nz[sp]
bfs1.sparse.gamma[sp] 
r = np.size(eigval)
sp_ind = np.arange(r)[bfs1.sparse.nonzero[:, sp]]

# %% Eigvalue Spectrum
var = var0
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
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
ax1.set_xlabel(r'$\Re(\mu_i)$')
ax1.set_ylabel(r'$\Im(\mu_i)$')
ax1.grid(b=True, which='both', linestyle=':')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(path+var+'DMDEigSpectrum.svg', bbox_inches='tight')
plt.show()

# %% discard the bad DMD modes
# bfs2 = bfs
bfs2 = bfs.reduce(0.999)
phi = bfs2.modes
freq = bfs2.omega/2/np.pi
beta = bfs2.beta
coeff = bfs2.amplitudes

# %% Mode frequency specturm
matplotlib.rc('font', size=textsize)
fig2, ax2 = plt.subplots(figsize=(6.5, 3.5))
psi = np.abs(coeff)/np.max(np.abs(coeff))
ind1 = freq > 0.0 
freq1 = freq[ind1]
psi1 = np.abs(coeff[ind1])/np.max(np.abs(coeff[ind1]))
ax2.set_xscale("log")
ax2.vlines(freq1, [0], psi1, color='k', linewidth=1.0)
# ind2 = bfs1.sparse.nonzero[:, sp] & ind1
ind2 = bfs1.sparse.nonzero[bfs2.ind, sp]
ind3 = ind2[ind1]
ax2.scatter(freq1[ind3], psi1[ind3], marker='o',
            facecolor='gray', edgecolors='gray', s=15)
ax2.set_ylim(bottom=0.0)
ax2.tick_params(labelsize=numsize, pad=6)
ax2.set_xlabel(r'$f \delta_0/u_\infty$')
ax2.set_ylabel(r'$|\psi_i|$')
ax2.grid(b=True, which='both', linestyle=':')
plt.savefig(path+var+'DMDFreqSpectrum.svg', bbox_inches='tight')
plt.show()

# %% specific mode in real space
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
var = var1
fa = 1.0 #1.7*1.7*1.4
ind = 13
num = sp_ind[ind] # ind from small to large->freq from low to high
freq1 = bfs.omega/2/np.pi
name = str(round(freq1[num], 3)).replace('.', '_') #.split('.')[1] # str(ind)
tempflow = bfs.modes[:, num].real
print('The frequency is', freq1[num])
modeflow = tempflow[varset[var][0]:varset[var][1]]
print("The limit value: ", np.min(modeflow)*fa, np.max(modeflow)*fa)
u = griddata((xval, yval), modeflow, (x, y))*fa
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(5, 2))
c1 = -0.007 #-0.024
c2 = 0.007  #0.018
lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, levels=lev1, cmap='RdBu_r') #, extend='both') 
#cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.28, 0.76, 0.34, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(numsize)
cbar1.update_ticks()
cbar1.set_label(r'$\Re(\phi_{})$'.format(var), rotation=0, 
                x=-0.23, labelpad=-29, fontsize=textsize)
cbaxes.tick_params(labelsize=numsize)
# Add shock wave
shock = np.loadtxt(path1+'Shock.dat', skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], 'g', linewidth=1.0)
# Add sonic line
sonic = np.loadtxt(path1+'SonicLine.dat', skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], 'g--', linewidth=1.0)
# Add boundary layer
boundary = np.loadtxt(path1+'BoundaryLayer.dat', skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], 'k', linewidth=1.0)
# Add dividing line(separation line)
dividing = np.loadtxt(path1+'DividingLine.dat', skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], 'k--', linewidth=1.0)

plt.savefig(path+var+'DMDMode'+name+'Real.svg', bbox_inches='tight')
plt.show()

# % specific mode in imaginary space
tempflow = bfs.modes[:, num].imag
imagflow = tempflow[varset[var][0]:varset[var][1]]
print("The limit value: ", np.min(imagflow)*fa, np.max(imagflow)*fa)
u = griddata((xval, yval), imagflow, (x, y))*fa
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=18)
fig, ax = plt.subplots(figsize=(5, 2))
c1 = -0.011 #-0.024
c2 = 0.011 #0.018
lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, levels=lev1, cmap='RdBu_r') #, extend='both') 
#cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.28, 0.76, 0.34, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(numsize)
cbar1.update_ticks()
cbar1.set_label(r'$\Im(\phi_{})$'.format(var), rotation=0, x=-0.23, labelpad=-29, fontsize=textsize)
cbaxes.tick_params(labelsize=numsize)
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
# Add shock wave
shock = np.loadtxt(path1+'Shock.dat', skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], 'g', linewidth=1.0)
# Add sonic line
sonic = np.loadtxt(path1+'SonicLine.dat', skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], 'g--', linewidth=1.0)
# Add boundary layer
boundary = np.loadtxt(path1+'BoundaryLayer.dat', skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], 'k', linewidth=1.0)
# Add dividing line(separation line)
dividing = np.loadtxt(path1+'DividingLine.dat', skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], 'k--', linewidth=1.0)

plt.savefig(path+var+'DMDMode'+name+'Imag.svg', bbox_inches='tight')
plt.show()

# %% growth rate of a specific mode
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(6, 4.5))
beta = bfs.beta
ind1 = freq > 0.0
ax1.set_xscale("log")
# ax1.vlines(freq[ind1], [0], phi1[ind1], color='k', linewidth=1.0)
# ind3 = bfs1.sparse.nonzero[:, sp] & ind1
ax1.scatter(freq[ind3], beta[ind3], marker='o',
            facecolor='gray', edgecolors='gray', s=15.0)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax1.tick_params(labelsize=numsize)
plt.xlabel(r'$f \delta_0/u_\infty$')
plt.ylabel(r'$\beta_i$')
ax1.grid(b=True, which='both', linestyle=':')
plt.savefig(path+var+'DMDGrowthRate.svg', bbox_inches='tight')
plt.show()

# %% save dataframe of reconstructing flow
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/DMD1/tec0.01/"
base = meanflow[col].values
base[:, 2] = meanflow['p'].values*1.7*1.7*1.4
ind = 1
num = sp_ind[ind-1] # ind from small to large->freq from low to high
print('The frequency is', freq[num])
phase = np.linspace(0, 2*np.pi, 16, endpoint=False).reshape(1, -1)
modeflow1 = phi[:,num].reshape(-1, 1) * coeff[num] \
           @ bfs.Vand[num, :].reshape(1, -1)
modeflow = phi[:,num].reshape(-1, 1) * coeff[num] * np.exp(phase*1j)
mag0 = 20
mag1 = 10
mag2 = 20
xarr = xval.values.reshape(-1, 1)
yarr = yval.values.reshape(-1, 1)
zarr = np.zeros((m, 1))
names = ['x', 'y', 'z', var0, var1, var2]
for ii in range(np.size(phase)):
    fluc = modeflow[:, ii].reshape((m, o), order='F')
    fluc[:, 0] = fluc[:, 0] * mag0
    fluc[:, 1] = fluc[:, 1] * mag1
    fluc[:, 2] = fluc[:, 2] * mag2
    newflow = base + fluc.real
    outfile = 'DMD'+str(timepoints[ii]+24)
    data = np.hstack((xarr, yarr, zarr, newflow))
    df = pd.DataFrame(data, columns=names)
    df1 = df.query("x>=0.0")
    with timer('save plt of t='+str(timepoints[ii]+24)):
        p2p.frame2plt(df1, path2, outfile, time=timepoints[ii]+24, zonename=1)

