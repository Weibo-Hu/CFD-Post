#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 18:57:20 2018
    This code for DMD post processing.
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
import os

plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',  # 'color' : 'k',
    'weight': 'normal',
}
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('font', **font)

# %% load data
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/4/"
SaveFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Test"
path = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
NewFrame = DataFrame.query("x>=-5.0 & x<=10.0 & y>=-3.0 & y<=5.0")
ind = NewFrame.index.values
xval = NewFrame['x']
yval = NewFrame['y']
del DataFrame
with timer("Load Data"):
    Snapshots = np.vstack(
        [pd.read_hdf(InFolder + dirs[i])['u'] for i in range(np.size(dirs))])
    Snapshots = Snapshots.T
Snapshots = Snapshots[ind, :]
m, n = np.shape(Snapshots)
# %% DMD
timepoints = np.arange(440, 549.5+0.5, 0.5)
test1 = DMD(Snapshots)
with timer("DMD computing"):
    eigval, phi, residual = test1.dmd_standard(fluc='True')
print("The residuals of DMD is ", residual)
coeff = test1.dmd_amplitude()
dynamics = test1.dmd_dynamics(timepoints)
coeff = test1.SPDMD_J()
# with timer("DMD computing"):
#     eigval, phi, U, eigvec, residual = \
#         rm.DMD_Standard(Snapshots, SaveFolder, fluc='True')
# print("The residuals of DMD is ", residual)
# coeff = rm.DMD_Amplitude(Snapshots, U, eigvec, phi, eigval) #, lstsq='False')
# dynamics = rm.DMD_Dynamics(eigval, coeff, timepoints)
 
# %% Eigvalue Spectrum
matplotlib.rc('font', size=14)
fig1, ax1 = plt.subplots(figsize=(6, 6))
unit_circle = plt.Circle((0., 0.), 1., color='grey', linestyle='-', fill=False,
                         label='unit circle', linewidth=5.0, alpha=0.7)
ax1.add_artist(unit_circle)
ax1.scatter(eigval.real, eigval.imag, marker='o',\
            facecolor='none', edgecolors='k', s=18)
limit = np.max(np.absolute(eigval))+0.1
ax1.set_xlim((-limit, limit))
ax1.set_ylim((-limit, limit))
plt.xlabel(r'$\Re(\lambda)$')
plt.ylabel(r'$\Im(\lambda)$')
ax1.grid(b=True, which='both', linestyle=':')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(path + 'DMDEigSpectrum.svg', bbox_inches='tight')
plt.show()
# %% specific mode in space
ind = 1
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
modeflow = phi[:, ind - 1].real*coeff[ind - 1].real
u = griddata((xval, yval), modeflow, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=18)
fig, ax = plt.subplots(figsize=(12, 4))
lev1 = [-0.01, 0.01]
cbar = ax.contourf(x, y, u, levels=lev1, \
                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'), extend='both') #blue, grey, red
cbar = ax.contourf(x, y, u, colors=('#66ccff', '#e6e6e6', '#ff4d4d')) #blue, grey, red
ax.grid(b=True, which='both', linestyle=':')
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=14)
cbar.cmap.set_under('b')
cbar.cmap.set_over('r')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
plt.savefig(path+'DMDMode'+str(ind)+'.svg', bbox_inches='tight')
plt.show()
# %% Time evolution of each mode (first several modes)
plt.figure(figsize=(10, 5))
matplotlib.rc('font', size=18)
for i in range(2):
    plt.plot(timepoints, dynamics[ind-1,:].real*phi[0,ind-1].real)
plt.xlabel(r'$tu_\infty/\delta_0/$')
plt.ylabel(r'$\phi$')
plt.grid(b=True, which='both', linestyle=':')
plt.savefig(path + 'DMDModeTemp' + str(ind) + '.svg', bbox_inches='tight')
plt.show()
# %% Reconstruct flow field using DMD
tind = 0
N_modes = np.shape(phi)[1]
reconstruct = test1.dmd_reconstruct(phi, dynamics)
meanflow = np.mean(Snapshots, axis=1)
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
newflow = phi[:,:N_modes]@dynamics[:N_modes, tind]
#newflow = reconstruct[:,tind]
u = griddata((xval, yval), meanflow+newflow.real, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=18)
fig, ax = plt.subplots(figsize=(12, 4))
lev1 = np.linspace(-0.20, 1.15, 18)
cbar = ax.contourf(x, y, u, cmap='rainbow', levels=lev1, extend="both")
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=14)
cbar.cmap.set_under('b')
cbar.cmap.set_over('r')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
# add colorbar
rg2 = np.linspace(-0.20, 1.15, 4)
cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar1.set_label(r'$u/u_{\infty}$', rotation=0, fontdict=font)
cbaxes.tick_params(labelsize=14)
plt.savefig(path+'DMDReconstructFlow.svg', bbox_inches='tight')
plt.show()

err = Snapshots - (reconstruct.real+np.tile(meanflow.reshape(m,1), (1, n)))
print("Errors of DMD: ", np.linalg.norm(err)/n)
# how about residuals???
# %% Test DMD using meaning flow
def DMDMeanflow(Snapshots):
    flow = DMD(Snapshots)
    m, n = np.shape(Snapshots)
    with timer("DMD computing"):
        eigval, phi, residual = flow.dmd_standard()
    print("The residuals of DMD is ", residual)
    coeff = flow.dmd_amplitude()
    dynamics = flow.dmd_dynamics(timepoints)
    # Reconstruct flow field using DMD
    tind = 0
    x, y = np.meshgrid(np.unique(xval), np.unique(yval))
    newflow = phi @ dynamics
    meanflow = np.mean(newflow.real, axis=1)
    #newflow = reconstruct[:,tind]
    u = griddata((xval, yval), meanflow, (x, y))
    corner = (x < 0.0) & (y < 0.0)
    u[corner] = np.nan
    matplotlib.rc('font', size=18)
    fig, ax = plt.subplots(figsize=(12, 4))
    lev1 = np.linspace(-0.20, 1.15, 18)
    cbar = ax.contourf(x, y, u, cmap='rainbow', levels=lev1, extend="both")
    ax.set_xlim(-10.0, 30.0)
    ax.set_ylim(-3.0, 10.0)
    ax.tick_params(labelsize=14)
    cbar.cmap.set_under('b')
    cbar.cmap.set_over('r')
    ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
    ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
    # add colorbar
    rg2 = np.linspace(-0.20, 1.15, 4)
    cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
    cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
    cbar1.set_label(r'$u/u_{\infty}$', rotation=0, fontdict=font)
    cbaxes.tick_params(labelsize=14)
    plt.savefig(path + 'DMDMeanFlow.svg', bbox_inches='tight')
    plt.show()
    # %% Original Meanflow
    origflow = np.mean(Snapshots, axis=1)
    u = griddata((xval, yval), origflow, (x, y))
    corner = (x < 0.0) & (y < 0.0)
    u[corner] = np.nan
    matplotlib.rc('font', size=18)
    fig, ax = plt.subplots(figsize=(12, 4))
    lev1 = np.linspace(-0.20, 1.15, 18)
    cbar = ax.contourf(x, y, u, cmap='rainbow', levels=lev1)  #, extend="both")
    ax.set_xlim(-10.0, 30.0)
    ax.set_ylim(-3.0, 10.0)
    ax.tick_params(labelsize=14)
    cbar.cmap.set_under('b')
    cbar.cmap.set_over('r')
    ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
    ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
    # add colorbar
    rg2 = np.linspace(-0.20, 1.15, 4)
    cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
    cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
    cbar1.set_label(r'$u/u_{\infty}$', rotation=0, fontdict=font)
    cbaxes.tick_params(labelsize=14)
    plt.savefig(path + 'OrigMeanFlow.svg', bbox_inches='tight')
    plt.show()
    print("Errors of MeanFlow: ", np.linalg.norm(meanflow-origflow)/n)
# %% DMD for mean flow
DMDMeanflow(Snapshots)
