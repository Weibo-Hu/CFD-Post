#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:00:39 2018
    This code for POD post processing.
@author: weibo
"""
#%% Load necessary module
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ReducedModel as rm
from timer import timer
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import griddata
import os

plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',
    #'color' : 'k',
    'weight': 'normal',
}
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('font', **font)

#%% load data
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/4/"
SaveFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Test"
path = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
NewFrame = DataFrame.query("x>=-5.0 & x<=10.0 & y>=-3.0 & y<=5.0")
ind = NewFrame.index.values
xval = NewFrame['x']
yval = NewFrame['y']
# set range to do POD and DMD: (x:-5~8, y:-3~5, make some tests)
# (x:8~15, y:-3~3)
del DataFrame
with timer("Load Data"):
    Snapshots = np.vstack(
        [pd.read_hdf(InFolder + dirs[i])['u'] for i in range(np.size(dirs))])
    Snapshots = Snapshots.T
Snapshots = Snapshots[ind, :]
m, n = np.shape(Snapshots)
#%% POD
timepoints = np.arange(330, 549.5 + 0.5, 0.5)
with timer("POD computing"):
    eigval, eigvec, phi, coeff = \
        rm.POD(Snapshots, SaveFolder, fluc='True', method='svd')

#%% Eigvalue Spectrum
EFrac, ECumu, N_modes = rm.POD_EigSpectrum(99, eigval)
matplotlib.rc('font', size=14)
fig1, ax1 = plt.subplots(figsize=(6,5))
xaxis = np.arange(0, N_modes + 1)
ax1.scatter(
    xaxis[1:],
    EFrac[:N_modes],
    c='black',
    marker='o',
    s=EFrac[:N_modes]*2,
)   # fraction energy of every eigval mode
#ax1.legend('E_i')
ax1.set_ylim(bottom=0)
ax1.set_xlabel(r'$i$')
ax1.set_ylabel(r'$E_i$')
ax1.grid(b=True, which='both', linestyle=':')

ax2 = ax1.twinx()   # cumulation energy of first several modes
#ax2.fill_between(xaxis, ECumu[:N_modes], color='grey', alpha=0.5)
ESum = np.zeros(N_modes+1)
ESum[1:] = ECumu[:N_modes]
ax2.plot(xaxis, ESum, color='grey', label=r'$ES_i$')
ax2.set_ylim([0, 100])
ax2.set_ylabel(r'$ES_i$')
fig1.set_size_inches(5, 4, forward=True)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(path+'PODEigSpectrum.svg', bbox_inches='tight')
plt.show()
#%% specific mode in space
ind = 1
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
newflow = phi[:, ind - 1]*coeff[ind - 1, 0]
u = griddata((xval, yval), newflow, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=18)
fig, ax = plt.subplots(figsize=(12, 4))
lev1 = [-0.01, 0.01]
cbar = ax.contourf(x, y, u, levels=lev1, \
                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'), extend='both') #blue, grey, red
ax.grid(b=True, which='both', linestyle=':')
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=14)
cbar.cmap.set_under('b')
cbar.cmap.set_over('r')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
plt.savefig(path+'PODMode'+str(ind)+'.svg', bbox_inches='tight')
plt.show()
#%% First several modes with time
plt.figure(figsize=(10, 5))
for i in range(2):
    plt.plot(timepoints, coeff[i, :])
plt.xlabel(r'$tu_\infty/\delta_0/$')
plt.ylabel(r'$\phi$')
plt.grid(b=True, which='both', linestyle=':')
plt.savefig(path + 'PODModeTemp' + str(ind) + '.svg', bbox_inches='tight')
plt.show()
#%% Reconstruct flow field using POD
tind = 0
meanflow = np.mean(Snapshots, axis=1)
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
newflow = phi[:,:N_modes]@coeff[:N_modes, tind] #\
#    np.reshape(phi[:, ind-1], (m,1))@np.reshape(coeff[ind-1, :], (1, n))
u = griddata((xval, yval), meanflow+newflow, (x, y))
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
plt.savefig(path+'PODReconstructFlow.svg', bbox_inches='tight')
plt.show()

reconstruct = phi @ coeff
err = Snapshots - (reconstruct + np.tile(meanflow.reshape(m, 1), (1, n)))
print("Errors of POD: ", np.linalg.norm(err)/n)
# %% Test POD using meaning flow
def PODMeanflow(Snapshots):
    with timer("POD mean flow computing"):
        eigval, eigvec, phi, coeff = \
            rm.POD(Snapshots, SaveFolder, method='svd')
    ind = 1
    m, n = np.shape(Snapshots)
    x, y = np.meshgrid(np.unique(xval), np.unique(yval))
    newflow = \
        np.reshape(phi[:, ind-1], (m,1))@np.reshape(coeff[ind-1, :], (1, n))
    meanflow = np.mean(newflow.real, axis=1)
    u = griddata((xval, yval), meanflow, (x, y))
    corner = (x < 0.0) & (y < 0.0)
    u[corner] = np.nan
    matplotlib.rc('font', size=18)
    fig, ax = plt.subplots(figsize=(12, 4))
    lev1 = np.linspace(-0.20, 1.15, 18)
    cbar = ax.contourf(x, y, u, cmap='rainbow', levels=lev1) #, extend="both")
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
    plt.savefig(path+'PODMeanFlow.svg', bbox_inches='tight')
    plt.show()
    #%% Original MeanFlow
    origflow = np.mean(Snapshots, axis=1)
    u = griddata((xval, yval), origflow, (x, y))
    corner = (x < 0.0) & (y < 0.0)
    u[corner] = np.nan
    matplotlib.rc('font', size=18)
    fig, ax = plt.subplots(figsize=(12, 4))
    lev1 = np.linspace(-0.20, 1.15, 18)
    cbar = ax.contourf(x, y, u, cmap='rainbow', levels=lev1) #, extend="both")
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
    plt.savefig(path+'OrigMeanFlow.svg', bbox_inches='tight')
    plt.show()
    print("Errors of MeanFlow: ", np.linalg.norm(meanflow - origflow)/n)
#%% POD for mean flow
PODMeanflow(Snapshots)
