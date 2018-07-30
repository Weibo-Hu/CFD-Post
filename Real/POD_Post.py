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
import FlowVar as fv
from timer import timer
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
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
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/5/"
SaveFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Test"
path = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/All/"
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
NewFrame = DataFrame.query("x>=-5.0 & x<=13.0 & y>=-3.0 & y<=5.0")
#NewFrame = DataFrame.query("x>=9.0 & x<=13.0 & y>=-3.0 & y<=5.0")
ind = NewFrame.index.values
xval = NewFrame['x']
yval = NewFrame['y']
x1 = -5.0
x2 = 13.0
y1 = -3.0
y2 = 5.0
# set range to do POD and DMD: (x:-5~8, y:-3~5, make some tests)
# (x:8~15, y:-3~3)
del DataFrame
with timer("Load Data"):
    Snapshots = np.vstack(
        [pd.read_hdf(InFolder + dirs[i])['u'] for i in range(np.size(dirs))])
    Snapshots = Snapshots.T
Snapshots = Snapshots[ind, :]
m, n = np.shape(Snapshots)
# %% POD
timepoints = np.arange(550, 659.5 + 0.5, 0.5)
with timer("POD computing"):
    eigval, eigvec, phi, coeff = \
        rm.POD(Snapshots, SaveFolder, fluc='True', method='svd')

# %% Eigvalue Spectrum
EFrac, ECumu, N_modes = rm.POD_EigSpectrum(95, eigval)
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
ax1.set_xlabel('Mode')
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
# %% specific mode in space
ind = 8
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
newflow = phi[:, ind - 1]*coeff[ind - 1, 0]
u = griddata((xval, yval), newflow, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=20)
fig, ax = plt.subplots(figsize=(12, 4))
c1 = -0.04
c2 = 0.04
lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, levels=lev1, cmap='RdBu_r', extend='both') 
ax.contour(x, y, u, levels=lev2, colors='k', linewidths=0.8, extend='both')
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'), extend='both')
# cbar = ax.contour(x, y, u, levels=lev2, extend='both')
# plt.clabel(cbar, inline=1, fontsize=16)
ax.grid(b=True, which='both', linestyle=':')  # blue, grey, red
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=16)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar1.set_label(r'$\varphi (u)$', rotation=0, fontdict=font)
cbaxes.tick_params(labelsize=16)
plt.savefig(path+'PODMode'+str(ind)+'.svg', bbox_inches='tight')
plt.show()
# %% First several modes with time and WPSD
fig, ax = plt.subplots(figsize=(8, 4))
matplotlib.rc('font', size=18)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
lab = []
NO = np.arange(7, 9, 1)
ax.plot(timepoints, coeff[NO[0]-1, :], 'k-')
lab.append('Mode '+str(NO[0]-1))
ax.plot(timepoints, coeff[NO[1]-1, :], 'k:')
lab.append('Mode '+str(NO[1]-1))
ax.legend(lab, frameon=False)
ax.set_xlabel(r'$tu_\infty/\delta_0$')
ax.set_ylabel(r'$\phi (u)$')
ax.tick_params(labelsize=16)
plt.grid(b=True, which='both', linestyle=':')
plt.savefig(path + 'PODModeTemp' + str(78) + '.svg', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
matplotlib.rc('font', size=18)
freq, psd = fv.FW_PSD(coeff[NO[0]-1, :], timepoints, 2)
ax.semilogx(freq, psd, 'k-')
freq, psd = fv.FW_PSD(coeff[NO[1]-1, :], timepoints, 2)
ax.semilogx(freq, psd, 'k:')
ax.legend(lab, frameon=False)
plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_xlabel(r'$f\delta_0/U_\infty$')
ax.set_ylabel('WPSD, unitless')
ax.tick_params(labelsize=16)
plt.grid(b=True, which='both', linestyle=':')
plt.savefig(path + 'POD_WPSDModeTemp' + str(78) + '.svg', bbox_inches='tight')
plt.show()
# %% Reconstruct flow field using POD
tind = 0
meanflow = np.mean(Snapshots, axis=1)
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
newflow = phi[:,:N_modes]@coeff[:N_modes, tind]  
# np.reshape(phi[:, ind-1], (m,1))@np.reshape(coeff[ind-1, :], (1, n))
u = griddata((xval, yval), meanflow+newflow, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=18)
fig, ax = plt.subplots(figsize=(12, 4))
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
lev1 = np.linspace(-0.20, 1.15, 18)
cbar = ax.contourf(x, y, u, cmap='rainbow', levels=lev1, extend="both")
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
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
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
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
    # %% Eigvalue Spectrum
    EFrac, ECumu, N_modes = rm.POD_EigSpectrum(95, eigval)
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
    ax1.set_xlabel('Mode')
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
    plt.savefig(path+'MeanPODEigSpectrum.svg', bbox_inches='tight')
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
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
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
