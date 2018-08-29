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
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Snapshots/"
SaveFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Test"
path = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/DMD/"
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
NewFrame = DataFrame.query("x>=-5.0 & x<=20.0 & y>=-3.0 & y<=5.0")
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
var = 'v'
fa = 1.7*1.7*1.4
Snapshots = DataFrame[var]
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(InFolder + dirs[i+1])
        if np.shape(TempFrame) != np.shape(DataFrame):
            sys.exit('The input snapshots does not match!!!')
        Snapshots = np.vstack((Snapshots, TempFrame[var]))
        DataFrame += TempFrame
Snapshots = Snapshots.T  
Snapshots = Snapshots[ind, :] 
Snapshots = Snapshots*fa
m, n = np.shape(Snapshots)
AveFlow = DataFrame/np.size(dirs)
meanflow = AveFlow.query("x>=-5.0 & x<=20.0 & y>=-3.0 & y<=5.0")

# %%   
Snapshots1 = Snapshots[:, :-1]
timepoints = np.arange(450, 699.5 + 0.5, 0.5)
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
# %% DMD
predmd = DMD(Snapshots)
with timer("DMD computing"):
    eigval, phi = predmd.dmd_standard(fluc=True)

# %% 
coeff = predmd.dmd_amplitude(opt='spdmd')
dynamics = predmd.dmd_dynamics(timepoints)
residual = predmd.dmd_residual
print("The residuals of DMD is ", residual)

with timer("Precompute SPDMD amplitudes"):
    predmd.spdmd_amplitude()
# %% SPDMD
gamma = [900, 1000] # np.logspace(2.7, 3.0, 5) # around the value of snapshots NO
with timer("SPDMD computing"):
    ans = predmd.compute_spdmd(gamma=gamma)
print("The nonzero amplitudes of each gamma:", ans.Nz)
# %% decide the value of gamma to show SPDMD
sp = 0
sp_no = ans.Nz[sp]
sp_gamma = ans.gamma[sp]
r = np.size(eigval)
sp_ind = np.arange(r)[ans.nonzero[:, sp]]
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

# %% Mode frequency specturm
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
phi_abs = np.linalg.norm(phi*coeff, axis=0)
phi_max = np.max(phi_abs)
phi1 = phi_abs/phi_max
freq = predmd.omega/2/np.pi
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
plt.savefig(path + 'DMDFreqSpectrum.svg', bbox_inches='tight')
plt.show()
# %%

# %% Add isoline for boudary layer edge
meanu = griddata((meanflow.x, meanflow.y), meanflow.u, (x, y))
umax = meanu[-1,:]
# umax = np.amax(u, axis = 0)
rg2  = (x[1,:]<10.375) # in front of the shock wave
umax[rg2] = 1.0
rg1  = (x[1,:]>=10.375)
umax[rg1] = 0.95
meanu  = meanu/(np.transpose(umax))
corner = (x < 0.0) & (y < 0.0)
meanu[corner] = np.nan # mask the corner
rg1 = (y>0.3*np.max(y)) # remove the upper part
meanu[rg1] = np.nan

# %% Add Mach isoline for boudary layer edge
Ma_inf = 1.7
c = meanflow.u**2+meanflow.v**2+meanflow.w**2
meanflow['Mach'] = Ma_inf * np.sqrt(c/meanflow['T'])
meanma = griddata((meanflow.x, meanflow.y), meanflow.Mach, (x, y))
meanma[corner] = np.nan

# %% specific mode in real space
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
ind = 21
num = sp_ind[ind-1] # ind from small to large->freq from low to high
modeflow = phi[:, num].real
print("The limit value: ", np.min(modeflow), np.max(modeflow))
u = griddata((xval, yval), modeflow, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=18)
fig, ax = plt.subplots(figsize=(6, 2))
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
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.18, 0.76, 0.24, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar1.set_label(r'$\Re(\phi_{})$'.format(var), rotation=0, fontdict=font)
cbaxes.tick_params(labelsize=numsize)
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
ax.contour(x, y, meanu, levels=0.0,
           linewidths=1.0, linestyles=':', colors='k')
ax.contour(x, y, meanma, levels=1.0,
           linewidths=1.0, linestyles=':', colors='green')
plt.savefig(path+'DMDMode'+str(ind)+'Real.svg', bbox_inches='tight')
plt.show()

# % specific mode in imaginary space
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
ax.contour(x, y, meanu, levels=0.0,
           linewidths=1.0, linestyles=':', colors='k')
ax.contour(x, y, meanma, levels=1.0,
           linewidths=1.0, linestyles=':', colors='green')
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
# %% Time evolution of each mode (first several modes)
"""
plt.figure(figsize=(10, 5))
matplotlib.rc('font', size=18)
plt.plot(timepoints[:-1], dynamics[0, :].real*phi[0, 0].real)
plt.plot(timepoints[:-1], dynamics[12, :].real*phi[0, 12].real)
plt.plot(timepoints[:-1], dynamics[22, :].real*phi[0, 22].real)
plt.xlabel(r'$tu_\infty/\delta_0/$')
plt.ylabel(r'$\phi$')
plt.grid(b=True, which='both', linestyle=':')
plt.savefig(path + 'DMDModeTemp' + str(ind) + '.svg', bbox_inches='tight')
plt.show()
"""
# %% Reconstruct flow field using DMD
"""
tind = 0
N_modes = np.shape(phi)[1]
reconstruct = predmd.reconstruct(predmd.modes, predmd.amplit, predmd.Vand)
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
newflow = phi[:, :N_modes]@dynamics[:N_modes, tind]
# newflow = reconstruct[:,tind]
u = griddata((xval, yval), meanflow+newflow.real, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(12, 4))
lev1 = np.linspace(-0.20, 1.15, 18)
cbar = ax.contourf(x, y, u, cmap='rainbow', levels=lev1, extend="both")
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('b')
cbar.cmap.set_over('r')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
# add colorbar
rg2 = np.linspace(-0.20, 1.15, 4)
cbaxes = fig.add_axes([0.16, 0.76, 0.18, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar1.set_label(r'$u/u_{\infty}$', rotation=0, fontdict=font)
cbaxes.tick_params(labelsize=numsize)
plt.savefig(path+'DMDReconstructFlow.svg', bbox_inches='tight')
plt.show()

err = Snapshots1 - (reconstruct.real+np.tile(meanflow.reshape(m, 1), (1, n-1)))
print("Errors of DMD: ", np.linalg.norm(err)/n)
# how about residuals???

# %% Test DMD using meaning flow
def DMDMeanflow(Snapshots):
    flow = DMD(Snapshots)
    n = np.shape(Snapshots)[1]
    with timer("DMD computing"):
        eigval, phi = flow.dmd_standard()
    
    coeff = flow.dmd_amplitude()
    dynamics = flow.dmd_dynamics(timepoints)
    residual = flow.dmd_residual
    print("The residuals of DMD is ", residual)
    # Reconstruct flow field using DMD
    # tind = 0
    x, y = np.meshgrid(np.unique(xval), np.unique(yval))
    newflow = np.reshape(phi[:, 0], (m,1))@np.reshape(dynamics[0, :], (1, n-1))
    meanflow = np.mean(newflow.real, axis=1)
    # newflow = reconstruct[:,tind]
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
    plt.savefig(path + 'DMDMeanFlow.svg', bbox_inches='tight')
    plt.show()
    ## Original Meanflow
    origflow = np.mean(Snapshots, axis=1)
    u = griddata((xval, yval), origflow, (x, y))
    corner = (x < 0.0) & (y < 0.0)
    u[corner] = np.nan
    matplotlib.rc('font', size=18)
    fig, ax = plt.subplots(figsize=(12, 4))
    lev1 = np.linspace(-0.20, 1.15, 18)
    cbar = ax.contourf(x, y, u, cmap='rainbow', levels=lev1)  # ,extend="both")
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
    plt.savefig(path + 'OrigMeanFlow.svg', bbox_inches='tight')
    plt.show()
    print("Errors of MeanFlow: ", np.linalg.norm(meanflow-origflow)/n)


#  DMD for mean flow
# DMDMeanflow(Snapshots)
"""
