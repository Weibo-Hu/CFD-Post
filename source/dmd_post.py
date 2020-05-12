#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:59:58 2018

@author: weibo
"""

# %% Load necessary module
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import numpy as np
import pandas as pd
from timer import timer
from DMD import DMD
from scipy.interpolate import griddata
from sparse_dmd import dmd, sparse
import os
from glob import glob
import sys
import plt2pandas as p2p
from matplotlib import colors as mcolors


# %% data path settings
path = "/media/weibo/IM1/BFS_M1.7Tur/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathD = path + "Domain/"
pathPOD = path + "POD/"
pathDMD = path + "DMD/"
pathSS = path + 'Slice/S_10a/'
timepoints = np.arange(951.00, 1350.50 + 0.5, 0.5)

# %% figures properties settings
plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',  # 'color' : 'k',
    'weight': 'normal',
}

matplotlib.rc('font', **font)
textsize = 12
numsize = 10
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# %% load data
dirs = glob(pathSS+'*.h5')
# dirs.sort( key=lambda f: int(''.join(filter(str.isdigit, f))) )
dirs = sorted(dirs)
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
# preload
DataFrame = pd.read_hdf(dirs[0])
DataFrame['walldist'] = DataFrame['y']
DataFrame.loc[DataFrame['x'] >= 0.0, 'walldist'] += 3.0
grouped = DataFrame.groupby(['x', 'y'])
DataFrame = grouped.mean().reset_index()
NewFrame = DataFrame.query("x>=-5.0 & x<=30.0 & walldist>=0.0 & y<=5.0")
ind = NewFrame.index.values
xval = DataFrame['x'][ind] # NewFrame['x']
yval = DataFrame['y'][ind] # NewFrame['y']
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
x1 = -5.0
x2 = 25.0
y1 = -3.0
y2 = 5.0
# load data
var0 = 'u'
var1 = 'v'
var2 = 'p'
var3 = 'T'
col = [var0, var1, var2, var3]
fa = 1 #/(1.7*1.7*1.4)
FirstFrame = DataFrame[col].values
Snapshots = FirstFrame[ind].ravel(order='F')
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(dirs[i+1])
        grouped = TempFrame.groupby(['x', 'y'])
        TempFrame = grouped.mean().reset_index()
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        NextFrame = TempFrame[col].values
        Snapshots = np.vstack((Snapshots, NextFrame[ind].ravel(order='F')))
        DataFrame += TempFrame
Snapshots = Snapshots.T  
# Snapshots = Snapshots[ind, :] 
# Snapshots = Snapshots*fa
m, n = np.shape(Snapshots)  # n: NO of snapshots
o = np.size(col)  # NO of variables
if (m % o != 0):
    sys.exit("Dimensions of snapshots are wrong!!!")
m = int(m/o)  # dimensions of every variable 
AveFlow = DataFrame/np.size(dirs)
meanflow = AveFlow.query("x>=-5.0 & x<=30.0 & y>=-3.0 & y<=5.0")

# %% DMD 
varset = { var0: [0, m],
           var1: [m, 2*m],
           var2: [2*m, 3*m],
           var3: [3*m, 4*m]
        }
dt = 0.5
bfs = dmd.DMD(Snapshots, dt=dt)
with timer("DMD computing"):
    bfs.compute()
print("The residuals of DMD is ", bfs.residuals)
eigval = bfs.eigval
# %% save results
meanflow.to_hdf(pathDMD + 'MeanFlow.h5', 'w', format='fixed')

np.save(pathDMD + 'eigval', bfs.eigval) # \mu
np.save(pathDMD + 'modes', bfs.modes) # \phi
np.save(pathDMD + 'lambda', bfs.frequencies) # \lambda
np.save(pathDMD + 'omega', bfs.omega) # Imag(\lambda)
np.save(pathDMD + 'beta', bfs.beta) # Real(\lambda)
np.save(pathDMD + 'amplitudes', bfs.amplitudes) # \alpha

# %% SPDMD
bfs1 = sparse.SparseDMD(Snapshots, bfs, dt=dt)
gamma = [400, 450, 500]
with timer("SPDMD computing"):
    bfs1.compute_sparse(gamma)
print("The nonzero amplitudes of each gamma:", bfs1.sparse.Nz)

# %% 
sp = 2
bfs1.sparse.Nz[sp]
bfs1.sparse.gamma[sp]
r = np.size(eigval)
sp_ind = np.arange(r)[bfs1.sparse.nonzero[:, sp]]

np.savez(pathDMD + 'sparse.npz',
         Nz=bfs1.sparse.Nz,
         gamma=bfs1.sparse.gamma,
         nonzero=bfs1.sparse.nonzero)

# %% Eigvalue Spectrum
# sp_ind = None
filtval = 0.99  # 0.0
var = var0
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(2.8, 3.0))
unit_circle = plt.Circle((0., 0.), 1., color='grey', linestyle='-', fill=False,
                         label='unit circle', linewidth=3.0, alpha=0.5)
ax1.add_artist(unit_circle)
ind = np.where(np.abs(eigval) > filtval)
ax1.scatter(eigval.real[ind], eigval.imag[ind], marker='o',
            facecolor='none', edgecolors='k', s=18)
if sp_ind is not None:
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
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathDMD+var+'DMDEigSpectrum.svg')
plt.show()

# %% discard the bad DMD modes
# bfs2 = bfs
bfs2 = bfs.reduce(filtval)
phi = bfs2.modes
freq = bfs2.omega/2/np.pi
beta = bfs2.beta
coeff = bfs2.amplitudes

# %% discard the bad DMD modes
#rm = np.abs(bfs.amplitudes)
#index = rm.argsort()[:-60]
#freq = bfs.omega[index]/2/np.pi
#coeff = bfs.amplitudes[index]

# %% Mode frequency specturm
matplotlib.rc('font', size=textsize)
fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
psi = np.abs(coeff)/np.max(np.abs(coeff))
ind1 = freq > 0.0 
freq1 = freq[ind1]
psi1 = np.abs(coeff[ind1]) / np.max(np.abs(coeff[ind1]))
ax2.set_xscale("log")
ax2.vlines(freq1, [0], psi1, color='k', linewidth=1.0)
# ind2 = bfs1.sparse.nonzero[:, sp][index]  # & index
if sp_ind is not None:
    ind2 = bfs1.sparse.nonzero[bfs2.ind, sp]
    ind3 = ind2[ind1]
    ax2.scatter(freq1[ind3], psi1[ind3], marker='o',
                facecolor='gray', edgecolors='gray', s=15)
ax2.set_ylim(bottom=0.0)
ax2.tick_params(labelsize=numsize, pad=6)
ax2.set_xlabel(r'$f \delta_0/u_\infty$')
ax2.set_ylabel(r'$|\psi_i|$')
ax2.grid(b=True, which='both', linestyle=':')
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathDMD+var+'DMDFreqSpectrum.svg')
plt.show()

# %% specific mode in real space
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
var = var2
fa =   1.7*1.7*1.4 # 1.0 # 
ind = 0
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
fig, ax = plt.subplots(figsize=(6.4, 2.8))
c1 = -0.007 #-0.024
c2 = -c1 # 0.010 #0.018
lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, levels=lev1, cmap='RdBu_r', extend='both') 
#cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$y/\delta_0$', fontsize=textsize)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.25, 0.76, 0.30, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal",
                     extendrect='False', ticks=rg2)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(numsize)
cbar1.update_ticks()
cbar1.set_label(r'$\Re(\phi_{})$'.format(var), rotation=0, 
                x=-0.20, labelpad=-26, fontsize=textsize)
cbaxes.tick_params(labelsize=numsize)
# Add shock wave
shock = np.loadtxt(pathM+'ShockLineFit.dat', skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], color='#32cd32ff', linewidth=1.2)
# Add sonic line
sonic = np.loadtxt(pathM+'SonicLine.dat', skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], color='#32cd32ff',
        linestyle='--', linewidth=1.5)
# Add boundary layer
boundary = np.loadtxt(pathM+'BoundaryEdge.dat', skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], 'k', linewidth=1.2)
# Add dividing line(separation line)
dividing = np.loadtxt(pathM+'BubbleLine.dat', skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], 'k--', linewidth=1.2)
# ax.annotate("(a)", xy=(-0.1, 1.), xycoords='axes fraction', fontsize=textsize)
plt.savefig(pathDMD+var+'DMDMode'+name+'Real.svg', bbox_inches='tight')
plt.show()

# % specific mode in imaginary space
tempflow = bfs.modes[:, num].imag
imagflow = tempflow[varset[var][0]:varset[var][1]]
print("The limit value: ", np.min(imagflow)*fa, np.max(imagflow)*fa)
u = griddata((xval, yval), imagflow, (x, y))*fa
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=18)
fig, ax = plt.subplots(figsize=(6.4, 2.8))
c1 = -0.007 #-0.024
c2 = -c1 # 0.012 #0.018
lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, levels=lev1, cmap='RdBu_r', extend='both') 
#cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.25, 0.76, 0.30, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal",
                     extendrect='False', ticks=rg2)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(numsize)
cbar1.update_ticks()
cbar1.set_label(r'$\Im(\phi_{})$'.format(var), rotation=0,
                x=-0.20, labelpad=-26, fontsize=textsize)
cbaxes.tick_params(labelsize=numsize)
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$y/\delta_0$', fontsize=textsize)
# Add shock wave
shock = np.loadtxt(pathM+'ShockLineFit.dat', skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], color='#32cd32ff', linewidth=1.2)
# Add sonic line
sonic = np.loadtxt(pathM+'SonicLine.dat', skiprows=1)
ax.plot(sonic[:, 0], sonic[:, 1], color='#32cd32ff',
        linestyle='--', linewidth=1.5)
# Add boundary layer
boundary = np.loadtxt(pathM+'BoundaryEdge.dat', skiprows=1)
ax.plot(boundary[:, 0], boundary[:, 1], 'k', linewidth=1.2)
# Add dividing line(separation line)
dividing = np.loadtxt(pathM+'BubbleLine.dat', skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], 'k--', linewidth=1.2)
# ax.annotate("(b)", xy=(-0.10, 1.0), xycoords='axes fraction', fontsize=numsize)
plt.savefig(pathDMD+var+'DMDMode'+name+'Imag.svg', bbox_inches='tight')
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

# %% create animation figure of reconstructing flow
path_mode = pathV + "Mode1"
base = meanflow[col].values
base[:, 2] = meanflow['p'].values * 1.7 * 1.7 * 1.4
ind = 27
num = sp_ind[ind] # ind from small to large->freq from low to high
print('The frequency is', bfs.omega[num]/2/np.pi)
phase = np.linspace(0, 4 * np.pi, 32, endpoint=False)
modeflow = bfs.modes[:,num].reshape(-1, 1) * bfs.amplitudes[num] \
           * np.exp(phase.reshape(1, -1)*1j)
corner = (x < 0.0) & (y < 0.0) | (x > x2)
x_coord = x[0, :]
y_coord = y[:, 0]
im = []
matplotlib.rc('font', size=textsize)
lev = np.linspace(0.0, 1.8, 37)

for i in range(np.size(phase)):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim((x1, x2))
    ax.set_ylim((y1, y2))
    ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
    ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
    fluc = modeflow[:, i].reshape((m, o), order='F')
    u_f = fluc[:, 0].real
    p_f = fluc[:, 2].real
    u_new = base[:, 0] + u_f * 10
    p_new = base[:, 2] + p_f * 90
    u = griddata((xval, yval), u_new, (x, y))
    u_fluc = griddata((xval, yval), u_f, (x, y))
    p = griddata((xval, yval), p_new, (x, y))
    gradyx = np.gradient(p, y_coord, x_coord)
    pgrady = gradyx[0]
    pgradx = gradyx[1]
    pgrad = np.sqrt(pgrady ** 2 + pgradx ** 2)
    u[corner] = np.nan
    u_fluc[corner] = np.nan
    pgrad[corner] = np.nan
    cs = ax.contour(x, y, u_fluc, levels=[-0.007, 0.007],
                    colors='k', linewidths=0.6)
    # ax.clabel(cs, inline=0, inline_spacing=numsize, fontsize=numsize-4)    
    cbar = ax.contour(x, y, u, levels=[0.0], colors='w', linewidths=1.0)
    cbar = ax.contourf(x, y, pgrad, levels=lev, cmap='bwr_r')
    # Add shock wave
    shock = np.loadtxt(pathM+'ShockLineFit.dat', skiprows=1)
    ax.plot(shock[:, 0], shock[:, 1], 'g', linewidth=1.0)
    # Add sonic line555
    sonic = np.loadtxt(pathM+'SonicLine.dat', skiprows=1)
    ax.plot(sonic[:, 0], sonic[:, 1], 'g--', linewidth=1.0)
    ax.set_title(r'$\omega t={}/8\pi$'.format(i), fontsize=textsize)
    cbar1 = plt.colorbar(cbar)
    cbar1.set_label(r'$|\nabla p|\delta_0/p_\infty$', rotation=90, 
                    x=0.0, labelpad=10, fontsize=textsize)
    plt.savefig(path_mode + 'DMDAnimat'+str(i)+'.svg', 
                dpi=300, bbox_inches='tight')
    plt.close()
# %% Convert plots to animation
import imageio
from natsort import natsorted, ns
path6 = pathV + "Mode3/"
dirs = os.listdir(path6)
dirs = natsorted(dirs, key=lambda y: y.lower())
with imageio.get_writer(pathV+'DMDAnima3.mp4', mode='I', fps=2) as writer:
    for filename in dirs:
        image = imageio.imread(path6 + filename)
        writer.append_data(image)

# %% save dataframe  to plt of reconstructing flow
path5 = "/media/weibo/Data1/BFS_M1.7L_0505/temp/video/0_059/"
base = meanflow[col].values
base[:, 2] = meanflow['p'].values*1.7*1.7*1.4
ind = 15
num = sp_ind[ind] # ind from small to large->freq from low to high
print('The frequency is', bfs.omega[num]/2/np.pi)
phase = np.linspace(0, 2*np.pi, 32, endpoint=False)
# modeflow1 = bfs.modes[:,num].reshape(-1, 1) * bfs.amplitudes[num] \
#             @ bfs.Vand[num, :].reshape(1, -1)
modeflow = bfs.modes[:,num].reshape(-1, 1) * bfs.amplitudes[num] \
           * np.exp(phase.reshape(1, -1)*1j)
xarr = xval.values.reshape(-1, 1)
yarr = yval.values.reshape(-1, 1)
zarr = np.zeros((m, 1))
names = ['x', 'y', 'z', var0, var1, var2, 'u`', 'v`', 'p`']
for ii in range(np.size(phase)):
    fluc = modeflow[:, ii].reshape((m, o), order='F')
#    fluc[:, 0] = fluc[:, 0] * mag0
#    fluc[:, 1] = fluc[:, 1] * mag1
#    fluc[:, 2] = fluc[:, 2] * mag2
    newflow = fluc.real
    outfile = 'DMD'+str(np.round(phase[ii], 2))
    data = np.hstack((xarr, yarr, zarr, base, newflow))
    df = pd.DataFrame(data, columns=names)
    df1 = df.query("x>=0.0")
    with timer('save plt of t='+str(phase[ii])):
        p2p.frame2plt(df1, path5, outfile+'_1', 
                      time=timepoints[ii]+24, zonename=1)
    df2 = df.query("x<=0.0 & y>=0")
    with timer('save plt of t='+str(phase[ii])):
        p2p.frame2plt(df2, path5, outfile+'_2', 
                      time=timepoints[ii]+24, zonename=2)

