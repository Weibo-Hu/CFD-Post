#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:00:05 2024
    pod for ramp cases
@author: weibo
"""

# %% Load necessary module
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pod as pod
import plt2pandas as p2p
from timer import timer
from scipy.interpolate import griddata
import os
import sys

tsize = 13
nsize = 10
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
matplotlib.rc("font", size=tsize)
plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
cm2in = 1 / 2.54


# %% prep data
path = "/media/weibo/VID2/AAS/ramp_st14_2nd/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathD = path + "Domain/"
pathPOD = path + "POD/"
pathSS = path + 'TP_2D_Z_001/'
timepoints = np.arange(600, 900 + 0.25, 0.25)
dirs = sorted(os.listdir(pathSS))
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
DataFrame = pd.read_hdf(pathSS + dirs[0])
ind0 = (DataFrame['y'] == 0.0)
DataFrame['walldist'][ind0] = 0.0

grouped = DataFrame.groupby(['x', 'y'])
DataFrame = grouped.mean().reset_index()
NewFrame = DataFrame.query("x>=-160.0 & x<=60.0 & walldist>=0.0 & y<=36.0")

ind = NewFrame.index.values
xval = DataFrame['x'][ind]
yval = DataFrame['y'][ind]
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
x1 = -160.0
x2 = 60.0
y1 = 0.0
y2 = 36.0

var0 = 'u'
var1 = 'v'
var2 = 'p'
col = [var0, var1, var2]
fa = 1
FirstFrame = DataFrame[col].values
Snapshots = FirstFrame[ind].ravel(order='F')

m, = np.shape(Snapshots)
n = np.size(timepoints)
o = np.size(col)
if (m % o != 0):
    sys.exit("Dimensions of snapshots are wrong!!!")
m = int(m/o)
varset = {var0: [0, m],
          var1: [m, 2*m],
          var2: [2*m, 3*m]
         }
# %% load data
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(pathSS + dirs[i+1])
        grouped = TempFrame.groupby(['x', 'y'])
        TempFrame = grouped.mean().reset_index()
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        NextFrame = TempFrame[col].values
        Snapshots = np.vstack((Snapshots, NextFrame[ind].ravel(order='F')))
        DataFrame += TempFrame
Snapshots = Snapshots.T

AveFlow = DataFrame/np.size(dirs)
meanflow = AveFlow.query("x>=-160.0 & x<=60.0 & walldist>=0.0 & y<=36.0")

# %% POD compute
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")

with timer("POD computing"):
    eigval, eigvec, phi, coeff = \
        pod.pod(Snapshots, fluc=True, method='svd')

meanflow.to_hdf(pathPOD + 'Meanflow.h5', 'w', format='fixed')
np.save(pathPOD + 'eigval', eigval)
np.save(pathPOD + 'eigvec', eigvec)
np.save(pathPOD + 'phi1', phi[:, :600])
np.save(pathPOD + 'phi2', phi[:, 600:])
np.save(pathPOD + 'coeff', coeff)

# %% optional: load POD results
eigval = np.load(pathPOD + 'eigval.npy')
eigvec = np.load(pathPOD + 'eigvec.npy')
coeff = np.load(pathPOD + 'coeff.npy')
# phi1 = np.load(pathPOD + 'phi1.npy')
# phi2 = np.load(pathPOD + 'phi2.npy')
phi = np.hstack((np.load(pathPOD + 'phi1.npy'), np.load(pathPOD + 'phi2.npy')))
# %%############################################################################
"""
    Plot eigenvalue spectrum
"""
EFrac, ECumu, N_modes = pod.pod_eigspectrum(97, eigval)
np.savetxt(pathPOD+'EnergyFraction97.dat', EFrac, fmt='%1.7e', delimiter='\t')
N_modes = -N_modes + 1201 + 2
print("the NO of modes is ", N_modes)
var = var0
matplotlib.rc('font', size=tsize)
fig1, ax1 = plt.subplots(figsize=(10*cm2in, 8*cm2in))
xaxis = np.arange(0, N_modes)
ax1.scatter(
    xaxis[1:],
    EFrac[1:N_modes],
    c='black',
    marker='o',
    s=10.0,
)   # fraction energy of every eigval mode
# ax1.legend('E_i')
ax1.set_ylim(bottom=0)
ax1.set_xlabel('Mode', fontsize=tsize)
ax1.set_ylabel(r'$E_i$', fontsize=tsize)
ax1.grid(visible=True, which='both', linestyle=':')
ax1.tick_params(labelsize=nsize)
ax2 = ax1.twinx()   # cumulation energy of first several modes
# ax2.fill_between(xaxis, ECumu[:N_modes], color='grey', alpha=0.5)
# ESum = np.zeros(N_modes)
ESum = ECumu[:N_modes]
ax2.plot(xaxis, ESum, color='grey', label=r'$ES_i$')
ax2.set_ylim([60, 100])
ax2.set_ylabel(r'$ES_i$', fontsize=tsize)
ax2.tick_params(labelsize=nsize)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(pathPOD+str(N_modes)+'_PODEigSpectrum80.svg', bbox_inches='tight')
plt.show()

# %% specific mode in space
ind = 20
var = var0
fa = 1.0  # 1.7*1.7*1.4 #
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
newflow = phi[:, ind]*coeff[ind, 0]
modeflow = newflow[varset[var][0]:varset[var][1]]
print("The limit value: ", np.min(modeflow)*fa, np.max(modeflow)*fa)
u = griddata((xval, yval), modeflow, (x, y))*fa
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=tsize)
fig, ax = plt.subplots(figsize=(14*cm2in, 5*cm2in))
c1 = -0.02  # -0.01 # -0.006
c2 = -c1  # 0.063

lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, cmap='coolwarm', levels=lev1, extend='both')
# ax.contour(x, y, u, levels=lev2, colors='k', linewidths=0.8, extend='both')
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'), extend='both')
ax.set_xlim(-100, 60)
ax.set_ylim(0, 25)
ax.set_xticks(np.linspace(-100, 60, 5))
ax.set_yticks(np.linspace(0, 25, 6))
ax.tick_params(labelsize=nsize)
# cbar.cmap.set_under('#053061')
# cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/l_r$', fontsize=tsize)
ax.set_ylabel(r'$y/l_r$', fontsize=tsize)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.2, 0.76, 0.30, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal",
                     ticks=rg2, extendrect='False')
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(nsize)
cbar1.update_ticks()
cbar1.set_label(r'$\varphi_{}$'.format(var), rotation=0,
                x=-0.18, labelpad=-20, fontsize=tsize)
cbaxes.tick_params(labelsize=nsize)
# Add shock wave
shock = pd.read_csv(pathM + "ShockLineFit.dat", skipinitialspace=True)
ax.plot(shock.x, shock.y, 'gray', linewidth=1.0)
# Add sonic line
sonic = pd.read_csv(pathM+'SonicLine.dat', skipinitialspace=True)
ax.plot(sonic.x, sonic.y, 'g--', linewidth=1.0)
# Add boundary layer
# boundary = np.loadtxt(pathM+'BoundaryEdge.dat', skiprows=1)
# ax.plot(boundary[:, 0], boundary[:, 1], 'k', linewidth=1.0)
# Add dividing line(separation line)
# dividing = np.loadtxt(pathM+'BubbleLine.dat', skiprows=1)
# ax.plot(dividing[:, 0], dividing[:, 1], 'k--', linewidth=1.0)

plt.savefig(pathPOD+var+'_PODMode'+str(ind)+'.svg', bbox_inches='tight')
plt.show()