#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:22:31 2018
    This code for plotting line/curve figures

@author: weibo
"""
# %% Load necessary module
import numpy as np
import pandas as pd
import plt2pandas as p2p
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, splev, splrep
from scipy import signal
import variable_analysis as va
from timer import timer
import sys
import os
from planar_field import PlanarField as pf
from triaxial_field import TriField as tf

plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}

path = "/media/weibo/VID2/BFS_M1.7TS_LA/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathSL = path + 'Slice/'

matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 13
numsize = 10
matplotlib.rc("font", size=textsize)

VarName = [
    "x",
    "y",
    "z",
    "u",
    "v",
    "w",
    "rho",
    "p",
    "T",
    "Q-criterion",
    "L2-criterion",
]

StepHeight = 3.0

# %%
MeanFlow = pf()
MeanFlow.load_meanflow(path)
MeanFlow.add_walldist(StepHeight)
stat = MeanFlow.PlanarData

# %%############################################################################
"""
    save coordinates of bubble line & max fluctuations points 
"""
# %% save dividing line coordinates
dividing = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)[:-2, :]
x2 = np.arange(dividing[-1, 0], 50.0+0.125, 0.125)
y2 = np.ones(np.size(x2))*(-2.99342)
x3 = np.concatenate((dividing[:,0], x2), axis=0)
y3 = np.concatenate((dividing[:,1], y2), axis=0) # streamline
xx = np.zeros(np.size(x3))
yy = np.zeros(np.size(y3))
frame1 = stat.loc[stat['z'] == 0, ['x', 'y']]
frame2 = frame1.query("x>=0.0 & y<=3.0")
xmesh = frame2['x'].values
ymesh = frame2['y'].values
for i in range(np.size(x3)):
    variance = np.sqrt(np.square(xmesh - x3[i]) + np.square(ymesh - y3[i]))
    idx = np.where(variance == np.min(variance))[0][0]
    xx[i] = xmesh[idx]
    yy[i] = ymesh[idx]
xy = np.vstack((xx, yy))
strmln = pd.DataFrame(xy.T, columns=['x', 'y'])
strmln = strmln.drop_duplicates(keep='last')
strmln.to_csv(pathM + "BubbleGrid.dat", sep=' ',
              float_format='%1.8e', index=False)

# draw dividing line
fig, ax = plt.subplots(figsize=(6.4, 3.0))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.plot(xx, yy, 'k')
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "BubbleGrid.svg", bbox_inches="tight", pad_inches=0.1
)

# %% compute coordinates where the BL profile has Max RMS along streamwise direction
varn = '<u`u`>'
if varn == '<u`u`>':
    savenm = "MaxRMS_u.dat"
elif varn =='<p`p`>':
    savenm = "MaxRMS_p.dat"
xnew = np.arange(-40.0, 40.0+0.125, 0.125)
znew = np.zeros(np.size(xnew))
varv = np.zeros(np.size(xnew))
ynew = np.zeros(np.size(xnew))
for i in range(np.size(xnew)):
    df = va.max_pert_along_y(stat, '<u`u`>', [xnew[i], znew[i]])
    varv[i] = df[varn]
    ynew[i] = df['y']
data = np.vstack((xnew, ynew, varv))
df = pd.DataFrame(data.T, columns=['x', 'y', varn])
# df = df.drop_duplicates(keep='last')
df.to_csv(pathM + savenm, sep=' ',
          float_format='%1.8e', index=False)

# %% save the grid points on the wall
xnew = np.arange(-40.0, 40.0+0.125, 0.125)
ynew = 0.001953125 * np.ones(np.size(xnew))
ind = np.where(xnew >= 0.0)
ynew[ind] = -2.997037172
data = np.vstack((xnew, ynew))
df = pd.DataFrame(data.T, columns=['x', 'y'])
df.to_csv(pathM + 'WallGrid.dat', sep=' ', float_format='%1.8e', index=False)

# %% draw maximum value curve
fig, ax = plt.subplots(figsize=(6.4, 3.0))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.plot(xnew, ynew, 'k', label=r'$q^\prime_\mathrm{max}$')
# ax.plot(xx, yy, 'k--', label='bubble')
legend = ax.legend(loc='upper right', shadow=False, fontsize=numsize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "MaxPertLoc.svg", bbox_inches="tight", pad_inches=0.1
)

# %%############################################################################
"""
    RMS distribution along streamwise and wall-normal direction
"""
# %% Plot RMS of velocity on the wall along streamwise direction
loc = ['z', 'y']
val = [0.0, -2.99704]
varnm = '<u`u`>'
pert = va.pert_at_loc(stat, varnm, loc, val)
fig, ax = plt.subplots(figsize=(6.4, 2.2))
matplotlib.rc('font', size=14)
ax.set_ylabel(r"$\sqrt{u^{\prime 2}}/u_{\infty}$", fontsize=textsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.plot(pert['x'], np.sqrt(pert[varnm]), 'k')
ax.set_xlim([0.0, 30])
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "PerturProfileX.svg", bbox_inches="tight", pad_inches=0.1
)

# %% Plot RMS of velocity on the wall along streamwise direction
xynew = pd.read_csv(pathM + 'MaxRMS.dat', sep=' ')
fig, ax = plt.subplots(figsize=(6.4, 2.2))
matplotlib.rc('font', size=numsize)
ylab = r"$\sqrt{u^{\prime 2}_\mathrm{max}}/u_{\infty}$"
ax.set_ylabel(ylab, fontsize=textsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.plot(xynew['x'], xynew['<u`u`>'], 'k')
ax.set_yscale('log')
ax.set_xlim([-5, 30])
# ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.grid(b=True, which="both", linestyle=":")
ax.tick_params(labelsize=numsize)
ax.yaxis.offsetText.set_fontsize(numsize-1)
plt.show()
plt.savefig(
    pathF + "MaxRMS_x.svg", bbox_inches="tight", pad_inches=0.1
)

# %% Plot RMS of velocity along wall-normal direction
loc = ['x', 'z']
val = [10.0, 0.0]
pert = va.pert_at_loc(stat, varnm, loc, val)
fig, ax = plt.subplots(figsize=(6.4, 2.2))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$u^{\prime}/u_{\infty}$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
ax.plot(pert[varnm], pert['y'])
ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "PerturProfileY.svg", bbox_inches="tight", pad_inches=0.1
)

# %%############################################################################
"""
    RMS distribution along spanwise direction
"""
# %% Plot RMS of velocity along spanwise on the dividing line
# load data method1
#TimeAve = tf()
#TimeAve.load_3data(pathT, FileList=pathT + 'TimeAve.h5', NameList='h5')
#stat3d = TimeAve.TriData
# load data method2
df0 = pd.read_hdf(pathT + 'MeanFlow0.h5')
df1 = pd.read_hdf(pathT + 'MeanFlow1.h5')
df2 = pd.read_hdf(pathT + 'MeanFlow2.h5')
df3 = pd.read_hdf(pathT + 'MeanFlow3.h5')
stat3d = pd.concat([df0, df1, df2, df3], ignore_index=True)
# %% plot
loc = ['x', 'y']
valarr = [[0.0078125, -0.00781],
          [2.5625, -0.43750],
          [5.5625, -1.1875],
          [7.625, -1.4375],
          [11.3125, -1.6875]]
#          [15.0, -1.6875]]

fig, ax = plt.subplots(1, 5, figsize=(6.4, 3.2))
fig.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.9, wspace=0.2)
matplotlib.rc('font', size=14)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
# a
val = valarr[0]
pert = va.pert_at_loc(stat3d, varnm, loc, val)
ax[0].plot(np.sqrt(pert[varnm]), pert['z'], 'k-')
ax[0].set_xlim([0.0, 1e-3])
ax[0].ticklabel_format(axis="x", style="sci", scilimits=(-1, 1))
ax[0].set_yticks([-8.0, -4.0, 0.0, 4.0, 8.0])
ax[0].set_ylabel(r"$z/\delta_0$", fontsize=textsize)
ax[0].tick_params(axis='both', labelsize=numsize)
ax[0].set_title(title[0], fontsize=numsize)
ax[0].grid(b=True, which="both", linestyle=":")
ax[0].xaxis.offsetText.set_fontsize(numsize)
# b
val = valarr[1]
pert = va.pert_at_loc(stat3d, varnm, loc, val)
ax[1].plot(np.sqrt(pert[varnm]), pert['z'], 'k-')
ax[1].set_xlim([1e-2, 3e-2])
ax[1].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[1].set_yticklabels('')
ax[1].tick_params(axis='both', labelsize=numsize)
ax[1].set_title(title[1], fontsize=numsize)
ax[1].grid(b=True, which="both", linestyle=":")
ax[1].xaxis.offsetText.set_fontsize(numsize)
# c
val = valarr[2]
pert = va.pert_at_loc(stat3d, varnm, loc, val)
ax[2].plot(np.sqrt(pert[varnm]), pert['z'], 'k-')
ax[2].set_xlim([0.05, 0.20])
ax[2].set_xlabel(r"$\sqrt{u^{\prime 2}}/u_{\infty}$",
                 fontsize=textsize, labelpad=18.0)
ax[2].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[2].set_yticklabels('')
ax[2].tick_params(labelsize=numsize)
ax[2].set_title(title[2], fontsize=numsize)
ax[2].grid(b=True, which="both", linestyle=":")
# d
val = valarr[3]
pert = va.pert_at_loc(stat3d, varnm, loc, val)
ax[3].plot(np.sqrt(pert[varnm]), pert['z'], 'k-')
ax[3].set_xlim([0.05, 0.20])
ax[3].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[3].set_yticklabels('')
ax[3].tick_params(labelsize=numsize)
ax[3].set_title(title[3], fontsize=numsize)
ax[3].grid(b=True, which="both", linestyle=":")
# e
val = valarr[4]
pert = va.pert_at_loc(stat3d, varnm, loc, val)
ax[4].plot(np.sqrt(pert[varnm]), pert['z'], 'k-')
ax[4].set_xlim([0.05, 0.20])
ax[4].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[4].set_yticklabels('')
ax[4].tick_params(labelsize=numsize)
ax[4].set_title(title[4], fontsize=numsize)
ax[4].grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "PerturProfileZ.svg", bbox_inches="tight"
)

# %%############################################################################
"""
    distribution of amplitude & amplication factor along a line
"""
# %% load time sequential snapshots
sp = 'S_010'
InFolder = path + 'Slice/' + sp + '/'
dirs = sorted(os.listdir(InFolder))
DataFrame = pd.read_hdf(InFolder + dirs[0])
grouped = DataFrame.groupby(['x', 'y'])
DataFrame = grouped.mean().reset_index()
var = 'p' # 'u'
Snapshots = DataFrame[['x', 'y', 'z', 'u', 'p']]

fa = 1.7*1.7*1.4
skip = 1
dt = 0.25  # 0.25
freq_samp = 1/dt  # 4.0
timepoints = np.arange(900, 1299.75 + dt, dt)
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        if i % skip == 0:
            TempFrame = pd.read_hdf(InFolder + dirs[i+1])
            grouped = TempFrame.groupby(['x', 'y'])
            Frame1 = grouped.mean().reset_index()
            Frame2 = Frame1[['x', 'y', 'z', 'u', 'p']]
            if np.shape(Frame1)[0] != np.shape(DataFrame)[0]:
                sys.exit('The input snapshots does not match!!!')
            Snapshots = pd.concat([Snapshots, Frame2])

# Snapshots.loc[var] = Snapshots.loc[var] * fa
Snapshots.assign(var=Snapshots[var] * fa)
m, n = np.shape(Snapshots)
# %% 1.500000  -0.500000
probe = Snapshots.loc[(Snapshots['x']==6.87500) & (Snapshots['y']==-1.781250)]
probe.insert(0, 'time', timepoints)
probe.to_csv(pathI + 'probe_14.dat', index=False, sep=' ', float_format='%1.8e')
# %% compute amplitude of variable along a line
# xynew = np.loadtxt(pathM + "BubbleGrid.dat", skiprows=1)
xynew = np.loadtxt(pathM + "MaxRMS_u.dat", skiprows=1)
# xynew = np.loadtxt(pathM + "WallGrid.dat", skiprows=1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
ind = (xynew[:, 0] <= 30.0) & (xynew[:, 0] >= -10.0)   # -40.0
xval = xynew[ind, 0]
yval = xynew[ind, 1]
amplit = np.zeros(np.size(xval))
for i in range(np.size(xval)):
    xyz = [xval[i], yval[i], 0.0]
    amplit[i] = va.amplit(Snapshots, xyz, var)

# %% plot amplitude of variable along a line
fig, ax = plt.subplots(figsize=(6.4, 3))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$A_{u^{\prime}}$", fontsize=textsize)
ax.plot(xval, amplit)
b, a = signal.butter(3, 0.15, btype='lowpass', analog=False)
amplit1 = signal.filtfilt(b, a, amplit)
ax.plot(xval, amplit1, 'r--')
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + var + "_AmplitX_max.svg", bbox_inches="tight", pad_inches=0.1
)

# %% compute RMS along a line
varnm = '<u`u`>'
loc = ['x', 'y']
uu = np.zeros(np.size(xval))
for i in range(np.size(xval)):
    xyz = [xval[i], yval[i]]
    uu[i] = va.pert_at_loc(stat, varnm, loc, xyz)[varnm]

# %% draw RMS & amplication factor along streamwise
fig, ax = plt.subplots(figsize=(6.4, 3.0))
matplotlib.rc('font', size=14)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\sqrt{u^{\prime 2}}/u_{\infty}$", fontsize=textsize)
ax.plot(xval, uu, 'k')
ax.set_xlim([-0.5, 20.5])
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

ax2 = ax.twinx()
ax2.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax2.set_ylabel(r"$A/A_0$", fontsize=textsize)
ax2.set_yscale('log')
ax2.plot(xval[1:], amplit1[1:]/amplit1[0], 'k--')
ax2.set_xlim([-0.5, 20.5])
# ax2.set_ylim([0.8, 100])
ax2.axvline(x=0.6, linewidth=1.0, linestyle='--', color='gray')
ax2.axvline(x=3.2, linewidth=1.0, linestyle='--', color='gray')
ax2.axvline(x=6.5, linewidth=1.0, linestyle='--', color='gray')
ax2.axvline(x=9.7, linewidth=1.0, linestyle='--', color='gray')
ax2.axvline(x=12., linewidth=1.0, linestyle='--', color='gray')
# ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax2.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + var + "GrowRateX_max.svg", bbox_inches="tight", pad_inches=0.1
)

# %%############################################################################
"""
    RMS distribution along a line, computed from temporal snapshots
"""
# %% Plot RMS map along a line
# compute RMS by temporal sequential data
varnm = 'p'
rms = np.zeros(np.size(xval))
for i in range(np.size(xval)):
    xyz = [xval[i], yval[i], 0.0]
    rms[i] = va.rms_map(Snapshots, xyz, varnm)

# plot
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
fig, ax = plt.subplots(figsize=(6.4, 3.0))
# ax.yaxis.major.formatter.set_powerlimits((-2, 3))
ax.plot(xval, rms, 'k-')
ax.set_xlim([0.0, 30.0])
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\mathrm{RMS}(u^{\prime}/u_{\infty})$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
ax.axvline(x=10.9, linewidth=1.0, linestyle='--', color='k')
plt.savefig(pathF + varnm + "_RMSMap_max_" + sp + ".svg", 
            bbox_inches="tight", pad_inches=0.1)
plt.show()

# %%############################################################################
"""
    frequency-weighted PSD along a line
"""
# %% compute
var = 'p'
skip = 1
samples = int(np.size(timepoints) / skip / 2 + 1)
FPSD = np.zeros((samples, np.size(xval)))
for i in range(np.size(xval)):
    xyz = [xval[i], yval[i], 0.0]
    freq, FPSD[:, i] = va.fw_psd_map(Snapshots, xyz, var, dt, freq_samp,
                                     opt=1, seg=4, overlap=2)
np.savetxt(pathSL + 'FWPSD_freq_' + sp + '.dat', freq, delimiter=' ')
np.savetxt(pathSL + 'FWPSD_x.dat', xval, delimiter=' ')
np.savetxt(pathSL + var + '_FWPSD_psd_' + sp + '.dat', FPSD, delimiter=' ')

freq = np.loadtxt(pathSL + 'FWPSD_freq_' + sp + '.dat', delimiter=' ')
xval = np.loadtxt(pathSL + 'FWPSD_x.dat', delimiter=' ')
FPSD = np.loadtxt(pathSL + var + '_FWPSD_psd_' + sp + '.dat', delimiter=' ')
freq = freq[1:]
FPSD = FPSD[1:, :]

# %% Plot frequency-weighted PSD map along a line
SumFPSD = np.min(FPSD, axis=0)  # np.sum(FPSD, axis=0)
FPSD1 = np.log(FPSD/SumFPSD)  # np.log(FPSD) # 
max_id = np.argmax(FPSD1, axis=0)
max_freq = [freq[i] for i in max_id]
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
fig, ax = plt.subplots(figsize=(6.4, 2.8))
# ax.yaxis.major.formatter.set_powerlimits((-2, 3))
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
print(np.max(FPSD1))
print(np.min(FPSD1))
cb1 =  0
cb2 = 9
lev = np.linspace(cb1, cb2, 41)
cbar = ax.contourf(xval, freq, FPSD1, extend='both',
                   cmap='bwr', levels=lev)  # seismic # bwr # coolwarm
ax.plot(xval, max_freq, 'C7:', linewidth=1.2)
# every stage of the transition\
ax.axvline(x=0.0, linewidth=1.0, linestyle='--', color='k')
# ax.axvline(x=9.2, linewidth=1.0, linestyle='--', color='k')
ax.set_yscale('log')
ax.set_xlim([-5, 20.0])
rg = np.linspace(cb1, cb2, 3)
cbar = plt.colorbar(cbar, ticks=rg, extendrect=True)
cbar.ax.xaxis.offsetText.set_fontsize(numsize)
cbar.ax.tick_params(labelsize=numsize)
cbar.update_ticks()
barlabel = r'$\log_{10} [f\cdot\mathcal{P}(f)/ \mathcal{P}(f)_\mathrm{min} ]$'
# barlabel = r'$\log_{10} [f\cdot\mathcal{P}(f)/\int \mathcal{P}(f) \mathrm{d}f]$'
ax.set_title(barlabel, pad=3, fontsize=numsize-1)
# cbar.set_label(barlabel, rotation=90, fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + var + "_FWPSDMap_max_" + sp + ".svg",
            bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Plot multiple frequency-weighted PSD curve along streamwise
xr = 8.9
def d2l(x):
    return x * xr

def l2d(x):
    return x / xr

fig, ax = plt.subplots(1, 7, figsize=(6.8, 2.4))
fig.subplots_adjust(hspace=0.5, wspace=0.0)
matplotlib.rc('font', size=numsize)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
xcoord = np.array([-2.0, 2.75, 3.0, 6.0, 6.375, 9.0, 10.0])
# xcoord = np.array([-5, 1.0, 3.75, 4.875, 6.5, 8.875, 18.25])
for i in range(np.size(xcoord)):
    ind = np.where(xval[:] == xcoord[i])[0]
    fpsd_x = FPSD[:, ind]
    ax[i].plot(fpsd_x, freq, "k-", linewidth=1.0)
    ax[i].set_yscale('log')
    ax[i].xaxis.major.formatter.set_powerlimits((-2, 2))
    ax[i].xaxis.offsetText.set_fontsize(numsize)
    if i != 0:
        ax[i].set_yticklabels('')
        ax[i].set_title(r'${}$'.format(xcoord[i]), fontsize=numsize-2)
        ax[i].patch.set_alpha(0.0)
        # ax[i].spines['left'].set_visible(False)
        ax[i].yaxis.set_ticks_position('none')
        xticks = ax[i].xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
    if i != np.size(xcoord) - 1:
        ax[i].spines['right'].set_visible(False)
    ax[i].set_xlim(left=0)
    # ax[i].set_xticklabels('')
    ax[i].tick_params(axis='both', which='major', labelsize=numsize)
    # ax[i].grid(b=True, which="both", axis='both', linestyle=":")
ax[0].set_title(r'$x/\delta_0={}$'.format(xcoord[0]), fontsize=numsize-2)
ax[0].set_ylabel(r"$f \delta_0 /u_\infty$", fontsize=textsize)
ax[3].set_xlabel(r'$f \mathcal{P}(f)$', fontsize=textsize, labelpad=15)
ax2 = ax[-1].secondary_yaxis('right', functions=(d2l, l2d)) 
ax2.set_ylabel(r"$f x_r /u_\infty$", fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "MulFWPSD_" + sp + var + ".svg", bbox_inches="tight", pad_inches=0.1
)

# %%############################################################################
# %
# % boundary layer profile of RMS
# %
# % plot BL fluctuations profile
varnm = '<u`u`>'
fig, ax = plt.subplots(1, 7, figsize=(6.4, 2.5))
fig.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.9, wspace=0.2)
matplotlib.rc('font', size=14)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
xcoord = np.array([-40, 0, 5, 10, 15, 20, 30])
loc = ['z', 'x']
for i in range(np.size(xcoord)):
    pert = va.pert_at_loc(stat, varnm, loc, [0.0, xcoord[i]])
    if xcoord[i] > 0.0:
        ax[i].plot(np.sqrt(pert[varnm]), pert['y']+3.0, 'k-')
    else:
        ax[i].plot(np.sqrt(pert[varnm]), pert['y'], 'k-')
    ax[i].set_ylim([0, 4])
    if i != 0:
        ax[i].set_yticklabels('')
    # ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].tick_params(axis='both', which='major', labelsize=numsize-1)
    ax[i].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    ax[i].set_title(r'$x/\delta_0={}$'.format(xcoord[i]),
                    fontsize=numsize-2, y=0.96)
    ax[i].grid(b=True, which="both", linestyle=":")
ax[0].set_ylabel(r"$\Delta y/\delta_0$", fontsize=textsize)
ax[3].set_xlabel(r'$u^{\prime}/u_\infty$', fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLProfileRMS.pdf", bbox_inches="tight", pad_inches=0.1
)
