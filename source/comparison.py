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
import matplotlib.pyplot as plt
import matplotlib
import plt2pandas as p2p
import matplotlib.ticker as ticker
import variable_analysis as va
from timer import timer
import warnings
import os
from planar_field import PlanarField as pf


path0 = "/media/weibo/VID2/BFS_M1.7L/"
path0F, path0P, path0M, path0S, path0T, path0I = p2p.create_folder(path0)
path1 = "/media/weibo/VID2/BFS_M1.7TS/"
path1F, path1P, path1M, path1S, path1T, path1I = p2p.create_folder(path1)
path2 = "/media/weibo/VID2/BFS_M1.7TS1/"
path2F, path2P, path2M, path2S, path2T, path2I = p2p.create_folder(path2)
pathC = path2 + 'Comparison/'

plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 13
numsize = 10

# %%############################################################################
"""
    load data
"""
StepHeight = 3.0
MeanFlow0 = pf()
MeanFlow0.load_meanflow(path0)
MeanFlow0.add_walldist(StepHeight)
MeanFlow1 = pf()
MeanFlow1.load_meanflow(path1)
MeanFlow1.add_walldist(StepHeight)
MeanFlow2 = pf()
MeanFlow2.load_meanflow(path2)
MeanFlow2.add_walldist(StepHeight)

# %%############################################################################
"""
    skin friction & pressure coefficiency/turbulent kinetic energy along streamwise
"""
# %% calculate
MeanFlow0.copy_meanval()
WallFlow0 = MeanFlow0.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow1.copy_meanval()
WallFlow1 = MeanFlow1.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow2.copy_meanval()
WallFlow2 = MeanFlow2.PlanarData.groupby("x", as_index=False).nth(1)
# WallFlow = WallFlow[WallFlow.x != -0.0078125]
xwall = WallFlow0["x"].values
mu0 = va.viscosity(13718, WallFlow0["T"])
Cf0 = va.skinfriction(mu0, WallFlow0["u"], WallFlow0["walldist"]).values
ind0 = np.where(Cf0[:] < 0.005)
mu1 = va.viscosity(13718, WallFlow1["T"])
Cf1 = va.skinfriction(mu1, WallFlow1["u"], WallFlow1["walldist"]).values
ind1 = np.where(Cf1[:] < 0.005)
mu2 = va.viscosity(13718, WallFlow2["T"])
Cf2 = va.skinfriction(mu2, WallFlow2["u"], WallFlow2["walldist"]).values
ind2 = np.where(Cf2[:] < 0.005)

# %% Plot streamwise skin friction
# fig2, ax2 = plt.subplots(figsize=(5, 2.5))
fig = plt.figure(figsize=(6.4, 4.7))
matplotlib.rc("font", size=textsize)
ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=textsize)
ax2.scatter(xwall[ind0][0::8], Cf0[ind0][0::8], s=10, marker='o',
            facecolors='w', edgecolors='C7', linewidths=0.8)
ax2.plot(xwall[ind1], Cf1[ind1], "k", linewidth=1.1)
ax2.plot(xwall[ind2], Cf2[ind2], "k--", linewidth=1.1)
# ax2.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=textsize)
ax2.set_xlim([-40.0, 40.0])
ax2.tick_params(axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=True,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off)
ax2.set_ylim([-0.002, 0.006])
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
# ax2.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
ax2.grid(b=True, which="both", linestyle=":")

ax2.yaxis.offsetText.set_fontsize(numsize)
ax2.annotate("(a)", xy=(-0.12, 1.04), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.savefig(pathC+'Cf.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# % turbulent kinetic energy
# xwall0 = Flow0['x'].values
tke0 = np.sqrt(WallFlow0['<p`p`>'].values)
tke1 = np.sqrt(WallFlow1['<p`p`>'].values)
tke2 = np.sqrt(WallFlow2['<p`p`>'].values)
# fig3, ax3 = plt.subplots(figsize=(6.4, 2.3))
ax3 = fig.add_subplot(212)
matplotlib.rc("font", size=textsize)
ax3.scatter(xwall[ind0][0::8], tke0[ind0][0::8], s=10, marker='o',
            facecolors='w', edgecolors='C7', linewidths=0.8)
ax3.plot(xwall[ind1], tke1[ind1], "k", linewidth=1.1)
ax3.plot(xwall[ind2], tke2[ind2], "k--", linewidth=1.1)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ylab = r"$2\sqrt{\langle p^{\prime}p^{\prime} \rangle}/\rho_\infty u^2_\infty$"
ax3.set_ylabel(ylab, fontsize=textsize)
ax3.set_xlim([-40.0, 40.0])
ax3.set_ylim([-0.001, 0.02])
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
# ax3.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
ax3.grid(b=True, which="both", linestyle=":")
ax3.yaxis.offsetText.set_fontsize(numsize)
ax3.annotate("(b)", xy=(-0.12, 1.04), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.subplots_adjust(hspace=0.2)  # adjust space between subplots
plt.savefig(pathC+'CfPrms.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

# %%############################################################################
"""
    time sequential data at some probes
"""
# %% load sequential slices
path_z3 = path0S + 'TP_2D_Z_03/'
stime = np.arange(700.0, 1000, 0.25) # n*61.5
newlist = ['x', 'y', 'z', 'u', 'p']
dirs = sorted(os.listdir(path_z3 ))
df_shape = np.shape(pd.read_hdf(path_z3  + dirs[0]))
sp = pd.DataFrame()
for i in range(np.size(dirs)):
    fm_temp = pd.read_hdf(path_z3  + dirs[i])
    if df_shape[0] != np.shape(fm_temp)[0]:
        warnings.warn("Shape of" + dirs[i] + " does not match!!!",
                      UserWarning)
    sp = sp.append(fm_temp.loc[:, newlist], ignore_index=True)
# %% extract probe data with time
x0 = [-0.1875, 0.203125, 0.5, 0.59375]
y0 = [0.03125, 0.0390625, -0.078125, -0.171875] 
num_samp = np.size(stime)
var_list = ['x', 'y', 'u', 'p']
var = np.zeros((num_samp, 2))
for j in range(np.size(x0)):
    file = path0P + 'timeline_' + str(x0[j]) + '.dat'
    var = sp.loc[(sp['x']==x0[j]) & (sp['y']==y0[j]), var_list].values
    df = pd.DataFrame(data=np.hstack((stime.reshape(-1, 1), var)), 
                      columns=['time', 'x', 'y', 'u', 'p'])
    df.to_csv(file, sep=' ', index=False, float_format='%1.8e')

# %%############################################################################
"""
    Development of variables with time
"""
# %% variables with time
varnm = 'u'
dt = 0.25
if varnm == 'u':
    ylab = r"$u^\prime / u_\infty$"
    ylab_sub = r"$_{u^\prime}$"
else:
    ylab = r"$p^\prime/(\rho_\infty u_\infty ^2)$"
    ylab_sub = r"$_{p^\prime}$"

xloc = [2.0] # [-0.1875] # [0.203125] #, 0.203125, 
curve0= ['g-'] # , 'b-', 'g-']
curve1= ['b-'] # , 'b:', 'g:']
fig3, ax3 = plt.subplots(figsize=(6.4, 1.5))
grid = plt.GridSpec(1, 3, wspace=0.4)
ax1 = plt.subplot(grid[0, :2])
ax2 = plt.subplot(grid[0, 2:])
matplotlib.rc("font", size=numsize)
for i in range(np.size(xloc)):
    filenm = 'timeline_' + str(xloc[i]) + '.dat'  
    var0 = pd.read_csv(path0P + filenm, sep=' ', skiprows=0,
                       index_col=False, skipinitialspace=True)
    val0 = var0[varnm] # - np.mean(var0[varnm])
    var1 = pd.read_csv(path1P + filenm, sep=' ', skiprows=0,
                       index_col=False, skipinitialspace=True)
    val1 = var1[varnm] # - np.mean(var1[varnm])
    # val = var[varnm] - base.loc[base['x']==xloc[i], [varnm]].values[0]
    ax1.plot(var0['time'], val0, curve0[i], linewidth=0.8)
    ax1.plot(var1['time'], val1, curve1[i], linewidth=0.8)
    
    fre0, fpsd0 = va.fw_psd(var0[varnm], dt, 1/dt, opt=2, seg=8, overlap=4)
    ax2.semilogx(fre0, fpsd0/np.max(fpsd0), curve0[i], linewidth=0.8)
    fre1, fpsd1 = va.fw_psd(var1[varnm], dt, 1/dt, opt=2, seg=8, overlap=4)
    ax2.semilogx(fre1, fpsd1/np.max(fpsd1), curve1[i], linewidth=0.8)

ax1.set_ylabel(ylab, fontsize=textsize)
ax1.set_xlabel(r"$t u_\infty / \delta_0$", fontsize=textsize)
ax1.set_xlim([700, 1000])
# ax1.set_ylim([-4e-3, 4.1e-3]) # ([-8.0e-4, 6.0e-4])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax1.tick_params(labelsize=numsize)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.grid(b=True, which="both", linestyle=":")
ax1.yaxis.offsetText.set_fontsize(numsize)
ax2.set_ylabel(r"$f \mathcal{P}/\mathcal{P}_\mathrm{max}$", fontsize=textsize-1)
ax2.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
ax2.tick_params(labelsize=numsize)
plt.savefig(pathC + varnm + "_time_d.svg", bbox_inches='tight', dpi=300)
plt.show()
