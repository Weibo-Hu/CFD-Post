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
from glob import glob
import os
from planar_field import PlanarField as pf
from triaxial_field import TriField as tf


path0 = "/media/weibo/VID2/BFS_M1.7L/"
path0F, path0P, path0M, path0S, path0T, path0I = p2p.create_folder(path0)
path1 = "/media/weibo/VID2/BFS_M1.7TS_LA/"
path1F, path1P, path1M, path1S, path1T, path1I = p2p.create_folder(path1)
path2 = "/media/weibo/VID2/BFS_M1.7TS1_HA/"
path2F, path2P, path2M, path2S, path2T, path2I = p2p.create_folder(path2)
path3 = "/media/weibo/IM1/BFS_M1.7Tur/"
path3F, path3P, path3M, path3S, path3T, path3I = p2p.create_folder(path3)
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
###
###    load data
###
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
MeanFlow3 = pf()
MeanFlow3.load_meanflow(path3)
MeanFlow3.add_walldist(StepHeight)

# %%############################################################################
###
### skin friction & pressure coefficiency/turbulent kinetic energy along streamwise
###
# %% calculate
MeanFlow0.copy_meanval()
WallFlow0 = MeanFlow0.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow1.copy_meanval()
WallFlow1 = MeanFlow1.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow2.copy_meanval()
WallFlow2 = MeanFlow2.PlanarData.groupby("x", as_index=False).nth(1)
MeanFlow3.copy_meanval()
WallFlow3 = MeanFlow3.PlanarData.groupby("x", as_index=False).nth(1)
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
mu3 = va.viscosity(13718, WallFlow3["T"])
Cf3 = va.skinfriction(mu3, WallFlow3["u"], WallFlow3["walldist"]).values
ind3 = np.where(Cf3[:] < 0.005)

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
ax2.plot(xwall[ind3], Cf3[ind3], "k:", linewidth=1.5)
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
# plt.savefig(pathC+'Cf.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# % wall pressure / turbulent kinetic energy
# xwall0 = Flow0['x'].values
tke0 = np.sqrt(WallFlow0['<p`p`>'].values)
tke1 = np.sqrt(WallFlow1['<p`p`>'].values)
tke2 = np.sqrt(WallFlow2['<p`p`>'].values)
tke3 = np.sqrt(WallFlow3['<p`p`>'].values)
# fig3, ax3 = plt.subplots(figsize=(6.4, 2.3))
ax3 = fig.add_subplot(212)
matplotlib.rc("font", size=textsize)
ax3.scatter(xwall[ind0][0::8], tke0[ind0][0::8], s=10, marker='o',
            facecolors='w', edgecolors='C7', linewidths=0.8)
ax3.plot(xwall[ind1], tke1[ind1], "k", linewidth=1.1)
ax3.plot(xwall[ind2], tke2[ind2], "k--", linewidth=1.1)
ax3.plot(xwall[ind2], tke3[ind3], "k:", linewidth=1.5)
ax3.set_yscale('log')
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ylab = r"$p_{\mathrm{rms}}$"  # 
# ylab=r"$2\sqrt{\langle p^{\prime}p^{\prime} \rangle}/\rho_\infty u^2_\infty$"
ax3.set_ylabel(ylab, fontsize=textsize)
ax3.set_xlim([-40.0, 40.0])
ax3.set_ylim([0.00003, 0.05])
# ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
# ax3.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
ax3.grid(b=True, which="both", linestyle=":")
ax3.yaxis.offsetText.set_fontsize(numsize)
ax3.annotate("(b)", xy=(-0.12, 1.04), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.subplots_adjust(hspace=0.2)  # adjust space between subplots
outfile = os.path.join(pathC, 'CfPrms.pdf')
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.1)
plt.show()

# %%############################################################################
"""
    time sequential data at some probes
"""
# %% load sequential slices
path1 = "/media/weibo/IM1/BFS_M1.7Tur/"
path1F, path1P, path1M, path1S, path1T, path1I = p2p.create_folder(path1)
path1Sl = path1 + 'Slice/'
path_z3 = path1Sl + 'TP_2D_Z_03/'
stime = np.arange(975, 1064.00 + 0.25, 0.25) # n*61.5
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
x0 = [-10.0, -0.1875, 0.203125, 0.5, 0.59375, 9.0]
y0 = [0.03125, 0.03125, 0.0390625, -0.078125, -0.171875, -2.34375] 
num_samp = np.size(stime)
var_list = ['x', 'y', 'u', 'p']
var = np.zeros((num_samp, 2))
for j in range(np.size(x0)):
    file = path1P + 'timeline_' + str(x0[j]) + '.dat'
    var = sp.loc[(sp['x']==x0[j]) & (sp['y']==y0[j]), var_list].values
    df = pd.DataFrame(data=np.hstack((stime.reshape(-1, 1), var)), 
                      columns=['time', 'x', 'y', 'u', 'p'])
    df.to_csv(file, sep=' ', index=False, float_format='%1.8e')

# %%############################################################################
"""
    Development of variables with time
"""
# %% variables with time
varnm = 'p'
dt = 0.25
if varnm == 'u':
    ylab = r"$u / u_\infty$"
    ylab_sub = r"$_{u^\prime}$"
else:
    ylab = r"$p^\prime/(\rho_\infty u_\infty ^2)$"
    ylab_sub = r"$_{p^\prime}$"

xloc = [9.0] # [0.59375] # [0.203125] #[-0.1875] # [2.0] #  , 
curve0= ['g-'] # , 'b-', 'g-']
curve1= ['b-'] # , 'b:', 'g:']
fig3, ax3 = plt.subplots(figsize=(6.4, 1.5))
grid = plt.GridSpec(1, 3, wspace=0.4)
ax1 = plt.subplot(grid[0, :2])
ax2 = plt.subplot(grid[0, 2:])
matplotlib.rc("font", size=numsize)
for i in range(np.size(xloc)):
    filenm = 'timeline_' + str(xloc[i]) + '.dat'  
    #var0 = pd.read_csv(path0P + filenm, sep=' ', skiprows=0,
    #                   index_col=False, skipinitialspace=True)
    #val0 = var0[varnm] # - np.mean(var0[varnm])
    var1 = pd.read_csv(path1P + filenm, sep=' ', skiprows=0,
                       index_col=False, skipinitialspace=True)
    val1 = var1[varnm] # - np.mean(var1[varnm])
    # val = var[varnm] - base.loc[base['x']==xloc[i], [varnm]].values[0]
    #ax1.plot(var0['time'], val0, curve0[i], linewidth=0.8)
    ax1.plot(var1['time'], val1, curve1[i], linewidth=0.8)
    
    #fre0, fpsd0 = va.fw_psd(var0[varnm], dt, 1/dt, opt=2, seg=8, overlap=4)
    #ax2.semilogx(fre0, fpsd0, curve0[i], linewidth=0.8)
    # ax2.set_yscale('log')
    fre1, fpsd1 = va.fw_psd(var1[varnm], dt, 1/dt, opt=2, seg=8, overlap=4)
    ax2.semilogx(fre1, fpsd1, curve1[i], linewidth=0.8)

ax1.set_ylabel(ylab, fontsize=textsize)
ax1.set_xlabel(r"$t u_\infty / \delta_0$", fontsize=textsize)
ax1.set_xlim([975, 1100])
# ax1.set_ylim([-4e-3, 4.1e-3]) # ([-8.0e-4, 6.0e-4])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax1.tick_params(labelsize=numsize)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.grid(b=True, which="both", linestyle=":")
ax1.yaxis.offsetText.set_fontsize(numsize-1)
ax2.set_xlim([0.01, 2.0])
ax2.set_ylabel(r"$f \mathcal{P}$", fontsize=textsize-1)  # /\mathcal{P}_\mathrm{max}
ax2.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize-1)
ax2.tick_params(labelsize=numsize)
plt.savefig(path1F + varnm + "_time_f.svg", bbox_inches='tight', dpi=300)
plt.show()

# %%############################################################################
"""
    frequency-weighted PSD
"""
# %% singnal of bubble
# -- temporal evolution
data1 = np.loadtxt(path1I + "BubbleArea.dat", skiprows=1)
data3 = np.loadtxt(path3I + "BubbleArea.dat", skiprows=1)
dt = 0.25
Xb1 = data1[:, 1]
Xb3 = data3[:, 1]
Lr1 = 10.9
Lr3 = 8.9
# -- FWPSD
fig, ax = plt.subplots(figsize=(3.0, 3.0))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f L_r/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre1, FPSD1 = va.fw_psd(Xb1, dt, 1/dt, opt=1, seg=8, overlap=4)
Fre3, FPSD3 = va.fw_psd(Xb3, dt, 1/dt, opt=1, seg=8, overlap=4)
ax.semilogx(Fre1*Lr1, FPSD1, "k-", linewidth=1.0)
# ax.scatter(Fre1*Lr1, FPSD1, s=10, marker='o',
#           facecolors='w', edgecolors='C7', linewidths=0.8)
ax.semilogx(Fre3*Lr3, FPSD3, "k:", linewidth=1.5)
# ax.set_xscale('log')
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathC + "XbFWPSD_com.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% singnal of bubble
# -- temporal evolution
xloc1 = np.loadtxt(path1S + "FWPSD_x_Z_003.dat", skiprows=0)
freq1 = np.loadtxt(path1S + "FWPSD_freq_Z_003.dat", skiprows=0)
psd1 = np.loadtxt(path1S + 'u_FWPSD_psd_Z_003.dat', skiprows=0)
ind1 = np.where(xloc1[:]==10.875)[0]

xloc3 = np.loadtxt(path3S + "FWPSD_x_Z_03.dat", skiprows=0)
freq3 = np.loadtxt(path3S + "FWPSD_freq_Z_03.dat", skiprows=0)
psd3 = np.loadtxt(path3S + 'u_FWPSD_psd_Z_03.dat', skiprows=0)
ind3 = np.where(xloc3[:]==8.875)[0]
# -- FWPSD
fig, ax = plt.subplots(figsize=(3.0, 3.0))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f L_r/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
ax.semilogx(freq1, psd1[:, ind1], "k-", linewidth=1.0)
# ax.scatter(Fre1*Lr1, FPSD1, s=10, marker='o',
#           facecolors='w', edgecolors='C7', linewidths=0.8)
ax.semilogx(freq3, psd3[:, ind3], "k:", linewidth=1.5)
# ax.set_xscale('log')
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathC + "VarFWPSD_com.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %%############################################################################
"""
    Development of variables with spanwise direction
"""
# %% load data
TimeFlow0 = tf()
TimeFlow0.load_3data(path0T, FileList=path0+'TP_912.plt') #, NameList='plt')
TimeFlow1 = tf()
TimeFlow1.load_3data(path1T, FileList=path1+'TP_912.plt') #, NameList='plt')
# %% save data
xyloc = [[-0.1875, 0.03125], [0.203125, 0.0390625], [0.59375, -0.171875], [2.0, 0.46875]]
for i in range(np.shape(xyloc)[0]):
    file = path0P + 'spanwise_' + str(xyloc[i][0]) + '.dat'
    var0 = TimeFlow0.TriData
    df = var0.loc[(var0['x']==xyloc[i][0]) &  (var0['y']==xyloc[i][1])]
    val0 = df.drop_duplicates(subset='z', keep='first')
    val0.to_csv(file, sep=' ', index=False, float_format='%1.8e')

    file = path1P + 'spanwise_' + str(xyloc[i][0]) + '.dat'
    var1 = TimeFlow1.TriData
    df = var1.loc[(var1['x']==xyloc[i][0]) &  (var1['y']==xyloc[i][1])]
    val1 = df.drop_duplicates(subset='z', keep='first')
    val1.to_csv(file, sep=' ', index=False, float_format='%1.8e')

# %% variables with time
varnm = 'u'
dz = 0.03125
if varnm == 'u':
    ylab = r"$u / u_\infty$"
    ylab_sub = r"$_{u^\prime}$"
else:
    ylab = r"$p /(\rho_\infty u_\infty ^2)$"
    ylab_sub = r"$_{p^\prime}$"

xyloc = [[-0.1875, 0.03125]] # [2.0] # [0.59375] # [0.203125] # , 
curve0= ['g-'] # , 'b-', 'g-']
curve1= ['b-'] # , 'b:', 'g:']
fig3, ax3 = plt.subplots(figsize=(6.4, 1.5))
grid = plt.GridSpec(1, 3, wspace=0.4)
ax1 = plt.subplot(grid[0, :2])
ax2 = plt.subplot(grid[0, 2:])
matplotlib.rc("font", size=numsize)
for i in range(np.shape(xyloc)[0]):
    filenm = 'spanwise_' + str(xyloc[i][0]) + '.dat'  
    val0 = pd.read_csv(path0P + filenm, sep=' ', skiprows=0,
                       index_col=False, skipinitialspace=True)
    ax1.plot(val0['z'], val0[varnm]-np.mean(val0[varnm]), curve0[i], linewidth=0.8)
    
    val1 = pd.read_csv(path1P + filenm, sep=' ', skiprows=0,
                       index_col=False, skipinitialspace=True)
    ax1.plot(val1['z'], val1[varnm]-np.mean(val1[varnm]), curve1[i], linewidth=0.8)
    
    fre0, fpsd0 = va.fw_psd(val0[varnm], dz, 1/dz, opt=2, seg=4, overlap=2)
    ax2.semilogx(fre0, fpsd0, curve0[i], linewidth=0.8)
    fre1, fpsd1 = va.fw_psd(val1[varnm], dz, 1/dz, opt=2, seg=4, overlap=2)
    ax2.semilogx(fre1, fpsd1, curve1[i], linewidth=0.8)

ax1.set_ylabel(ylab, fontsize=textsize)
ax1.set_xlabel(r"$z / \delta_0$", fontsize=textsize)
ax1.set_xlim([-8.0, 8.0])
# ax1.set_ylim([-4e-3, 4.1e-3]) # ([-8.0e-4, 6.0e-4])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax1.tick_params(labelsize=numsize)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.grid(b=True, which="both", linestyle=":")
ax1.yaxis.offsetText.set_fontsize(numsize-1)
ax2.set_ylabel(r"$\lambda_z \mathcal{P}$", fontsize=textsize-1)  # /\mathcal{P}_\mathrm{max}
ax2.set_xlabel(r"$\lambda_z \delta_0$", fontsize=textsize)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize-1)
ax2.tick_params(labelsize=numsize)
plt.savefig(pathC + "u_z_a_per.svg", bbox_inches='tight', dpi=300)
plt.show()

# %%############################################################################
"""
    Growth of RMS along streamwise direction
"""
# %% Plot RMS of velocity on the wall along streamwise direction
xynew0 = pd.read_csv(path0M + 'MaxRMS.dat', sep=' ')
xynew1 = pd.read_csv(path1M + 'MaxRMS.dat', sep=' ')
fig, ax = plt.subplots(figsize=(6.4, 2.6))
matplotlib.rc('font', size=numsize)
ylab = r"$\sqrt{u^{\prime 2}_\mathrm{max}}/u_{\infty}$"
ax.set_ylabel(ylab, fontsize=textsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
# ax.plot(xynew0['x'], np.sqrt(xynew0['<u`u`>']), 'g:', linewidth=1.0)
ax.scatter(xynew0['x'], np.sqrt(xynew0['<u`u`>']), s=7, marker='o',
           facecolors='w', edgecolors='g', linewidths=0.8)
ax.plot(xynew1['x'], np.sqrt(xynew1['<u`u`>']), 'b-', linewidth=1.0)
ax.set_yscale('log')
ax.set_xlim([-5.0, 20.0])
ax.set_ylim([0.005, 0.5])
# ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.grid(b=True, which="both", linestyle=":")
ax.tick_params(labelsize=numsize)
ax.yaxis.offsetText.set_fontsize(numsize-1)
plt.show()
plt.savefig(
    pathC + "MaxRMS_x1.svg", bbox_inches="tight", pad_inches=0.1
)

# %%
