#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:22:31 2018
    This code for plotting line/curve figures

@author: weibo
"""
#%% Load necessary module
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter
from scipy.interpolate import griddata
import copy
from DataPost import DataPost
import FlowVar as fv

plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman', #'color' : 'k',
    'weight': 'normal',
}

path = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

#%% Load Data
VarName  = ['x', 'y', 'u', 'v', 'w', 'rho', 'p', 'T', 'uu', \
            'uv', 'uw', 'vv', 'vw', 'ww', 'Q-criterion', 'L2-criterion']
MeanFlow = DataPost()
MeanFlow.UserData(VarName, path2+'Meanflow.dat', 1, Sep='\t')
MeanFlow.AddWallDist(3.0)

# %% Plot BL profile along streamwise
xcoord = np.array([-40, -20, 0, 5, 10, 15, 20, 30, 40])
num = np.size(xcoord)
xtick = np.zeros(num+1)
xtick[-1] = 1.0
fig, ax = plt.subplots(figsize=(7, 3))
matplotlib.rc('font', size=14)
ax.plot(np.arange(num+1), np.zeros(num+1), 'w-')
ax.set_xlim([0, num+0.5])
ax.set_ylim([0, 4])
ax.set_xticks(np.arange(num+1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = xtick
#ax.set_xticklabels(labels, fontdict=font)
ax.set_xticklabels(["$%d$"%f for f in xtick])
for i in range(num):
    y0, q0 = MeanFlow.BLProfile('x', xcoord[i], 'u')
    ax.plot(q0+i, y0, 'k-')
    ax.text(i+0.75, 3.0, r'$x/\delta_0={}$'.format(xcoord[i]), rotation=90, fontdict=font)
plt.tick_params(labelsize=12)
ax.set_xlabel(r'$u/u_{\infty}$', fontsize=14)
ax.set_ylabel(r'$y/\delta_0$', fontsize=14)
ax.grid (b=True, which='both', linestyle=':')
plt.show()
plt.savefig(path2 + 'StreamwiseBLProfile.svg', bbox_inches='tight', pad_inches=0.1)

# %% Compare van Driest transformed mean velocity profile
# xx = np.arange(27.0, 60.0, 0.25)
z0 = -0.5
MeanFlow.UserDataBin(path + 'MeanFlow4.h5')
MeanFlow.ExtraSeries('z', z0, z0)
MeanFlow.AddWallDist(3.0)
MeanFlow.AddMu(13718)
x0 = 43.0
# %%
# for x0 in xx:
BLProf = copy.copy(MeanFlow)
BLProf.ExtraSeries('x', x0, x0)
# BLProf.SpanAve()
u_tau = fv.UTau(BLProf.walldist, BLProf.u, BLProf.rho, BLProf.mu)
Re_tau = BLProf.rho[0]*u_tau/BLProf.mu[0]*13718
print("Re_tau=", Re_tau)
Re_theta = '2000'
StdUPlus1, StdUPlus2 = fv.StdWallLaw()
ExpUPlus = fv.ExpWallLaw(Re_theta)[0]
CalUPlus = fv.DirestWallLaw(BLProf.walldist, BLProf.u, BLProf.rho, BLProf.mu)
fig, ax = plt.subplots(figsize=(4, 3.6))
textsize = 16
numsize = 14
matplotlib.rc('font', size=textsize)
ax.plot(StdUPlus1[:, 0], StdUPlus1[:,1], 'k--', \
            StdUPlus2[:, 0], StdUPlus2[:,1], 'k--', linewidth=1.5)
ax.scatter(ExpUPlus[:, 0], ExpUPlus[:,1], linewidth=0.8, \
           s=8.0, facecolor="none", edgecolor='gray')
ax.plot(CalUPlus[:, 0], CalUPlus[:, 1], 'k', linewidth=1.5)
ax.set_xscale('log')
ax.set_xlim([1, 2000])
ax.set_ylim([0, 30])
ax.set_ylabel(r'$\langle u_{VD}^+ \rangle$', fontsize=16)
ax.set_xlabel(r'$y^+$', fontsize=16)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.grid(b=True, which='both', linestyle=':')
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.tick_params(labelsize=numsize)
plt.savefig(
    path2 + 'WallLaw' + str(x0) + '.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# %% Compare Reynolds stresses in Morkovin scaling
ExpUPlus, ExpUVPlus, ExpUrmsPlus, ExpVrmsPlus, ExpWrmsPlus = \
    fv.ExpWallLaw(Re_theta)
fig, ax = plt.subplots(figsize=(4, 3.6))
matplotlib.rc('font', size=textsize)
plt.tick_params(labelsize=numsize)
ax.scatter(ExpUrmsPlus[:, 0], ExpUrmsPlus[:, 1], linewidth = 0.8, \
           s=8.0, facecolor="none", edgecolor='k')
ax.scatter(ExpVrmsPlus[:, 0], ExpVrmsPlus[:, 1], linewidth = 0.8, \
           s=8.0, facecolor="none", edgecolor='gray')
ax.scatter(ExpWrmsPlus[:, 0], ExpWrmsPlus[:, 1], linewidth = 0.8, \
           s=8.0, facecolor="none", edgecolor='gray')
ax.scatter(ExpUVPlus[:, 0], ExpUVPlus[:, 1], linewidth = 0.8, \
           s=8.0, facecolor="none", edgecolor='gray')
xi = np.sqrt(BLProf.rho / BLProf.rho[0])
UUPlus = BLProf.uu / u_tau
uu = xi * np.sqrt(UUPlus)
ax.plot(CalUPlus[:, 0], uu, 'k', linewidth=1.5)
ax.set_xscale('log')
ax.set_xlim([1, 2000])
# ax.set_ylim([0, 30])
ax.set_ylabel(r'$\langle u^{\prime}_i u^{\prime}_j \rangle$', fontsize=16)
ax.set_xlabel(r'$y^+$', fontsize=16)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.grid(b=True, which='both', linestyle=':')
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(path2 + 'ReynoldStress' + str(x0) + '.svg',
            bbox_inches='tight', pad_inches=0.1)
plt.show()
