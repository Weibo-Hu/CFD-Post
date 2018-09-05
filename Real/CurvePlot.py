#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:22:31 2018
    This code for plotting line/curve figures

@author: weibo
"""
#%% Load necessary module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d, splev, splrep
import scipy.optimize as opt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
import copy
from DataPost import DataPost
import FlowVar as fv
from timer import timer
import os
import sys

plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman', #'color' : 'k',
    'weight': 'normal',
    'size': 'large'
}
font1 = {
    'family': 'Times New Roman', #'color' : 'k',
    'weight': 'normal',
    'size': 'medium'
}
path = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
path3 = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/"
path4 = "/media/weibo/Data1/BFS_M1.7L_0505/MeanFlow/"

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
textsize = 18
numsize = 15
matplotlib.rc('font', size=textsize)
#%% Load Data
VarName = [
    'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'T', 'uu', 'uv', 'uw', 'vv', 'vw',
    'ww', 'Q-criterion', 'L2-criterion', 'gradp'
]
MeanFlow = DataPost()
MeanFlow.UserData(VarName, path4+'MeanFlow.dat', 1, Sep='\t')
MeanFlow.AddWallDist(3.0)

# %% Plot BL profile along streamwise
xcoord = np.array([-40, -5, 0, 5, 10, 15, 20, 30, 40])
num = np.size(xcoord)
xtick = np.zeros(num+1)
xtick[-1] = 1.0
fig, ax = plt.subplots(figsize=(10, 3.5))
# matplotlib.rc('font', size=14)
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
    ax.text(i+0.75, 3.0, r'$x/\delta_0={}$'.format(xcoord[i]),
            rotation=90, fontsize=numsize) #fontdict=font)
plt.tick_params(labelsize=numsize)
ax.set_xlabel(r'$u/u_{\infty}$', fontsize=textsize)
ax.set_ylabel(r'$\Delta y/\delta_0$', fontsize=textsize)
ax.grid (b=True, which='both', linestyle=':')
plt.show()
plt.savefig(path2 + 'StreamwiseBLProfile.svg', bbox_inches='tight', pad_inches=0.1)

# %% Compare van Driest transformed mean velocity profile
# xx = np.arange(27.0, 60.0, 0.25)
#z0 = -0.5
#MeanFlow.UserDataBin(path + 'MeanFlow4.h5')
#MeanFlow.ExtraSeries('z', z0, z0)
MeanFlow.AddWallDist(3.0)
MeanFlow.AddMu(13718)
x0 = 63.0 #43.0
# for x0 in xx:
BLProf = copy.copy(MeanFlow)
BLProf.ExtraSeries('x', x0, x0)
# BLProf.SpanAve()
u_tau = fv.UTau(BLProf.walldist, BLProf.u, BLProf.rho, BLProf.mu)
Re_tau = BLProf.rho[0]*u_tau/BLProf.mu[0]*2
print("Re_tau=", Re_tau)
Re_theta = '2000'
StdUPlus1, StdUPlus2 = fv.StdWallLaw()
ExpUPlus = fv.ExpWallLaw(Re_theta)[0]
CalUPlus = fv.DirestWallLaw(BLProf.walldist, BLProf.u, BLProf.rho, BLProf.mu)
fig, ax = plt.subplots(figsize=(6, 5))
# matplotlib.rc('font', size=textsize)
ax.plot(StdUPlus1[:, 0], StdUPlus1[:,1], 'k--', \
        StdUPlus2[:, 0], StdUPlus2[:,1], 'k--', linewidth=1.5)
ax.scatter(ExpUPlus[:, 0], ExpUPlus[:,1], linewidth=0.8, \
           s=8.0, facecolor="none", edgecolor='gray')
#spl = splrep(CalUPlus[:, 0], CalUPlus[:, 1], s=0.1)
#uplus = splev(CalUPlus[:, 0], spl)
#ax.plot(CalUPlus[:, 0], uplus, 'k', linewidth=1.5)
ax.plot(CalUPlus[:, 0], CalUPlus[:, 1], 'k', linewidth=1.5)
ax.set_xscale('log')
ax.set_xlim([1, 2000])
ax.set_ylim([0, 30])
ax.set_ylabel(r'$\langle u_{VD}^+ \rangle$', fontdict=font)
ax.set_xlabel(r'$\Delta y^+$', fontdict=font)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.grid(b=True, which='both', linestyle=':')
plt.tick_params(labelsize='medium')
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
# plt.tick_params(labelsize=numsize)
plt.savefig(
    path2 + 'WallLaw' + str(x0) + '.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# %% Compare Reynolds stresses in Morkovin scaling
U_inf = 469.852
ExpUPlus, ExpUVPlus, ExpUrmsPlus, ExpVrmsPlus, ExpWrmsPlus = \
    fv.ExpWallLaw(Re_theta)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(ExpUrmsPlus[:, 0], ExpUrmsPlus[:, 1], linewidth=0.8, \
           s=8.0, facecolor="none", edgecolor='k')
ax.scatter(ExpVrmsPlus[:, 0], ExpVrmsPlus[:, 1], linewidth=0.8, \
           s=8.0, facecolor="none", edgecolor='r')
ax.scatter(ExpWrmsPlus[:, 0], ExpWrmsPlus[:, 1], linewidth=0.8, \
           s=8.0, facecolor="none", edgecolor='b')
ax.scatter(ExpUVPlus[:, 0], ExpUVPlus[:, 1], linewidth = 0.8, \
           s=8.0, facecolor="none", edgecolor='gray')
xi = np.sqrt(BLProf.rho / BLProf.rho[0])
# UUPlus = BLProf.uu / u_tau
uu = xi * np.sqrt(BLProf.uu* U_inf) 
vv = xi * np.sqrt(BLProf.vv* U_inf) 
ww = xi * np.sqrt(BLProf.ww* U_inf) 
uv = xi * np.sqrt(np.abs(BLProf.uv)* U_inf) * (-1)
#spl = splrep(CalUPlus[:, 0], uu, s=0.1)
#uu = splev(CalUPlus[:, 0], spl)
ax.plot(CalUPlus[:, 0], uu, 'k', linewidth=1.5)
ax.plot(CalUPlus[:, 0], vv, 'r', linewidth=1.5)
ax.plot(CalUPlus[:, 0], ww, 'b', linewidth=1.5)
ax.plot(CalUPlus[:, 0], uv, 'gray', linewidth=1.5)
ax.set_xscale('log')
ax.set_xlim([1, 2000])
# ax.set_ylim([0, 30])
ax.set_ylabel(r'$\langle u^{\prime}_i u^{\prime}_j \rangle$', fontdict=font)
ax.set_xlabel(r'$y^+$', fontdict=font)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.grid(b=True, which='both', linestyle=':')
plt.tick_params(labelsize='medium')
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(path2 + 'ReynoldStress' + str(x0) + '.svg',
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# %% Plot streamwise skin friction
MeanFlow.AddWallDist(3.0)
WallFlow = MeanFlow.DataTab.groupby("x", as_index=False).nth(1)
mu = fv.Viscosity(13718, WallFlow['T'])
Cf = fv.SkinFriction(mu, WallFlow['u'], WallFlow['walldist'])
fig2, ax2 = plt.subplots(figsize=(5, 4))
# fig = plt.figure(figsize=(8, 3.5))
matplotlib.rc('font', size=textsize)
# plt.tick_params(labelsize=numsize)
# ax2 = fig.add_subplot(121)
matplotlib.rc('font', size=textsize)
ax2.plot(WallFlow['x'], Cf, 'k', linewidth=1.5)
ax2.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax2.set_ylabel(r'$\langle C_f \rangle$', fontsize=textsize)
# ax2.set_xlim([-40.0, 70.0])
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax2.axvline(x=10.8, color='gray', linestyle='--', linewidth=1.0)
ax2.grid(b=True, which='both', linestyle=':')
ax2.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.savefig(path2+'Cf.svg',bbox_inches='tight', pad_inches=0.1)
plt.show()

# %% pressure coefficiency
fa = 1.7 * 1.7 * 1.4
fig3, ax3 = plt.subplots(figsize=(5, 4))
# ax3 = fig.add_subplot(122)
ax3.plot(WallFlow['x'], WallFlow['p'] * fa, 'k', linewidth=1.5)
ax3.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax3.set_ylabel(r'$\langle p_w \rangle/p_{\infty}$', fontsize=textsize)
# ax3.set_xlim([-40.0, 70.0])
ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax3.axvline(x=10.8, color='gray', linestyle='--', linewidth=1.0)
ax3.grid(b=True, which='both', linestyle=':')
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(path2 + 'Cp.svg', dpi=300)
plt.show()

# %% Load data for Computing intermittency factor
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots1/"
dirs = sorted(os.listdir(InFolder))
data = pd.read_hdf(InFolder + dirs[0])
data['walldist'] = data['y']
data.loc[data['x'] >= 0.0, 'walldist'] += 3.0
NewFrame = data.query("walldist<=0.0")
ind = NewFrame.index.values
xzone = data['x'][ind].values
# xzone = np.linspace(-40.0, 70.0, 111)
with timer("Load Data"):
    Snapshots = np.vstack(
        [pd.read_hdf(InFolder + dirs[i])['p'] for i in range(np.size(dirs))])
Snapshots = Snapshots.T
Snapshots = Snapshots[ind, :]

# %% calculate
gamma = np.zeros(np.size(xzone))
alpha = np.zeros(np.size(xzone))
p0 = Snapshots[0, :]
sigma = np.std(p0)
timezone = np.arange(600, 999.5 + 0.5, 0.5)
for j in range(np.size(Snapshots[:, 0])):
    gamma[j] = fv.Intermittency(sigma, p0, Snapshots[j, :], timezone)
    alpha[j] = fv.Alpha3(Snapshots[j, :])
# %% Plot Intermittency factor
xarr = np.linspace(-40.0, 70.0, 111)
spl = splrep(xzone, gamma, s=0.35)
yarr = splev(xarr, spl)
fig3, ax3 = plt.subplots(figsize=(10, 3))
# ax3.plot(xzone, gamma, 'k-')
ax3.plot(xarr, yarr, 'k-')
ax3.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax3.set_ylabel(r'$\gamma$', fontdict=font)
ax3.set_xlim([-40.0, 60.0])
#ax3.set_ylim([0.0, 1.0])
ax3.grid(b=True, which='both', linestyle=':')
ax3.axvline(x=0.0, color='k', linestyle='--', linewidth=1.0)
ax3.axvline(x=10.8, color='k', linestyle='--', linewidth=1.0)
plt.tick_params(labelsize='medium')
plt.savefig(path2 + 'Intermittency.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()


#%% Temporal variation of reattachment point
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots1/"
dirs = sorted(os.listdir(InFolder))
data = pd.read_hdf(InFolder + dirs[0])
NewFrame = data.query("x>=9.0 & x<=13.0 & y==-2.99703717231750488")
ind = NewFrame.index.values
timezone = np.arange(600, 999.5 + 0.5, 0.5)
xarr = np.zeros(np.size(timezone))
# xzone = np.linspace(-40.0, 70.0, 111)
with timer("Computing reattaching point"):
    for i in range(np.size(dirs)):
        frame = pd.read_hdf(InFolder + dirs[i])
        frame = frame.iloc[ind]
        xarr[i] = frame.loc[frame['u']>=0.0, 'x'].head(1)
reatt = np.vstack((timezone, xarr))
np.savetxt(path2+"Reattach.dat", reatt, fmt='%.8e', delimiter='  ', header="t, x")
#%%  Plot reattachment point with time      
fig, ax = plt.subplots(figsize=(10, 2))
#spl = splrep(timezone, xarr, s=0.35)
#xarr1 = splev(timezone[0::5], spl)
ax.plot(timezone, xarr, 'k-')
ax.set_xlabel(r'$t u_\infty/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$x/\delta_0$', fontsize=textsize)
ax.grid(b=True, which='both', linestyle=':')
xmean = np.mean(xarr)
ax.axhline(y=xmean, color='k', linestyle='--', linewidth=1.0)
plt.tick_params(labelsize='medium')
plt.savefig(path2 + 'Xr.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
ax.set_ylabel ('Weighted PSD, unitless', fontsize=textsize)
ax.grid(b=True, which='both', linestyle=':')
Fre, FPSD = fv.FW_PSD(xarr, timezone, 2.0)
ax.semilogx(Fre, FPSD, 'k', linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(path2+'XrFWPSD.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

#%% Temporal shock position
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/5/"
dirs = sorted(os.listdir(InFolder))
fig1, ax1 = plt.subplots(figsize=(10, 4))
matplotlib.rc('font', size=textsize)
x0 = np.unique(data['x'])
x1 = x0[x0 > 10.0]
x1 = x1[x1 <= 30.0]
y0 = np.unique(data['y'])
y1 = y0[y0 > -2.5]
xini, yini = np.meshgrid(x1, y1)
corner = (xini<0.0) & (yini<0.0)
shock1 = np.empty(shape=[0, 2])
shock2 = np.empty(shape=[0, 2])
timepoints = np.arange(650.0, 899.5 + 0.5, 0.5)
if np.size(timepoints) != np.size(dirs):
    sys.exit('The input snapshots does not match!!!')
for i in range(np.size(dirs)):
    with timer('Shock position at '+dirs[i]):
        frame = pd.read_hdf(InFolder + dirs[i])
        gradp = griddata((frame['x'], frame['y']), frame['|gradp|'], (xini, yini))
        gradp[corner] = np.nan
        cs = ax1.contour(xini, yini, gradp, levels=0.06, linewidths=1.2, colors='gray')
        xycor = np.empty(shape=[0, 2])
        for isoline in cs.collections[0].get_paths():
            xy = isoline.vertices
            xycor = np.append(xycor, xy, axis=0)
            ax1.plot(xy[:, 0], xy[:, 1], 'r:')  
        ind1 = np.where(xycor[:, 1]==0.0)[0]
        x1 = np.mean(xycor[ind1, 0])
        shock1 = np.append(shock1, [timepoints[i], x1, 0.0])
        ind2 = np.where(xycor[:, 1]==5.0)[0]
        x2 = np.mean(xycor[ind2, 0])
        shock2 = np.append(shock2, [timepoints[i], x2, 5.0])     
np.savetxt(path2+"Shock1.dat", shock1, fmt='%.8e', delimiter='  ', header="t, x, y")
np.savetxt(path2+"Shock2.dat", shock2, fmt='%.8e', delimiter='  ', header="t, x, y")


# %% Streamwise evolution of variable
# Load Data for time-averaged results
MeanFlow = DataPost()
MeanFlow.UserDataBin(path+'MeanFlow.h5')
MeanFlow.AddWallDist(3.0)
tab = MeanFlow.DataTab
# %%
tab1 = tab.loc[tab['z']==0.0]
tab2 = tab1.loc[tab1['walldist']==0.0]
newflow = tab2.query("u<=0.05")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(newflow.x, newflow.u, 'k-')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$u/u_\infty$', fontdict=font)
ax.grid(b=True, which='both', linestyle=':')
plt.tick_params(labelsize='medium')
plt.savefig(path2 + '123.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()
 
# %% POD convergence
fig, ax = plt.subplots(figsize=(5,3)) 
data = np.loadtxt(path2+'POD/PODConvergence.dat', skiprows=1)    
ax.semilogy(data[0, :], data[1, :]/100, marker='o', color='k', linewidth=1.0) 
ax.semilogy(data[0, :], data[2, :]/100, marker='^', color='k', linewidth=1.0) 
ax.semilogy(data[0, :], data[3, :]/100, marker='*', color='k', linewidth=1.0) 
ax.semilogy(data[0, :], data[4, :]/100, marker='s', color='k', linewidth=1.0)  
lab = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']
ax.legend(lab, ncol=2, loc='upper right', fontsize=15)
#          bbox_to_anchor=(1., 1.12), borderaxespad=0., frameon=False)
ax.set_ylim([0.01, 1.0])
ax.set_xlabel(r'$N$', fontsize=textsize)
ax.set_ylabel(r'$\lambda_i/\sum_{k=1}^N \lambda_k$', fontsize=textsize)
ax.grid(b=True, which='both', linestyle=':')
plt.tick_params(labelsize=numsize)
plt.savefig(path2 + 'POD/PODConvergence.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

#%% Vortex Wavelength along streamwise direction
time = '779.50'
wave = pd.read_csv(path3+'L'+time+'.txt', sep='\t', index_col=False, 
                   skipinitialspace=True, keep_default_na=False)
meanwave = pd.read_csv(path3+'LMean.txt', sep='\t', index_col=False, 
                       skipinitialspace=True, keep_default_na=False)
xarr = wave['x']
func = interp1d(meanwave['x'], meanwave['u'])
yarr = func(xarr)
#wave = np.loadtxt(path3+'L'+time+'.txt', delimiter='\t', skiprows=1)
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(wave['x'], wave['u']-yarr, linewidth=1.2)
ax.set_xlim([2.0, 10.0])
ax.set_ylim([-0.08, 0.1])
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$u^\prime/u_\infty$', fontsize=textsize)
ax.grid(b=True, which='both', linestyle=':')
plt.tick_params(labelsize=numsize)
plt.savefig(path2 + 'Wavelen' + time + '.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()
