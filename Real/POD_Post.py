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

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
                self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('font', **font)
textsize = 18
numsize = 15
# %% load data
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Snapshots1/"
SaveFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/Test"
path = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/POD/"
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
# set range to do POD and DMD: (x:-5~8, y:-3~5, make some tests)
# (x:8~15, y:-3~3)
#del DataFrame
#with timer("Load Data"):
#    Snapshots = np.vstack(
#        [pd.read_hdf(InFolder + dirs[i])['u'] for i in range(np.size(dirs))])
var = 'u'    
fa = 1.0 #1.7*1.7*1.4
timepoints = np.arange(650, 974.5 + 0.5, 0.5)
Snapshots = DataFrame[var]
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(InFolder + dirs[i+1])
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        Snapshots = np.vstack((Snapshots, TempFrame[var]))
        DataFrame += TempFrame
Snapshots = Snapshots.T   
Snapshots = Snapshots[ind, :]
Snapshots = Snapshots*fa
m, n = np.shape(Snapshots)
AveFlow = DataFrame/np.size(dirs)
meanflow = AveFlow.query("x>=-5.0 & x<=20.0 & y>=-3.0 & y<=5.0")
# %% POD
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
    
with timer("POD computing"):
    eigval, eigvec, phi, coeff = \
        rm.POD(Snapshots, SaveFolder, fluc='True', method='svd')

# %% Eigvalue Spectrum
EFrac, ECumu, N_modes = rm.POD_EigSpectrum(90, eigval)
np.savetxt(path+'EnergyFraction650.dat', EFrac, fmt='%1.7e', delimiter='\t')
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(6,5))
xaxis = np.arange(0, N_modes + 1)
ax1.scatter(
    xaxis[1:],
    EFrac[:N_modes],
    c='black',
    marker='o',
    s=15.0,
)   # fraction energy of every eigval mode
#ax1.legend('E_i')
ax1.set_ylim(bottom=0)
ax1.set_xlabel('Mode')
ax1.set_ylabel(r'$E_i$')
ax1.grid(b=True, which='both', linestyle=':')
ax1.tick_params(labelsize=numsize)
ax2 = ax1.twinx()   # cumulation energy of first several modes
#ax2.fill_between(xaxis, ECumu[:N_modes], color='grey', alpha=0.5)
ESum = np.zeros(N_modes+1)
ESum[1:] = ECumu[:N_modes]
ax2.plot(xaxis, ESum, color='grey', label=r'$ES_i$')
ax2.set_ylim([0, 100])
ax2.set_ylabel(r'$ES_i$')
ax2.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(path+var+'_PODEigSpectrum.svg', bbox_inches='tight')
plt.show()

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

# %% specific mode in space
ind = 8
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
newflow = phi[:, ind - 1]*coeff[ind - 1, 0]
u = griddata((xval, yval), newflow, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(6, 2))
c1 = -4.2e-3
c2 = 4.2e-3
print("The limit value: ", np.min(newflow), np.max(newflow))
lev1 = np.linspace(c1, c2, 11)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, cmap='RdBu_r', levels=lev1) #, extend='both') 
#ax.contour(x, y, u, levels=lev2, colors='k', linewidths=0.8, extend='both')
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'), extend='both')
# cbar = ax.contour(x, y, u, levels=lev2, extend='both')
# plt.clabel(cbar, inline=1, fontsize=16)
ax.grid(b=True, which='both', linestyle=':')  # blue, grey, red
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
ax.set_ylabel(r'$y/\delta_0$', fontdict=font)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.18, 0.76, 0.24, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", 
                     ticks=rg2)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(numsize)
cbar1.update_ticks()
cbar1.set_label(r'$\varphi_{}$'.format(var), rotation=0, fontdict=font)
cbaxes.tick_params(labelsize=numsize)
ax.contour(x, y, meanu, levels=0.0,
           linewidths=1.0, linestyles=':', colors='k')
ax.contour(x, y, meanma, levels=1.0,
           linewidths=1.0, linestyles=':', colors='green')
plt.savefig(path+var+'_PODMode'+str(ind)+'.svg', bbox_inches='tight')
plt.show()
# %% First several modes with time and WPSD
fig, ax = plt.subplots(figsize=(6, 3))
matplotlib.rc('font', size=textsize)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
lab = []
NO = [3, 4]
ax.plot(timepoints, coeff[NO[0]-1, :], 'k-')
lab.append('Mode '+str(NO[0]))
ax.plot(timepoints, coeff[NO[1]-1, :], 'k:')
lab.append('Mode '+str(NO[1]))
ax.legend(lab, ncol=2, loc='upper right', fontsize=14,
          bbox_to_anchor=(1., 1.12), borderaxespad=0., frameon=False)
ax.set_xlabel(r'$tu_\infty/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$a_{}$'.format(var), fontsize=textsize)
ax.tick_params(labelsize=numsize)
plt.grid(b=True, which='both', linestyle=':')
plt.savefig(path+var+'_PODModeTemp' + str(NO[0]) + '.svg', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
matplotlib.rc('font', size=numsize)
freq, psd = fv.FW_PSD(coeff[NO[0]-1, :], timepoints, 2)
ax.semilogx(freq, psd, 'k-')
freq, psd = fv.FW_PSD(coeff[NO[1]-1, :], timepoints, 2)
ax.semilogx(freq, psd, 'k:')
ax.legend(lab, fontsize=15, frameon=False)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_xlabel(r'$f\delta_0/U_\infty$', fontsize=textsize)
ax.set_ylabel('WPSD, unitless', fontsize=textsize)
ax.tick_params(labelsize=numsize)
plt.grid(b=True, which='both', linestyle=':')
plt.savefig(path+var+'_POD_WPSDModeTemp' + str(NO[0]) + '.svg', bbox_inches='tight')
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
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(12, 4))
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
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
plt.savefig(path+var+'_PODReconstructFlow.svg', bbox_inches='tight')
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
# PODMeanflow(Snapshots)
