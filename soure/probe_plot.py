#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 18:58:21 2017
    This code for post-processing data from instantaneous/time-average plane
    data, need data file.                                                                   
@author: weibo
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter
from scipy.interpolate import griddata
from scipy.interpolate import spline, interp1d, splev, splrep
from scipy import integrate, signal
from data_post import DataPost
import variable_analysis as fv
from line_field import LineField as lf
import os

# %% data path settings
path = "/media/weibo/Data2/BFS_M1.7TS/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"

# %% figures properties settings
plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 15
numsize = 12

# %% Time evolution of a specific variable at several streamwise locations
probe = lf()
fa = 1.7 * 1.7 * 1.4
var = 'p'
timezone = [400, 600]
xloc = [-40.0, -30.0, -20.0, -10.0, -1.0]
fig, ax = plt.subplots(5, 1, figsize=(6.4, 5.6))
fig.subplots_adjust(hspace=0.6, wspace=0.15)
matplotlib.rc('font', size=numsize)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
for i in range(np.size(xloc)):
    probe.load_probe(pathP, (xloc[i], 0.0, 0.0))
    probe.extract_data(timezone)
    temp = ( getattr(probe, var) - np.mean(getattr(probe, var)) ) * fa
    ax[i].plot(probe.time, temp, 'k')
    ax[i].ticklabel_format(axis='y', style='sci', 
                           useOffset=False, scilimits=(-2, 2))
    ax[i].set_xlim([400, 600])
    # ax[i].set_ylim([-0.001, 0.001])
    if i != np.size(xloc) - 1:
        ax[i].set_xticklabels('')
    ax[i].set_ylabel(r"$p^\prime/p_\infty$", 
                     fontsize=textsize)
    ax[i].grid(b=True, which='both', linestyle=':')
    ax[i].set_title(r'$x/\delta_0={}$'.format(xloc[i]), fontsize=numsize-1)
ax[-1].set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
plt.show()
plt.savefig(pathF + var + '_TimeEvolveX.svg', 
            bbox_inches='tight', pad_inches=0.1)

# %% Streamwise evolution of a specific variable
probe = lf()
fa = 1.0 # 1.7 * 1.7 * 1.4
var = 'u'
timepoint = 0.60001185E+03 # 0.60000985E+03 # 
xloc = np.arange(-40.0, 0.0, 1.0)
var_val = np.zeros(np.size(xloc))
for i in range(np.size(xloc)):
    probe.load_probe(pathP, (xloc[i], 0.0, 0.0) )
    meanval = getattr(probe, var + '_m')
    temp = probe.ProbeSignal
    var_val[i] = temp.loc[temp['time']==timepoint, var] - meanval

fig, ax = plt.subplots(figsize=(6.4, 2.8))
matplotlib.rc('font', size=numsize)
ax.plot(xloc, var_val * fa, 'k')
ax.set_xlim([-40.0, 0.0])
ax.set_ylim([-0.001, 0.001])
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$u^\prime/u_\infty$', fontsize=textsize)
ax.ticklabel_format(axis='y', style='sci', 
                    useOffset=False, scilimits=(-2, 2))
ax.grid(b=True, which='both', linestyle=':')
plt.show()
plt.savefig(pathF + var + '_StreamwiseEvolve.svg', 
            bbox_inches='tight', pad_inches=0.1)
# %% Frequency Weighted Power Spectral Density
Freq_samp = 50
fig = plt.figure(figsize=(10,5))
matplotlib.rc('font', size=textsize)
ax = fig.add_subplot(121)
ax.set_title(xlabel.format(xloc[1], yloc[1]), fontsize=numsize, loc='right')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.grid(b=True, which='both', linestyle=':')
Fre10, FPSD10 = fv.FW_PSD(getattr(Probe10, var), Probe10.time, 
                          Freq_samp, opt=1)
ax.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
ax.semilogx(Fre10, FPSD10, 'k', linewidth=1.0)
ax.set_ylabel('Weighted PSD, unitless', fontsize=textsize-4)
ax.annotate("(a)", xy=(-0.10, 1.04), xycoords='axes fraction', fontsize=numsize)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)

ax = fig.add_subplot(122)
ax.set_title(xlabel.format(xloc[2], yloc[2]), fontsize=numsize, loc='right')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
ax.grid(b=True, which='both', linestyle=':')
Fre30, FPSD30 = fv.FW_PSD(getattr(Probe20, var), Probe20.time, 
                          Freq_samp, opt=1)
ax.semilogx(Fre30, FPSD30, 'k', linewidth=1.0)
ax.annotate("(b)", xy=(-0.10, 1.04), xycoords='axes fraction', fontsize=numsize)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF+'StreamwiseFWPSD.svg', bbox_inches='tight')
plt.show()

#%% Compute intermittency factor
xzone = np.linspace(-40.0, 40.0, 41)
gamma = np.zeros(np.size(xzone))
alpha = np.zeros(np.size(xzone))
sigma = np.std(Probe0.p)
p0    = Probe0.p
ProbeID = DataPost()
for j in range(np.size(xzone)):
    if xzone[j] <= 0.0:
        ProbeID.LoadProbeData(xzone[j], 0.0, 0.0, pathP)
    else:
        ProbeID.LoadProbeData(xzone[j], -3.0, 0.0, pathP)
    ProbeID.ExtraSeries('time', t1, t2)
    gamma[j] = fv.Intermittency(sigma, p0, ProbeID.p, ProbeID.time)
    alpha[j] = fv.Alpha3(ProbeID.p)

fig3, ax3 = plt.subplots(figsize=(4, 3.5))
ax3.plot(xzone, gamma, 'ko')
ax3.set_xlabel (r'$x/\delta_0$', fontdict=font3)
ax3.set_ylabel (r'$\gamma$', fontdict=font3)
#ax3.set_ylim([0.0, 1.0])
ax3.grid(b=True, which = 'both', linestyle = ':')
ax3.axvline(x=0.0, color='k', linestyle='--', linewidth=1.0)
ax3.axvline(x=12.7, color='k', linestyle='--', linewidth=1.0)
plt.tight_layout(pad = 0.5, w_pad=0.5, h_pad =0.3)
plt.tick_params(labelsize=14)
plt.savefig (pathF+'IntermittencyFactor.svg', dpi = 300)
plt.show()

#%% Skewness coefficient
fig4, ax4 = plt.subplots()
ax4.plot(gamma, alpha, 'ko')
ax4.set_xlabel (r'$\gamma$', fontdict = font3)
ax4.set_ylabel (r'$\alpha_3$', fontdict = font3)
#ax3.set_ylim([0.0, 1.0])
ax4.grid (b=True, which = 'both', linestyle = ':')
#ax4.axvline(x=0.0, color='k', linestyle='--', linewidth=1.0)
#ax4.axvline(x=12.7, color='k', linestyle='--', linewidth=1.0)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 1)
fig4.set_size_inches(6, 5, forward=True)
plt.savefig (pathF+'SkewnessCoeff.svg', dpi = 300)
plt.show()

#%% Spanwise distribution of a specific varibles
# load data
import pandas as pd
loc = ['x', 'y']
valarr = [[-40.0, 0.00781],
          [-20.0, 0.00781],
          [-10.0, 0.00781]]
var = 'p'
fa = 1.7*1.7*1.4
# varlabel = 
#          [15.0, -1.6875]]
#plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
fig, ax = plt.subplots(1, 3, figsize=(8, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.20)
matplotlib.rc('font', size=14)
title = [r'$(a)$', r'$(b)$', r'$(c)$']
# a
val = valarr[0]
fluc = pd.read_csv(pathF+'x=-40.txt', sep=' ', skipinitialspace=True)
pert = fv.PertAtLoc(fluc, var, loc, val)
ax[0].plot(pert[var]-np.mean(pert[var]), pert['z'], 'k-')
# ax[0].set_xlim([0.0, 2e-3])
ax[0].ticklabel_format(axis="x", style="sci", scilimits=(-1, 1))
# ax[0].set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])      
ax[0].set_ylabel(r"$z/\delta_0$", fontsize=textsize)
ax[0].tick_params(labelsize=numsize)
ax[0].set_title(title[0], fontsize=numsize)
ax[0].grid(b=True, which="both", linestyle=":")
# b 
val = valarr[1]
fluc = pd.read_csv(pathF+'x=-20.txt', sep=' ', skipinitialspace=True)
pert = fv.PertAtLoc(fluc, var, loc, val)
ax[1].plot(pert[var]-np.mean(pert[var]), pert['z'], 'k-')
ax[1].set_xlabel(r"$p^{\prime}/p_{\infty}$",
                 fontsize=textsize, labelpad=18.0)
# ax[1].set_xlim([0.0, 2.0e-2])
ax[1].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[1].set_yticklabels('')
ax[1].tick_params(labelsize=numsize)
ax[1].set_title(title[1], fontsize=numsize)
ax[1].grid(b=True, which="both", linestyle=":")
# c
val = valarr[2]
fluc = pd.read_csv(pathF+'x=-10.txt', sep=' ', skipinitialspace=True)
pert = fv.PertAtLoc(fluc, var, loc, val)
ax[2].plot(pert[var]-np.mean(pert[var]), pert['z'], 'k-')
# ax[2].set_xlim([0.025, 0.20])
ax[2].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
ax[2].set_yticklabels('')
ax[2].tick_params(labelsize=numsize)
ax[2].set_title(title[2], fontsize=numsize)
ax[2].grid(b=True, which="both", linestyle=":")

plt.show()
plt.savefig(
    pathF + "PerturProfileZ.svg", bbox_inches="tight", pad_inches=0.1
)

#%% Streamwise distribution of a specific varibles
# load data
import pandas as pd
pathP = '/media/weibo/Data2/BFS_M1.7C_TS/probes2/'
var = 'u'
fa = 1.0 #1.7 * 1.7 * 1.4
t3 = 108
t4 = 170
xzone = np.linspace(-40.0, -5.0, 36)
yval = 0.0
zval = 0.0
lastval = np.zeros(np.size(xzone))
meanval = np.zeros(np.size(xzone))
flucval = np.zeros(np.size(xzone))
ProbeID = DataPost()
for j in range(np.size(xzone)):
    ProbeID.LoadProbeData(xzone[j], yval, zval, pathP)
    ProbeID.ExtraSeries('time', t3, t4)
    lastval[j] = getattr(ProbeID, var)[-1]
    meanval[j] = np.mean(getattr(ProbeID, var))
    flucval[j] = lastval[j] - meanval[j]
# fit curve
def func(t, A, B, C, D):
    return A * np.sin( B * t + C) + D
popt, pcov = DataPost.fit_func(func, xzone, flucval, guess=None)
A, B, C, D = popt
fitfunc = lambda t: A * np.sin( B * t + C) + D
# smooth curve
spl = splrep(xzone, flucval, k=5, s=0.5)
fitfluc = splev(xzone, spl)
# filter curve
b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
filfluc = signal.filtfilt(b, a, flucval)

fig, ax = plt.subplots(figsize=(8, 3))
matplotlib.rc('font', size=14)
ax.plot(xzone, flucval*fa, 'k-', linewidth=1.5)
# ax.plot(xzone, filfluc*fa, 'k-', linewidth=1.5)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$u^{\prime}/u_{\infty}$",
              fontsize=textsize, labelpad=18.0)
# ax[0].set_xlim([0.0, 2e-3])
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
# ax[0].set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])      
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")

plt.show()
plt.savefig(
    pathF + "UPerturProfileX.svg", bbox_inches="tight", pad_inches=0.1
)

#%% Streamwise distribution of a specific varibles
# load data
import pandas as pd
loc = ['z', 'y']
valarr = [[0.0, 0.00781]]
var = 'u'
fa = 1.0 # 1.7 * 1.7 * 1.4
# varlabel = 
#          [15.0, -1.6875]]
#plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
fig, ax = plt.subplots(figsize=(8, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.20)
matplotlib.rc('font', size=14)
title = [r'$(a)$', r'$(b)$', r'$(c)$']
# a
val = valarr[0]
orig = pd.read_csv(pathF+'z=0.txt', sep=' ', skipinitialspace=True)
meanflow = pd.read_csv(pathF+'meanflow.txt', sep=' ', skipinitialspace=True)
fluc = orig.copy()
fluc[var] = orig[var] - meanflow[var]
pert = fv.PertAtLoc(fluc, var, loc, val)
ax.plot(pert['x'], pert[var]*fa, 'k-')
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$u^{\prime}/u_{\infty}$",
              fontsize=textsize, labelpad=18.0)
# ax[0].set_xlim([0.0, 2e-3])
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
# ax[0].set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])      
ax.tick_params(labelsize=numsize)
ax.set_title(title[0], fontsize=numsize)
ax.grid(b=True, which="both", linestyle=":")

plt.show()
plt.savefig(
    pathF + "PerturProfileX.svg", bbox_inches="tight", pad_inches=0.1
)

# %% 
path0 = '/media/weibo/Data2/BFS_M1.7C_TS/probes/'
pathP = '/media/weibo/Data2/BFS_M1.7C_TS/probes1/'
pathF = '/media/weibo/Data2/BFS_M1.7C_TS/probes2/'
dirs = sorted(os.listdir(path0))
for i in range(np.size(dirs)):
    with open (path0 + dirs[i]) as f:
        title1 = f.readline ().split ('\t')
        title2 = f.readline ().split ()
        title = '\n'.join([str(title1), str(title2)])
    f.close()
    file0 = np.genfromtxt(path0+dirs[i], skip_header=2, filling_values=0.0)
    file1 = np.loadtxt(pathP+dirs[i])
    file2 = np.concatenate([file0, file1])
    np.savetxt(pathF+dirs[i], file2, \
                fmt='%1.8e', delimiter = " ", header = str(title))
