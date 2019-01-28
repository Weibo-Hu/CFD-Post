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
import sys
from DataPost import DataPost
import FlowVar as fv
import os

plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',  # 'color' : 'k',
    'weight': 'normal',
}

matplotlib.rc('font', **font)
textsize = 18
numsize = 15

font0 = {'family' : 'Times New Roman',
        'color' : 'k',
        'weight' : 'normal',
        'size' : 12,
        }
font1 = {'family' : 'Times New Roman',
        'color' : 'k',
        'weight' : 'normal',
        'size' : 14,}

font2 = {'family' : 'Times New Roman',
         'color' : 'k',
         'weight' : 'normal',
         'size' : 16,
        }

font3 = {'family' : 'Times New Roman',
        'color' : 'k',
        'weight' : 'normal',
        'size' : 18,
}

path = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
path1 = "/media/weibo/Data2/BFS_M1.7C_TS/probes/"
path2 = "/media/weibo/Data2/BFS_M1.7C_TS/DataPost/"
path3 = "/media/weibo/Data2/BFS_M1.7C_TS/DataPost/"
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

textsize = 22
labsize = 20
t0 = 600
t1 = 100 #600 #960
t2 = 180 #1000.0 #

#%% Read data for Streamwise variations of frequency of a specific variable
Probe0 = DataPost()
xloc = [-40.0, -30.0, -20.0, -10.0]
yloc = [0.0, 0.0, 0.0, 0.0]
#xloc = [-30.0, -20, -10.0, -5.0]
#yloc = [0.0, 0.0, 0.0, 0.0]
Probe0.LoadProbeData(xloc[0], yloc[0], 0.0, path1, Uniq=True)
Probe0.ExtraSeries('time', t1, t2)
#Probe0.DataTab.to_csv(OutFolder+'ProbeE3.dat', sep='\t', index=False, float_format='%.8e')
#Probe0.AveAtSameXYZ('All')
#time1 = Probe0.time
Probe10 = DataPost()
Probe10.LoadProbeData(xloc[1], yloc[1], 0.0, path1, Uniq=True)
#Probe10.unique_rows()
Probe10.ExtraSeries('time', t1, t2)
#Probe10.AveAtSameXYZ('All')

Probe20 = DataPost()
Probe20.LoadProbeData(xloc[2], yloc[2], 0.0, path1, Uniq=True)
#Probe20.unique_rows()
Probe20.ExtraSeries('time', t1, t2)
#Probe20.AveAtSameXYZ('All')
#time2 = Probe20.time

Probe30 = DataPost()
Probe30.LoadProbeData(xloc[3], yloc[3], 0.0, path1, Uniq=True)
#Probe30.unique_rows()
Probe30.ExtraSeries('time', t1, t2)
#Probe30.AveAtSameXYZ('All')


#%% Streamwise variations of time evolution of a specific variable
fa = 1.0 #1.7*1.7*1.4
matplotlib.rc('font', size=textsize)
fig = plt.figure(figsize=(10, 8))
matplotlib.rc('font', size=18)
ax = fig.add_subplot(411)
xlabel = r'$x/\delta_0={}, \ y/\delta_0={}$'
ytitle = r'$u/u_\infty$' # 'r'$p/p_\infty$'
var = 'u'
ax.set_title(xlabel.format(xloc[0], yloc[0]), fontsize=numsize)
ax.set_xlim([t1, t2])
#ax.set_ylim([0.99, 1.01])
ax.set_xticklabels('')
#ax.set_xlabel (r'$t u_\infty/\delta$', fontdict = font1)
ax.set_ylabel(ytitle, fontsize=textsize)
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax.yaxis.offsetText.set_fontsize(numsize)
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#ax.text (850, 6.5, 'x=0', fontdict = font2)
ax.grid(b=True, which='both', linestyle=':')
#Probe0P = Probe0.p-BProbe0P
#grow0, time0 = Probe0.GrowthRate(Probe0.time, Probe0P)
#ax.plot (time0, grow0, 'k', linewidth = 1.5)
#Probe0.AddUGrad(0.015625)
meanval = np.mean(getattr(Probe0, var)*fa)
ax.plot(Probe0.time, getattr(Probe0, var)*fa-meanval, 'k', linewidth=1.0)
ax.tick_params(labelsize=numsize)

#ax.annotate("(a)", xy=(-0.05, 1.2), xycoords='axes fraction', fontsize=labsize)
ax.tick_params(labelsize=numsize)
# fit curve
def func(t, A, B):
    return A / t + B
popt, pcov = Probe0.fit_func(func, Probe0.time, getattr(Probe0, var), guess=None)
A, B = popt
fitfunc = lambda t: A / t + B
#ax.plot(Probe0.time, fitfunc(Probe0.time), 'b', linewidth=1.5)


ax = fig.add_subplot(412)
ax.set_title(xlabel.format(xloc[1], yloc[1]), fontsize=numsize)
ax.set_xlim([t1, t2])
ax.set_xticklabels('')
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax.yaxis.offsetText.set_fontsize(numsize)
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.set_ylabel(ytitle, fontsize=textsize)
ax.grid(b=True, which='both', linestyle = ':')
meanval = np.mean(getattr(Probe10, var)*fa)
ax.plot(Probe10.time, getattr(Probe10, var)*fa-meanval, 'k', linewidth=1.0)
#ax.annotate("(b)", xy=(-0.05, 1.15), xycoords='axes fraction', fontsize=labsize)
ax.tick_params(labelsize=numsize)

ax = fig.add_subplot(413)
ax.set_title(xlabel.format(xloc[2], yloc[2]), fontsize=numsize)
ax.set_xlim([t1, t2])
ax.set_xticklabels('')
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax.yaxis.offsetText.set_fontsize(numsize)
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.set_ylabel(ytitle, fontsize=textsize)
ax.grid(b=True, which ='both', linestyle =':')
meanval = np.mean(getattr(Probe20, var)*fa)
ax.plot(Probe20.time, getattr(Probe20, var)*fa-meanval, 'k', linewidth=1.0)
#ax.annotate("(c)", xy=(-0.05, 1.15), xycoords='axes fraction', fontsize=labsize)
ax.tick_params(labelsize=numsize)

ax = fig.add_subplot(414)
ax.set_title(xlabel.format(xloc[3], yloc[3]), fontsize=numsize)
ax.set_xlim([t1, t2])
#ax.set_xticklabels ('')
ax.set_xlabel(r'$t u_\infty/\delta_0$', fontsize=textsize)
ax.set_ylabel(ytitle, fontsize=textsize)
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax.yaxis.offsetText.set_fontsize(numsize)
#ax.text (850, 6.5, 'x=0', fontdict = font2)
ax.grid(b=True, which='both', linestyle=':')
meanval = np.mean(getattr(Probe30, var)*fa)
ax.plot(Probe30.time, getattr(Probe30, var)*fa-meanval, 'k', linewidth=1.0)
#ax.annotate("(d)", xy=(-0.05, 1.15), xycoords='axes fraction', fontsize=labsize)
#ax.plot (Probe40.time, Probe40.p, 'k', linewidth = 1.5)
ax.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(path3+'UStreamwiseTimeEvolution.svg', bbox_inches='tight')
plt.show()


# %% Frequency Weighted Power Spectral Density
Freq_samp = 50
fig = plt.figure(figsize=(10,8))
matplotlib.rc('font', size=textsize)
ax = fig.add_subplot(221)
ax.set_title(xlabel.format(xloc[0], yloc[0]), fontdict=font1)
#ax.set_xlim ([720, 960])
#ax.set_xticklabels ('')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax = plt.gca()
#ax.yaxis.get_major_formatter().set_powerlimits((0,1))
ax.set_ylabel('WPSD, unitless', fontsize=textsize)
ax.grid(b=True, which='both', linestyle=':')
#Fre0, FPSD0 = fv.FW_PSD(getattr(Probe0, var), Probe0.time, Freq_samp)
Fre0, FPSD0 = fv.FW_PSD(getattr(Probe0, var)-fitfunc(Probe0.time), 
                        Probe0.time, Freq_samp, opt=1)
ax.semilogx(Fre0, FPSD0, 'k', linewidth=1.0)
#ax.annotate("(a)", xy=(-0.05, 1.04), xycoords='axes fraction', fontsize=labsize)
ax.yaxis.offsetText.set_fontsize(labsize)
plt.tick_params(labelsize=labsize)
#ax.psd(Probe0.p-np.mean(Probe0.p), 100, 10)
ax = fig.add_subplot(222)
ax.set_title(xlabel.format(xloc[1], yloc[1]), fontdict=font1)
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#ax.set_xlabel (r'$f\delta_0/U_\infty$', fontdict = font1)
#ax.set_ylabel ('Weighted PSD, unitless', fontdict = font1)
ax.grid(b=True, which='both', linestyle=':')
Fre10, FPSD10 = fv.FW_PSD (getattr(Probe10, var), Probe10.time, 
                           Freq_samp, opt=1)
ax.semilogx(Fre10, FPSD10, 'k', linewidth=1.0)
#ax.annotate("(b)", xy=(-0.05, 1.04), xycoords='axes fraction', fontsize=labsize)
ax.yaxis.offsetText.set_fontsize(labsize)
plt.tick_params(labelsize=labsize)

ax = fig.add_subplot(223)
ax.set_title(xlabel.format(xloc[2], yloc[2]), fontdict=font1)
#ax.set_xlim ([720, 960])
#ax.set_xticklabels ('')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
ax.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
ax.set_ylabel('WPSD, unitless', fontsize=textsize)
ax.grid(b=True, which='both', linestyle=':')
Fre20, FPSD20 = fv.FW_PSD(getattr(Probe20, var), Probe20.time, 
                          Freq_samp, opt=1)
ax.semilogx (Fre20, FPSD20, 'k', linewidth = 1.0)
#ax.annotate("(c)", xy=(-0.05, 1.04), xycoords='axes fraction', fontsize=labsize)
ax.yaxis.offsetText.set_fontsize(labsize)
plt.tick_params(labelsize=labsize)

ax = fig.add_subplot(224)
ax.set_title(xlabel.format(xloc[3], yloc[3]), fontdict=font1)
#ax.set_xlim ([720, 960])
#ax.set_xticklabels ('')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
#ax.set_ylabel ('Weighted PSD, unitless', fontdict = font1)
ax.grid(b=True, which='both', linestyle=':')
Fre30, FPSD30 = fv.FW_PSD(getattr(Probe30, var), Probe30.time, 
                          Freq_samp, opt=1)
ax.semilogx(Fre30, FPSD30, 'k', linewidth=1.0)
#ax.annotate("(d)", xy=(-0.05, 1.04), xycoords='axes fraction', fontsize=labsize)
ax.yaxis.offsetText.set_fontsize(labsize)
plt.tick_params(labelsize=labsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig (path3+'StreamwiseFWPSD.svg', bbox_inches='tight')
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
        ProbeID.LoadProbeData(xzone[j], 0.0, 0.0, path1)
    else:
        ProbeID.LoadProbeData(xzone[j], -3.0, 0.0, path1)
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
plt.savefig (path3+'IntermittencyFactor.svg', dpi = 300)
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
plt.savefig (path3+'SkewnessCoeff.svg', dpi = 300)
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
fluc = pd.read_csv(path3+'x=-40.txt', sep=' ', skipinitialspace=True)
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
fluc = pd.read_csv(path3+'x=-20.txt', sep=' ', skipinitialspace=True)
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
fluc = pd.read_csv(path3+'x=-10.txt', sep=' ', skipinitialspace=True)
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
    path3 + "PerturProfileZ.svg", bbox_inches="tight", pad_inches=0.1
)

#%% Streamwise distribution of a specific varibles
# load data
import pandas as pd
path1 = '/media/weibo/Data2/BFS_M1.7C_TS/probes2/'
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
    ProbeID.LoadProbeData(xzone[j], yval, zval, path1)
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
    path3 + "UPerturProfileX.svg", bbox_inches="tight", pad_inches=0.1
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
orig = pd.read_csv(path3+'z=0.txt', sep=' ', skipinitialspace=True)
meanflow = pd.read_csv(path3+'meanflow.txt', sep=' ', skipinitialspace=True)
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
    path3 + "PerturProfileX.svg", bbox_inches="tight", pad_inches=0.1
)

# %% 
path0 = '/media/weibo/Data2/BFS_M1.7C_TS/probes/'
path1 = '/media/weibo/Data2/BFS_M1.7C_TS/probes1/'
path2 = '/media/weibo/Data2/BFS_M1.7C_TS/probes2/'
dirs = sorted(os.listdir(path0))
for i in range(np.size(dirs)):
    with open (path0 + dirs[i]) as f:
        title1 = f.readline ().split ('\t')
        title2 = f.readline ().split ()
        title = '\n'.join([str(title1), str(title2)])
    f.close()
    file0 = np.genfromtxt(path0+dirs[i], skip_header=2, filling_values=0.0)
    file1 = np.loadtxt(path1+dirs[i])
    file2 = np.concatenate([file0, file1])
    np.savetxt(path2+dirs[i], file2, \
                fmt='%1.8e', delimiter = " ", header = str(title))
