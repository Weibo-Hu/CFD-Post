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
from scipy.interpolate import spline
from scipy import integrate
import sys
from DataPost import DataPost
import FlowVar as fv

plt.close ("all")
plt.rc('text', usetex=True)
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
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
path3 = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/3AF/"
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

textsize = 24
numsize = 20
t0 = 600
t1 = 600 #600 #960
t2 = 1000 #1000.0 #

#%% Read data for Streamwise variations of frequency of a specific variable
Probe0 = DataPost()
xloc = [-40.0, 5.359, 10.0, 60.0]
yloc = [0.0, -0.889, -3.0, -3.0]
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
fa = 1.7*1.7*1.4
xlabel = r'$x/\delta_0={}, \ y/\delta_0={}$'
ytitle = r'$p/p_\infty$'
var = 'p'
fig = plt.figure(figsize=(10,5.5))
matplotlib.rc('font', size=textsize)
ax = fig.add_subplot(211)
ax.set_title(xlabel.format(xloc[1], yloc[1]), fontsize=numsize)
ax.set_xlim([t1, t2])
ax.set_xticklabels('')
ax.set_ylabel(ytitle, fontsize=textsize)
ax.yaxis.set_major_formatter(FormatStrFormatter(r'${%.2f}$'))
ax.grid(b=True, which='both', linestyle=':')
ax.plot(Probe10.time, getattr(Probe10, var)*fa, 'k', linewidth=1.0)
ax.annotate("(a)", xy=(-0.10, 1.1), xycoords='axes fraction', fontsize=numsize)
plt.tick_params(labelsize=numsize)

ax = fig.add_subplot(212)
ax.set_title(xlabel.format(xloc[2], yloc[2]), fontsize=numsize)
ax.set_xlim([t1, t2])
ax.yaxis.set_major_formatter(FormatStrFormatter(r'${%.2f}$'))
ax.set_ylabel(ytitle, fontsize=textsize)
ax.grid(b=True, which='both', linestyle = ':')
ax.plot(Probe20.time, getattr(Probe20, var)*fa, 'k', linewidth=1.0)
ax.annotate("(b)", xy=(-0.10, 1.1), xycoords='axes fraction', fontsize=numsize)
ax.set_xlabel(r'$t u_\infty/\delta_0$', fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)

plt.savefig(path3+'StreamwiseTimeEvolution.svg', bbox_inches='tight')
plt.show()


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
plt.savefig(path3+'StreamwiseFWPSD.svg', bbox_inches='tight')
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
