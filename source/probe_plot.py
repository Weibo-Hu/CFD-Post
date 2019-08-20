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
yloc = 0.0
zloc = 0.0
fig, ax = plt.subplots(5, 1, figsize=(6.4, 5.6))
fig.subplots_adjust(hspace=0.6, wspace=0.15)
matplotlib.rc('font', size=numsize)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
for i in range(np.size(xloc)):
    probe.load_probe(pathP, (xloc[i], yloc, zloc))
    probe.extract_series(timezone)
    temp = (getattr(probe, var) - np.mean(getattr(probe, var))) * fa
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
    ax[i].set_title(r'$x/\delta_0={}$'.format(xloc[i]), fontsize=numsize - 1)
ax[-1].set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
plt.show()
plt.savefig(pathF + var + '_TimeEvolveX.svg' + str(zloc) + '.svg',
            bbox_inches='tight', pad_inches=0.1)

# %% Time evolution of a specific variable at several spanwise locations
probe = lf()
fa = 1.7 * 1.7 * 1.4
var = 'p'
timezone = [400, 600]
xloc = -10.0
yloc = 0.0
zloc = [-4.0, 0.0, 4.0]
fig, ax = plt.subplots(3, 1, figsize=(6.4, 5.6))
fig.subplots_adjust(hspace=0.6, wspace=0.15)
matplotlib.rc('font', size=numsize)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
for i in range(np.size(zloc)):
    probe.load_probe(pathP, (xloc, yloc, zloc[i]))
    probe.extract_series(timezone)
    temp = (getattr(probe, var) - np.mean(getattr(probe, var))) * fa
    ax[i].plot(probe.time, temp, 'k')
    ax[i].ticklabel_format(axis='y', style='sci',
                           useOffset=False, scilimits=(-2, 2))
    ax[i].set_xlim([400, 600])
    # ax[i].set_ylim([-0.001, 0.001])
    if i != np.size(zloc) - 1:
        ax[i].set_xticklabels('')
    ax[i].set_ylabel(r"$p^\prime/p_\infty$",
                     fontsize=textsize)
    ax[i].grid(b=True, which='both', linestyle=':')
    ax[i].set_title(r'$z/\delta_0={}$'.format(zloc[i]), fontsize=numsize - 1)
ax[-1].set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
plt.show()
plt.savefig(pathF + var + '_TimeEvolveZ' + str(xloc) + '.svg',
            bbox_inches='tight', pad_inches=0.1)

# %% Streamwise evolution of a specific variable
probe = lf()
fa = 1.0  # 1.7 * 1.7 * 1.4
var = 'u'
timeval = 0.60001185E+03  # 0.60000985E+03 #
xloc = np.arange(-40.0, 0.0, 1.0)
yloc = 0.0
zloc = 0.0
var_val = np.zeros(np.size(xloc))
for i in range(np.size(xloc)):
    probe.load_probe(pathP, (xloc[i], yloc, zloc))
    meanval = getattr(probe, var + '_m')
    var_val[i] = (probe.extract_point(timeval)[var] - meanval) * fa

fig, ax = plt.subplots(figsize=(6.4, 2.8))
matplotlib.rc('font', size=numsize)
ax.plot(xloc, var_val, 'k')
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

# %% Spanwise evolution of a specific variable
probe = lf()
fa = 1.7 * 1.7 * 1.4
var = 'p'
timeval = 0.60001185E+03  # 0.60000985E+03 #
xloc = -40.0
yloc = 0.0
zloc = np.arange(-8.0, 8.0 + 1.0, 1.0)
var_val = np.zeros(np.size(zloc))
for i in range(np.size(zloc)):
    probe.load_probe(pathP, (xloc, yloc, zloc[i]))
    meanval = getattr(probe, var + '_m')
    var_val[i] = (probe.extract_point(timeval)[var] - meanval) * fa

fig, ax = plt.subplots(figsize=(6.4, 2.8))
matplotlib.rc('font', size=numsize)
ax.plot(zloc, var_val, 'k')
ax.set_xlim([-8.0, 8.0])
# ax.set_ylim([-0.001, 0.001])
ax.set_xlabel(r'$z/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$u^\prime/u_\infty$', fontsize=textsize)
ax.ticklabel_format(axis='y', style='sci',
                    useOffset=False, scilimits=(-2, 2))
ax.grid(b=True, which='both', linestyle=':')
plt.show()
plt.savefig(pathF + var + '_SpanwiseEvolve.svg',
            bbox_inches='tight', pad_inches=0.1)

# %% Frequency Weighted Power Spectral Density
probe = lf()
probe.load_probe(pathP, (-10.0, 0.0, 0.0))
dt = 0.02
freq_samp = 50
var = 'u'
freq, fpsd = fv.fw_psd(getattr(probe, var), probe.time, freq_samp)
# fig, ax1 = plt.subplots(figsize=(6.4, 2.8))
fig = plt.figure(figsize=(6.4, 2.8))
ax1 = fig.add_subplot(121)
matplotlib.rc('font', size=textsize)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax1.grid(b=True, which='both', linestyle=':')
ax1.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
ax1.semilogx(freq, fpsd, 'k', linewidth=1.0)
ax1.set_ylabel('Weighted PSD, unitless', fontsize=textsize - 4)
ax1.annotate("(a)", xy=(-0.12, 1.04), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
ax1.yaxis.offsetText.set_fontsize(numsize)
plt.savefig(pathF + var + '_ProbeFWPSD_a.svg', bbox_inches='tight')
plt.show()

probe1 = lf()
probe1.load_probe(pathP, (-20.0, 0.0, 0.0))
freq1, fpsd1 = fv.fw_psd(getattr(probe1, var), probe1.time, freq_samp)
ax2 = fig.add_subplot(122)
matplotlib.rc('font', size=textsize)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax2.grid(b=True, which='both', linestyle=':')
ax2.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
ax2.semilogx(freq1, fpsd1, 'k', linewidth=1.0)
ax2.annotate("(b)", xy=(-0.12, 1.04), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
ax2.yaxis.offsetText.set_fontsize(numsize)
plt.savefig(pathF + var + '_ProbeFWPSD.svg', bbox_inches='tight')
plt.show()

# %% Compute intermittency factor
probe.load_probe(pathP, (-10.0, 0.0, 0.0))
xzone = np.linspace(-40.0, 40.0, 41)
gamma = np.zeros(np.size(xzone))
alpha = np.zeros(np.size(xzone))
sigma = np.std(probe.p)
p0 = probe.p
t1 = 400
t2 = 600
for j in range(np.size(xzone)):
    if xzone[j] <= 0.0:
        probe.load_probe(pathP, (xzone[j], 0.0, 0.0))
    else:
        probe.load_probe(pathP, (xzone[j], -3.0, 0.0))
    probe.extract_series((t1, t2))
    gamma[j] = fv.Intermittency(sigma, p0, probe.p, probe.time)
    alpha[j] = fv.Alpha3(probe.p)

fig3, ax3 = plt.subplots(figsize=(4, 3.5))
ax3.plot(xzone, gamma, 'ko')
ax3.set_xlabel (r'$x/\delta_0$', fontdict=textsize)
ax3.set_ylabel (r'$\gamma$', fontdict=textsize)
#ax3.set_ylim([0.0, 1.0])
ax3.grid(b=True, which = 'both', linestyle = ':')
ax3.axvline(x=0.0, color='k', linestyle='--', linewidth=1.0)
ax3.axvline(x=12.7, color='k', linestyle='--', linewidth=1.0)
plt.tight_layout(pad = 0.5, w_pad=0.5, h_pad =0.3)
plt.tick_params(labelsize=14)
plt.savefig (pathF+'IntermittencyFactor.svg', dpi = 300)
plt.show()

# %% Skewness coefficient
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

# %% Spanwise distribution of a specific varibles
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

