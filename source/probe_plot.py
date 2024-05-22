#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 18:58:21 2017
    This code for post-processing data from instantaneous/time-average plane
    data, need data file.
@author: weibo
"""
# %% load necessary module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plt2pandas as p2p
import matplotlib
from scipy.interpolate import interp1d  # spline, splev, splrep
# from scipy import signal  # integrate
import variable_analysis as va
from line_field import LineField as lf

# -- data path settings
# path = "/media/weibo/IM2/FFS_M1.7SFD120/"
#path = 'E:/cases/wavy_1009/'
path = '/media/weibo/Weibo_data/2023cases/heating2/'
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"

# -- figures properties settings
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
    streamwise evolution of signals along an axis
"""
# %% Streamwise evolution of a specific variable
probe = lf()
fa = 1.0  # 1.7 * 1.7 * 1.4
var = 'u'
timeval = 0.18000284E+04  # 0.60000985E+03 #
xloc = np.arange(0.0, 360.0, 4.0)
yloc = 0.0
zloc = 0.0
var_val = np.zeros(np.size(xloc))
for i in range(np.size(xloc)):
    probe.load_probe(pathP, (xloc[i], yloc, zloc))
    meanval = getattr(probe, var + '_m')
    var_val[i] = (probe.extract_point(timeval)[var] - meanval) * fa
    # probe.extract_series([600, 1000])
    # temp = (getattr(probe, var) - np.mean(getattr(probe, var))) * fa
    var_val[i] = temp
# %%
fig, ax = plt.subplots(figsize=(6.4, 2.8))
matplotlib.rc('font', size=numsize)
ax.plot(xloc, var_val, 'k')
ax.set_xlim([0, 400.0])
# ax.set_ylim([-0.001, 0.001])
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$u^\prime/u_\infty$', fontsize=textsize)
ax.ticklabel_format(axis='y', style='sci',
                    useOffset=False, scilimits=(-2, 2))
ax.grid(b=True, which='both', linestyle=':')
plt.show()
plt.savefig(pathF + var + '_StreamwiseEvolve.svg',
            bbox_inches='tight', pad_inches=0.1)

# %%############################################################################
"""
    frequency weighted power spectral density
"""
# %% Frequency Weighted Power Spectral Density
Lsep = 1.0 # for 34.8 laminar # 12.6 for turbulent
def d_l(x):
    return x / Lsep
def l_d(x):
    return x * Lsep

sh = 1.0
def d_h(x):
    return x / sh
def h_d(x):
    return x * sh

Hsep = 1.0 # 11.6 for laminar # 4.2 for turbulent
def h_l(x):
    return x / Hsep

def l_h(x):
    return x * Hsep

dt = 0.125
freq_samp = 1/dt  # 50
var = 'p'

probe = lf()
x0 = 160
t1, t2 = [1000, 2000]
probe.load_probe(pathP, (x0, 0.015625, 0.0))
probe.extract_series((t1, t2))
freq, fpsd = va.fw_psd(getattr(probe, var), probe.time, freq_samp,
                       opt=1, seg=8, overlap=4)
# probe = pd.read_csv(pathI + 'probe_03.dat', sep=' ')
# freq, fpsd = va.fw_psd(probe[var].values, dt, 1/dt, opt=1)
# fig, ax1 = plt.subplots(figsize=(6.4, 2.8))
fig = plt.figure(figsize=(6.4, 3.0))
ax1 = fig.add_subplot(121)
matplotlib.rc('font', size=numsize)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax1.grid(b=True, which='both', linestyle=':')
ax1.set_xlabel(r"$f l /u_\infty$", fontsize=textsize)
ax1.semilogx(h_d(freq), fpsd, 'k', linewidth=1.0)
ax1.set_ylabel('Weighted PSD, unitless', fontsize=textsize - 4)
ax1b = ax1.secondary_xaxis('top', functions=(l_h, h_l))
ax1b.tick_params(axis='x', which='major', pad=1)
ax1b.xaxis.set_ticklabels([])
# ax1b.set_xlabel(r'$f L_r/u_\infty$', fontsize=textsize)
ax1.annotate("(a)", xy=(-0.15, 1.05), xycoords='axes fraction',
             fontsize=numsize)
ax1.set_title(r"$x={:.1f}$".format(x0/sh), fontsize=numsize, 
              position=(0.5, 0.78) ) # pad=0.1)
ax1.get_yaxis().get_offset_text().set_visible(False)
ax_max = max(ax1.get_yticks())
exponent_axis = np.floor(np.log10(ax_max)).astype(int)
ax1.annotate(r'$\times$10$^{%i}$'%(exponent_axis),
             xy=(.05, .92), xycoords='axes fraction')
plt.tick_params(labelsize=numsize)
ax1.yaxis.offsetText.set_fontsize(numsize)
# plt.savefig(pathF + var + '_ProbeFWPSD_03.svg', bbox_inches='tight')
# plt.show()

probe1 = lf()
x1 = 380
probe1.load_probe(pathP, (x1, 0.0, 0.0))
probe1.extract_series((t1, t2))
freq1, fpsd1 = va.fw_psd(getattr(probe1, var), probe1.time, freq_samp,
                         opt=1, seg=8, overlap=4)
ax2 = fig.add_subplot(122)
matplotlib.rc('font', size=numsize)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax2.grid(b=True, which='both', linestyle=':')
ax2.set_xlabel(r"$f l /u_\infty$", fontsize=textsize)
ax2.semilogx(h_d(freq1), fpsd1, 'k', linewidth=1.0)
ax2b = ax2.secondary_xaxis('top', functions=(l_h, h_l))
ax2b.tick_params(axis='x', which='major', pad=1)
ax2b.xaxis.set_ticklabels([])
# ax2b.set_xlabel(r'$f L_r / u_\infty$', fontsize=textsize)
ax2.annotate("(b)", xy=(-0.12, 1.05), xycoords='axes fraction',
             fontsize=numsize)
ax2.set_title(r"$x={:.1f}$".format(x1/sh), fontsize=numsize,
              position=(0.5, 0.78) ) # pad=0.1)
ax2.get_yaxis().get_offset_text().set_visible(False)
ax_max = max(ax2.get_yticks())
exponent_axis = np.floor(np.log10(ax_max)).astype(int)
ax2.annotate(r'$\times$10$^{%i}$'%(exponent_axis),
             xy=(.05, .92), xycoords='axes fraction')
plt.tick_params(labelsize=numsize)
ax2.yaxis.offsetText.set_fontsize(numsize)
plt.savefig(pathF + var + '_ProbeFWPSD_04.svg', bbox_inches='tight')
plt.show()

# %%############################################################################
"""
Multiple FWPSD Map
"""
# %% Plot multiple frequency-weighted PSD curve along streamwise
sh = 1.0
t1, t2 = [1750, 2000] # [600, 1100] # turbulent
var = 'p'
freq_samp = 1/0.125
fig, ax = plt.subplots(1, 7, figsize=(7.2, 2.4))
fig.subplots_adjust(hspace=0.5, wspace=0)
matplotlib.rc('font', size=numsize)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
xcoord = np.array([10, 50.25, 92.25, 120, 160, 260, 350])
ycoord = np.array([0.015625, -1.03125, -0.59375, 0.015625, 0.015625, 0.015625, 0.015625])
pow_arr = [-7, -6, -6, -7, -6, -4, -5] # [-7, -7, -6, -5, -4, -5, -5]
for i in range(np.size(xcoord)):
    probe.load_probe(pathP, (xcoord[i], ycoord[i], 0.0))
    probe.extract_series((t1, t2))
    varval = getattr(probe, var)
    freq, fpsd_x = va.fw_psd(varval, probe.time, freq_samp,
                             opt=1, seg=8, overlap=4)
    maxs = "{:.0e}".format(np.max(fpsd_x))
    parts = str(maxs).split('e', 2)
    if int(parts[0]) < 5:
        pownm = int(parts[-1]) - 1
    else:
        pownm = int(parts[-1])
    pownm = pow_arr[i] # -6
    expon = 10**pownm
    if i % 2 == 0:
        ax[i].plot(fpsd_x/expon, h_d(freq), "k-", linewidth=1.0)
    else:
        ax[i].plot(fpsd_x/expon, h_d(freq), "k-", linewidth=1.0)
    ax[i].set_ylim([0.003, 15])
    ax[i].set_yscale('log')
    # ax[i].set_xscale('log')
    # ax[i].set_xticks([0, 0.25*10**(-4), 0.5*10**(-4), 0.75*10**(-4), 10**(-4)])
    # ax[i].set_xticks([0, 0.5*10**(-4), 10**(-4)])
    # ax[i].set_xticks(np.linspace(0, 1, 11)*10**(-4), minor=True)
    # ax[i].set_xlim([0, 3*10**(-4)])
    ax[i].set_xlim([0, 10])
    ax[i].set_xticks([0, 5, 10])
    xloc = np.round(xcoord[i]/sh, 1)
    ax[i].set_title(r'${}$'.format(xloc), fontsize=numsize-1, loc='left')
    ax[i].annotate(r'$\times 10^{{{}}}$'.format(int(pownm)), 
                   xy=(0.1, 0.1), fontsize=numsize-2, xycoords='axes fraction')
    # ax[i].xaxis.major.formatter.set_powerlimits((-2, 2))
    # ax[i].xaxis.offsetText.set_fontsize(numsize)
    if i != 0:
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].patch.set_alpha(0.0)
        ax[i].xaxis.set_ticks_position('none')
        ax[i].yaxis.set_ticks_position('none')
        xticks = ax[i].xaxis.get_major_ticks()
        ax[i].spines['left'].set_color('gray')
        ax[i].spines['left'].set_linestyle('--')
        ax[i].spines['left'].set_linewidth(0.5)
        # ax[i].set_frame_on(False)
        # ax[i].yaxis.set_visible(False)
        # xticks[0].label.set_visible(False) # xticks[0].label1.set_visible(False)
    if i != np.size(xcoord) - 1:
        ax[i].spines['right'].set_visible(False)
    # ax[i].set_xlim(left=0)
    # ax[i].set_xticklabels('')
    # ax[i].tick_params(axis='both', which='major', labelsize=numsize)
    ax[i].grid(visible=True, which="both", axis='y', linestyle=":")
ax[0].spines['right'].set_visible(False)
ax[0].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
# ax[0].set_xticklabels([r'$10^{-8}$','',r'$10^{-6}$','', r'$10^{-4}$'])
ax[0].annotate(r'$x/l=$', xy=(-0.48, 1.04), xycoords='axes fraction')
ax[3].set_xlabel(r'$f \mathcal{P}(f)$', fontsize=textsize, labelpad=10)
ax[0].set_ylabel(r"$f l /u_\infty$", fontsize=textsize)
ax2 = ax[-1].secondary_yaxis('right', functions=(l_h, h_l)) 
ax2.set_ylabel(r"$f L_r /u_\infty$", fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "ProbeFWPSD_" + var + ".svg", bbox_inches="tight", pad_inches=0.1
)
# %%############################################################################
"""
    intermittency factor
"""
# %% Compute intermittency factor
probe.load_probe(pathP, (0.0, 0.015625, 0.0))
xzone = np.arange(0.0, 360.0, 5.0)
gamma = np.zeros(np.size(xzone))
alpha = np.zeros(np.size(xzone))
sigma = np.std(probe.p)
p0 = probe.p
t1 = 1000
t2 = 2000
for j in range(np.size(xzone)):
    if xzone[j] <= 0.0:
        probe.load_probe(pathP, (xzone[j], 0.015625, 0.0))
    else:
        probe.load_probe(pathP, (xzone[j], 0.015625, 0.0))
    probe.extract_series((t1, t2))
    gamma[j] = va.intermittency(sigma, p0, probe.p, probe.time)
    alpha[j] = va.alpha3(probe.p)
# -- plot
fig3, ax3 = plt.subplots(figsize=(6.4, 3.0))
ax3.plot(xzone, gamma, 'ko')
ax3.set_xlabel(r'$x/l$', fontdict=textsize)
ax3.set_ylabel(r'$\gamma$', fontdict=textsize)
# ax3.set_ylim([0.0, 1.0])
ax3.grid(b=True, which='both', linestyle=':')
ax3.axvline(x=0.0, color='k', linestyle='--', linewidth=1.0)
ax3.axvline(x=12.7, color='k', linestyle='--', linewidth=1.0)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.tick_params(labelsize=14)
plt.savefig(pathF+'IntermittencyFactor.svg', bbox_inches='tight')
plt.show()

# %%############################################################################
"""
    skewness coefficient
"""
# %% Skewness coefficient
fig4, ax4 = plt.subplots(figsize=(6.4, 3.0))
ax4.plot(gamma, alpha, 'ko')
ax4.set_xlabel(r'$\gamma$', fontdict=textsize)
ax4.set_ylabel(r'$\alpha_3$', fontdict=textsize)
# ax3.set_ylim([0.0, 1.0])
ax4.grid(b=True, which='both', linestyle=':')
# ax4.axvline(x=0.0, color='k', linestyle='--', linewidth=1.0)
# ax4.axvline(x=12.7, color='k', linestyle='--', linewidth=1.0)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
fig4.set_size_inches(6, 5, forward=True)
plt.savefig(pathF+'SkewnessCoeff.svg', bbox_inches='tight')
plt.show()

# %%############################################################################
"""
    critical aerodynamic parameters (reattachment, shock, bubble)
"""
# %% Calculate singnal of Xr, Xsl, Xsf, Xb
InFolder = path + 'Slice/TP_2D_S_009/'
timezone = np.arange(1100, 1600.0 + 0.25, 0.25)
skip = 1
# -- separation location
reatt = va.separate_loc(InFolder, pathI, timezone, skip=skip, opt=2) #, loc=-0.5, opt=1)
# --- separation location by velocity on the wall
ywall = 0.0023002559
va.shock_foot(InFolder, pathI, timezone, ywall, 0.00099, outfile='NewSep.dat', skip=skip)
# -- shock location outside the boundary layer
va.shock_loc(InFolder, pathI, timezone, skip=skip, var='|grad(rho)|', lev=0.42) #, opt=2, val=[0.91, 0.92])
# -- shock foot within the boudnary layer
va.shock_foot(InFolder, pathI, timezone, 1.5, 0.875, skip=skip)  # 0.82 for laminar
# -- area of the separation bubble
va.bubble_area(InFolder, pathI, timezone, step=0, skip=skip)
#%% -- a specific location, for example in the shear layer
InFolder = path + 'Slice/TP_2D_S_009/'
xy = [-37.5   ,  1.25]  # [-10,2.59375] # [-4.375, 3.53125] #[-4.6875, 2.9375]  #  
va.extract_point(InFolder, pathI, timezone, xy, skip=skip)
dt = 0.25
fs = 4.0
lh = 3.0
# x1x2 = [1100/lh, 1600/lh]
x1x2 = [600/lh, 1100/lh]
# %%############################################################################
"""
    Temporal evolution & Power spectral density
"""
# %% probe within shear layer (Kelvin Helholmz fluctuation)
# -- temporal evolution
fa = 1.0 #1.7*1.7*1.4
var = 'u'
# probe = np.loadtxt(pathI + "ProbeKH.dat", skiprows=1)
# func = interp1d(probe[:, 1], probe[:, 7])
# Xk = func(timezone)  # probe[:, 8]
probe = pd.read_csv(pathI + 'Xk5.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
Xk = probe[var].values
fig, ax = plt.subplots(figsize=(6.4, 2.2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
ax.plot(probe['t']/lh, Xk*fa, "k-", linewidth=0.8)
ax.set_xlim(x1x2) # (950, 1350) # 
ax.set_xlabel(r"$t u_\infty/h$", fontsize=textsize)
ax.set_ylabel(r"$u/u_\infty$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xk*fa)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xk5.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# -- FWPSD
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f h/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xk*fa, dt, 1/dt, opt=1, seg=8, overlap=4)
ax.semilogx(Fre*lh, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
# ax2nd = ax.secondary_xaxis('top', functions=(d2l, l2d))
# ax2nd.set_xlabel(r'$f x_r / u_\infty$')
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "Xk5FWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% singnal of bubble
# -- temporal evolution
bubble = np.loadtxt(pathI + "BubbleArea.dat", skiprows=1)
Xb = bubble[:, 1] / lh /lh
fig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.plot(bubble[:, 0]/lh, Xb, "k-", linewidth=0.8)
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/h$", fontsize=textsize)
ax.set_ylabel(r"$A/h^2$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xb)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xb.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# -- FWPSD
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.set_xlabel(r"$f h/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xb, dt, 1/dt, opt=1, seg=10, overlap=6)
ax.semilogx(Fre*lh, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XbFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% probablity
df = pd.DataFrame(data=Xb, columns=['A'])
posi = np.zeros(np.size(Xb))
nega = np.zeros(np.size(Xb))
Xb_grad = np.gradient(Xb, 0.083333333)  # va.sec_ord_fdd(bubble[:, 0]/lh, Xb)
posi_mean = np.mean(Xb_grad[Xb_grad > 0])
nega_mean = np.mean(Xb_grad[Xb_grad < 0])
for ii in range(np.size(Xb)):
    if Xb_grad[ii] >= 0:
        if ii == 0:
            posi[ii] = posi[ii] + 1
            nega[ii] = nega[ii] + 0
        else:
            posi[ii] = posi[ii-1] + 1
            nega[ii] = nega[ii-1] + 0
    elif Xb_grad[ii]  < 0:
        if ii == 0:
            posi[ii] = posi[ii] + 0
            nega[ii] = nega[ii] + 1
        else:
            posi[ii] = posi[ii-1] + 0
            nega[ii] = nega[ii-1] + 1
    
ig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.plot(bubble[:, 0]/lh, posi/np.size(Xb), "k-", linewidth=0.8)
ax.plot(bubble[:, 0]/lh, nega/np.size(Xb), "b-", linewidth=0.8)
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/h$", fontsize=textsize)
ax.set_ylabel(r"$P(dA/dt)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
# xmean = np.mean(Xb)
print("The mean value: ", xmean)
# ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xb_prob.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% signal of shock information
# -- load data
shock1 = np.loadtxt(pathI + "ShockA.dat", skiprows=1)
shock2 = np.loadtxt(pathI + "ShockB.dat", skiprows=1)
foot = np.loadtxt(pathI + "ShockFoot.dat", skiprows=1)
delta_x = shock2[0, 2] - shock1[0, 2]
angle = np.arctan(delta_x / (shock2[:, 1] - shock1[:, 1])) / np.pi * 180
shockloc = shock2[:, 1] - 8.0 / np.tan(angle/180*np.pi)  # outside the BL
Xl = shockloc
Xf = foot[:, 1]
# %% temporal evolution
Xa = angle # shock1[:, 1] # shock2[:, 1] # Xf # angle # Xf  # Xl  #      
if np.array_equal(Xa, Xl):
    output = 'Shockloc'
    ylabel = r"$x_l/h$"
    Xa1 = Xa/lh
elif np.array_equal(Xa, Xf):
    output = 'Shockfoot'
    ylabel = r"$x_f/h$"
    Xa1 = Xa/lh
elif np.array_equal(Xa, angle):
    output = 'Shockangle'
    ylabel = r"$\eta(^{\circ})$"
    Xa1 = Xa
fig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.plot(shock1[:, 0]/lh, Xa1, "k-", linewidth=0.8)
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/h$", fontsize=textsize)
ax.set_ylabel(ylabel, fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xa1)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + output + ".svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# print("Corelation: ", va.correlate(Xs, Xk))

# -- FWPSD
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
# ax.yaxis.major.formatter.set_powerlimits((-2, 3))
ax.set_xlabel(r"$f h/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xb, dt, 1/dt, opt=1, seg=8, overlap=4)
ax.semilogx(Fre*lh, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + output + "FWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% singnal of reattachment point
# -- temporal evolution
reatt = np.loadtxt(pathI+"Reattach.dat", skiprows=1)  # separate
Xr = reatt[:, 1] / lh
fig, ax = plt.subplots(figsize=(6.4, 2.2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
ax.plot(reatt[:, 0]/lh, Xr, "k-", linewidth=0.8)
ax.set_xlim(x1x2) # (1150, 1200)  # 
ax.set_xlabel(r"$t u_\infty/h$", fontsize=textsize)
ax.set_ylabel(r"$y_r/h$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xr)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xr.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# -- FWPSD
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.set_xlabel(r"$f h/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xr, dt, 1/dt, opt=1, seg=10, overlap=8)
ax.semilogx(Fre*lh, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XrFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% probablity
df = pd.DataFrame(data=Xr, columns=['A'])
posi = np.zeros(np.size(Xr))
nega = np.zeros(np.size(Xr))
Xr_grad = va.sec_ord_fdd(reatt[:, 0]/lh, Xr)
posi_mean = np.mean(Xr_grad[Xr_grad > 0])
nega_mean = np.mean(Xr_grad[Xr_grad < 0])
for ii in range(np.size(Xr)):
    if Xr_grad[ii] >= 0:
        if ii == 0:
            posi[ii] = posi[ii] + 1
            nega[ii] = nega[ii] + 0
        else:
            posi[ii] = posi[ii-1] + 1
            nega[ii] = nega[ii-1] + 0
    elif Xr_grad[ii] < 0:
        if ii == 0:
            posi[ii] = posi[ii] + 0
            nega[ii] = nega[ii] + 1
        else:
            posi[ii] = posi[ii-1] + 0
            nega[ii] = nega[ii-1] + 1
    
ig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.plot(reatt[:, 0]/lh, posi/np.size(Xb), "k-", linewidth=0.8)
ax.plot(reatt[:, 0]/lh, nega/np.size(Xb), "b-", linewidth=0.8)
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/h$", fontsize=textsize)
ax.set_ylabel(r"$P(dy_r/dt)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
# xmean = np.mean(Xb)
print("The mean value: ", xmean)
# ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "yr_prob.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% singnal of separation point
# -- temporal evolution
sepa = np.loadtxt(pathI+"Separate.dat", skiprows=1)  # separate
Xs = sepa[:, 1]/lh
fig, ax = plt.subplots(figsize=(6.4, 2.2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
ax.plot(sepa[:, 0]/lh, Xs, "k-", linewidth=0.8)
ax.set_xlim(x1x2)
# ax.set_xlim(x1x2) # (1150, 1200)  # 
ax.set_xlabel(r"$t u_\infty/h$", fontsize=textsize)
ax.set_ylabel(r"$x_s/h$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xs)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xs.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# -- FWPSD
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
ax.set_xlabel(r"$f h/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xs, dt, 1/dt, opt=1, seg=8, overlap=4)
ax.semilogx(Fre*lh, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XsFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% probablity
df = pd.DataFrame(data=Xs, columns=['A'])
posi = np.zeros(np.size(Xs))
nega = np.zeros(np.size(Xs))
Xs_grad = va.sec_ord_fdd(sepa[:, 0]/lh, Xs)
posi_mean = np.mean(Xs_grad[Xs_grad > 0])
nega_mean = np.mean(Xs_grad[Xs_grad < 0])
for ii in range(np.size(Xs)):
    if (Xs_grad[ii] >= 0):
        if ii == 0:
            posi[ii] = posi[ii] + 1
            nega[ii] = nega[ii] + 0
        else:
            posi[ii] = posi[ii-1] + 1
            nega[ii] = nega[ii-1] + 0
    elif (Xs_grad[ii] < 0):
        if ii == 0:
            posi[ii] = posi[ii] + 0
            nega[ii] = nega[ii] + 1
        else:
            posi[ii] = posi[ii-1] + 0
            nega[ii] = nega[ii-1] + 1
    
ig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.plot(sepa[:, 0]/lh, posi/np.size(Xs), "k-", linewidth=0.8)
ax.plot(sepa[:, 0]/lh, nega/np.size(Xs), "b-", linewidth=0.8)
# ax.plot(sepa[:, 0]/lh, nega/np.size(Xs), "b-", linewidth=0.8)
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/h$", fontsize=textsize)
ax.set_ylabel(r"$P(d A/d t)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
# xmean = np.mean(Xb)
print("The mean value: ", xmean)
# ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xs_prob.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% gradient of Xr
# x1x2 = [-40, 10]
timezone = sepa[:, 0]/lh
fig, ax = plt.subplots(figsize=(6.4, 3.0))
dxr = va.sec_ord_fdd(timezone, Xr)
ax.plot(timezone, dxr, "k-")
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\mathrm{d} x_r/\mathrm{d} t$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
dx_pos = dxr[np.where(dxr > 0.0)]
mean_pos = np.mean(dx_pos)
dx_neg = dxr[np.where(dxr < 0.0)]
mean_neg = np.mean(dx_neg)
ax.axhline(y=mean_pos, color="k", linestyle="--", linewidth=1.0)
ax.axhline(y=mean_neg, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "GradientXr.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% histogram of probability
num = 9
timezone = reatt[:,0]
t_no = 20
aa = int(np.size(Xb)/t_no)
new_t = np.zeros(aa)
new_xr = np.zeros(aa)
for i in range(aa):
    new_t[i] = np.mean(timezone[i*t_no:(i+1)*t_no])
    new_xr[i] = np.mean(Xb[i*t_no:(i+1)*t_no])

dxr = va.sec_ord_fdd(new_t, new_xr)
posi_mean = np.mean(dxr[dxr > 0])
nega_mean = np.mean(dxr[dxr < 0])
tnum = np.size(dxr)
xr_max = np.round(np.max(np.abs(dxr)), 4)
dxr_arr = np.linspace(-xr_max, xr_max, num, endpoint=True)
dxdt = np.linspace(-2.0, 2.0, num)
nsize = np.zeros(num-1)
proba = np.zeros(num-1)
for i in range(num-1):
    ind = np.all([dxr_arr[i]<=dxr, dxr<dxr_arr[i+1]], axis=0)
    nsize[i] = np.size(dxr[ind])
    proba[i] = nsize[i]/tnum

fig, ax = plt.subplots(figsize=(10, 3.5))
# ax.hist(dxr, bins=num, range=(-2.0, 2.0), edgecolor='k', linestyle='-',
#         facecolor='#D3D3D3', alpha=0.98, density=True)
hist, edges = np.histogram(dxr, bins=num, range=(-2.0, 2.0), density=True)
# binwid = edges[1] - edges[0]
# plt.bar(edges[:-1], hist*binwid, width=binwid, edgecolor='k', linestyle='-',
#         facecolor='#D3D3D3', alpha=0.98)
#plt.bar(edges[:-1], hist, width=binwid, edgecolor='k', linestyle='-',
#         facecolor='#D3D3D3', alpha=0.98)
binwid = dxr_arr[1] - dxr_arr[0]
plt.bar(dxr_arr[:4]+0.5*binwid, proba[:4], width=binwid, edgecolor='k', linestyle='-',
        facecolor='#40B14B', alpha=0.98)
plt.bar(dxr_arr[4:-1]+0.5*binwid, proba[4:], width=binwid, edgecolor='k', linestyle='-',
        facecolor='#F87E7D', alpha=0.98)
# ax.set_xlim([-2.0, 2.0])
ax.set_ylabel(r"$P$", fontsize=textsize)
ax.set_xlabel(r"$\mathrm{d} A/\mathrm{d} t$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "ProbGradientXb_0.2hz.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %%############################################################################
"""
    cross-correlation (coherence & phase)
"""
# %% laod data
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/Data/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/temp/"
probe1 = np.loadtxt(OutFolder + "ShockFootE.dat", skiprows=1)
probe2 = np.loadtxt(OutFolder + "ShockFootC.dat", skiprows=1)
probe11 = np.loadtxt(OutFolder + "ShockFootE.dat", skiprows=1)
probe21 = np.loadtxt(OutFolder + "ShockFootD.dat", skiprows=1)
probe12 = np.loadtxt(OutFolder + "ShockFootE.dat", skiprows=1)
probe22 = np.loadtxt(OutFolder + "ShockFootF.dat", skiprows=1)
timezone = np.arange(600, 1000 + 0.5, 0.5)
# %% coherence
dt = 0.25
fs = 4
Xs0 = Xs  # probe1[:, 1]
Xr0 = Xl # probe2[:, 1]
#Xs1 = probe11[:, 1]
#Xr1 = probe21[:, 1]
#Xs2 = probe12[:, 1]
#Xr2 = probe22[:, 1]

fig = plt.figure(figsize=(6.4, 3.0))
matplotlib.rc("font", size=textsize)
ax = fig.add_subplot(121)
Fre, coher = va.coherence(Xs0, Xr0, dt, fs, opt=1, seg=8, overlap=4)
#Fre1, coher1 = va.coherence(Xr1, Xs1, dt, fs)
#Fre2, coher2 = va.coherence(Xr2, Xs2, dt, fs)
ax.semilogx(Fre * lh, coher, "k-", linewidth=1.0)
# ax.semilogx(Fre1, coher1, "k:", linewidth=1.0)
# ax.semilogx(Fre2, coher2, "k--", linewidth=1.0)
# ax.set_ylim([0.0, 1.0])
# ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.set_xlabel(r"$fh/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$C$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
ax.annotate("(a)", xy=(-0.17, 1.0), xycoords="axes fraction", fontsize=numsize)

ax = fig.add_subplot(122)
Fre, cpsd = va.cro_psd(Xs0, Xr0, dt, fs, opt=1, seg=8, overlap=2)
#Fre1, cpsd1 = va.cro_psd(Xr1, Xs1, dt, fs)
#Fre2, cpsd2 = va.cro_psd(Xr2, Xs2, dt, fs)
ax.semilogx(Fre * lh, np.arctan2(cpsd.imag, cpsd.real) / np.pi, 
            "k-", linewidth=1.0)
# ax.semilogx(Fre, np.arctan(cpsd1.imag, cpsd1.real), "k:", linewidth=1.0)
# ax.semilogx(Fre, np.arctan(cpsd2.imag, cpsd2.real), "k--", linewidth=1.0)
# ax.set_ylim([-1.0, 1.0])
ax.set_xlabel(r"$fh/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$\theta/\pi$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
ax.annotate("(b)", xy=(-0.20, 1.0), xycoords="axes fraction", fontsize=numsize)
#lab = [
#    r"$\Delta y/\delta_0 = 1.0$",
#    r"$\Delta y/\delta_0 = 1.5$",
#    r"$\Delta y/\delta_0 =4.0$",
#]
# ax.legend(lab, ncol=1, loc="upper right", fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "Statistic"+"XsXl.svg",
            bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% plot cross-correlation coeffiency of two variables with time delay
x1 = Xb
x2 = Xk
delay = np.arange(-100.0, 100+0.5, 0.5)
cor = np.zeros(np.size(delay))
for i, dt in enumerate(delay):
    cor[i] = va.delay_correlate(x1, x2, 0.5, dt)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(delay, cor, "k-")
ax.set_xlim([delay[0], delay[-1]])
ax.set_xlabel(r"$\Delta t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$R_{ij}$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "Cor_XbXk.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
