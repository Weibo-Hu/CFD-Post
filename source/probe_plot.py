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
path = "/media/weibo/IM2/FFS_M1.7TB/"
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
    temporal evolution of signals along an axis
"""
# %% Time evolution of a specific variable at several streamwise locations
probe = lf()
fa = 1.7 * 1.7 * 1.4
var = 'p'
timezone = np.arange(850.25, 850.00 + 0.25, 0.25)
xloc = [-70.0, -50.0, -30.0, -20.0, 0.0]
yloc = 0.0
zloc = 0.0
fig, ax = plt.subplots(np.size(xloc), 1, figsize=(6.4, 5.6))
fig.subplots_adjust(hspace=0.6, wspace=0.15)
matplotlib.rc('font', size=numsize)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
for i in range(np.size(xloc)):
    probe.load_probe(pathP, (xloc[i], yloc, zloc))
    probe.extract_series([timezone[0], timezone[-1]])
    temp = (getattr(probe, var) - np.mean(getattr(probe, var))) * fa
    ax[i].plot(probe.time, temp, 'k')
    ax[i].ticklabel_format(axis='y', style='sci',
                           useOffset=False, scilimits=(-2, 2))
    ax[i].set_xlim([timezone[0], timezone[-1]])
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
            bbox_inches='tight')

# %%############################################################################
"""
    streamwise evolution of signals along an axis
"""
# %% Streamwise evolution of a specific variable
probe = lf()
fa = 1.0  # 1.7 * 1.7 * 1.4
var = 'u'
timeval = 0.60001185E+03  # 0.60000985E+03 #
xloc = np.arange(-70.0, 0.0, 1.0)
yloc = 0.0
zloc = 0.0
var_val = np.zeros(np.size(xloc))
for i in range(np.size(xloc)):
    probe.load_probe(pathP, (xloc[i], yloc, zloc))
    meanval = getattr(probe, var + '_m')
    var_val[i] = (probe.extract_point(timeval)[var] - meanval) * fa
# %%
fig, ax = plt.subplots(figsize=(6.4, 2.8))
matplotlib.rc('font', size=numsize)
ax.plot(xloc, var_val, 'k')
ax.set_xlim([-70.0, 0.0])
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
Lsep = 8.9
def d2l(x):
    return x * Lsep

def l2d(x):
    return x / Lsep

dt = 0.25
freq_samp = 1/dt  # 50
var = 'u'

# probe = lf()
# probe.load_probe(pathP, (-60.0, 0.0, 0.0))
# freq, fpsd = va.fw_psd(getattr(probe, var), probe.time, freq_samp)
probe = pd.read_csv(pathI + 'probe_03.dat', sep=' ')
freq, fpsd = va.fw_psd(probe[var].values, dt, 1/dt, opt=1)
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
plt.savefig(pathF + var + '_ProbeFWPSD_03.svg', bbox_inches='tight')
plt.show()

# probe1 = lf()
# probe1.load_probe(pathP, (-50.0, 0.0, 0.0))
# freq1, fpsd1 = va.fw_psd(getattr(probe1, var), probe1.time, freq_samp)
ax2 = fig.add_subplot(122)
matplotlib.rc('font', size=textsize)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax2.grid(b=True, which='both', linestyle=':')
ax2.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
# ax2.semilogx(freq1, fpsd1, 'k', linewidth=1.0)
# ax2nd = ax2.secondary_xaxis('top', functions=(d2l, l2d))
# ax2nd.set_xlabel(r'$f x_r / u_\infty$')
# ax2.annotate("(b)", xy=(-0.12, 1.04), xycoords='axes fraction',
#              fontsize=numsize)
plt.tick_params(labelsize=numsize)
ax2.yaxis.offsetText.set_fontsize(numsize)
plt.savefig(pathF + var + '_ProbeFWPSD_03.svg', bbox_inches='tight')
plt.show()

# %%############################################################################
"""
    intermittency factor
"""
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
    gamma[j] = va.intermittency(sigma, p0, probe.p, probe.time)
    alpha[j] = va.alpha3(probe.p)
# -- plot
fig3, ax3 = plt.subplots(figsize=(6.4, 3.0))
ax3.plot(xzone, gamma, 'ko')
ax3.set_xlabel(r'$x/\delta_0$', fontdict=textsize)
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
InFolder = path + 'Slice/S_009/'
timezone = np.arange(600.0, 1100.00 + 0.25, 0.25)
skip = 1
# -- separation location
reatt = va.separate_loc(InFolder, pathI, timezone, skip=skip, opt=1) #, loc=-0.5, opt=1)
# -- shock location outside the boundary layer
va.shock_loc(InFolder, pathI, timezone, skip=skip) #, opt=2, val=[0.91, 0.92])
# -- shock foot within the boudnary layer
va.shock_foot(InFolder, pathI, timezone, 1.5, 0.87, skip=skip)  # 0.82 for laminar
# -- area of the separation bubble
va.bubble_area(InFolder, pathI, timezone, step=0, skip=skip)
# -- a specific location, for example in the shear layer
InFolder = path + 'Slice/Z_003/'
xy = [-11.0625, 2.5] # [-4.375, 3.53125] # [-11.0625, 2.125]  # [-4.6875, 2.9375]  # 
va.extract_point(InFolder, pathI, timezone, xy, skip=skip)
dt = 0.25
fs = 4.0
x1x2 = [600, 1100]
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
probe = pd.read_csv(pathI + 'Xk5_old.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
Xk = probe[var].values
fig, ax = plt.subplots(figsize=(6.4, 2.2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
ax.plot(probe['t'], Xk*fa, "k-", linewidth=0.8)
ax.set_xlim(x1x2) # (950, 1350) # 
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$u/u_\infty$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xk*fa)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xk.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# -- FWPSD
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xk*fa, dt, 1/dt, opt=1, seg=8, overlap=6)
ax.semilogx(Fre, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
# ax2nd = ax.secondary_xaxis('top', functions=(d2l, l2d))
# ax2nd.set_xlabel(r'$f x_r / u_\infty$')
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XkFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% singnal of bubble
# -- temporal evolution
bubble = np.loadtxt(pathI + "BubbleArea.dat", skiprows=1)
Xb = bubble[:, 1]
fig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.plot(bubble[:, 0], Xb, "k-", linewidth=0.8)
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$A/\delta_0^2$", fontsize=textsize)
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
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xb, dt, 1/dt, opt=1, seg=10, overlap=6)
ax.semilogx(Fre, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XbFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
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
Xa = angle  # shock2[:, 1] # Xf # angle # Xf  # Xl  #      
if np.array_equal(Xa, Xl):
    output = 'Shockloc'
    ylabel = r"$x_l/\delta_0$"
elif np.array_equal(Xa, Xf):
    output = 'Shockfoot'
    ylabel = r"$x_f/\delta_0$"
elif np.array_equal(Xa, angle):
    output = 'Shockangle'
    ylabel = r"$\eta(^{\circ})$"
fig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.plot(shock1[:, 0], Xa, "k-", linewidth=0.8)
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(ylabel, fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xa)
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
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xa, dt, 1/dt, opt=1, seg=8, overlap=4)
ax.semilogx(Fre, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + output + "FWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% singnal of reattachment point
# -- temporal evolution
reatt = np.loadtxt(pathI+"Reattach.dat", skiprows=1)  # separate
Xr = reatt[:, 1]
fig, ax = plt.subplots(figsize=(6.4, 2.2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
ax.plot(reatt[:, 0], Xr, "k-", linewidth=0.8)
ax.set_xlim(x1x2) # (1150, 1200)  # 
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y_r/\delta_0$", fontsize=textsize)
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
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xr, dt, 1/dt, opt=1, seg=10, overlap=8)
ax.semilogx(Fre, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XrFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% singnal of separation point
# -- temporal evolution
sepa = np.loadtxt(pathI+"Separate.dat", skiprows=1)  # separate
Xs = sepa[:, 1]
fig, ax = plt.subplots(figsize=(6.4, 2.2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
ax.plot(sepa[:, 0], Xs, "k-", linewidth=0.8)
ax.set_xlim(x1x2) # (1150, 1200)  # 
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$x_s/\delta_0$", fontsize=textsize)
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
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$f\ \mathcal{P}(f)$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = va.fw_psd(Xs, dt, 1/dt, opt=1, seg=8, overlap=4)
ax.semilogx(Fre, FPSD, "k", linewidth=0.8)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XsFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% gradient of Xr
x1x2 = [-40, 10]
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
num = 12
tnum = np.size(dxr)
dxdt = np.linspace(-2.0, 2.0, num)
nsize = np.zeros(num)
proba = np.zeros(num)
for i in range(num):
    ind = np.where(np.round(dxr, 1) == np.round(dxdt[i], 1))
    nsize[i] = np.size(ind)
    proba[i] = np.size(ind)/tnum

fig, ax = plt.subplots(figsize=(10, 3.5))
# ax.hist(dxr, bins=num, range=(-2.0, 2.0), edgecolor='k', linestyle='-',
#         facecolor='#D3D3D3', alpha=0.98, density=True)
hist, edges = np.histogram(dxr, bins=num, range=(-2.0, 2.0), density=True)
binwid = edges[1] - edges[0]
# plt.bar(edges[:-1], hist*binwid, width=binwid, edgecolor='k', linestyle='-',
#         facecolor='#D3D3D3', alpha=0.98)
plt.bar(edges[:-1], hist, width=binwid, edgecolor='k', linestyle='-',
        facecolor='#D3D3D3', alpha=0.98)
# ax.set_xlim([-2.0, 2.0])
ax.set_ylabel(r"$\mathcal{P}$", fontsize=textsize)
ax.set_xlabel(r"$\mathrm{d} x_r/\mathrm{d} t$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "ProbGradientXr.svg", bbox_inches="tight", pad_inches=0.1)
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
Xr0 = Xa # probe2[:, 1]
#Xs1 = probe11[:, 1]
#Xr1 = probe21[:, 1]
#Xs2 = probe12[:, 1]
#Xr2 = probe22[:, 1]

fig = plt.figure(figsize=(6.4, 3.0))
matplotlib.rc("font", size=textsize)
ax = fig.add_subplot(121)
Fre, coher = va.coherence(Xr0, Xs0, dt, fs, opt=1, seg=8, overlap=2)
#Fre1, coher1 = va.coherence(Xr1, Xs1, dt, fs)
#Fre2, coher2 = va.coherence(Xr2, Xs2, dt, fs)
ax.semilogx(Fre, coher, "k-", linewidth=1.0)
# ax.semilogx(Fre1, coher1, "k:", linewidth=1.0)
# ax.semilogx(Fre2, coher2, "k--", linewidth=1.0)
# ax.set_ylim([0.0, 1.0])
# ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$C$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
ax.annotate("(a)", xy=(-0.15, 1.0), xycoords="axes fraction", fontsize=numsize)

ax = fig.add_subplot(122)
Fre, cpsd = va.cro_psd(Xr0, Xs0, dt, fs, opt=1, seg=8, overlap=4)
#Fre1, cpsd1 = va.cro_psd(Xr1, Xs1, dt, fs)
#Fre2, cpsd2 = va.cro_psd(Xr2, Xs2, dt, fs)
ax.semilogx(Fre, np.arctan(cpsd.imag, cpsd.real), "k-", linewidth=1.0)
# ax.semilogx(Fre, np.arctan(cpsd1.imag, cpsd1.real), "k:", linewidth=1.0)
# ax.semilogx(Fre, np.arctan(cpsd2.imag, cpsd2.real), "k--", linewidth=1.0)
# ax.set_ylim([-1.0, 1.0])
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$\theta$" + "(rad)", fontsize=textsize)
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
plt.savefig(pathF + "Statistic"+"XsXa.svg",
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
