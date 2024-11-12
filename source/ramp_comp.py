#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:06:09 2024
    post-process data for ramp cases
@author: weibo
"""

# %% load necessary modules
import os
from natsort import natsorted
import imageio as iio
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import variable_analysis as va
from planar_field import PlanarField as pf
import pytecio as pytec
import plt2pandas as p2p
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import splprep, splev, interp1d
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from line_field import LineField as lf
from IPython import get_ipython
from scipy import fft
# get_ipython().run_line_magic("matplotlib", "qt")

# %% set path and basic parameters
path = "/media/weibo/VID2/AAS/ramp_st14_optimal/"
# path = 'E:/cases/wavy_1009/'
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + "Vortex/"
pathSL = path + "Slice/"
pathSN = path + "snapshots/"
pathC = path + "comparison/"
tsize = 13
nsize = 10
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
matplotlib.rc("font", size=tsize)
plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
cm2in = 1 / 2.54


# %% mean flow contour in x-y plane
MeanFlow = pf()
MeanFlow.load_meanflow(path)
MeanFlow.copy_meanval()
ind0 = (MeanFlow.PlanarData.y == 0.0)
MeanFlow.PlanarData['walldist'][ind0] = 0.0

# %% set path and basic parameters
path2 = "/media/weibo/VID2/AAS/ramp_st14_2nd/"
# path = 'E:/cases/wavy_1009/'
p2p.create_folder(path)
pathP2 = path2 + "probes/"
pathF2 = path2 + "Figures/"
pathM2 = path2 + "MeanFlow/"
pathS2 = path2 + "SpanAve/"
pathT2 = path2 + "TimeAve/"
pathI2 = path2 + "Instant/"
pathV2 = path2 + "Vortex/"
pathSL2 = path2 + "Slice/"
pathSN2 = path2 + "snapshots/"
pathC2 = path2 + "comparison/"

MeanFlow2 = pf()
MeanFlow2.load_meanflow(path2)
MeanFlow2.copy_meanval()
ind0 = (MeanFlow2.PlanarData.y == 0.0)
MeanFlow2.PlanarData['walldist'][ind0] = 0.0

# %% dataframe of the first level points
# corner = MeanFlow.PlanarData.walldist < 0.0
firstval = 0.015625
ind0 = (MeanFlow.PlanarData.y == 0.0)
MeanFlow.PlanarData['walldist'][ind0] = 0.0
xy_val = va.wall_line(MeanFlow.PlanarData, path, val=firstval)  # mask=corner)
# onelev = pd.DataFrame(data=xy_val, columns=["x", "y"])
# onelev.drop_duplicates(subset='x', keep='first', inplace=True)
FirstLev = pd.DataFrame(data=xy_val, columns=["x", "y"])
# ind = FirstLev.index[FirstLev['x'] < 0]
# FirstLev.loc[ind, 'y'] = firstval
xx = np.arange(-200.0, 80.0, 0.25)
FirstLev = FirstLev[FirstLev["x"].isin(xx)]
FirstLev.to_csv(pathM + "FirstLev.dat", index=False, float_format="%9.6f")

# %%
T_inf = 86.6
Re = 7736
df = MeanFlow.PlanarData
ramp_wall = pd.read_csv(pathM + "FirstLev.dat", skipinitialspace=True)
ramp_wall = va.add_variable(df, ramp_wall)
ramp_up = ramp_wall.loc[ramp_wall['x'] <= 0.0]
ramp_down = ramp_wall.loc[ramp_wall['x'] > 0.0]
# for flat plate
mu = va.viscosity(Re, ramp_up["T"])
xwall_up = ramp_up['x']
Cf_up = va.skinfriction(mu, ramp_up['u'], firstval, factor=1)
# for ramp
angle = 15 / 180 * np.pi
mu = va.viscosity(Re, ramp_down["T"])
ramp_down_u = ramp_down['u'] * np.cos(angle) + ramp_down['v'] * np.sin(angle)
xwall_down = ramp_down['x']
Cf_down = va.skinfriction(mu, ramp_down_u, firstval*np.cos(angle), factor=1)
ind = np.argmin(np.abs(Cf_up[:-20]))
xline_se = xwall_up[ind]
xline_re = np.round(np.interp(0.0, Cf_down, xwall_down), 2)
yline = np.max(Cf_down) * 0.88
# xwall, Cf = va.skinfric_wavy(pathM, wavy, Re, T_inf, firstval)
# xw1_fit, Cf1_fit = va.curve_fit(xwall1, Cf1, [-72.0, -46.0, -2.5], deg=[3, 4, 5, 4])
# xw2_fit, Cf2_fit = va.curve_fit(xwall2, Cf2, [20.0, 60.0], deg=[3, 4, 5])
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.25)
Cf_up_fit = signal.filtfilt(b, a, Cf_up)
Cf_down_fit = signal.filtfilt(b, a, Cf_down)
# %% 2nd
ind2 = (MeanFlow2.PlanarData.y == 0.0)
MeanFlow2.PlanarData['walldist'][ind2] = 0.0
xy_val2 = va.wall_line(MeanFlow2.PlanarData, path2, val=firstval)  # mask=corner)
# onelev = pd.DataFrame(data=xy_val, columns=["x", "y"])
# onelev.drop_duplicates(subset='x', keep='first', inplace=True)
FirstLev2 = pd.DataFrame(data=xy_val2, columns=["x", "y"])
# ind = FirstLev.index[FirstLev['x'] < 0]
# FirstLev.loc[ind, 'y'] = firstval
FirstLev2 = FirstLev[FirstLev["x"].isin(xx)]
FirstLev2.to_csv(pathM2 + "FirstLev.dat", index=False, float_format="%9.6f")

df2 = MeanFlow2.PlanarData
ramp_wall2 = pd.read_csv(pathM2 + "FirstLev.dat", skipinitialspace=True)
ramp_wall2 = va.add_variable(df2, ramp_wall2)
ramp2_up = ramp_wall2.loc[ramp_wall2['x'] <= 0.0]
ramp2_down = ramp_wall2.loc[ramp_wall2['x'] > 0.0]
# for flat plate
mu2 = va.viscosity(Re, ramp2_up["T"])
xwall2_up = ramp2_up['x']
Cf2_up = va.skinfriction(mu2, ramp2_up['u'], firstval, factor=1)
# for ramp
angle = 15 / 180 * np.pi
mu2 = va.viscosity(Re, ramp2_down["T"])
ramp2_down_u = ramp2_down['u'] * np.cos(angle) + ramp2_down['v'] * np.sin(angle)
xwall2_down = ramp2_down['x']
Cf2_down = va.skinfriction(mu2, ramp2_down_u, firstval*np.cos(angle), factor=1)
ind = np.argmin(np.abs(Cf2_up[:-20]))
xline2_se = xwall2_up[ind]
xline2_re = np.round(np.interp(0.0, Cf2_down, xwall2_down), 1)
yline2 = np.max(Cf2_down) * 0.88
Cf2_up_fit = signal.filtfilt(b, a, Cf2_up)
Cf2_down_fit = signal.filtfilt(b, a, Cf2_down)
# %% skin friction
fig2, ax2 = plt.subplots(figsize=(15*cm2in, 5*cm2in), dpi=500)
matplotlib.rc("font", size=nsize)
# ax2.plot(xwall1, Cf1, "b--", linewidth=1.5)
ax2.plot(xwall_up, Cf_up_fit, "r", linewidth=1.5)
ax2.plot(xwall_up, Cf2_up_fit, "b", linewidth=1.5)
# ax2.plot(xwall2, Cf2, "b--", linewidth=1.5)
ax2.plot(xwall_down, Cf_down_fit, "r", linewidth=1.5)
# ax2.plot(xwall2, Cf2, "b--", linewidth=1.5)
ax2.plot(xwall_down, Cf2_down_fit, "b", linewidth=1.5)
ax2.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=tsize)
ax2.set_xlim([-160.0, 80.0])
ax2.set_xticks(np.arange(-160, 80.0, 40))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=xline_se, color="r", linestyle=":", linewidth=1.0)
ax2.axvline(x=xline2_se, color="b", linestyle=":", linewidth=1.0)
# ax2.annotate(r"$x_s={}$".format(xline_se), (xline_se-16, yline))
ax2.axvline(x=xline_re, color="r", linestyle=":", linewidth=1.0)
ax2.axvline(x=xline2_re, color="b", linestyle=":", linewidth=1.0)
# ax2.annotate(r"$x_r={}$".format(xline_re), (xline_re-16, yline))
# ax2.annotate(r"$x_r={}$".format(xline2_re), (xline_re-16, yline))
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
ax2.legend(["optimal", "2nd"], loc='upper left', fontsize=nsize-2, framealpha=0.4)
#            ncols=2, columnspacing=0.6, )
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cf_comp.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Cp
Ma = 6.0
fa = 1 / ramp_wall2["p"][0]
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.25)
Cp_fit = signal.filtfilt(b, a, ramp_wall["p"])

# %%
Ma = 6.0
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.25)
Cp2_fit = signal.filtfilt(b, a, ramp_wall2["p"])

# %%
fig3, ax3 = plt.subplots(figsize=(15*cm2in, 5*cm2in), dpi=500)
# ax3 = fig.add_subplot(212)
# ax3.plot(ramp_wall['x'], ramp_wall["p"] * fa, "b--", linewidth=1.5)
ax3.plot(ramp_wall["x"], Cp_fit * fa, "r", linewidth=1.5)
ax3.plot(ramp_wall2["x"], Cp2_fit * fa, "b", linewidth=1.5)
ax3.set_xlabel(r"$x/l_r$", fontsize=tsize)
# ylab = r"$\langle p_w \rangle/\rho_{\infty} u_{\infty}^2$"
ax3.set_ylabel(r"$\langle C_p \rangle$", fontsize=tsize)
ax3.set_xlim([-160.0, 80])
ax3.set_xticks(np.arange(-160, 80.0, 40))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=xline_se, color="r", linestyle=":", linewidth=1.0)
ax3.axvline(x=xline2_se, color="b", linestyle=":", linewidth=1.0)
# ax3.annotate(r"$x_s={}$".format(xline1), (xline1-16, yline))
ax3.axvline(x=xline_re, color="r", linestyle=":", linewidth=1.0)
ax3.axvline(x=xline2_re, color="b", linestyle=":", linewidth=1.0)
# ax3.annotate(r"$x_r={}$".format(xline2), (xline2-16, yline))
ax3.grid(visible=True, which="both", linestyle=":")
ax3.annotate("(b)", xy=(-0.13, 1.0), xycoords="axes fraction", fontsize=nsize)
ax3.legend(["optimal", "2nd"], loc='upper left', fontsize=nsize-2, framealpha=0.4)
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cp_comp.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Cp certificate
p_ref = pd.read_csv(pathC + "p-x.txt", sep=' ', skipinitialspace=True)
fig3, ax3 = plt.subplots(figsize=(7*cm2in, 6*cm2in), dpi=500)
xarr = ramp_wall2["x"] / (xline2_re - xline2_se)
ax3.plot(xarr, Cp2_fit * fa, "b", linewidth=1.5)
ind = 8
xref = (p_ref['x'] - 1.0) * 252 / 192
pref = p_ref['p']/p_ref['p'][ind]
ax3.scatter(xref, pref, s=10, marker='o', c='g')
ax3.set_xlabel(r"$x/l_{sep}$", fontsize=tsize)
ax3.set_ylabel(r"$\langle C_p/C_{p0} \rangle$", fontsize=tsize)
ax3.set_xlim([-1.2, 1.2])
ax3.set_ylim([0.5, 6.5])
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.grid(visible=True, which="both", linestyle=":")
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cp_ref.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% Stanton number
firstval = 0.015625
T_wall = 3.35
Ma = 6.0
mu = va.viscosity(Re, ramp_wall["T"], T_inf=T_inf, law="Suther")
kt = va.thermal(mu, 0.72)
Tt = va.stat2tot(Ma=6.0, Ts=1.0, opt="t")
Cs = va.Stanton(ramp_wall["T"], T_wall, firstval,
                Re, Ma, T_inf=1.0, factor=1)
xwall = ramp_wall['x']
b, a = signal.butter(6, Wn=1 / 12, fs=1 / 0.25)
Cs_fit = signal.filtfilt(b, a, Cs)
# %%
mu = va.viscosity(Re, ramp_wall2["T"], T_inf=T_inf, law="Suther")
kt = va.thermal(mu, 0.72)
Tt = va.stat2tot(Ma=6.0, Ts=1.0, opt="t")
Cs2 = va.Stanton(ramp_wall2["T"], T_wall, firstval,
                 Re, Ma, T_inf=1.0, factor=1)
Cs2_fit = signal.filtfilt(b, a, Cs2)
# %% low-pass filter
# ind = np.where(Cf[:] < 0.008)
fig2, ax2 = plt.subplots(figsize=(15*cm2in, 5*cm2in), dpi=500)
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
# ax2.plot(xwall, np.abs(Cs), "b--", linewidth=1.5)
ax2.plot(xwall, np.abs(Cs), "r", linewidth=1.5)
ax2.plot(xwall, np.abs(Cs2), "b", linewidth=1.5)
ax2.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_s \rangle$", fontsize=tsize)
ax2.set_xlim([-160.0, 80])
ax2.set_xticks(np.arange(-160, 80.0, 40))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=xline_se, color="r", linestyle=":", linewidth=1.0)
ax2.axvline(x=xline2_se, color="b", linestyle=":", linewidth=1.0)
# ax2.annotate(r"$x_s={}$".format(xline1), (xline1-16, yline))
ax2.axvline(x=xline_re, color="r", linestyle=":", linewidth=1.0)
ax2.axvline(x=xline2_re, color="b", linestyle=":", linewidth=1.0)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(c)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
ax2.legend(["optimal", "2nd"], loc='upper left', fontsize=nsize-2,
           columnspacing=0.6, framealpha=0.4)
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cs_comp.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% growth rate
varn = '<p`p`>'
if varn == '<u`u`>':
    savenm = "MaxRMS_u"
    ylab = r"$\sqrt{u^{\prime 2}_\mathrm{max}}/u_{\infty}$"
elif varn == '<p`p`>':
    savenm = "MaxRMS_p"
    ylab = r"$\sqrt{p^{\prime 2}_\mathrm{max}}/\rho_{\infty} u_{\infty}^2$"
ramp_max = pd.read_csv(pathM + savenm + ".dat", sep=' ', skipinitialspace=True)
ramp_max2 = pd.read_csv(pathM2 + savenm + ".dat", sep=' ', skipinitialspace=True)
fig, ax = plt.subplots(figsize=(15*cm2in, 5.0*cm2in))
matplotlib.rc('font', size=nsize)
ax.set_ylabel(ylab, fontsize=tsize)
ax.set_xlabel(r"$x/l_r$", fontsize=tsize)
ax.plot(ramp_max['x'], ramp_max[varn], 'r')
ax.plot(ramp_max2['x'], ramp_max2[varn], 'b')
ax.set_xlim([-180.0, 60])
# ax.set_ylim([1e-7, 1e-1])
ax.set_xticks(np.linspace(-180.0, 60.0, 13))
ax.set_yscale('log')
# ax.axvline(x=xline1, color="gray", linestyle=":", linewidth=1.5)
# ax.axvline(x=xline2, color="gray", linestyle=":", linewidth=1.5)
ax.grid(visible=True, which="both", linestyle=":")
ax.legend(["optimal", "2nd"], loc='upper left', fontsize=nsize-1,
           columnspacing=0.6, framealpha=0.4)
ax.tick_params(labelsize=nsize)
plt.show()
plt.savefig(
    pathC + "MaxFluc" + savenm + ".svg", bbox_inches="tight", pad_inches=0.1
)
# %%
