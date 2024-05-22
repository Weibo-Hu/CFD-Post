#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 27 10:39:40 2023
    post-process data for wavy cases

@author: weibo
"""

# %% Load necessary module

# %% load necessary modules
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import variable_analysis as va
from planar_field import PlanarField as pf
import plt2pandas as p2p
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from line_field import LineField as lf
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "qt")


path0 = "/media/weibo/VID21/wavy_flat/"
path0F, path0P, path0M, path0S, path0T, path0I = p2p.create_folder(path0)
path1 = "/media/weibo/VID21/wavy_018_small/"
path1F, path1P, path1M, path1S, path1T, path1I = p2p.create_folder(path1)
path2 = "/media/weibo/VID21/wavy_029_middle/"
path2F, path2P, path2M, path2S, path2T, path2I = p2p.create_folder(path2)
path3 = "/media/weibo/VID21/wavy_030_high/"
path3F, path3P, path3M, path3S, path3T, path3I = p2p.create_folder(path3)
pathC = path0 + "Comparison/"

plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",
    "weight": "normal",
    "size": "large",
}
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
tsize = 16
nsize = 11

# %% calculate
MeanFlow0 = pf()
MeanFlow0.load_meanflow(path0)
MeanFlow0.copy_meanval()
WallFlow0 = MeanFlow0.PlanarData.groupby("x", as_index=False).nth(1)

MeanFlow1 = pf()
MeanFlow1.load_meanflow(path1)
MeanFlow1.copy_meanval()

MeanFlow2 = pf()
MeanFlow2.load_meanflow(path2)
MeanFlow2.copy_meanval()

MeanFlow3 = pf()
MeanFlow3.load_meanflow(path3)
MeanFlow3.copy_meanval()

# if there is duplicate points
if np.size(np.unique(WallFlow0["y"])) > 2:
    maxy = np.max(WallFlow0["y"])
    WallFlow0 = WallFlow0.drop(WallFlow0[WallFlow0["y"] == maxy].index)

xwall = WallFlow0["x"].values
mu0 = va.viscosity(10000, WallFlow0["T"].values, T_inf=45, law="Suther")
Cf0 = va.skinfriction(mu0, WallFlow0["u"], WallFlow0["walldist"])
# commont parameters
firstval = 0.03125
T_inf = 45
Re = 10000
# case 1
df1 = MeanFlow1.PlanarData
wavy1 = pd.read_csv(path1M + "FirstLev.dat", skipinitialspace=True)
wavy1 = va.add_variable(df1, wavy1)
xwall1, Cf1 = va.skinfric_wavy(path1M, wavy1, Re, T_inf, firstval)

df2 = MeanFlow2.PlanarData
wavy2 = pd.read_csv(path2M + "FirstLev.dat", skipinitialspace=True)
wavy2 = va.add_variable(df2, wavy2)
xwall2, Cf2 = va.skinfric_wavy(path2M, wavy2, Re, T_inf, firstval)

df3 = MeanFlow3.PlanarData
wavy3 = pd.read_csv(path3M + "FirstLev.dat", skipinitialspace=True)
wavy3 = va.add_variable(df3, wavy3)
xwall3, Cf3 = va.skinfric_wavy(path3M, wavy3, Re, T_inf, firstval)

# %% Cf
b, a = signal.butter(6, Wn=1 / 10, fs=1 / 0.125)
fig2, ax2 = plt.subplots(figsize=(8, 2.5), dpi=500)
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
ax2.scatter(xwall[0::8], Cf0[0::8], s=10, marker='o',
            facecolors='w', edgecolors='C7', linewidths=0.8)
ax2.plot(xwall1, Cf1, "k-.", linewidth=1.5)
ax2.plot(xwall2, Cf2, "r:", linewidth=1.5)
ax2.plot(xwall3, Cf3, "g--", linewidth=1.5)
ax2.legend(labels=["FP", "W1", "W2", "W3"], ncols=2)
ax2.set_xlabel(r"$x$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=tsize)
ax2.set_xlim([0, 350])
ax2.set_xticks(np.arange(0, 351, 50))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=75, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cf.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Cs
Tt = va.stat2tot(Ma=6.0, Ts=45, opt="t") / 45
WallFlow0w = MeanFlow0.PlanarData.groupby("x", as_index=False).nth(0)

kt0 = va.thermal(mu0, 0.72)
Cs0 = va.Stanton(
    kt0,
    WallFlow0["T"].values,
    WallFlow0w["T"].values,
    WallFlow0["walldist"].values,
    Tt,
)

T_wall = 6.66
xwall1, Cs1 = va.Stanton_wavy(path1M, wavy1, Re, T_inf, T_wall, firstval)
xwall2, Cs2 = va.Stanton_wavy(path2M, wavy2, Re, T_inf, T_wall, firstval)
xwall3, Cs3 = va.Stanton_wavy(path3M, wavy3, Re, T_inf, T_wall, firstval)

# b, a = signal.butter(6, Wn=1 / 10, fs=1 / 0.125)
# Cs0 = np.abs(Cs0)
# Cs0[1200:] = signal.filtfilt(b, a, Cs0[1200:])
# Cs1 = np.abs(Cs1)
# Cs1[1200:] = signal.filtfilt(b, a, Cs1[1200:])
# Cs2 = np.abs(Cs2)
# Cs2[1200:] = signal.filtfilt(b, a, Cs2[1200:])
# Cs3 = np.abs(Cs3)
# Cs3[1200:] = signal.filtfilt(b, a, Cs3[1200:])

fig2, ax2 = plt.subplots(figsize=(8, 2.5), dpi=500)
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
# ax2.scatter(xwall[0::8], Cs0[0::8], s=10, marker='o',
#             facecolors='y', edgecolors='C7', linewidths=0.8)
ax2.scatter(
    xwall[0::8],
    np.abs(Cs0[0::8]),  # signal.filtfilt(b, a, Cs0[1200:])[0::8], # Cs0[0::8], 
    s=10,
    marker="o",
    facecolors="w",
    edgecolors="C7",
    linewidths=0.8,
)
ax2.plot(xwall1, np.abs(Cs1), "k-.", linewidth=1.5)
ax2.plot(xwall2, np.abs(Cs2), "r:", linewidth=1.5)
ax2.plot(xwall3, np.abs(Cs3), "b--", linewidth=1.5)
# ax2.plot(xwall, signal.filtfilt(b, a, Cs1), "r:", linewidth=1.5)
# ax2.plot(xwall, signal.filtfilt(b, a, Cs2), "r-", linewidth=1.5)
# ax2.plot(xwall, signal.filtfilt(b, a, Cs3), "b:", linewidth=1.5)
# ax2.plot(xwall, signal.filtfilt(b, a, Cs4), "b-", linewidth=1.5)
ax2.legend(labels=["FP", "W1", "W2", "W3"], ncols=2)
ax2.set_xlabel(r"$x$", fontsize=tsize)
ax2.set_ylabel(r"$\langle C_s \rangle$", fontsize=tsize)
ax2.set_xlim([0, 350])
# ax2.set_ylim([-0.001, 0.007])
ax2.set_xticks(np.arange(0, 351, 50))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=75, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(nsize)
ax2.annotate("(c)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cs.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %% Cp
fig1, ax1 = plt.subplots(figsize=(8, 2.5), dpi=500)
Ma = 6.0
fa = 1.0  # Ma * Ma *1.4
# fig = plt.figure(figsize=(8, 5), dpi=500)
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=nsize)
xwall = WallFlow0["x"].values
ax1.scatter(xwall[0::8], WallFlow0['p'][0::8], s=10, marker='o',
            facecolors='w', edgecolors='C7', linewidths=0.8)

ax1.plot(xwall1, wavy1['p'], "k-.", linewidth=1.5)
ax1.plot(xwall2, wavy2['p'], "r:", linewidth=1.5)
ax1.plot(xwall3, wavy3['p'], "g--", linewidth=1.5)
ax1.legend(labels=["FP", "W1", "W2", "W3"], ncols=2)
ax1.set_xlabel(r"$x$", fontsize=tsize)
ax1.set_ylabel(r"$\langle C_p \rangle$", fontsize=tsize)
ax1.set_xlim([0, 350])
ax1.set_xticks(np.arange(0, 351, 50))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.axvline(x=75, color="gray", linestyle=":", linewidth=1.5)
ax1.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax1.grid(visible=True, which="both", linestyle=":")
ax1.yaxis.offsetText.set_fontsize(nsize)
ax1.annotate("(b)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=nsize)
plt.tick_params(labelsize=nsize)
plt.savefig(pathC + "Cp.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()


# %%############################################################################
"""
    streamwise evolution of signals along an axis
"""
# %% Streamwise evolution of a specific variable
df0 = pd.read_hdf(path0M + 'TP_fluc.h5')
df1 = pd.read_hdf(path1M + 'TP_fluc.h5')
df2 = pd.read_hdf(path2M + 'TP_fluc.h5')
df3 = pd.read_hdf(path3M + 'TP_fluc.h5')
Wall0 = df0.groupby("x", as_index=False).nth(1)
nms = ["p", "T", "u", "v", "walldist", "p`"]
wavy1 = pd.read_csv(path1M + "FirstLev.dat", skipinitialspace=True)
wavy1 = va.add_variable(df1, wavy1, nms=nms)
wavy2 = pd.read_csv(path2M + "FirstLev.dat", skipinitialspace=True)
wavy2 = va.add_variable(df2, wavy2, nms=nms)
wavy3 = pd.read_csv(path3M + "FirstLev.dat", skipinitialspace=True)
wavy3 = va.add_variable(df3, wavy3, nms=nms)
# %%
fa = 1.0  # 1.7 * 1.7 * 1.4
var = 'p`'
fig, ax = plt.subplots(figsize=(8.2, 2.8))
matplotlib.rc('font', size=nsize)
ax.scatter(
    Wall0.x[::4],
    fa * (Wall0[var]-np.mean(Wall0[var]))[::4],
    s=10,
    marker="o",
    facecolors="w",
    edgecolors="C7",
    linewidths=0.8,
)
ax.plot(wavy1.x, fa * (wavy1[var]-np.mean(wavy1[var])), 'k-', linewidth=1.5)
ax.plot(wavy2.x, fa * (wavy2[var]-np.mean(wavy2[var])), 'r-', linewidth=1.5)
ax.plot(wavy3.x, fa * (wavy3[var]-np.mean(wavy3[var])), 'g-', linewidth=1.5)
ax.legend(labels=["FP", "W1", "W2", "W3"], ncols=2)
ax.set_xlim([50, 350.0])
# ax.set_ylim([-0.001, 0.001])
ax.set_xlabel(r'$x/\delta_0$', fontsize=tsize)
ax.set_ylabel(r'${}^\prime$'.format(var[:-1]), fontsize=tsize)
# ax.set_ylabel(r'$2p^\prime/\rho_\infty u_\infty^2$', fontsize=tsize)
ax.ticklabel_format(axis='y', style='sci',
                    useOffset=False, scilimits=(-2, 2))
ax.grid(visible='both', linestyle=':')
plt.show()
plt.savefig(pathC + var + '_streamwise.svg',
            bbox_inches='tight', pad_inches=0.1)

# %%
"""
#2 the growth of the fluctuations on the wall
"""

var='<u`u`>'
df1 = pd.read_hdf(path1M + 'MeanFlow.h5')
df2 = pd.read_hdf(path2M + 'MeanFlow.h5')
df3 = pd.read_hdf(path3M + 'MeanFlow.h5')
wavy1 = va.add_variable(df1, wavy1, nms=[var])
wavy2 = va.add_variable(df2, wavy2, nms=[var])
wavy3 = va.add_variable(df3, wavy3, nms=[var])
fig, ax = plt.subplots(figsize=(8, 2.5))
matplotlib.rc("font", size=nsize)
ax.set_xlabel(r"$x$", fontsize=tsize)
ax.set_ylabel("RMS(" + r"${}$".format(var[1:2]) + ")", fontsize=tsize)
ax.plot(
    WallFlow0["x"], np.sqrt(WallFlow0[var]), "k:"
)  # , label=r'$q^\prime_\mathrm{max}$')
ax.plot(
    wavy1["x"], np.sqrt(wavy1[var]), "k-"
)  
ax.plot(
    wavy2["x"], np.sqrt(wavy2[var]), "r-"
) 
ax.plot(
    wavy3["x"], np.sqrt(wavy3[var]), "g-"
) 
ax.set_yscale('log')
# ax.plot(xx, yy, 'k--', label='bubble')
# legend = ax.legend(loc="upper right", shadow=False, fontsize=nsize)
ax.legend(labels=["FP", "W1", "W2", "W3"], ncols=2)
ax.set_xlim([0, 350])
ax.set_xticks(np.arange(0, 351, 50))
# ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(visible="both", linestyle=":")
plt.show()
plt.savefig(pathC + var[1:2] + "RMS.svg", bbox_inches="tight", pad_inches=0.1)


# %% fluctuations of probe
"""
#1 coordinates where the fluctuations is max
"""
stat = MeanFlow1.PlanarData
var = "RMS_p"
if var == "RMS_u":
    varnm = "<u`u`>"
    savenm = "MaxRMS_u"
elif var == "RMS_p":
    varnm = "<p`p`>"
    savenm = "MaxRMS_p"
xloc = np.arange(0, 360.0 + 0.25, 0.25)
zloc = np.zeros(np.size(xloc))
varv = np.zeros(np.size(xloc))
yloc = np.zeros(np.size(xloc))
for i in range(np.size(xloc)):
    df = va.max_pert_along_y(stat, varnm, [xloc[i], zloc[i]])
    varv[i] = df[varnm]
    yloc[i] = df["y"]
data = np.vstack((xloc, yloc, varv))
df = pd.DataFrame(data.T, columns=["x", "y", varnm])
# df = df.drop_duplicates(keep='last')
df.to_csv(path1M + savenm + ".dat", sep=" ", float_format="%1.8e", index=False)

fig, ax = plt.subplots(figsize=(8, 2.5))
matplotlib.rc("font", size=14)
ax.set_ylabel(r"$y$", fontsize=tsize)
ax.set_xlabel(r"$x$", fontsize=tsize)
ax.plot(df["x"], df["y"], "b")
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    path1F + "loc_" + savenm + ".svg", bbox_inches="tight", pad_inches=0.1
)
# %%
"""
#2 the growth of the fluctuations
"""
df = pd.read_csv(path1M + savenm + ".dat", sep=" ")
fig, ax = plt.subplots(figsize=(8, 2.5))
matplotlib.rc("font", size=nsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=tsize)
ax.set_ylabel(r"$q^\prime_\mathrm{max}$", fontsize=tsize)
ax.plot(
    df["x"], np.sqrt(df[varnm]), "k"
)  # , label=r'$q^\prime_\mathrm{max}$')
# ax.plot(xx, yy, 'k--', label='bubble')
legend = ax.legend(loc="upper right", shadow=False, fontsize=nsize)
ax.set_xlim([0, 350])
ax.set_xticks(np.arange(0, 351, 50))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(path1F + var + ".svg", bbox_inches="tight", pad_inches=0.1)

