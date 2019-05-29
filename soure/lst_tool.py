#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:52:46 2019
    post-processing LST data

@author: weibo
"""

# %% Load necessary module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, splev, splrep
from data_post import DataPost
import variable_analysis as fv
from timer import timer
from scipy.interpolate import griddata


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

# %% Contour of TS mode
ts_mode = pd.read_csv(path + 'TSmode1_-40.0.dat', sep=' ',
                      index_col=False, skipinitialspace=True)

phi = np.arange(50.0, 60.0 + 0.1, 0.1)
beta = np.arange(0.0001, 0.9, 0.001)
omega = np.arange(0.0001, 0.9, 0.0001)
ts_core = ts_mode # .query("phi >= 49.0 & phi <= 61.0")
x, y = np.meshgrid(beta, omega)
growth = griddata((ts_core['beta'], ts_core['omega']),
                  ts_core['alpha_i'], (x, y))
wave = griddata((ts_core['beta'], ts_core['omega']),
                ts_core['alpha_r'], (x, y))
# beta = np.tan(y/180*np.pi) * wave
print("growth_max=", np.max(growth))
print("growth_min=", np.min(growth))

fig, ax = plt.subplots(figsize=(6.4, 3.2))
matplotlib.rc("font", size=textsize)
rg1 = np.linspace(-0.018, 0.008, 27)
cbar = ax.contourf(x, y, growth, cmap='rainbow_r', levels=rg1)  # rainbow_r
ax.contour(x, y, growth, levels=[-0.016, -0.015, -0.012],
           linewidths=1.2, colors='w')
ax.set_xlim(0.2, 0.80)
ax.set_ylim(0.05, 0.15)
ax.set_yticks(np.arange(0.05, 0.15 + 0.025, 0.025))
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$\beta \delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\omega \delta_0 / u_\infty$", fontsize=textsize)
ax.axvline(x=0.40, ymin=0, ymax=0.5,
           color="gray", linestyle="--", linewidth=1.0)
ax.axhline(y=0.10, xmin=0, xmax=0.33,
           color="gray", linestyle="--", linewidth=1.0)
# Add colorbar
rg2 = np.linspace(-0.018, 0.008, 3)
cbaxes = fig.add_axes([0.56, 0.80, 0.3, 0.06])  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar.formatter.set_powerlimits((-2, 2))
cbar.ax.xaxis.offsetText.set_fontsize(numsize)
cbar.update_ticks()
cbar.set_label(
    r"$\alpha_i \delta_0$", rotation=0, x=-0.15,
    labelpad=-25, fontsize=textsize
)
cbaxes.tick_params(labelsize=numsize)
plt.savefig(pathF + "TS_mode_contour.svg", bbox_inches="tight")
plt.show()

# %% Plot LST spectrum
spectrum = pd.read_csv(path + 'SpatEigVal-40.0.dat', sep=' ',
                       index_col=False, na_values=['Inf', '-Inf'],
                       skipinitialspace=True, keep_default_na=False)
spectrum.dropna()
fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc("font", size=textsize)
# plot lines
ax.scatter(spectrum['alpha_r'], spectrum['alpha_i'], s=15, marker='o',
           facecolors='k', edgecolors='k', linewidths=0.5)
ax.set_xlim([-2, 2])
ax.set_ylim([-0.1, 0.3])
ax.set_xlabel(r'$\alpha_r \delta_0$', fontsize=textsize)
ax.set_ylabel(r'$\alpha_i \delta_0$', fontsize=textsize)
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")
plt.savefig(pathF + "TS_mode_spectrum.svg", bbox_inches="tight")
plt.show()

# %% Plot LST perturbations profiles
blasius = 1.579375403141185e-04
l_ref = 1.0e-3
ts_profile = pd.read_csv(path + 'UnstableMode.inp', sep=' ',
                         index_col=False, skiprows=4,
                         skipinitialspace=True)
ts_profile['u'] = np.sqrt(ts_profile['u_r']**2+ts_profile['u_i']**2)
ts_profile['v'] = np.sqrt(ts_profile['v_r']**2+ts_profile['v_i']**2)
ts_profile['w'] = np.sqrt(ts_profile['w_r']**2+ts_profile['w_i']**2)
ts_profile['t'] = np.sqrt(ts_profile['t_r']**2+ts_profile['t_i']**2)
ts_profile['p'] = np.sqrt(ts_profile['p_r']**2+ts_profile['p_i']**2)
# normalized
var_ref = np.max(ts_profile['u'])
ts_profile['u'] = ts_profile['u'] / var_ref
ts_profile['v'] = ts_profile['v'] / var_ref
ts_profile['w'] = ts_profile['w'] / var_ref
ts_profile['p'] = ts_profile['p'] / var_ref
ts_profile['t'] = ts_profile['t'] / var_ref
fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc("font", size=textsize)
# plot lines
ax.plot(ts_profile.u, ts_profile.y, 'k', linewidth=1.2,
        marker='x', markersize=3, label=r'$u$')
ax.plot(ts_profile.v, ts_profile.y, 'r', linewidth=1.2)
ax.plot(ts_profile.w, ts_profile.y, 'g', linewidth=1.2)
ax.plot(ts_profile.p, ts_profile.y, 'b', linewidth=1.2)
ax.plot(ts_profile.t, ts_profile.y, 'c', linewidth=1.2)
# plot scatter
#ax.scatter(ts_profile.u[0::4], ts_profile.y[0::4], s=15, marker='x',
#           facecolors='k', edgecolors='k', linewidths=0.5, label=r'$u$')
ax.scatter(ts_profile.v[0::8], ts_profile.y[0::8], s=15, marker='o',
           facecolors='r', edgecolors='r', linewidths=0.5, label=r'$v$')
ax.scatter(ts_profile.w[0::4], ts_profile.y[0::4], s=12, marker='s',
           facecolors='g', edgecolors='g', linewidths=0.5, label=r'$w$')
ax.scatter(ts_profile.p[0::8], ts_profile.y[0::8], s=30, marker='*',
           facecolors='b', edgecolors='b', linewidths=0.5, label=r'$p$')
ax.scatter(ts_profile.t[0::8], ts_profile.y[0::8], s=30, marker='^',
           facecolors='c', edgecolors='c', linewidths=0.5, label=r'$T$')
ax.set_xlim(0.0, 1.0)
ax.tick_params(axis='x', which='major', pad=5)
ax.set_ylim(0.0, 5.0)
ax.set_xlabel(r'$q^{\prime}/q^{\prime}_{max}$', fontsize=textsize)
ax.set_ylabel(r'$y/\delta_0$', fontsize=textsize)
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")
plt.legend()
plt.savefig(pathF + "TS_mode_profile.pdf", bbox_inches="tight")
plt.show()
