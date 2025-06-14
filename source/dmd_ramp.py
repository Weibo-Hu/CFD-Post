#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:17:43 2024

@author: weibo
"""

# %% Load necessary module
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sparse_dmd import dmd, sparse, dmd_orig
import plt2pandas as p2p
from timer import timer
from scipy.interpolate import griddata
import os
import sys

matplotlib.use('ipympl')

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

# %% prep data
path = "L:/AAS/ramp_st14_2nd/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathD = path + "Domain/"
pathDMD = path + "DMD/"
pathSS = path + "TP_2D_Z_001/"
timepoints = np.arange(600.00, 900.00 + 0.50, 0.50)
dirs = sorted(os.listdir(pathSS))
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
DataFrame = pd.read_hdf(pathSS + dirs[0])
ind0 = DataFrame["y"] == 0.0
DataFrame.loc[ind0, "walldist"] = 0.0

grouped = DataFrame.groupby(["x", "y"])
DataFrame = grouped.mean().reset_index()
NewFrame = DataFrame.query("x>=-180.0 & x<=60.0 & walldist>=0.0 & y<=30.0")

ind = NewFrame.index.values
xval = DataFrame["x"][ind]
yval = DataFrame["y"][ind]

x = np.unique(xval)
y = np.unique(yval)

var0 = "u"
var1 = "v"
var2 = "p"
var3 = 'rho'
col = [var0, var1, var2, var3]
fa = 1
FirstFrame = DataFrame[col].values
Snapshots = FirstFrame[ind].ravel(order="F")

(m,) = np.shape(Snapshots)
n = np.size(timepoints)
o = np.size(col)
if m % o != 0:
    sys.exit("Dimensions of snapshots are wrong!!!")
m = int(m / o)
varset = {
    var0: [0, m], var1: [m, 2 * m], var2: [2 * m, 3 * m],
    var3: [3*m, 4*m]
         }
# %% load data
with timer("Load Data"):
    for i in range(np.size(dirs) - 1):
        TempFrame = pd.read_hdf(pathSS + dirs[i + 1])
        grouped = TempFrame.groupby(["x", "y"])
        TempFrame = grouped.mean().reset_index()
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit("The input snapshots does not match!!!")
        NextFrame = TempFrame[col].values
        Snapshots = np.vstack((Snapshots, NextFrame[ind].ravel(order="F")))
        DataFrame += TempFrame
Snapshots = Snapshots.T

AveFlow = DataFrame / np.size(dirs)
meanflow = AveFlow.query("x>=-180.0 & x<=60.0 & walldist>=0.0 & y<=30.0")

# %%
dt = 0.5
bfs = dmd.DMD(Snapshots, dt=dt)
with timer("DMD computing"):
    bfs.compute()
print("The residuals of DMD is ", bfs.residuals)
eigval = bfs.eigval

meanflow.to_hdf(pathDMD + "MeanFlow.h5", "w", format="fixed")

np.save(pathDMD + "eigval", bfs.eigval)  # \mu
np.save(pathDMD + "modes", bfs.modes)  # \phi
np.save(pathDMD + "lambda", bfs.frequencies)  # \lambda
np.save(pathDMD + "omega", bfs.omega)  # Imag(\lambda)
np.save(pathDMD + "beta", bfs.beta)  # Real(\lambda)
np.save(pathDMD + "amplitudes", bfs.amplitudes)  # \alpha

omega = bfs.omega
coeff = bfs.amplitudes
phi = bfs.modes
# %% SPDMD
bfs1 = sparse.SparseDMD(Snapshots, bfs, dt=dt)
gamma = [200, 400, 800]
with timer("SPDMD computing"):
    bfs1.compute_sparse(gamma)
print("The nonzero amplitudes of each gamma:", bfs1.sparse.Nz)

# %%
sp = 1
bfs1.sparse.Nz[sp]
bfs1.sparse.gamma[sp]
r = np.size(eigval)
sp_ind = np.arange(r)[bfs1.sparse.nonzero[:, sp]]

np.savez(
    pathDMD + "sparse.npz",
    Nz=bfs1.sparse.Nz,
    gamma=bfs1.sparse.gamma,
    nonzero=bfs1.sparse.nonzero,
)

spdmd = np.load(pathDMD + "sparse.npz")
sp_ind = np.arange(r)[spdmd['nonzero'][:, sp]]
# %% optional: load DMD results
eigval = np.load(pathDMD + "eigval.npy")
phi = np.load(pathDMD + "modes.npy")
omega = np.load(pathDMD + "omega.npy")
coeff = np.load(pathDMD + "amplitudes.npy")

# %% Eigvalue Spectrum
sp_ind = None
filtval = 0.0  # 0.99  # 0.0
var = var0
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rc("font", size=tsize)
plt.close("all")
fig1, ax1 = plt.subplots(figsize=(2.8, 3.0))
unit_circle = plt.Circle(
    (0.0, 0.0),
    1.0,
    color="grey",
    linestyle="-",
    fill=False,
    label="unit circle",
    linewidth=3.0,
    alpha=0.5,
)
ax1.add_artist(unit_circle)
ind = np.where(np.abs(eigval) > filtval)
ax1.scatter(
    eigval.real[ind],
    eigval.imag[ind],
    marker="o",
    facecolor="none",
    edgecolors="k",
    s=18,
)
if sp_ind is not None:
    sp_eigval = eigval[sp_ind]
    ax1.scatter(
        sp_eigval.real,
        sp_eigval.imag,
        marker="o",
        facecolor="gray",
        edgecolors="gray",
        s=18,
    )
limit = np.max(np.absolute(eigval)) + 0.1
ax1.set_xlim((-limit, limit))
ax1.set_ylim((-limit, limit))
ax1.tick_params(labelsize=nsize)
ax1.set_xlabel(r"$\Re(\mu_i)$")
ax1.set_ylabel(r"$\Im(\mu_i)$")
ax1.grid(visible=True, which="both", linestyle=":")
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathDMD + var + "DMDEigSpectrum.svg")
plt.show()
# %% discard the bad DMD modes
discard = False
if discard is True:
    bfs2 = bfs.reduce(filtval)
    phi = bfs2.modes
    freq = bfs2.omega / 2 / np.pi
    beta = bfs2.beta
    coeff = bfs2.amplitudes
else:
    freq = omega / 2 / np.pi

ind_pos = freq > 0.0
freq_pos = freq[ind_pos]
coeff_pos = coeff[ind_pos]
psi_pos = np.abs(coeff_pos) / np.max(np.abs(coeff_pos))
ind_st = np.argsort(freq_pos)
np.savetxt(pathDMD + 'FreqSpectrum.dat',
           np.transpose(np.vstack((freq_pos[ind_st], psi_pos[ind_st]))),
           header="freq, amplit",
           fmt='%.8f')
# %% Mode frequency specturm
matplotlib.rc("font", size=tsize)
fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
ax2.set_xscale("log")
ax2.vlines(freq_pos, [0], psi_pos, color="k", linewidth=1.0)
if sp_ind is not None:
    if discard is True:
        ind2 = bfs1.sparse.nonzero[bfs2.ind, sp]
    else:
        ind2 = spdmd['nonzero'][:, sp]
    ind3 = ind2[ind_pos]
    ax2.scatter(
        freq_pos[ind3], psi_pos[ind3],
        marker="o", facecolor="gray",
        edgecolors="gray", s=15
    )
ax2.set_xlim([1e-3, 2])
ax2.set_ylim(bottom=0.0)
ax2.tick_params(labelsize=nsize, pad=6)
ax2.set_xlabel(r"$f \delta_0/u_\infty$")
ax2.set_ylabel(r"$|\psi_i|$")
ax2.grid(visible=True, which="both", linestyle=":")
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathDMD + var + "DMDFreqSpectrum.svg")
plt.show()

# %% specific mode in real space
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
x1 = -180.0
x2 = -20.0
y1 = 0.0
y2 = 20.0
xs = x[(x >= x1) & (x <= x2)]
ys = y[(y >= y1) & (y <= y2)]
xx, yy = np.meshgrid(xs, ys)

var = var3
if var == 'p':
    fa = 6.0 * 6.0 * 1.4  # 1.0 #
else:
    fa = 1.0
ind = 0
# num = sp_ind[ind]  # ind from small to large->freq from low to high
num = np.argmin(np.abs(freq - 0.006))
name = str(round(freq[num], 3)).replace(".", "_")  # .split('.')[1] # str(ind)
tempflow = phi[:, num].real
print("The frequency is", freq[num])
modeflow = tempflow[varset[var][0]:varset[var][1]]
u = griddata((xval, yval), modeflow, (xx, yy)) * fa
print("The limit value: ", np.nanmin(u), np.nanmax(u))
corner = (xx < 0.0) & (yy < 0.0)
u[corner] = np.nan

matplotlib.rc("font", size=tsize)
fig, ax = plt.subplots(figsize=(7.5, 2.5))
c1 = -0.0005  # -0.024
c2 = -c1  # 0.010 #0.018
lev1 = np.linspace(c1, c2, 21)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(xx, yy, u, levels=lev1, cmap="viridis", extend="both")
# cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.set_xlim(x1, x2)
ax.set_ylim(y1, 12)
ax.set_xticks(np.linspace(x1, x2, 5))
ax.set_yticks(np.linspace(y1, 12, 4))
ax.tick_params(labelsize=nsize)
# cbar.cmap.set_under('#053061')
# cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/l_r$', fontsize=tsize)
ax.set_ylabel(r'$y/l_r$', fontsize=tsize)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.25, 0.76, 0.30, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(
    cbar, cax=cbaxes, orientation="horizontal", extendrect="False", ticks=rg2
)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(nsize)
cbar1.update_ticks()
cbar1.set_label(
    r"$\Re(\phi_{})$".format(var[0]),
    rotation=0, x=-0.20, labelpad=-26, fontsize=tsize
)
cbaxes.tick_params(labelsize=nsize)
# Add boundary layer edge
boundary = pd.read_csv(
    pathM + "ThermalBoundaryEdge.dat", skipinitialspace=True)
ax.plot(boundary.x, boundary.y, 'w', linewidth=1.0)
# Add shock wave
shock = pd.read_csv(pathM + "ShockLineFit.dat", skipinitialspace=True)
ax.plot(shock.x, shock.y, 'r', linewidth=1.0)
# Add sonic line
sonic = pd.read_csv(pathM+'SonicLine.dat', skipinitialspace=True)
ax.plot(sonic.x, sonic.y, 'w--', linewidth=1.0)

# ax.annotate("(a)", xy=(-0.1, 1.), xycoords='axes fraction', fontsize=tsize)
plt.savefig(pathDMD + var + "DMDMode" + name + "Real.svg", bbox_inches="tight")
plt.show()

# % specific mode in imaginary space
tempflow = phi[:, num].imag
imagflow = tempflow[varset[var][0]:varset[var][1]]
u = griddata((xval, yval), imagflow, (xx, yy)) * fa
print("The limit value: ", np.nanmin(u), np.nanmax(u))
corner = (xx < 0.0) & (yy < 0.0)
u[corner] = np.nan
matplotlib.rc("font", size=18)
fig, ax = plt.subplots(figsize=(7.5, 2.5))
c1 = -0.0005  # -0.024
c2 = -c1  # 0.012 #0.018
lev1 = np.linspace(c1, c2, 21)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(xx, yy, u, levels=lev1, cmap="viridis", extend="both")
# cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red

ax.set_xlim(x1, x2)
ax.set_ylim(y1, 12)
ax.set_xticks(np.linspace(x1, x2, 5))
ax.set_yticks(np.linspace(y1, 12, 4))
ax.tick_params(labelsize=nsize)
# cbar.cmap.set_under('#053061')
# cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/l_r$', fontsize=tsize)
ax.set_ylabel(r'$y/l_r$', fontsize=tsize)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.25, 0.76, 0.30, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(
    cbar, cax=cbaxes, orientation="horizontal", extendrect="False", ticks=rg2
)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(nsize)
cbar1.update_ticks()
cbar1.set_label(
    r"$\Im(\phi_{})$".format(var[0]),
    rotation=0, x=-0.20, labelpad=-26, fontsize=tsize
)
cbaxes.tick_params(labelsize=nsize)

# Add boundary layer edge
boundary = pd.read_csv(
    pathM + "ThermalBoundaryEdge.dat", skipinitialspace=True)
ax.plot(boundary.x, boundary.y, 'w', linewidth=1.0)
# ax.plot(boundary.x, boundary.y, "g--", linewidth=1.0)
# Add shock wave
shock = pd.read_csv(pathM + "ShockLineFit.dat", skipinitialspace=True)
ax.plot(shock.x, shock.y, 'r', linewidth=1.0)
# Add sonic line
sonic = pd.read_csv(pathM+'SonicLine.dat', skipinitialspace=True)
ax.plot(sonic.x, sonic.y, 'w--', linewidth=1.0)
# ax.annotate("(b)", xy=(-0.10, 1.0), xycoords='axes fraction', fontsize=nsize)
plt.savefig(pathDMD + var + "DMDMode" + name + "Imag.svg", bbox_inches="tight")
plt.show()
