#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:22:31 2018
    This code for plotting line/curve figures

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
import os
from planar_field import PlanarField as pf


# %% data path settings
path = "/media/weibo/VID2/BFS_M1.7L/"
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
textsize = 13
numsize = 10

# %% Load Data
VarName = [
    "x",
    "y",
    "z",
    "u",
    "v",
    "w",
    "rho",
    "p",
    "T",
    "uu",
    "uv",
    "uw",
    "vv",
    "vw",
    "ww",
    "Q-criterion",
    "L2-criterion",
    "gradp",
]

timezone = np.arange(700, 999.75 + 0.25, 0.25)
x1x2 = [700, 1000]
StepHeight = 3.0
MeanFlow = pf()
MeanFlow.load_meanflow(path)
MeanFlow.add_walldist(StepHeight)

# %% 
path1 = "/media/weibo/VID2/BFS_M1.7L/MeanFlow/"
MeanFlow = pf()
MeanFlow.load_meanflow(path)
MeanFlow.add_walldist(StepHeight)

# %%############################################################################
"""
    boundary layer profile along streamwise direction
"""
# %% plot BL profile along streamwise
MeanFlow.copy_meanval()
fig, ax = plt.subplots(1, 7, figsize=(6.4, 2.2))
fig.subplots_adjust(hspace=0.5, wspace=0.15)
matplotlib.rc('font', size=numsize)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
xcoord = np.array([-40, 0, 5, 10, 15, 20, 30])
for i in range(np.size(xcoord)):
    df = MeanFlow.yprofile("x", xcoord[i])
    y0 = df['walldist']
    q0 = df['u']
    ax[i].plot(q0, y0, "k-")
    ax[i].set_ylim([0, 3])
    if i != 0:
        ax[i].set_yticklabels('')
    ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].tick_params(axis='both', which='major', labelsize=numsize)
    ax[i].set_title(r'$x/\delta_0={}$'.format(xcoord[i]), fontsize=numsize - 2)
    ax[i].grid(b=True, which="both", linestyle=":")
ax[0].set_ylabel(r"$\Delta y/\delta_0$", fontsize=textsize)
ax[3].set_xlabel(r'$u/u_\infty$', fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLProfile.svg", bbox_inches="tight", pad_inches=0.1
)


# %% Compute reference data
# ExpData = np.loadtxt(path + "vel_1060_LES_M1p6.dat", skiprows=0)
# m, n = ExpData.shape
# y_delta = ExpData[:, 0]
# u = ExpData[:, 1]
# rho = ExpData[:, 3]
# T = ExpData[:, 4]   # how it normalized?
# urms = ExpData[:, 5]
# vrms = ExpData[:, 6]
# wrms = ExpData[:, 7]
# uv = ExpData[:, 8]
# mu = 1.0/13506*np.power(T, 0.75)
# u_tau = fv.UTau(y_delta, u, rho, mu)
# Re_tau = rho[0]*u_tau/mu[0]
# ExpUPlus = fv.DirestWallLaw(y_delta, u, rho, mu)
# xi = np.sqrt(rho / rho[0])
# uu = np.sqrt(urms) / u_tau * xi
# vv = np.sqrt(vrms) / u_tau * xi
# ww = np.sqrt(wrms) / u_tau * xi
# uv = np.sqrt(np.abs(uv)) / u_tau * xi * (-1)
# y_plus = ExpUPlus[:, 0]
# ExpUVPlus = np.column_stack((y_plus, uv))
# ExpUrmsPlus = np.column_stack((y_plus, uu))
# ExpVrmsPlus = np.column_stack((y_plus, vv))
# ExpWrmsPlus = np.column_stack((y_plus, ww))
# %%############################################################################
"""
    boundary layer profile along streamwise direction
"""
# %% Compare wall law:
# %% computation, theory, experiment by van Driest transformation
x0 = 30.0
MeanFlow.copy_meanval()
BLProf = MeanFlow.yprofile('x', x0)
#BLProf1 = MeanFlow1.yprofile('x', x0)
#u_tau1 = fv.u_tau(BLProf1, option='mean')
u_tau = fv.u_tau(BLProf, option='mean')
mu_inf = BLProf['<mu>'].values[-1]
delta, u_inf = fv.BLThickness(BLProf['walldist'], BLProf['<u>'])
delta_star, u_inf, rho_inf = fv.BLThickness(
        BLProf['walldist'], BLProf['<u>'], 
        rho=BLProf['<rho>'], opt='displacement')
theta, u_inf, rho_inf = fv.BLThickness(
        BLProf['walldist'], BLProf['<u>'], 
        rho=BLProf['<rho>'], opt='momentum')
Re_theta = rho_inf * u_inf * theta / mu_inf
Re_delta_star = rho_inf * u_inf * delta_star / mu_inf
Re_tau = BLProf['<rho>'].values[0] * u_tau / BLProf['<mu>'].values[0] * delta
print("Re_tau=", Re_tau)
print("Re_theta=", Re_theta)
print("Re_delta*=", Re_delta_star)
Re_theta = 2000 # 1400# 2000
StdUPlus1, StdUPlus2 = fv.std_wall_law()
ExpUPlus = fv.ref_wall_law(Re_theta)[0]
CalUPlus = fv.direst_transform(BLProf, option='mean')
#CalUPlus1 = fv.direst_transform(BLProf1, option='mean')
fig, ax = plt.subplots(figsize=(3.2, 3))
# matplotlib.rc('font', size=textsize)
ax.plot(
    StdUPlus1[:, 0],
    StdUPlus1[:, 1],
    "k--",
    StdUPlus2[:, 0],
    StdUPlus2[:, 1],
    "k--",
    linewidth=1.5,
)
ax.scatter(
    ExpUPlus[:, 0],
    ExpUPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="gray",
)
# spl = splrep(CalUPlus[:, 0], CalUPlus[:, 1], s=0.1)
# uplus = splev(CalUPlus[:, 0], spl)
# ax.plot(CalUPlus[:, 0], uplus, 'k', linewidth=1.5)
ax.plot(CalUPlus[:, 0], CalUPlus[:, 1], "k", linewidth=1.5)
#ax.plot(CalUPlus1[:, 0], CalUPlus1[:, 1], "r:", linewidth=1.5)
ax.set_xscale("log")
ax.set_xlim([0.5, 2000])
ax.set_ylim([0, 30])
ax.set_ylabel(r"$\langle u_{VD}^+ \rangle$", fontdict=font)
ax.set_xlabel(r"$\Delta y^+$", fontdict=font)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize="medium")
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
# plt.tick_params(labelsize=numsize)
plt.savefig(
    pathF + 'WallLaw' + str(x0) + '.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# %% Compare Reynolds stresses in Morkovin scaling
ExpUPlus, ExpUVPlus, ExpUrmsPlus, ExpVrmsPlus, ExpWrmsPlus = \
    fv.ref_wall_law(Re_theta)
fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.scatter(
    ExpUrmsPlus[:, 0],
    ExpUrmsPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="k",
)
ax.scatter(
    ExpVrmsPlus[:, 0],
    ExpVrmsPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="r",
)
ax.scatter(
    ExpWrmsPlus[:, 0],
    ExpWrmsPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="b",
)
ax.scatter(
    ExpUVPlus[:, 0],
    ExpUVPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="gray",
)
xi = np.sqrt(BLProf['<rho>'] / BLProf['<rho>'][0])
uu = np.sqrt(BLProf['<u`u`>']) / u_tau * xi 
vv = np.sqrt(BLProf['<v`v`>']) / u_tau * xi 
ww = np.sqrt(BLProf['<w`w`>']) / u_tau * xi
uv = BLProf['<u`v`>'] / u_tau**2 * xi**2

#xi1 = np.sqrt(BLProf1['<rho>'] / BLProf1['<rho>'][0])
#uu1 = np.sqrt(BLProf1['<u`u`>']) / u_tau * xi
#vv1 = np.sqrt(BLProf1['<v`v`>']) / u_tau * xi
#ww1 = np.sqrt(BLProf1['<w`w`>']) / u_tau * xi
#uv1 = BLProf1['<u`v`>'] / u_tau**2 * xi**2

# spl = splrep(CalUPlus[:, 0], uu, s=0.1)
# uu = splev(CalUPlus[:, 0], spl)
ax.plot(CalUPlus[:, 0], uu, "k", linewidth=1.5)
ax.plot(CalUPlus[:, 0], vv, "r", linewidth=1.5)
ax.plot(CalUPlus[:, 0], ww, "b", linewidth=1.5)
ax.plot(CalUPlus[:, 0], uv, "gray", linewidth=1.5)
#ax.plot(CalUPlus1[:, 0], uu1, "ko", linewidth=1.5)
#ax.plot(CalUPlus1[:, 0], vv1, "ro", linewidth=1.5)
#ax.plot(CalUPlus1[:, 0], ww1, "bo", linewidth=1.5)
#ax.plot(CalUPlus1[:, 0], uv1, ":", linewidth=1.5)
ax.set_xscale("log")
ax.set_xlim([1, 2000])
# ax.set_ylim([0, 30])
ax.set_ylabel(r"$\langle u^{\prime}_i u^{\prime}_j \rangle$", fontdict=font)
ax.set_xlabel(r"$y^+$", fontdict=font)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize="medium")
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(
    pathF + "ReynoldStress" + str(x0) + ".svg",
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()

# %% calculate yplus along streamwise
wallval = -3.0 # 0.0 
dy =  -2.997037172317505 #  # 0.001953125
# -2.997037172317505  # 0.00390625
frame = MeanFlow.PlanarData.loc[(MeanFlow.PlanarData['y']==dy
                                 ) & (MeanFlow.PlanarData['x']>0.0)]
#frame = frame.iloc[:414]
frame1 = MeanFlow.PlanarData.loc[(MeanFlow.PlanarData['y']==wallval
                                  ) & (MeanFlow.PlanarData['x']>0.0)]
#frame1 = frame1.iloc[:414]
rho = frame1['<rho>'].values
mu = frame1['<mu>'].values
delta_u = (frame['<u>'].values-frame1['<u>'].values) / (dy - wallval)
tau = mu * delta_u
u_tau = np.sqrt(np.abs(tau / rho))
yplus = (dy - wallval) * u_tau * rho / mu
x = frame['x'].values
res = np.vstack((x, yplus))
frame2 = pd.DataFrame(data=res.T, columns=['x', 'yplus'])
frame2.to_csv(pathM + 'YPLUS2.dat',
              index=False, float_format='%1.8e', sep=' ')

# %% plot yplus along streamwise
yp = pd.read_csv(pathM + 'YPLUS.dat', sep=' ',
                 index_col=False, skipinitialspace=True)
fig3, ax3 = plt.subplots(figsize=(6.4, 3.0))
ax3.plot(yp['x'], yp['yplus'], "k", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$\Delta y^{+}$", fontsize=textsize)
ax3.set_xlim([-40.0, 70.0])
ax3.set_ylim([0.0, 2.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5)
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "yplus.svg", dpi=300)
plt.show()

# %% Compute BL edge & Gortler number
# compute
xd = np.arange(0.5, 40, 0.25)
num = np.size(xd)
delta = np.zeros(num)
delta_star = np.zeros(num)
theta = np.zeros(num)

for i in range(num):
    y0, u0 = MeanFlow.BLProfile("x", xd[i], "u")
    y0, rho0 = MeanFlow.BLProfile("x", xd[i], "rho")
    delta[i] = fv.BLThickness(y0.values, u0.values)
    delta_star[i] = fv.BLThickness(y0.values, u0.values,
                                   rho0.values, opt='displacement')
    theta[i] = fv.BLThickness(
        y0.values, u0.values, rho0.values, opt='momentum')

stream = np.loadtxt(pathM + 'Streamline.dat', skiprows=1)
stream[:, -1] = stream[:, -1] + 3.0
func = interp1d(stream[:, 0], stream[:, 1], bounds_error=False, fill_value=0.0)
yd = func(xd)
xmax = np.max(stream[:, 0])


# fit curve
def func(t, A, B, C, D):
    return A * t ** 3 + B * t ** 2 + C * t + D


popt, pcov = DataPost.fit_func(func, xd, delta, guess=None)
A, B, C, D = popt
fitfunc = lambda t: A * t ** 3 + B * t ** 2 + C * t + D
delta_fit = fitfunc(xd)

popt, pcov = DataPost.fit_func(func, xd, yd, guess=None)
A, B, C, D = popt
fitfunc = lambda t: A * t ** 3 + B * t **2 + C * t + D
yd = fitfunc(xd)
# gortler = fv.Gortler(1.3718e7, xd, delta1, theta)
radius = fv.Radius(xd[:-1], delta_fit[:-1])
# radius = fv.Radius(xd[:-1], yd[:-1])
gortler = fv.GortlerTur(theta[:-1], delta_star[:-1], radius)

fig3, ax3 = plt.subplots(figsize=(5, 2.5))
ax3.plot(xd, delta, 'k-', linewidth=1.5)
ax3.plot(xd, delta_fit, 'k--', linewidth=1.5)
ax3.plot(xd, yd, 'k--', linewidth=1.5)
ax3.plot(xd, theta, 'k:', linewidth=1.5)
# ax3.plot(xd[:-1], radius, 'k:', linewidth=1.5)
ax3.plot(stream[:, 0], stream[:, 1], 'b--', linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
# ax3.set_xlim([0.0, 30.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=10.9, color="gray", linestyle="--", linewidth=1.0)
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "BLEdge.svg", dpi=300)
plt.show()
# %% Plot Gortler number
# plot figure for delta/R
fig = plt.figure(figsize=(10, 4))
matplotlib.rc("font", size=textsize)
ax2 = fig.add_subplot(121)
matplotlib.rc("font", size=textsize)
ax2.plot(xd[:-1], delta_fit[:-1] / radius, "k", linewidth=1.5)
ax2.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax2.set_ylabel(r"$\delta/R$", fontsize=textsize)
ax2.set_xlim([0.0, 25.0])
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
# plt.savefig(path2 + "Cf.svg", bbox_inches="tight", pad_inches=0.1)
# plt.show()

# %% plot figure for Gortler number
fig3, ax3 = plt.subplots(figsize=(5, 4))
# ax3 = fig.add_subplot(122)
ax3.plot(xd[:-1], gortler, "k", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$G_t$", fontsize=textsize)
ax3.set_xlim([0.0, 25.0])
ax3.set_ylim([0.0, 1.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axhline(y=0.58, color="gray", linestyle="--", linewidth=1.5)
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "Gortler.svg", dpi=300)
plt.show()

# %% Plot streamwise skin friction
MeanFlow.copy_meanval()
WallFlow = MeanFlow.PlanarData.groupby("x", as_index=False).nth(1)
WallFlow = WallFlow[WallFlow.x != -0.0078125]
mu = fv.viscosity(13718, WallFlow["T"])
Cf = fv.skinfriction(mu, WallFlow["u"], WallFlow["walldist"]).values
ind = np.where(Cf[:] < 0.005)
# fig2, ax2 = plt.subplots(figsize=(5, 2.5))
fig = plt.figure(figsize=(6.4, 2.2))
matplotlib.rc("font", size=textsize)
ax2 = fig.add_subplot(121)
matplotlib.rc("font", size=textsize)
xwall = WallFlow["x"].values
ax2.plot(xwall[ind], Cf[ind], "k", linewidth=1.5)
ax2.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=textsize)
ax2.set_xlim([-20.0, 40.0])
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
ax2.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
ax2.annotate("(a)", xy=(-0.12, 1.04), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF+'Cf.svg',bbox_inches='tight', pad_inches=0.1)
plt.show()

# % pressure coefficiency
fa = 1.7 * 1.7 * 1.4
# fig3, ax3 = plt.subplots(figsize=(5, 2.5))
ax3 = fig.add_subplot(122)
ax3.plot(WallFlow["x"], WallFlow["p"] * fa, "k", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$\langle p_w \rangle/p_{\infty}$", fontsize=textsize)
ax3.set_xlim([-20.0, 40.0])
ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
ax3.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
ax3.grid(b=True, which="both", linestyle=":")
ax3.annotate("(b)", xy=(-0.12, 1.04), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
# plt.savefig(path2 + "Cp.svg", dpi=300)
plt.savefig(pathF + "CfCp.svg", dpi=300)
plt.show()

# %% Load data for Computing intermittency factor
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots2/"
dirs = sorted(os.listdir(InFolder))
data = pd.read_hdf(InFolder + dirs[0])
data["walldist"] = data["y"]
data.loc[data["x"] >= 0.0, "walldist"] += 3.0
NewFrame = data.query("walldist<=0.0")
ind = NewFrame.index.values
xzone = data["x"][ind].values
# xzone = np.linspace(-40.0, 70.0, 111)
with timer("Load Data"):
    Snapshots = np.vstack(
        [pd.read_hdf(InFolder + dirs[i])["p"] for i in range(np.size(dirs))]
    )
Snapshots = Snapshots.T
Snapshots = Snapshots[ind, :]

# %% calculate
gamma = np.zeros(np.size(xzone))
alpha = np.zeros(np.size(xzone))
p0 = Snapshots[0, :]
sigma = np.std(p0)
timezone = np.arange(800, 1149.50 + 0.5, 0.5)
dt = 0.5
for j in range(np.size(Snapshots[:, 0])):
    gamma[j] = fv.Intermittency(sigma, p0, Snapshots[j, :], timezone)
    alpha[j] = fv.Alpha3(Snapshots[j, :])

# %% Plot Intermittency factor
# universal intermittency distribution
x2 = np.linspace(15, 60, 50)
ksi = (x2 - 15) / 15
ga2 = 1 - np.exp(-0.412 * ksi ** 2)
x1 = np.linspace(-40, 15, 50)
ga1 = x1 - x1

XminorLocator = ticker.MultipleLocator(10)
YminorLocator = ticker.MultipleLocator(0.1)
xarr = np.linspace(-40.0, 70.0, 111)
spl = splrep(xzone, gamma, s=0.35)
yarr = splev(xarr, spl)
fig3, ax3 = plt.subplots(figsize=(10, 3))
# ax3.plot(xzone, gamma, 'k-')
ax3.plot(xarr, yarr, "k-")
ax3.scatter(x1, ga1, c="black", marker="o", s=10)
ax3.scatter(x2, ga2, c="black", marker="o", s=10)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$\gamma$", fontsize=textsize)
ax3.set_xlim([-40.0, 60.0])
ax3.set_ylim([-0.1, 1.0])
ax3.xaxis.set_minor_locator(XminorLocator)
ax3.yaxis.set_minor_locator(YminorLocator)
ax3.grid(b=True, which="both", linestyle=":")
ax3.axvline(x=0.0, color="k", linestyle="--", linewidth=1.0)
ax3.axvline(x=10.9, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Intermittency.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Kelvin Helholmz fluctuation
# B:K-H, C:X_reattach, D:turbulence
probe = np.loadtxt(pathI + "ProbeKH.dat", skiprows=1)
func = interp1d(probe[:, 1], probe[:, 7])
# timezone = probe[:, 1]
Xk = func(timezone)  # probe[:, 8]
dt = 0.5
fig, ax = plt.subplots(figsize=(6.4, 2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
fa = 1.7*1.7*1.4
ax.plot(timezone, Xk*fa, "k-")
ax.set_xlim([700, 1000])
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$p/p_\infty$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xk*fa)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xk.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()


fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = fv.fw_psd(Xk*fa, dt, 1/dt, opt=1)
ax.semilogx(Fre, FPSD, "k", linewidth=1.0)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XkFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% load data of separation bubble size
bubble = np.loadtxt(pathI + "BubbleArea.dat", skiprows=1)
Xb = bubble[:, 1]

#%% reattachment location with time
reatt = np.loadtxt(pathI+"Reattach.dat", skiprows=1)
timezone = reatt[:, 0]
Xr = reatt[:, 1]
dt = 0.5
fig, ax = plt.subplots(figsize=(6.4, 2.0))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
ax.plot(timezone, Xr, "k-")
ax.set_xlim([700, 1000])
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$x_r/\delta_0$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xr)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xr.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# FWPSD
fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = fv.fw_psd(Xr, dt, 1/dt, opt=1)
ax.semilogx(Fre, FPSD, "k", linewidth=1.0)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XrFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
# %% gradient of Xr
fig, ax = plt.subplots(figsize=(10, 2))
dxr = DataPost.SecOrdFDD(timezone, Xr)
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
    ind = np.where(np.round(dxr, 1) == np.round(dxdt[i],1))
    nsize[i] = np.size(ind)
    proba[i] = np.size(ind)/tnum

fig, ax = plt.subplots(figsize=(10, 3.5))
#ax.hist(dxr, bins=num, range=(-2.0, 2.0), edgecolor='k', linestyle='-',
#        facecolor='#D3D3D3', alpha=0.98, density=True)
hist, edges = np.histogram(dxr, bins=num, range=(-2.0, 2.0), density=True)
binwid = edges[1] - edges[0]
#plt.bar(edges[:-1], hist*binwid, width=binwid, edgecolor='k', linestyle='-',
#        facecolor='#D3D3D3', alpha=0.98)
plt.bar(edges[:-1], hist, width=binwid, edgecolor='k', linestyle='-',
        facecolor='#D3D3D3', alpha=0.98)
# ax.set_xlim([-2.0, 2.0])
ax.set_ylabel(r"$\mathcal{P}$", fontsize=textsize)
ax.set_xlabel(r"$\mathrm{d} x_r/\mathrm{d} t$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "ProbGradientXr.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
# %%  Plot Xb with time
bubble = np.loadtxt(pathI + "BubbleArea.dat", skiprows=1)
dt = 0.5
Xb = bubble[:, 1]
fig, ax = plt.subplots(figsize=(6.4, 2))
ax.plot(bubble[:, 0], Xb, "k-")
ax.set_xlim([700, 1000])
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$A/\delta_0^2$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xb)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xb.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = fv.fw_psd(Xb, dt, 1/dt, opt=1)
ax.semilogx(Fre, FPSD, "k", linewidth=1.0)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XbFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% load data of shock location with time
shock1 = np.loadtxt(pathI + "ShockA.dat", skiprows=1)
shock2 = np.loadtxt(pathI + "ShockB.dat", skiprows=1)
delta_x = shock2[0, 2] - shock1[0, 2]
angle = np.arctan(delta_x/ (shock2[:, 1] - shock1[:, 1]))/np.pi*180
shockloc = shock2[:, 1] - 8.0 / np.tan(angle/180*np.pi)
foot = np.loadtxt(pathI + "ShockFoot.dat", skiprows=1)
Xl = shockloc
Xf = foot[:, 1]
Xs = shockloc
# %% plot Xf with time
fig, ax = plt.subplots(figsize=(6.4, 2.0))
ax.plot(shock1[:, 0], Xs, "k-")
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
#ax.set_ylabel(r"$x_l/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\alpha(^{\circ})$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xs)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Shockloc.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
print("Corelation: ", fv.correlate(Xr, Xf))

dt = 0.5
fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax.yaxis.major.formatter.set_powerlimits((-2, 3))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = fv.fw_psd(Xs, dt, 1/dt, opt=1)
ax.semilogx(Fre, FPSD, "k", linewidth=1.0)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "ShocklocFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% plot coherence and phase for var1 & var2
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/Data/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/temp/"
probe1 = np.loadtxt(OutFolder + "ShockFootE.dat", skiprows=1)
probe2 = np.loadtxt(OutFolder + "ShockFootC.dat", skiprows=1)
probe11 = np.loadtxt(OutFolder + "ShockFootE.dat", skiprows=1)
probe21 = np.loadtxt(OutFolder + "ShockFootD.dat", skiprows=1)
probe12 = np.loadtxt(OutFolder + "ShockFootE.dat", skiprows=1)
probe22 = np.loadtxt(OutFolder + "ShockFootF.dat", skiprows=1)
timezone = np.arange(600, 1000 + 0.5, 0.5)
# %%
dt = 0.5
fs = 2
Xs0 = Xr # probe1[:, 1]
Xr0 = Xb # probe2[:, 1]
Xs1 = probe11[:, 1]
Xr1 = probe21[:, 1]
Xs2 = probe12[:, 1]
Xr2 = probe22[:, 1]

fig = plt.figure(figsize=(10, 4))
matplotlib.rc("font", size=textsize)
ax = fig.add_subplot(121)
Fre, coher = fv.Coherence(Xr0, Xs0, dt, fs)
Fre1, coher1 = fv.Coherence(Xr1, Xs1, dt, fs)
Fre2, coher2 = fv.Coherence(Xr2, Xs2, dt, fs)
ax.semilogx(Fre, coher, "k-", linewidth=1.0)
#ax.semilogx(Fre1, coher1, "k:", linewidth=1.0)
#ax.semilogx(Fre2, coher2, "k--", linewidth=1.0)
#ax.set_ylim([0.0, 1.0])
# ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$C$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
ax.annotate("(a)", xy=(-0.15, 1.0), xycoords="axes fraction", fontsize=numsize)

ax = fig.add_subplot(122)
Fre, cpsd = fv.Cro_PSD(Xr0, Xs0, dt, fs)
Fre1, cpsd1 = fv.Cro_PSD(Xr1, Xs1, dt, fs)
Fre2, cpsd2 = fv.Cro_PSD(Xr2, Xs2, dt, fs)
ax.semilogx(Fre, np.arctan(cpsd.imag, cpsd.real), "k-", linewidth=1.0)
#ax.semilogx(Fre, np.arctan(cpsd1.imag, cpsd1.real), "k:", linewidth=1.0)
#ax.semilogx(Fre, np.arctan(cpsd2.imag, cpsd2.real), "k--", linewidth=1.0)
#ax.set_ylim([-1.0, 1.0])
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$\theta$" + "(rad)", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
ax.annotate("(b)", xy=(-0.20, 1.0), xycoords="axes fraction", fontsize=numsize)
lab = [
    r"$\Delta y/\delta_0 = 1.0$",
    r"$\Delta y/\delta_0 = 1.5$",
    r"$\Delta y/\delta_0 =4.0$",
]
#ax.legend(lab, ncol=1, loc="upper right", fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(path2 + "Statistic"+"XrXb.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% plot cross-correlation coeffiency of two variables with time delay
x1 = Xr
x2 = Xk
delay = np.arange(-60.0, 60+0.5, 0.5)
cor = np.zeros(np.size(delay))
for i, dt in enumerate(delay):
    cor[i] = fv.DelayCorrelate(x1, x2, 0.5, dt)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(delay, cor, "k-")
ax.set_xlim([delay[0], delay[-1]])
ax.set_xlabel(r"$\Delta t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$R_{ij}$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "Cor_XrXk.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% Streamwise evolution of variable
# Load Data for time-averaged results
MeanFlow = DataPost()
MeanFlow.UserDataBin(path + "MeanFlow.h5")
MeanFlow.AddWallDist(3.0)
tab = MeanFlow.DataTab
# %%
tab1 = tab.loc[tab["z"] == 0.0]
tab2 = tab1.loc[tab1["walldist"] == 0.0]
newflow = tab2.query("u<=0.05")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(newflow.x, newflow.u, "k-")
ax.set_xlabel(r"$x/\delta_0$", fontdict=font)
ax.set_ylabel(r"$u/u_\infty$", fontdict=font)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize="medium")
plt.savefig(path2 + "123.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% POD convergence
fig, ax = plt.subplots(figsize=(5, 3))
data = np.loadtxt(path2 + "POD/PODConvergence.dat", skiprows=1)
ax.semilogy(data[0, :], data[1, :] / 100, marker="o", color="k", linewidth=1.0)
ax.semilogy(data[0, :], data[2, :] / 100, marker="^", color="k", linewidth=1.0)
ax.semilogy(data[0, :], data[3, :] / 100, marker="*", color="k", linewidth=1.0)
ax.semilogy(data[0, :], data[4, :] / 100, marker="s", color="k", linewidth=1.0)
lab = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]
ax.legend(lab, ncol=2, loc="upper right", fontsize=15)
#          bbox_to_anchor=(1., 1.12), borderaxespad=0., frameon=False)
ax.set_ylim([0.01, 1.0])
ax.set_xlim([280, 700])
ax.set_xlabel(r"$N$", fontsize=textsize)
ax.set_ylabel(r"$\lambda_i/\sum_{k=1}^N \lambda_k$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.savefig(
    path2 + "POD/PODConvergence.svg", bbox_inches="tight", pad_inches=0.1
)
plt.show()

# %% Vortex Wavelength along streamwise direction
time = "779.50"
wave = pd.read_csv(
    path3 + "L" + time + ".txt",
    sep="\t",
    index_col=False,
    skipinitialspace=True,
    keep_default_na=False,
)
meanwave = pd.read_csv(
    path3 + "LMean.txt",
    sep="\t",
    index_col=False,
    skipinitialspace=True,
    keep_default_na=False,
)
xarr = wave["x"]
func = interp1d(meanwave["x"], meanwave["u"])
yarr = func(xarr)
# wave = np.loadtxt(path3+'L'+time+'.txt', delimiter='\t', skiprows=1)
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(wave["x"], wave["u"] - yarr, linewidth=1.2)
ax.set_xlim([2.0, 10.0])
ax.set_ylim([-0.08, 0.1])
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$u^\prime/u_\infty$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.savefig(
    path2 + "Wavelen" + time + ".svg", bbox_inches="tight", pad_inches=0.1
)
plt.show()


# %% Calculate Xr, Xb, Xsf, Xsl, Xk
# %% Temporal variation of reattachment point
InFolder = path + 'Slice/TP_2D_S_10/'
timezone = np.arange(700, 999.75 + 0.25, 0.25)
reatt = fv.reattach_loc(InFolder, pathI, timezone[0::2], skip=2, opt=2)
# %%
fv.shock_loc(InFolder, pathI, timezone[0::2], skip=2)
# %%
fv.shock_foot(InFolder, pathI, timezone[0::2], -1.875, 0.82, skip=2)
# %%
fv.bubble_area(InFolder, pathI, timezone[0::2], skip=2)



