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
import copy
from DataPost import DataPost
import FlowVar as fv
from timer import timer
import os

plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
font1 = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "medium",
}
path = "/media/weibo/Data2/DF_example/8/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 18
numsize = 15
matplotlib.rc("font", size=textsize)
timezone = np.arange(800, 1149.50 + 0.5, 0.5)
x1x2 = [800, 1150]
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

MeanFlow = DataPost()
MeanFlow.LoadMeanFlow(path)
# MeanFlow.UserDataBin(pathM + "MeanFlow.h5")
# MeanFlow.UserData(VarName, path4 + "MeanFlow2.dat", 1, Sep="\t")
MeanFlow.AddWallDist(3.0)

# %% Plot BL profile along streamwise
xcoord = np.array([-40, -5, 0, 5.0, 10, 15, 20, 30, 40])
num = np.size(xcoord)
xtick = np.zeros(num + 1)
xtick[-1] = 1.0
fig, ax = plt.subplots(figsize=(10, 3.5))
# matplotlib.rc('font', size=14)
ax.plot(np.arange(num + 1), np.zeros(num + 1), "w-")
ax.set_xlim([0, num + 0.5])
ax.set_ylim([0, 4])
ax.set_xticks(np.arange(num + 1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = xtick
# ax.set_xticklabels(labels, fontdict=font)
ax.set_xticklabels(["$%d$" % f for f in xtick])
for i in range(num):
    y0, q0 = MeanFlow.BLProfile("x", xcoord[i], "u")
    ax.plot(q0 + i, y0, "k-")
    ax.text(
        i + 0.75,
        3.0,
        r"$x/\delta_0={}$".format(xcoord[i]),
        rotation=90,
        fontsize=numsize,
    )
plt.tick_params(labelsize=numsize)
ax.set_xlabel(r"$u/u_{\infty}$", fontsize=textsize)
ax.set_ylabel(r"$\Delta y/\delta_0$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.show()
plt.savefig(
    pathF + "StreamwiseBLProfile.svg", bbox_inches="tight", pad_inches=0.1
)

# %% Compute reference data
ExpData = np.loadtxt(path + "vel_1060_LES_M1p6.dat", skiprows=14)
m, n = ExpData.shape
y_delta = ExpData[:, 0]
u = ExpData[:, 1]
rho = ExpData[:, 3]
T = ExpData[:, 4]   # how it normalized?
urms = ExpData[:, 5]
vrms = ExpData[:, 6]
wrms = ExpData[:, 7]
uv = ExpData[:, 8]
mu = 1.0/13506*np.power(T, 0.75)
u_tau = fv.UTau(y_delta, u, rho, mu)
Re_tau = rho[0]*u_tau/mu[0]*2
ExpUPlus = fv.DirestWallLaw(y_delta, u, rho, mu)
xi = np.sqrt(rho / rho[0])
U_inf = 1.6 * np.sqrt(1.4*288*200)
uu = xi * np.sqrt(urms * U_inf) 
vv = xi * np.sqrt(vrms * U_inf) 
ww = xi * np.sqrt(wrms * U_inf) 
uv = xi * np.sqrt(np.abs(uv) * U_inf) * (-1)
y_plus = ExpUPlus[:, 0]
ExpUVPlus = np.column_stack((y_plus, uv))
ExpUrmsPlus = np.column_stack((y_plus, uu))
ExpVrmsPlus = np.column_stack((y_plus, vv))
ExpWrmsPlus = np.column_stack((y_plus, ww))

# %% Compare van Driest transformed mean velocity profile
# xx = np.arange(27.0, 60.0, 0.25)
# z0 = -0.5
# MeanFlow.UserDataBin(path + 'MeanFlow4.h5')
# MeanFlow.ExtraSeries('z', z0, z0)
MeanFlow.AddVariable('u', MeanFlow.DataTab['<u>'])
MeanFlow.AddVariable('rho', MeanFlow.DataTab['<rho>'])
MeanFlow.AddVariable('mu', MeanFlow.DataTab['<mu>'])
MeanFlow.AddVariable('T', MeanFlow.DataTab['<T>'])
MeanFlow.AddWallDist(3.0)
MeanFlow.AddMu(13718)
x0 = -10.0 #43.0
# for x0 in xx:
BLProf = copy.copy(MeanFlow)
BLProf.ExtraSeries('x', x0, x0)
BLProf.SpanAve()
u_tau = fv.UTau(BLProf.walldist, BLProf.u, BLProf.rho, BLProf.mu)
Re_tau = BLProf.rho[0] * u_tau / BLProf.mu[0] * 2
print("Re_tau=", Re_tau)
Re_theta = "2000"
StdUPlus1, StdUPlus2 = fv.StdWallLaw()
#ExpUPlus = fv.ExpWallLaw(Re_theta)[0]
CalUPlus = fv.DirestWallLaw(BLProf.walldist, BLProf.u, BLProf.rho, BLProf.mu)
fig, ax = plt.subplots(figsize=(5, 4))
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
spl = splrep(CalUPlus[:, 0], CalUPlus[:, 1], s=0.1)
uplus = splev(CalUPlus[:, 0], spl)
# ax.plot(CalUPlus[:, 0], uplus, 'k', linewidth=1.5)
ax.plot(CalUPlus[:, 0], CalUPlus[:, 1], "k", linewidth=1.5)
ax.set_xscale("log")
ax.set_xlim([1, 2000])
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
U_inf = 469.852
BLProf.AddVariable('uu', BLProf.DataTab['<u`u`>'])
BLProf.AddVariable('vv', BLProf.DataTab['<v`v`>'])
BLProf.AddVariable('ww', BLProf.DataTab['<w`w`>'])
BLProf.AddVariable('uv', BLProf.DataTab['<u`v`>'])
#ExpUPlus, ExpUVPlus, ExpUrmsPlus, ExpVrmsPlus, ExpWrmsPlus = \
#    fv.ExpWallLaw(Re_theta)
fig, ax = plt.subplots(figsize=(5, 4))
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
xi = np.sqrt(BLProf.rho / BLProf.rho[0])
# UUPlus = BLProf.uu / u_tau
uu = xi * np.sqrt(BLProf.uu * U_inf)
vv = xi * np.sqrt(BLProf.vv * U_inf)
ww = xi * np.sqrt(BLProf.ww * U_inf)
uv = xi * np.sqrt(np.abs(BLProf.uv) * U_inf) * (-1)
# spl = splrep(CalUPlus[:, 0], uu, s=0.1)
# uu = splev(CalUPlus[:, 0], spl)
ax.plot(CalUPlus[:, 0], uu, "k", linewidth=1.5)
ax.plot(CalUPlus[:, 0], vv, "r", linewidth=1.5)
ax.plot(CalUPlus[:, 0], ww, "b", linewidth=1.5)
ax.plot(CalUPlus[:, 0], uv, "gray", linewidth=1.5)
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
    theta[i] = fv.BLThickness(y0.values, u0.values, rho0.values, opt='momentum')

stream = np.loadtxt(path4+'Streamline.dat', skiprows=1)   
stream[:, -1] = stream[:, -1] + 3.0
func = interp1d(stream[:, 0], stream[:, 1], bounds_error=False, fill_value=0.0)
yd = func(xd)
xmax = np.max(stream[:, 0])
# fit curve
def func(t, A, B, C, D):
    return A * t ** 3 + B * t **2 + C * t + D
popt, pcov = DataPost.fit_func(func, xd, delta, guess=None)
A, B, C, D = popt
fitfunc = lambda t: A * t ** 3 + B * t **2 + C * t + D
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
ax2.plot(xd[:-1], delta_fit[:-1]/radius, "k", linewidth=1.5)
ax2.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax2.set_ylabel(r"$\delta/R$", fontsize=textsize)
ax2.set_xlim([0.0, 25.0])
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
#plt.savefig(path2 + "Cf.svg", bbox_inches="tight", pad_inches=0.1)
#plt.show()

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
MeanFlow.AddWallDist(3.0)
WallFlow = MeanFlow.DataTab.groupby("x", as_index=False).nth(1)
WallFlow = WallFlow[WallFlow.x != -0.0078125]
mu = fv.Viscosity(13718, WallFlow["T"])
Cf = fv.SkinFriction(mu, WallFlow["u"], WallFlow["walldist"]).values
ind = np.where(Cf[:] < 0.005)
# fig2, ax2 = plt.subplots(figsize=(5, 2.5))
fig = plt.figure(figsize=(10, 3))
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
probe = np.loadtxt(pathI + "ProbeB.dat", skiprows=1)
func = interp1d(probe[:, 1], probe[:, 8])
# timezone = probe[:, 1]
Xk = func(timezone)  # probe[:, 8]

fig, ax = plt.subplots(figsize=(10, 2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
fa = 1.7*1.7*1.4
ax.plot(timezone, Xk*fa, "k-")
ax.set_xlim([800, 1150])
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$p/p_\infty$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xk*fa)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xk.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

dt = 0.5
fig, ax = plt.subplots(figsize=(5, 4))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = fv.FW_PSD(Xk*fa, dt, 2.0, opt=1)
ax.semilogx(Fre, FPSD, "k", linewidth=1.0)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XkFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% load data of separation bubble size
bubble = np.loadtxt(OutFolder + "BubbleArea.dat", skiprows=1)
Xb = bubble[:, 1]

#%% reattachment location with time 
reatt = np.loadtxt(OutFolder+"Reattach.dat", skiprows=1)
timezone = reatt[:, 0]
Xr = reatt[:, 1]
dt = 0.5
fig, ax = plt.subplots(figsize=(10, 2))
# spl = splrep(timezone, xarr, s=0.35)
# xarr1 = splev(timezone[0::5], spl)
ax.plot(timezone, Xr, "k-")
ax.set_xlim([800, 1150])
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
fig, ax = plt.subplots(figsize=(5, 4))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = fv.FW_PSD(Xr, dt, 2.0, opt=1)
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
Xb = bubble[:, 1]
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(timezone, Xb, "k-")
ax.set_xlim([800, 1150])
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$A/\delta_0^2$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xb)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Xb.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = fv.FW_PSD(Xb, dt, 2.0, opt=1)
ax.semilogx(Fre, FPSD, "k", linewidth=1.0)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "XbFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% load data of shock location with time
shock1 = np.loadtxt(pathI + "ShockA.dat", skiprows=1)
shock2 = np.loadtxt(pathI + "ShockB.dat", skiprows=1)
angle = np.arctan(5 / (shock2[:, 1] - shock1[:, 1]))/np.pi*180
shockloc = shock2[:, 1] - 8.0 / np.tan(angle/180*np.pi)
foot = np.loadtxt(pathI + "ShockFoot.dat", skiprows=1)
Xl = shockloc
Xf = foot[:, 1]
Xs = angle
# %% plot Xf with time
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(timezone, Xs, "k-")
ax.set_xlim(x1x2)
ax.set_xlabel(r"$t u_\infty/\delta_0$", fontsize=textsize)
#ax.set_ylabel(r"$x_l/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$\alpha(^{\circ})$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
xmean = np.mean(Xs)
print("The mean value: ", xmean)
ax.axhline(y=xmean, color="k", linestyle="--", linewidth=1.0)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Shockangle.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
print("Corelation: ", fv.Correlate(Xr, Xf))

fig, ax = plt.subplots(figsize=(5, 4))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax.yaxis.major.formatter.set_powerlimits((-2, 3))
ax.set_xlabel(r"$f\delta_0/u_\infty$", fontsize=textsize)
ax.set_ylabel("Weighted PSD, unitless", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
Fre, FPSD = fv.FW_PSD(Xs, dt, 2.0, opt=1)
ax.semilogx(Fre, FPSD, "k", linewidth=1.0)
ax.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
plt.savefig(pathF + "ShockangleFWPSD.svg", bbox_inches="tight", pad_inches=0.1)
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
InFolder = path + 'Snapshots/'
timezone = np.arange(800, 1149.50 + 0.5, 0.5)
fv.ReattachLoc(InFolder, pathI, timezone, opt=2)
# %%
fv.ShockLoc(InFolder, pathI, timezone)
# %%
fv.ShockFoot(InFolder, pathI, timezone, -1.875, 0.82)
# %% 
fv.BubbleArea(InFolder, pathI, timezone)



