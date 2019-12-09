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
import plt2pandas as p2p
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, splev, splrep
from data_post import DataPost
import variable_analysis as va
from timer import timer
import os
from planar_field import PlanarField as pf


## data path settings
path = "/media/weibo/IM1/BFS_M1.7Tur/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"

## figures properties settings
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
#MeanFlow.load_data(path + 'inca_out/')
MeanFlow.load_meanflow(path)
MeanFlow.add_walldist(StepHeight)

# %% Load laminar data for comparison
path1 = "/media/weibo/VID2/BFS_M1.7L/MeanFlow/"
MeanFlowL = pf()
MeanFlowL.load_meanflow(path)
MeanFlowL.add_walldist(StepHeight)

# %%############################################################################
"""
    boundary layer profile along streamwise direction
"""
# %% plot BL profile along streamwise
MeanFlow.copy_meanval()
fig, ax = plt.subplots(1, 8, figsize=(6.4, 2.4))
fig.subplots_adjust(hspace=0.5, wspace=0.18)
matplotlib.rc('font', size=numsize)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
xcoord = np.array([-30, 2.0, 4.625, 6.25, 9.25, 10, 20, 30])
for i in range(np.size(xcoord)):
    df = MeanFlow.yprofile("x", xcoord[i])
    y0 = df['walldist']
    q0 = df['u']
    ax[i].plot(q0, y0, "k-")
    ax[i].set_ylim([0, 3])
    if i != 0:
        ax[i].set_yticklabels('')
        ax[i].set_title(r'${}$'.format(xcoord[i]), fontsize=numsize-2)
    ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].tick_params(axis='both', which='major', labelsize=numsize)   
    ax[i].grid(b=True, which="both", linestyle=":")
ax[0].set_title(r'$x/\delta_0={}$'.format(xcoord[0]), fontsize=numsize-2)
ax[0].set_ylabel(r"$\Delta y/\delta_0$", fontsize=textsize)
ax[4].set_xlabel(r'$u /u_\infty$', fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLProfile.svg", bbox_inches="tight", pad_inches=0.1
)


# %%############################################################################
"""
    Compare the law of wall: theoretical by van Driest, experiments, LES
"""
# %% velocity profile, computation
# df = p2p.ReadAllINCAResults(path + 'inca_out/', path + 'inca_out/', 
#                             FileName=path + 'inca_out/MeanFlow.szplt',
#                             SpanAve=True, OutFile='MeanFlow')
# path = "/media/weibo/VID2/FlatTur_coarse/"
#df = np.loadtxt(path + 'boundary_input_000001.dat')
# varnm = ['x', 'y', 'z', '<u>', '<v>', '<w>', '<rho>', '<T>',
#          '<u`u`>', '<v`v`>', '<w`w`>', '<u`v`>', 'u`w`', 'v`w`']
#BLProf = pd.DataFrame(data=df.values[2:, :], columns=varnm)
#grouped = BLProf.groupby('y')
#BLProf = grouped.mean().reset_index()
#BLProf['walldist'] = BLProf['y']
#BLProf['<mu>'] = va.viscosity(13500, BLProf['<T>'])
# %% velocity profile, computation
x0 = -30.0
## results from LES
MeanFlow.copy_meanval()
BLProf = MeanFlow.yprofile('x', x0)
# %%
u_tau = va.u_tau(BLProf, option='mean')
mu_inf = BLProf['<mu>'].values[-1]
delta, u_inf = va.bl_thickness(BLProf['walldist'], BLProf['<u>'])
delta_star, u_inf, rho_inf = va.bl_thickness(
        BLProf['walldist'], BLProf['<u>'], 
        rho=BLProf['<rho>'], opt='displacement')
theta, u_inf, rho_inf = va.bl_thickness(
        BLProf['walldist'], BLProf['<u>'], 
        rho=BLProf['<rho>'], opt='momentum')
Re_theta = rho_inf * u_inf * theta / mu_inf
Re_delta_star = rho_inf * u_inf * delta_star / mu_inf
Re_tau = BLProf['<rho>'].values[0] * u_tau / BLProf['<mu>'].values[0] * delta
print("Re_tau=", Re_tau)
print("Re_theta=", Re_theta)
print("Re_delta*=", Re_delta_star)
CalUPlus = va.direst_transform(BLProf, option='mean')
## results from theory by van Driest
StdUPlus1, StdUPlus2 = va.std_wall_law()
## results from known DNS
Re_theta = 800 # 1400 #  
ExpUPlus = va.ref_wall_law(Re_theta)[0]
## plot velocity profile
fig = plt.figure(figsize=(6.4, 3.0))
ax = fig.add_subplot(121)
matplotlib.rc('font', size=numsize)
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
ax.set_ylabel(r"$\langle u_{VD}^+ \rangle$", fontsize=textsize)
ax.set_xlabel(r"$\Delta y^+$", fontsize=textsize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
ax.annotate("(a)", xy=(-0.16, 0.98), xycoords='axes fraction', fontsize=numsize)
plt.tick_params(labelsize="medium")
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
# plt.tick_params(labelsize=numsize)
plt.savefig(
    pathF + 'WallLaw' + str(x0) + '.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

#  Reynolds stresses in Morkovin scaling
## results from known DNS
ExpUPlus, ExpUVPlus, ExpUrmsPlus, ExpVrmsPlus, ExpWrmsPlus = \
    va.ref_wall_law(Re_theta)
## results from current LES
xi = np.sqrt(BLProf['<rho>'] / BLProf['<rho>'][0])
uu = np.sqrt(BLProf['<u`u`>']) / u_tau * xi 
vv = np.sqrt(BLProf['<v`v`>']) / u_tau * xi 
ww = np.sqrt(BLProf['<w`w`>']) / u_tau * xi
uv = BLProf['<u`v`>'] / u_tau**2 * xi**2
# spl = splrep(CalUPlus[:, 0], uu, s=0.1)
# uu = splev(CalUPlus[:, 0], spl)
## plot Reynolds stress
# fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax2 = fig.add_subplot(122)
ax2.scatter(
    ExpUrmsPlus[:, 0],
    ExpUrmsPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="k",
)
ax2.scatter(
    ExpVrmsPlus[:, 0],
    ExpVrmsPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="k", #"r",
)
ax2.scatter(
    ExpWrmsPlus[:, 0],
    ExpWrmsPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="k", #"b",
)
ax2.scatter(
    ExpUVPlus[:, 0],
    ExpUVPlus[:, 1],
    linewidth=0.8,
    s=8.0,
    facecolor="none",
    edgecolor="k", #"gray",
)
ax2.plot(CalUPlus[:, 0], uu, "k", linewidth=1.5)
ax2.plot(CalUPlus[:, 0], vv, "k", linewidth=1.5)
ax2.plot(CalUPlus[:, 0], ww, "k", linewidth=1.5)
ax2.plot(CalUPlus[:, 0], uv, "k", linewidth=1.5)
ax2.set_xscale("log")
ax2.set_xlim([1, 2000])
# ax.set_ylim([0, 30])
ax2.set_ylabel(r"$\langle u^{\prime}_i u^{\prime}_j \rangle$", fontdict=font)
ax2.set_xlabel(r"$y^+$", fontdict=font)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.grid(b=True, which="both", linestyle=":")
ax2.annotate("(b)", xy=(-0.18, 0.98), xycoords='axes fraction',fontsize=numsize)
plt.subplots_adjust(wspace=0.8)
plt.tick_params(labelsize="medium")
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.3)
plt.savefig(
    pathF + "ReynoldStress" + str(x0) + ".svg",
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()

# %%############################################################################
"""
    y+ along streamwise
"""
# %% calculate yplus ahead/behind the step
wallval = 0.0 # -3.0 # 
dy = 0.001953125 #-2.997037172317505 #   #  
# -2.997037172317505  # 0.00390625
frame = MeanFlow.PlanarData.loc[(MeanFlow.PlanarData['y']==dy
                                 ) & (MeanFlow.PlanarData['x']<0.0)]
frame1 = MeanFlow.PlanarData.loc[(MeanFlow.PlanarData['y']==wallval
                                  ) & (MeanFlow.PlanarData['x']<0.0)]
rho = frame1['<rho>'].values
mu = frame1['<mu>'].values
delta_u = (frame['<u>'].values-frame1['<u>'].values) / (dy - wallval)
tau = mu * delta_u
u_tau = np.sqrt(np.abs(tau / rho))
yplus = (dy - wallval) * u_tau * rho / mu
x = frame['x'].values
res = np.vstack((x, yplus))
frame2 = pd.DataFrame(data=res.T, columns=['x', 'yplus'])
frame2.to_csv(pathM + 'YPLUS1.dat',
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

# %%############################################################################
"""
    Compute BL edge & Gortler number
"""
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
    delta[i] = va.BLThickness(y0.values, u0.values)
    delta_star[i] = va.BLThickness(y0.values, u0.values,
                                   rho0.values, opt='displacement')
    theta[i] = va.BLThickness(
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
# gortler = va.Gortler(1.3718e7, xd, delta1, theta)
radius = va.Radius(xd[:-1], delta_fit[:-1])
# radius = va.Radius(xd[:-1], yd[:-1])
gortler = va.GortlerTur(theta[:-1], delta_star[:-1], radius)

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

# %%############################################################################
"""
    skin friction & pressure coefficiency/turbulent kinetic energy along streamwise
"""
# %% Plot streamwise skin friction
MeanFlow.copy_meanval()
WallFlow = MeanFlow.PlanarData.groupby("x", as_index=False).nth(1)
# WallFlow = WallFlow[WallFlow.x != -0.0078125]
mu = va.viscosity(13718, WallFlow["T"])
Cf = va.skinfriction(mu, WallFlow["u"], WallFlow["walldist"]).values
ind = np.where(Cf[:] < 0.005)
# fig2, ax2 = plt.subplots(figsize=(5, 2.5))
fig = plt.figure(figsize=(6.4, 2.6))
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
ax2.axvline(x=9.2, color="gray", linestyle="--", linewidth=1.0)
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
ax2.annotate("(a)", xy=(-0.12, 1.04), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF+'Cf.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# % pressure coefficiency
fa = 1.7 * 1.7 * 1.4
# fig3, ax3 = plt.subplots(figsize=(5, 2.5))
ax3 = fig.add_subplot(122)
ax3.plot(WallFlow["x"], WallFlow["p"] * fa, "k", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$\langle p_w \rangle/p_{\infty}$", fontsize=textsize)
ax3.set_xlim([-20.0, 40.0])
ax3.set_ylim([0.25, 1.25])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
ax3.axvline(x=9.2, color="gray", linestyle="--", linewidth=1.0)
ax3.grid(b=True, which="both", linestyle=":")
ax3.annotate("(b)", xy=(-0.16, 1.04), xycoords='axes fraction',
              fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
 # plt.savefig(path2 + "Cp.svg", dpi=300)
plt.savefig(pathF + "CfCp.svg", dpi=300)
plt.show()

# % turbulent kinetic energy
#tke = va.tke(WallFlow).values
#ax3 = fig.add_subplot(122)
#matplotlib.rc("font", size=textsize)
#ax3.plot(xwall[ind], tke[ind], "k", linewidth=1.5)
#ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
#ax3.set_ylabel(r"$k/u^2_\infty$", fontsize=textsize)
#ax3.set_xlim([-20.0, 40.0])
## ax3.set_ylim([-0.001, 0.002])
#ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
#ax3.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
#ax3.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
#ax3.grid(b=True, which="both", linestyle=":")
#ax3.yaxis.offsetText.set_fontsize(numsize)
#ax3.annotate("(b)", xy=(-0.12, 1.04), xycoords='axes fraction',
#             fontsize=numsize)
#plt.tick_params(labelsize=numsize)
#plt.subplots_adjust(wspace=0.3)
#plt.savefig(pathF+'CfTk.svg', bbox_inches='tight', pad_inches=0.1)
#plt.show()

# %%############################################################################
"""
    Intermittency
"""
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
    gamma[j] = va.Intermittency(sigma, p0, Snapshots[j, :], timezone)
    alpha[j] = va.Alpha3(Snapshots[j, :])

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


# %%############################################################################
"""
    POD convergence
"""
# %% POD convergence
fig, ax = plt.subplots(figsize=(5, 3))
data = np.loadtxt(pathF + "POD/PODConvergence.dat", skiprows=1)
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
    pathF + "POD/PODConvergence.svg", bbox_inches="tight", pad_inches=0.1
)
plt.show()

# %%############################################################################
"""
    Vortex Wavelength
"""
# %% Vortex Wavelength along streamwise direction
time = "779.50"
wave = pd.read_csv(
    pathF + "L" + time + ".txt",
    sep="\t",
    index_col=False,
    skipinitialspace=True,
    keep_default_na=False,
)
meanwave = pd.read_csv(
    pathF + "LMean.txt",
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
    pathF + "Wavelen" + time + ".svg", bbox_inches="tight", pad_inches=0.1
)
plt.show()




