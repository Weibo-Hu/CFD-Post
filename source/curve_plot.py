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


# %% data path settings
# host = "/run/user/1000/gvfs/sftp:host=cartesius.surfsara.nl,user="
# path = host + "weibohu/nfs/home6/weibohu/weibo/FFS_M1.7TB/"
# path = "E:/cases/wavy_1009/"
path = "/mnt/work/cases/wavy_1009/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathSL = path + "Slice/"

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
cm2in = 1 / 2.54
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
x1x2 = [600, 1100]
StepHeight = 0.0
MeanFlow = pf()
# MeanFlow.load_data(path + 'inca_out/')
MeanFlow.load_meanflow(path)
# %% rescaled if necessary
# MeanFlow.add_walldist(StepHeight)
lh = 1.0
MeanFlow.rescale(lh)
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
corner1 = (x < 25) & (y < 0.0)
corner2 = (x > 110) & (y < 0.0)
corner = corner1 | corner2
# %% Load laminar data for comparison
path0 = "/mnt/work/Fourth/FFS_M1.7TB1/"
path0F, path0P, path0M, path0S, path0T, path0I = p2p.create_folder(path0)
MeanFlow0 = pf()
MeanFlow0.load_meanflow(path0)
MeanFlow0.add_walldist(StepHeight)
MeanFlow0.copy_meanval()
MeanFlow0.rescale(lh)
# %%############################################################################
"""
    boundary layer profile along streamwise direction
"""
# %% plot BL profile along streamwise
MeanFlow.copy_meanval()
fig, ax = plt.subplots(1, 9, figsize=(14 * cm2in, 4 * cm2in), dpi=500)
fig.subplots_adjust(hspace=0.5, wspace=0.18)
matplotlib.rc("font", size=numsize)
title = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$"]
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
xcoord = np.array([20, 34, 53, 71.5, 82, 95, 200, 300, 360])
for i in range(np.size(xcoord)):
    df = MeanFlow.yprofile("x", xcoord[i])
    y0 = df["walldist"]
    q0 = df["u"]
    if xcoord[i] == 0.0:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind], y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    else:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind], y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    if i != 0:
        ax[i].set_yticklabels("")
        ax[i].set_title(r"${}$".format(xcoord[i]), fontsize=numsize - 2)
    ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].tick_params(axis="both", which="major", labelsize=numsize)
    ax[i].grid(visible=True, which="both", linestyle=":")
ax[0].set_title(r"$x={}$".format(xcoord[0]), fontsize=numsize - 2)
ax[0].set_ylabel(r"$\Delta y$", fontsize=textsize)
ax[4].set_xlabel(r"$u /u_\infty$", fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(pathF + "BLProfile.svg", bbox_inches="tight", pad_inches=0.1)

# %%############################################################################
"""
    Compare the law of wall: theoretical by van Driest, experiments, LES
"""
# %% velocity profile, computation
# df = p2p.ReadAllINCAResults(path + 'inca_out/', path + 'inca_out/',
#                             FileName=path + 'inca_out/MeanFlow.szplt',
#                             SpanAve=True, OutFile='MeanFlow')
# path = "/media/weibo/VID2/FlatTur_coarse/"
# df = np.loadtxt(path + 'boundary_input_000001.dat')
# varnm = ['x', 'y', 'z', '<u>', '<v>', '<w>', '<rho>', '<T>',
#          '<u`u`>', '<v`v`>', '<w`w`>', '<u`v`>', 'u`w`', 'v`w`']
# BLProf = pd.DataFrame(data=df.values[2:, :], columns=varnm)
# grouped = BLProf.groupby('y')
# BLProf = grouped.mean().reset_index()
# BLProf['walldist'] = BLProf['y']
# BLProf['<mu>'] = va.viscosity(13500, BLProf['<T>'])
# %% comparison

path1 = "/media/weibo/IM2/FFS_grid/GX/"
pathM1 = path1 + "MeanFlow/"
MeanFlow1 = pf()
MeanFlow1.load_meanflow(path1)
MeanFlow1.add_walldist(StepHeight)
MeanFlow1.copy_meanval()

path2 = "/media/weibo/IM2/FFS_M1.7TB1/"
pathM2 = path2 + "MeanFlow/"
MeanFlow2 = pf()
MeanFlow2.load_meanflow(path2)
MeanFlow2.add_walldist(StepHeight)
MeanFlow2.copy_meanval()


# %%
x1 = -50.0
BLProf1 = MeanFlow1.yprofile("x", x1)  # -30.0
CalUPlus1 = va.direst_transform(BLProf1, option="mean")
BLProf1 = MeanFlow1.yprofile("x", x1)
u_tau1 = va.u_tau(BLProf1, option="mean", grad=True)
xi1 = np.sqrt(BLProf1["<rho>"] / BLProf1["<rho>"].values[1])
uu1 = np.sqrt(BLProf1["<u`u`>"]) / u_tau1 * xi1
vv1 = np.sqrt(BLProf1["<v`v`>"]) / u_tau1 * xi1
ww1 = np.sqrt(BLProf1["<w`w`>"]) / u_tau1 * xi1
uv1 = BLProf1["<u`v`>"] / u_tau1 ** 2 * xi1 ** 2
BLProf2 = MeanFlow2.yprofile("x", x1)  # -30.0
CalUPlus2 = va.direst_transform(BLProf2, option="mean")
BLProf2 = MeanFlow2.yprofile("x", x1)
u_tau2 = va.u_tau(BLProf2, option="mean", grad=True)
xi2 = np.sqrt(BLProf2["<rho>"] / BLProf2["<rho>"].values[1])
uu2 = np.sqrt(BLProf2["<u`u`>"]) / u_tau2 * xi2
vv2 = np.sqrt(BLProf2["<v`v`>"]) / u_tau2 * xi2
ww2 = np.sqrt(BLProf2["<w`w`>"]) / u_tau2 * xi2
uv2 = BLProf2["<u`v`>"] / u_tau2 ** 2 * xi2 ** 2

# %% velocity profile, computation
x0 = -50.0
incomp = True
# results from LES
MeanFlow.copy_meanval()
BLProf = MeanFlow.yprofile("x", x0)
u_tau = va.u_tau(BLProf, option="mean")
mu_inf = BLProf["<mu>"].values[-1]
delta, u_inf = va.bl_thickness(BLProf["walldist"], BLProf["<u>"])
delta_star, u_inf, rho_inf = va.bl_thickness(
    BLProf["walldist"], BLProf["<u>"], rho=BLProf["<rho>"], opt="displacement"
)
theta, u_inf, rho_inf = va.bl_thickness(
    BLProf["walldist"], BLProf["<u>"], rho=BLProf["<rho>"], opt="momentum"
)
Re_theta = rho_inf * u_inf * theta / mu_inf
Re_delta_star = rho_inf * u_inf * delta_star / mu_inf
Re_tau = BLProf["<rho>"].values[0] * u_tau / BLProf["<mu>"].values[0] * delta
print("Re_tau=", Re_tau)
print("Re_theta=", Re_theta)
print("Re_delta*=", Re_delta_star)
CalUPlus = va.direst_transform(BLProf, option="mean", grad=True)
# results from theory by van Driest
StdUPlus1, StdUPlus2 = va.std_wall_law()
# results from known DNS
Re_theta = 1000  # 1000  # 800 # 1400 #
ExpUPlus = va.ref_wall_law(Re_theta)[0]
# plot velocity profile
fig = plt.figure(figsize=(6.4, 2.6))
ax = fig.add_subplot(121)
matplotlib.rc("font", size=numsize)
ax.plot(
    StdUPlus1[:, 0],
    StdUPlus1[:, 1],
    "k-.",
    StdUPlus2[:, 0],
    StdUPlus2[:, 1],
    "k-.",
    linewidth=1.5,
)
ax.scatter(
    ExpUPlus[:, 0],
    ExpUPlus[:, 1],
    linewidth=0.8,
    s=16.0,
    facecolor="none",
    edgecolor="gray",
)
# spl = splrep(CalUPlus[:, 0], CalUPlus[:, 1], s=0.1)
# uplus = splev(CalUPlus[:, 0], spl)
# ax.plot(CalUPlus[:, 0], uplus, 'k', linewidth=1.5)
# ax.scatter(CalUPlus[:, 0], CalUPlus[:, 1], s=15)
ax.plot(CalUPlus[:, 0], CalUPlus[:, 1], "k", linewidth=1.0)
ax.plot(CalUPlus1[:, 0], CalUPlus1[:, 1], "k:", linewidth=1.0)
ax.plot(CalUPlus2[:, 0], CalUPlus2[:, 1], "k--", linewidth=1.0)
ax.set_xscale("log")
ax.set_xlim([0.5, 2000])
ax.set_ylim([0, 30])
ax.set_ylabel(r"$\langle u_{VD}^+ \rangle$", fontsize=textsize)
ax.set_xlabel(r"$y^+$", fontsize=textsize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
ax.annotate("(a)", xy=(-0.16, 0.98), xycoords="axes fraction", fontsize=numsize)
plt.tick_params(labelsize="medium")
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
# plt.tick_params(labelsize=numsize)
# plt.savefig(
#    pathF + 'WallLaw' + str(x0) + '.svg', bbox_inches='tight', pad_inches=0.1)
# plt.show()

#  Reynolds stresses in Morkovin scaling
# results from known DNS
ExpUPlus, ExpUVPlus, ExpUrmsPlus, ExpVrmsPlus, ExpWrmsPlus, XI = va.ref_wall_law(
    Re_theta
)
if incomp == True:
    XI = 1.0
# results from current LES
BLProf = MeanFlow.yprofile("x", x0)
xi = np.sqrt(BLProf["<rho>"] / BLProf["<rho>"].values[1])
uu = np.sqrt(BLProf["<u`u`>"]) / u_tau * xi
vv = np.sqrt(BLProf["<v`v`>"]) / u_tau * xi
ww = np.sqrt(BLProf["<w`w`>"]) / u_tau * xi
uv = BLProf["<u`v`>"] / u_tau ** 2 * xi ** 2
# spl = splrep(CalUPlus[:, 0], uu, s=0.1)
# uu = splev(CalUPlus[:, 0], spl)
# plot Reynolds stress
# fig, ax = plt.subplots(figsize=(3.2, 3.2))
Xi = 1.0
ax2 = fig.add_subplot(122)
ax2.scatter(
    ExpUrmsPlus[:, 0],
    ExpUrmsPlus[:, 1] * XI,
    linewidth=0.8,
    s=18,
    facecolor="none",
    edgecolor="gray",
)
ax2.scatter(
    ExpVrmsPlus[:, 0],
    ExpVrmsPlus[:, 1] * XI,
    linewidth=0.8,
    s=18,
    facecolor="none",
    edgecolor="gray",  # "r",
)
ax2.scatter(
    ExpWrmsPlus[:, 0],
    ExpWrmsPlus[:, 1] * XI,
    linewidth=0.8,
    s=18,
    facecolor="none",
    edgecolor="gray",  # "b",
)
ax2.scatter(
    ExpUVPlus[:, 0],
    ExpUVPlus[:, 1] * XI,
    linewidth=0.8,
    s=18,
    facecolor="none",
    edgecolor="gray",  # "gray",
)
ax2.plot(CalUPlus[:, 0], uu[1:], "k", linewidth=1.0)
ax2.plot(CalUPlus[:, 0], vv[1:], "k", linewidth=1.0)
ax2.plot(CalUPlus[:, 0], ww[1:], "k", linewidth=1.0)
ax2.plot(CalUPlus[:, 0], uv[1:], "k", linewidth=1.0)
ax2.plot(CalUPlus1[:, 0], uu1[1:], "k:", linewidth=1.0)
ax2.plot(CalUPlus1[:, 0], vv1[1:], "k:", linewidth=1.0)
ax2.plot(CalUPlus1[:, 0], ww1[1:], "k:", linewidth=1.0)
ax2.plot(CalUPlus1[:, 0], uv1[1:], "k:", linewidth=1.0)
ax2.plot(CalUPlus2[:, 0], uu2[1:], "k--", linewidth=1.0)
ax2.plot(CalUPlus2[:, 0], vv2[1:], "k--", linewidth=1.0)
ax2.plot(CalUPlus2[:, 0], ww2[1:], "k--", linewidth=1.0)
ax2.plot(CalUPlus2[:, 0], uv2[1:], "k--", linewidth=1.0)
ax2.set_xscale("log")
ax2.set_ylim([-1.5, 3.5])
ax2.set_xlim([1, 2000])
ax2.set_ylim([-1.5, 3.5])
vna1 = r"$u^\prime u^\prime$"
vna2 = r"$w^\prime w^\prime$"
vna3 = r"$v^\prime v^\prime$"
vna4 = r"$u^\prime v^\prime$"
ax2.text(3.5, 3.0, vna1, fontsize=numsize + 1)
ax2.text(12, 1.5, vna2, fontsize=numsize + 1)
ax2.text(65, 0.5, vna3, fontsize=numsize + 1)
ax2.text(55, -0.6, vna4, fontsize=numsize + 1)
ax2.set_ylabel(
    r"$\sqrt{\langle u^{\prime}_i u^{\prime}_j\rangle^{+}}$", fontsize=textsize
)
ax2.set_xlabel(r"$y^+$", fontsize=textsize)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.grid(b=True, which="both", linestyle=":")
ax2.annotate("(b)", xy=(-0.18, 0.98), xycoords="axes fraction", fontsize=numsize)
plt.subplots_adjust(wspace=0.8)
plt.tick_params(labelsize="medium")
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.3)
plt.savefig(
    pathF + "ReynoldStress" + str(x0) + ".svg", bbox_inches="tight", pad_inches=0.1,
)
plt.show()

# %%############################################################################
"""
    y+ along streamwise
"""
# %% calculate yplus ahead/behind the step


def yplus(MeanFlow, dy, wallval, opt):
    if opt == 1:
        TempFlow = MeanFlow.PlanarData.loc[MeanFlow.PlanarData["x"] < 5.0]
    elif opt == 2:
        TempFlow = MeanFlow.PlanarData.loc[MeanFlow.PlanarData["x"] > 100.0]
    frame = TempFlow.loc[np.round(TempFlow["y"], 6) == np.round(dy, 6)]
    frame1 = TempFlow.loc[np.round(TempFlow["y"], 6) == np.round(wallval, 6)]
    x = frame["x"].values
    rho = frame1["<rho>"].values
    mu = frame1["<mu>"].values
    delta_u = (frame["<u>"].values - frame1["<u>"].values) / (dy - wallval)
    tau = mu * delta_u
    u_tau = np.sqrt(np.abs(tau / rho))
    y_plus = (dy - wallval) * u_tau * rho / mu
    frame.assign(yplus=y_plus)
    return (x, y_plus, frame)


# 0.002300256 upsteam the step
x1, yplus1, frame1 = yplus(MeanFlow, 0.033333, 0.0, opt=1)
x2, yplus2, frame2 = yplus(MeanFlow, 0.033333, 0.0, opt=2)  # 3.001953125 downstream
res = np.vstack((np.hstack((x1, x2)), np.hstack((yplus1, yplus2))))
frame3 = pd.DataFrame(data=res.T, columns=["x", "yplus"])
frame3.to_csv(pathM + "YPLUS.dat", index=False, float_format="%1.8e", sep=" ")

# %% plot yplus along streamwise
yp = pd.read_csv(pathM + "YPLUS.dat", sep=" ", index_col=False, skipinitialspace=True)
fig3, ax3 = plt.subplots(figsize=(6.4, 3.0))
ax3.plot(yp["x"], yp["yplus"], "k", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$\Delta y^{+}$", fontsize=textsize)
ax3.set_xlim([0.0, 360])
ax3.set_ylim([0.0, 2.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5)
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "yplus.svg", dpi=300)
plt.show()
# %% plot u along streamwise
fig3, ax3 = plt.subplots(figsize=(6.4, 3.0))
ax3.plot(frame1["x"], frame1["<u>"], "k", linewidth=1.5)
ax3.plot(frame2["x"], frame2["<u>"], "k", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$u / u_\infty$", fontsize=textsize)
ax3.set_xlim([-70.0, 40.0])
# ax3.set_ylim([0.0, 2.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.5)
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "u_stream2.svg", dpi=300)
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
    df = MeanFlow.yprofile("x", xd[i])
    y0 = df["y"]
    u0 = df["<u>"]
    rho0 = df["<rho>"]
    delta[i] = va.bl_thickness(y0.values, u0.values)[0]
    delta_star[i] = va.bl_thickness(
        y0.values, u0.values, rho=rho0.values, opt="displacement"
    )[0]
    theta[i] = va.bl_thickness(y0.values, u0.values, rho=rho0.values, opt="momentum")[0]

stream = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
stream[:, -1] = stream[:, -1] + 3.0
func = interp1d(stream[:, 0], stream[:, 1], bounds_error=False, fill_value=0.0)
yd = func(xd)
xmax = np.max(stream[:, 0])


# fit curve
def func(t, A, B, C, D):
    return A * t ** 3 + B * t ** 2 + C * t + D


popt, pcov = DataPost.fit_func(func, xd, delta, guess=None)
A, B, C, D = popt


def fitfunc(t):
    return A * t ** 3 + B * t ** 2 + C * t + D


delta_fit = fitfunc(xd)

popt, pcov = DataPost.fit_func(func, xd, yd, guess=None)
A, B, C, D = popt


def fitfunc(t):
    return A * t ** 3 + B * t ** 2 + C * t + D


yd = fitfunc(xd)
# gortler = va.Gortler(1.3718e7, xd, delta1, theta)
radius = va.radius(xd[:-1], delta_fit[:-1])
# radius = va.Radius(xd[:-1], yd[:-1])
gortler = va.gortler_tur(theta[:-1], delta_star[:-1], radius)

fig3, ax3 = plt.subplots(figsize=(5, 2.5))
ax3.plot(xd, delta, "k-", linewidth=1.5)  # boundary layer
ax3.plot(xd, delta_fit, "k--", linewidth=1.5)
ax3.plot(xd, yd, "k--", linewidth=1.5)  # bubble line
ax3.plot(xd, theta, "k:", linewidth=1.5)  # momentum
# ax3.plot(xd[:-1], radius, 'k:', linewidth=1.5)
ax3.plot(stream[:, 0], stream[:, 1], "b--", linewidth=1.5)  # bubble line
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
fig = plt.figure(figsize=(3.2, 3.0))
matplotlib.rc("font", size=textsize)
ax2 = fig.add_subplot(111)
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
plt.savefig(pathF + "delta_radius.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% plot figure for Gortler number
fig3, ax3 = plt.subplots(figsize=(3.2, 3.0))
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
#
# skin friction & pressure coefficiency/turbulent kinetic energy along streamwise
#
# %% compare
temp_path = path + "snapshots/S_10a/"
MeanFlow1 = pf()
MeanFlow1.load_data(pathSL, FileList=temp_path + "TP_2D_S_10_01100.00.plt")
MeanFlow1.add_walldist(StepHeight)
df = MeanFlow1.PlanarData
grouped = df.groupby(["x", "y", "z"])
df = grouped.mean().reset_index()
WallFlow1 = MeanFlow1.PlanarData.groupby("x", as_index=False).nth(1)
mu1 = va.viscosity(13718, WallFlow1["T"])
Cf1 = va.skinfriction(mu1, WallFlow1["u"], WallFlow1["walldist"]).values
ind1 = np.where(Cf1[:] < 0.008)
xwall1 = WallFlow1["x"].values
#
temp_path = path + "snapshots/S_10a/"
MeanFlow2 = pf()
MeanFlow2.load_data(pathSL, FileList=temp_path + "TP_2D_S_10_01300.00.plt")
MeanFlow2.add_walldist(StepHeight)
WallFlow2 = MeanFlow2.PlanarData.groupby("x", as_index=False).nth(1)
mu2 = va.viscosity(13718, WallFlow1["T"])
Cf2 = va.skinfriction(mu2, WallFlow2["u"], WallFlow2["walldist"]).values
ind2 = np.where(Cf1[:] < 0.006)
xwall2 = WallFlow2["x"].values
# %% comparison with laminar case
WallFlow0 = MeanFlow0.PlanarData.groupby("x", as_index=False).nth(1)
if np.size(np.unique(WallFlow0["y"])) > 2:
    maxy = np.max(WallFlow0["y"])
    WallFlow0 = WallFlow0.drop(WallFlow0[WallFlow0["y"] == maxy].index)
mu0 = va.viscosity(13718, WallFlow0["T"])
Cf0 = va.skinfriction(mu0, WallFlow0["u"], WallFlow0["walldist"]).values
ind0 = np.where(Cf0[:] < 0.008)
xwall0 = WallFlow0["x"].values
# %%
# WallFlow = MeanFlow.PlanarData[np.round(
#     MeanFlow.PlanarData["y"], 6) == 3.3333e-02]
# FirstLev = WallFlow[["x", "y"]]
xy_val = va.wall_line(MeanFlow.PlanarData, path, mask=corner, val=0.03125)  # 0.033333)
FirstLev = pd.DataFrame(data=xy_val, columns=["x", "y"])
xx = np.arange(0.0, 400.0, 0.25)
FirstLev = FirstLev[FirstLev["x"].isin(xx)]
FirstLev.to_csv(pathM + "FirstLev.dat", index=False, float_format="%9.6f")
# %% wall buondary


def add_variable(col, df):
    for i in range(np.size(col)):
        val = griddata(
            (MeanFlow.x, MeanFlow.y),
            MeanFlow.PlanarData[col[i]],
            (df.x, df.y),
            method="cubic",
        )
        df[col[i]] = val
    return df


MeanFlow.copy_meanval()
wavy = pd.read_csv(pathM + "FirstLev.dat", delimiter=",", skipinitialspace=True)
add_variable(["p", "T"], wavy)
add_variable("u", wavy)
add_variable("v", wavy)
mu = va.viscosity(25600, wavy["T"])
Cf = va.skinfriction(mu, wavy["u"], 0.03125).values  # 0.0333333
ddx = np.diff(wavy["x"])
ddy = np.diff(wavy["y"])
tang = np.transpose(np.vstack((ddx, ddy)))
xposi = np.array([1, 0])
modules = np.linalg.norm(tang, axis=1) * np.linalg.norm(xposi)
cos_val = np.dot(tang, xposi) / modules
sin_val = np.cross(tang, xposi) / modules
wavy1 = wavy.drop(index=0)
# ddy[np.where(ddy == 0.0)] = 0.0166667
delta_y = 0.03125 * np.abs(cos_val)
delta_x = 0.03125 * np.abs(sin_val)
Cf = (
    va.skinfriction(mu[1:], wavy1["u"], delta_y) * cos_val
    + va.skinfriction(mu[1:], wavy1["v"], delta_x) * sin_val
)

# %% Plot streamwise skin friction
# WallFlow = MeanFlow.PlanarData.groupby("x", as_index=False).nth(1)
# if np.size(np.unique(WallFlow["y"])) > 2:
#     maxy = np.max(WallFlow["y"])
#     WallFlow = WallFlow.drop(WallFlow[WallFlow["y"] == maxy].index)
# mu = va.viscosity(13718, WallFlow["T"])
# Cf = va.skinfriction(mu, WallFlow["u"], WallFlow["walldist"]).values
# ind = np.where(Cf[:] < 0.008)
fig2, ax2 = plt.subplots(figsize=(15 * cm2in, 5 * cm2in))
# fig = plt.figure(figsize=(6.4, 4.6))
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=numsize)
# xwall = WallFlow["x"].values
ax2.plot(wavy1["x"], Cf, "k", linewidth=1.5)
# ax2.plot(xwall0[ind0], Cf0[ind0], "b--", linewidth=1.5)
# ax2.plot(xwall1[ind1], Cf1[ind1],
#          color='gray', linestyle=':', linewidth=1.2) #
# ax2.plot(xwall2[ind2], Cf2[ind2],
#          color='gray', linestyle=':', linewidth=1.2) #
ax2.set_xlabel(r"$x/h$", fontsize=textsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=textsize)
ax2.set_xlim([0, 400])
# ax2.set_ylim([-0.0001, 0.0003])
# ax2.set_yticks(np.arange(-0.002, 0.008, 0.002))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=-25, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF+'Cf_comp.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# %% zoom part
ax1 = fig.add_subplot(212)
matplotlib.rc("font", size=numsize)
ax1.plot(xwall[ind], Cf[ind], "k", linewidth=1.5)
ax1.plot(xwall0[ind0], Cf0[ind0], "b--", linewidth=1.5)
ax1.set_xlabel(r"$x/h$", fontsize=textsize)
ax1.set_ylabel(r"$\langle C_f \rangle$", fontsize=textsize)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.set_xlim([-1.0, 1.0])
ax1.set_ylim([-0.003, 0.005])
ax1.axvline(x=0.16, color="gray", linestyle=":", linewidth=1.5)
ax1.axvline(x=0.23, color="blue", linestyle=":", linewidth=1.5)
ax1.grid(b=True, which="both", linestyle=":")
ax1.annotate("(b)", xy=(-0.09, 0.98), xycoords="axes fraction", fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(pathF + "Cf_comp.svg")

# %% pressure coefficiency
Ma = 6.0
fa = Ma * Ma * 1.4
wavy = pd.read_csv(pathM + "FirstLev.dat", delimiter=",", skipinitialspace=True)
wavy["p"] = griddata(
    (MeanFlow.x, MeanFlow.y), MeanFlow.p, (wavy.x, wavy.y), method="cubic"
)
# %%
fig3, ax3 = plt.subplots(figsize=(15 * cm2in, 5 * cm2in))
ax3.plot(wavy["x"], wavy["p"] * fa, "k", linewidth=1.5)
# ax3.plot(WallFlow0["x"], WallFlow0["p"] * fa, "b--", linewidth=1.5)
# p_ref = np.loadtxt(pathM + "PressureRef1.dat", skiprows=4)
# lref = 1
# ax3.scatter(p_ref[:11, 0]/lref, p_ref[:11, 1],
#             linewidth=0.8,
#             s=20.0,
#             edgecolor="gray",
#             facecolor="none")
# ax3.plot(WallFlow1['x'], WallFlow1['p']*fa,
#          color='gray', linestyle=':', linewidth=1.2) #
# ax3.plot(WallFlow2['x'], WallFlow2['p']*fa,
#          color='gray', linestyle=':', linewidth=1.2) #
ax3.set_xlabel(r"$x$", fontsize=textsize)
ax3.set_ylabel(r"$\langle p_w \rangle/p_{\infty}$", fontsize=textsize)
ax3.set_xlim([0, 400])
# ax3.set_ylim([0.98, 0.99])
# ax3.set_yticks(np.arange(0.2, 2.6, 0.4))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=-25, color="gray", linestyle=":", linewidth=1.5)
ax3.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax3.grid(visible=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
# plt.savefig(pathF + "Cp_comp.svg", dpi=300)
plt.savefig(pathF + "Cp_comp.svg", dpi=300)
plt.show()

# % turbulent kinetic energy


# tke = va.tke(WallFlow).values
# ax3 = fig.add_subplot(122)
# matplotlib.rc("font", size=textsize)
# ax3.plot(xwall[ind], tke[ind], "k", linewidth=1.5)
# ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
# ax3.set_ylabel(r"$k/u^2_\infty$", fontsize=textsize)
# ax3.set_xlim([-20.0, 40.0])
## ax3.set_ylim([-0.001, 0.002])
# ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax3.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)delta
# ax3.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
# ax3.grid(b=True, which="both", linestyle=":")
# ax3.yaxis.offsetText.set_fontsize(numsize)
# ax3.annotate("(b)", xy=(-0.12, 1.04), xycoords='axes fraction',
#             fontsize=numsize)
# plt.tick_params(labelsize=numsize)
# plt.subplots_adjust(wspace=0.3)
# plt.savefig(pathF+'CfTk.svg', bbox_inches='tight', pad_inches=0.1)
# plt.show()

# %%############################################################################
"""
    Intermittency
"""
# %% Load data for Computing intermittency factor
InFolder = pathSL + "TP_2D_Z_03/"
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
timezone = np.arange(975, 1064.00 + 0.25, 0.25)
dt = 0.5
for j in range(np.size(Snapshots[:, 0])):
    gamma[j] = va.intermittency(sigma, p0, Snapshots[j, :], timezone)
    alpha[j] = va.alpha3(Snapshots[j, :])

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
fig3, ax3 = plt.subplots(figsize=(6.4, 2.6))
ax3.plot(xzone, gamma, "k--")
ax3.plot(xarr, yarr, "k-")  # fitting values
ax3.scatter(x1, ga1, c="black", marker="o", s=10)  # universal distribution
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
plt.savefig(pathF + "POD/PODConvergence.svg", bbox_inches="tight", pad_inches=0.1)
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
plt.savefig(pathF + "Wavelen" + time + ".svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$u^\prime/u_\infty$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Wavelen" + time + ".svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.savefig(pathF + "Wavelen" + time + ".svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
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


# %% data path settings
# host = "/run/user/1000/gvfs/sftp:host=cartesius.surfsara.nl,user="
# path = host + "weibohu/nfs/home6/weibohu/weibo/FFS_M1.7TB/"
# path = "E:/cases/wavy_1009/"
path = "/mnt/work/cases/wavy_1009/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathSL = path + "Slice/"

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
cm2in = 1 / 2.54
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
x1x2 = [600, 1100]
StepHeight = 0.0
MeanFlow = pf()
# MeanFlow.load_data(path + 'inca_out/')
MeanFlow.load_meanflow(path)
# %% rescaled if necessary
# MeanFlow.add_walldist(StepHeight)
lh = 1.0
MeanFlow.rescale(lh)
x, y = np.meshgrid(np.unique(MeanFlow.x), np.unique(MeanFlow.y))
corner1 = (x < 25) & (y < 0.0)
corner2 = (x > 110) & (y < 0.0)
corner = corner1 | corner2
# %% Load laminar data for comparison
path0 = "/mnt/work/Fourth/FFS_M1.7TB1/"
path0F, path0P, path0M, path0S, path0T, path0I = p2p.create_folder(path0)
MeanFlow0 = pf()
MeanFlow0.load_meanflow(path0)
MeanFlow0.add_walldist(StepHeight)
MeanFlow0.copy_meanval()
MeanFlow0.rescale(lh)
# %%############################################################################
"""
    boundary layer profile along streamwise direction
"""
# %% plot BL profile along streamwise
MeanFlow.copy_meanval()
fig, ax = plt.subplots(1, 9, figsize=(14 * cm2in, 4 * cm2in), dpi=500)
fig.subplots_adjust(hspace=0.5, wspace=0.18)
matplotlib.rc("font", size=numsize)
title = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$"]
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
xcoord = np.array([20, 34, 53, 71.5, 82, 95, 200, 300, 360])
for i in range(np.size(xcoord)):
    df = MeanFlow.yprofile("x", xcoord[i])
    y0 = df["walldist"]
    q0 = df["u"]
    if xcoord[i] == 0.0:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind], y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    else:
        ind = np.where(y0 >= 0.0)[0]
        ax[i].plot(q0[ind], y0[ind], "k-")
        ax[i].set_ylim([0, 6])
    if i != 0:
        ax[i].set_yticklabels("")
        ax[i].set_title(r"${}$".format(xcoord[i]), fontsize=numsize - 2)
    ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].tick_params(axis="both", which="major", labelsize=numsize)
    ax[i].grid(visible=True, which="both", linestyle=":")
ax[0].set_title(r"$x={}$".format(xcoord[0]), fontsize=numsize - 2)
ax[0].set_ylabel(r"$\Delta y$", fontsize=textsize)
ax[4].set_xlabel(r"$u /u_\infty$", fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(pathF + "BLProfile.svg", bbox_inches="tight", pad_inches=0.1)

# %%############################################################################
"""
    Compare the law of wall: theoretical by van Driest, experiments, LES
"""
# %% velocity profile, computation
# df = p2p.ReadAllINCAResults(path + 'inca_out/', path + 'inca_out/',
#                             FileName=path + 'inca_out/MeanFlow.szplt',
#                             SpanAve=True, OutFile='MeanFlow')
# path = "/media/weibo/VID2/FlatTur_coarse/"
# df = np.loadtxt(path + 'boundary_input_000001.dat')
# varnm = ['x', 'y', 'z', '<u>', '<v>', '<w>', '<rho>', '<T>',
#          '<u`u`>', '<v`v`>', '<w`w`>', '<u`v`>', 'u`w`', 'v`w`']
# BLProf = pd.DataFrame(data=df.values[2:, :], columns=varnm)
# grouped = BLProf.groupby('y')
# BLProf = grouped.mean().reset_index()
# BLProf['walldist'] = BLProf['y']
# BLProf['<mu>'] = va.viscosity(13500, BLProf['<T>'])
# %% comparison

path1 = "/media/weibo/IM2/FFS_grid/GX/"
pathM1 = path1 + "MeanFlow/"
MeanFlow1 = pf()
MeanFlow1.load_meanflow(path1)
MeanFlow1.add_walldist(StepHeight)
MeanFlow1.copy_meanval()

path2 = "/media/weibo/IM2/FFS_M1.7TB1/"
pathM2 = path2 + "MeanFlow/"
MeanFlow2 = pf()
MeanFlow2.load_meanflow(path2)
MeanFlow2.add_walldist(StepHeight)
MeanFlow2.copy_meanval()


# %%
x1 = -50.0
BLProf1 = MeanFlow1.yprofile("x", x1)  # -30.0
CalUPlus1 = va.direst_transform(BLProf1, option="mean")
BLProf1 = MeanFlow1.yprofile("x", x1)
u_tau1 = va.u_tau(BLProf1, option="mean", grad=True)
xi1 = np.sqrt(BLProf1["<rho>"] / BLProf1["<rho>"].values[1])
uu1 = np.sqrt(BLProf1["<u`u`>"]) / u_tau1 * xi1
vv1 = np.sqrt(BLProf1["<v`v`>"]) / u_tau1 * xi1
ww1 = np.sqrt(BLProf1["<w`w`>"]) / u_tau1 * xi1
uv1 = BLProf1["<u`v`>"] / u_tau1 ** 2 * xi1 ** 2
BLProf2 = MeanFlow2.yprofile("x", x1)  # -30.0
CalUPlus2 = va.direst_transform(BLProf2, option="mean")
BLProf2 = MeanFlow2.yprofile("x", x1)
u_tau2 = va.u_tau(BLProf2, option="mean", grad=True)
xi2 = np.sqrt(BLProf2["<rho>"] / BLProf2["<rho>"].values[1])
uu2 = np.sqrt(BLProf2["<u`u`>"]) / u_tau2 * xi2
vv2 = np.sqrt(BLProf2["<v`v`>"]) / u_tau2 * xi2
ww2 = np.sqrt(BLProf2["<w`w`>"]) / u_tau2 * xi2
uv2 = BLProf2["<u`v`>"] / u_tau2 ** 2 * xi2 ** 2

# %% velocity profile, computation
x0 = -50.0
incomp = True
# results from LES
MeanFlow.copy_meanval()
BLProf = MeanFlow.yprofile("x", x0)
u_tau = va.u_tau(BLProf, option="mean")
mu_inf = BLProf["<mu>"].values[-1]
delta, u_inf = va.bl_thickness(BLProf["walldist"], BLProf["<u>"])
delta_star, u_inf, rho_inf = va.bl_thickness(
    BLProf["walldist"], BLProf["<u>"], rho=BLProf["<rho>"], opt="displacement"
)
theta, u_inf, rho_inf = va.bl_thickness(
    BLProf["walldist"], BLProf["<u>"], rho=BLProf["<rho>"], opt="momentum"
)
Re_theta = rho_inf * u_inf * theta / mu_inf
Re_delta_star = rho_inf * u_inf * delta_star / mu_inf
Re_tau = BLProf["<rho>"].values[0] * u_tau / BLProf["<mu>"].values[0] * delta
print("Re_tau=", Re_tau)
print("Re_theta=", Re_theta)
print("Re_delta*=", Re_delta_star)
CalUPlus = va.direst_transform(BLProf, option="mean", grad=True)
# results from theory by van Driest
StdUPlus1, StdUPlus2 = va.std_wall_law()
# results from known DNS
Re_theta = 1000  # 1000  # 800 # 1400 #
ExpUPlus = va.ref_wall_law(Re_theta)[0]
# plot velocity profile
fig = plt.figure(figsize=(6.4, 2.6))
ax = fig.add_subplot(121)
matplotlib.rc("font", size=numsize)
ax.plot(
    StdUPlus1[:, 0],
    StdUPlus1[:, 1],
    "k-.",
    StdUPlus2[:, 0],
    StdUPlus2[:, 1],
    "k-.",
    linewidth=1.5,
)
ax.scatter(
    ExpUPlus[:, 0],
    ExpUPlus[:, 1],
    linewidth=0.8,
    s=16.0,
    facecolor="none",
    edgecolor="gray",
)
# spl = splrep(CalUPlus[:, 0], CalUPlus[:, 1], s=0.1)
# uplus = splev(CalUPlus[:, 0], spl)
# ax.plot(CalUPlus[:, 0], uplus, 'k', linewidth=1.5)
# ax.scatter(CalUPlus[:, 0], CalUPlus[:, 1], s=15)
ax.plot(CalUPlus[:, 0], CalUPlus[:, 1], "k", linewidth=1.0)
ax.plot(CalUPlus1[:, 0], CalUPlus1[:, 1], "k:", linewidth=1.0)
ax.plot(CalUPlus2[:, 0], CalUPlus2[:, 1], "k--", linewidth=1.0)
ax.set_xscale("log")
ax.set_xlim([0.5, 2000])
ax.set_ylim([0, 30])
ax.set_ylabel(r"$\langle u_{VD}^+ \rangle$", fontsize=textsize)
ax.set_xlabel(r"$y^+$", fontsize=textsize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.grid(b=True, which="both", linestyle=":")
ax.annotate("(a)", xy=(-0.16, 0.98), xycoords="axes fraction", fontsize=numsize)
plt.tick_params(labelsize="medium")
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
# plt.tick_params(labelsize=numsize)
# plt.savefig(
#    pathF + 'WallLaw' + str(x0) + '.svg', bbox_inches='tight', pad_inches=0.1)
# plt.show()

#  Reynolds stresses in Morkovin scaling
# results from known DNS
ExpUPlus, ExpUVPlus, ExpUrmsPlus, ExpVrmsPlus, ExpWrmsPlus, XI = va.ref_wall_law(
    Re_theta
)
if incomp == True:
    XI = 1.0
# results from current LES
BLProf = MeanFlow.yprofile("x", x0)
xi = np.sqrt(BLProf["<rho>"] / BLProf["<rho>"].values[1])
uu = np.sqrt(BLProf["<u`u`>"]) / u_tau * xi
vv = np.sqrt(BLProf["<v`v`>"]) / u_tau * xi
ww = np.sqrt(BLProf["<w`w`>"]) / u_tau * xi
uv = BLProf["<u`v`>"] / u_tau ** 2 * xi ** 2
# spl = splrep(CalUPlus[:, 0], uu, s=0.1)
# uu = splev(CalUPlus[:, 0], spl)
# plot Reynolds stress
# fig, ax = plt.subplots(figsize=(3.2, 3.2))
Xi = 1.0
ax2 = fig.add_subplot(122)
ax2.scatter(
    ExpUrmsPlus[:, 0],
    ExpUrmsPlus[:, 1] * XI,
    linewidth=0.8,
    s=18,
    facecolor="none",
    edgecolor="gray",
)
ax2.scatter(
    ExpVrmsPlus[:, 0],
    ExpVrmsPlus[:, 1] * XI,
    linewidth=0.8,
    s=18,
    facecolor="none",
    edgecolor="gray",  # "r",
)
ax2.scatter(
    ExpWrmsPlus[:, 0],
    ExpWrmsPlus[:, 1] * XI,
    linewidth=0.8,
    s=18,
    facecolor="none",
    edgecolor="gray",  # "b",
)
ax2.scatter(
    ExpUVPlus[:, 0],
    ExpUVPlus[:, 1] * XI,
    linewidth=0.8,
    s=18,
    facecolor="none",
    edgecolor="gray",  # "gray",
)
ax2.plot(CalUPlus[:, 0], uu[1:], "k", linewidth=1.0)
ax2.plot(CalUPlus[:, 0], vv[1:], "k", linewidth=1.0)
ax2.plot(CalUPlus[:, 0], ww[1:], "k", linewidth=1.0)
ax2.plot(CalUPlus[:, 0], uv[1:], "k", linewidth=1.0)
ax2.plot(CalUPlus1[:, 0], uu1[1:], "k:", linewidth=1.0)
ax2.plot(CalUPlus1[:, 0], vv1[1:], "k:", linewidth=1.0)
ax2.plot(CalUPlus1[:, 0], ww1[1:], "k:", linewidth=1.0)
ax2.plot(CalUPlus1[:, 0], uv1[1:], "k:", linewidth=1.0)
ax2.plot(CalUPlus2[:, 0], uu2[1:], "k--", linewidth=1.0)
ax2.plot(CalUPlus2[:, 0], vv2[1:], "k--", linewidth=1.0)
ax2.plot(CalUPlus2[:, 0], ww2[1:], "k--", linewidth=1.0)
ax2.plot(CalUPlus2[:, 0], uv2[1:], "k--", linewidth=1.0)
ax2.set_xscale("log")
ax2.set_ylim([-1.5, 3.5])
ax2.set_xlim([1, 2000])
ax2.set_ylim([-1.5, 3.5])
vna1 = r"$u^\prime u^\prime$"
vna2 = r"$w^\prime w^\prime$"
vna3 = r"$v^\prime v^\prime$"
vna4 = r"$u^\prime v^\prime$"
ax2.text(3.5, 3.0, vna1, fontsize=numsize + 1)
ax2.text(12, 1.5, vna2, fontsize=numsize + 1)
ax2.text(65, 0.5, vna3, fontsize=numsize + 1)
ax2.text(55, -0.6, vna4, fontsize=numsize + 1)
ax2.set_ylabel(
    r"$\sqrt{\langle u^{\prime}_i u^{\prime}_j\rangle^{+}}$", fontsize=textsize
)
ax2.set_xlabel(r"$y^+$", fontsize=textsize)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.grid(b=True, which="both", linestyle=":")
ax2.annotate("(b)", xy=(-0.18, 0.98), xycoords="axes fraction", fontsize=numsize)
plt.subplots_adjust(wspace=0.8)
plt.tick_params(labelsize="medium")
plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.3)
plt.savefig(
    pathF + "ReynoldStress" + str(x0) + ".svg", bbox_inches="tight", pad_inches=0.1,
)
plt.show()

# %%############################################################################
"""
    y+ along streamwise
"""
# %% calculate yplus ahead/behind the step


def yplus(MeanFlow, dy, wallval, opt):
    if opt == 1:
        TempFlow = MeanFlow.PlanarData.loc[MeanFlow.PlanarData["x"] < 5.0]
    elif opt == 2:
        TempFlow = MeanFlow.PlanarData.loc[MeanFlow.PlanarData["x"] > 100.0]
    frame = TempFlow.loc[np.round(TempFlow["y"], 6) == np.round(dy, 6)]
    frame1 = TempFlow.loc[np.round(TempFlow["y"], 6) == np.round(wallval, 6)]
    x = frame["x"].values
    rho = frame1["<rho>"].values
    mu = frame1["<mu>"].values
    delta_u = (frame["<u>"].values - frame1["<u>"].values) / (dy - wallval)
    tau = mu * delta_u
    u_tau = np.sqrt(np.abs(tau / rho))
    y_plus = (dy - wallval) * u_tau * rho / mu
    frame.assign(yplus=y_plus)
    return (x, y_plus, frame)


# 0.002300256 upsteam the step
x1, yplus1, frame1 = yplus(MeanFlow, 0.033333, 0.0, opt=1)
x2, yplus2, frame2 = yplus(MeanFlow, 0.033333, 0.0, opt=2)  # 3.001953125 downstream
res = np.vstack((np.hstack((x1, x2)), np.hstack((yplus1, yplus2))))
frame3 = pd.DataFrame(data=res.T, columns=["x", "yplus"])
frame3.to_csv(pathM + "YPLUS.dat", index=False, float_format="%1.8e", sep=" ")

# %% plot yplus along streamwise
yp = pd.read_csv(pathM + "YPLUS.dat", sep=" ", index_col=False, skipinitialspace=True)
fig3, ax3 = plt.subplots(figsize=(6.4, 3.0))
ax3.plot(yp["x"], yp["yplus"], "k", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$\Delta y^{+}$", fontsize=textsize)
ax3.set_xlim([0.0, 360])
ax3.set_ylim([0.0, 2.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5)
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "yplus.svg", dpi=300)
plt.show()
# %% plot u along streamwise
fig3, ax3 = plt.subplots(figsize=(6.4, 3.0))
ax3.plot(frame1["x"], frame1["<u>"], "k", linewidth=1.5)
ax3.plot(frame2["x"], frame2["<u>"], "k", linewidth=1.5)
ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax3.set_ylabel(r"$u / u_\infty$", fontsize=textsize)
ax3.set_xlim([-70.0, 40.0])
# ax3.set_ylim([0.0, 2.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.5)
ax3.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "u_stream2.svg", dpi=300)
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
    df = MeanFlow.yprofile("x", xd[i])
    y0 = df["y"]
    u0 = df["<u>"]
    rho0 = df["<rho>"]
    delta[i] = va.bl_thickness(y0.values, u0.values)[0]
    delta_star[i] = va.bl_thickness(
        y0.values, u0.values, rho=rho0.values, opt="displacement"
    )[0]
    theta[i] = va.bl_thickness(y0.values, u0.values, rho=rho0.values, opt="momentum")[0]

stream = np.loadtxt(pathM + "BubbleLine.dat", skiprows=1)
stream[:, -1] = stream[:, -1] + 3.0
func = interp1d(stream[:, 0], stream[:, 1], bounds_error=False, fill_value=0.0)
yd = func(xd)
xmax = np.max(stream[:, 0])


# fit curve
def func(t, A, B, C, D):
    return A * t ** 3 + B * t ** 2 + C * t + D


popt, pcov = DataPost.fit_func(func, xd, delta, guess=None)
A, B, C, D = popt


def fitfunc(t):
    return A * t ** 3 + B * t ** 2 + C * t + D


delta_fit = fitfunc(xd)

popt, pcov = DataPost.fit_func(func, xd, yd, guess=None)
A, B, C, D = popt


def fitfunc(t):
    return A * t ** 3 + B * t ** 2 + C * t + D


yd = fitfunc(xd)
# gortler = va.Gortler(1.3718e7, xd, delta1, theta)
radius = va.radius(xd[:-1], delta_fit[:-1])
# radius = va.Radius(xd[:-1], yd[:-1])
gortler = va.gortler_tur(theta[:-1], delta_star[:-1], radius)

fig3, ax3 = plt.subplots(figsize=(5, 2.5))
ax3.plot(xd, delta, "k-", linewidth=1.5)  # boundary layer
ax3.plot(xd, delta_fit, "k--", linewidth=1.5)
ax3.plot(xd, yd, "k--", linewidth=1.5)  # bubble line
ax3.plot(xd, theta, "k:", linewidth=1.5)  # momentum
# ax3.plot(xd[:-1], radius, 'k:', linewidth=1.5)
ax3.plot(stream[:, 0], stream[:, 1], "b--", linewidth=1.5)  # bubble line
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
fig = plt.figure(figsize=(3.2, 3.0))
matplotlib.rc("font", size=textsize)
ax2 = fig.add_subplot(111)
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
plt.savefig(pathF + "delta_radius.svg", bbox_inches="tight", pad_inches=0.1)
plt.show()

# %% plot figure for Gortler number
fig3, ax3 = plt.subplots(figsize=(3.2, 3.0))
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
#
# skin friction & pressure coefficiency/turbulent kinetic energy along streamwise
#
# %% compare
temp_path = path + "snapshots/S_10a/"
MeanFlow1 = pf()
MeanFlow1.load_data(pathSL, FileList=temp_path + "TP_2D_S_10_01100.00.plt")
MeanFlow1.add_walldist(StepHeight)
df = MeanFlow1.PlanarData
grouped = df.groupby(["x", "y", "z"])
df = grouped.mean().reset_index()
WallFlow1 = MeanFlow1.PlanarData.groupby("x", as_index=False).nth(1)
mu1 = va.viscosity(13718, WallFlow1["T"])
Cf1 = va.skinfriction(mu1, WallFlow1["u"], WallFlow1["walldist"]).values
ind1 = np.where(Cf1[:] < 0.008)
xwall1 = WallFlow1["x"].values
#
temp_path = path + "snapshots/S_10a/"
MeanFlow2 = pf()
MeanFlow2.load_data(pathSL, FileList=temp_path + "TP_2D_S_10_01300.00.plt")
MeanFlow2.add_walldist(StepHeight)
WallFlow2 = MeanFlow2.PlanarData.groupby("x", as_index=False).nth(1)
mu2 = va.viscosity(13718, WallFlow1["T"])
Cf2 = va.skinfriction(mu2, WallFlow2["u"], WallFlow2["walldist"]).values
ind2 = np.where(Cf1[:] < 0.006)
xwall2 = WallFlow2["x"].values
# %% comparison with laminar case
WallFlow0 = MeanFlow0.PlanarData.groupby("x", as_index=False).nth(1)
if np.size(np.unique(WallFlow0["y"])) > 2:
    maxy = np.max(WallFlow0["y"])
    WallFlow0 = WallFlow0.drop(WallFlow0[WallFlow0["y"] == maxy].index)
mu0 = va.viscosity(13718, WallFlow0["T"])
Cf0 = va.skinfriction(mu0, WallFlow0["u"], WallFlow0["walldist"]).values
ind0 = np.where(Cf0[:] < 0.008)
xwall0 = WallFlow0["x"].values
# %%
# WallFlow = MeanFlow.PlanarData[np.round(
#     MeanFlow.PlanarData["y"], 6) == 3.3333e-02]
# FirstLev = WallFlow[["x", "y"]]
xy_val = va.wall_line(MeanFlow.PlanarData, path, mask=corner, val=0.03125)  # 0.033333)
FirstLev = pd.DataFrame(data=xy_val, columns=["x", "y"])
xx = np.arange(0.0, 400.0, 0.25)
FirstLev = FirstLev[FirstLev["x"].isin(xx)]
FirstLev.to_csv(pathM + "FirstLev.dat", index=False, float_format="%9.6f")
# %% wall buondary


def add_variable(col, df):
    for i in range(np.size(col)):
        val = griddata(
            (MeanFlow.x, MeanFlow.y),
            MeanFlow.PlanarData[col[i]],
            (df.x, df.y),
            method="cubic",
        )
        df[col[i]] = val
    return df


MeanFlow.copy_meanval()
wavy = pd.read_csv(pathM + "FirstLev.dat", delimiter=",", skipinitialspace=True)
add_variable(["p", "T"], wavy)
add_variable("u", wavy)
add_variable("v", wavy)
mu = va.viscosity(25600, wavy["T"])
Cf = va.skinfriction(mu, wavy["u"], 0.03125).values  # 0.0333333
ddx = np.diff(wavy["x"])
ddy = np.diff(wavy["y"])
tang = np.transpose(np.vstack((ddx, ddy)))
xposi = np.array([1, 0])
modules = np.linalg.norm(tang, axis=1) * np.linalg.norm(xposi)
cos_val = np.dot(tang, xposi) / modules
sin_val = np.cross(tang, xposi) / modules
wavy1 = wavy.drop(index=0)
# ddy[np.where(ddy == 0.0)] = 0.0166667
delta_y = 0.03125 * np.abs(cos_val)
delta_x = 0.03125 * np.abs(sin_val)
Cf = (
    va.skinfriction(mu[1:], wavy1["u"], delta_y) * cos_val
    + va.skinfriction(mu[1:], wavy1["v"], delta_x) * sin_val
)

# %% Plot streamwise skin friction
# WallFlow = MeanFlow.PlanarData.groupby("x", as_index=False).nth(1)
# if np.size(np.unique(WallFlow["y"])) > 2:
#     maxy = np.max(WallFlow["y"])
#     WallFlow = WallFlow.drop(WallFlow[WallFlow["y"] == maxy].index)
# mu = va.viscosity(13718, WallFlow["T"])
# Cf = va.skinfriction(mu, WallFlow["u"], WallFlow["walldist"]).values
# ind = np.where(Cf[:] < 0.008)
fig2, ax2 = plt.subplots(figsize=(15 * cm2in, 5 * cm2in))
# fig = plt.figure(figsize=(6.4, 4.6))
# ax2 = fig.add_subplot(211)
matplotlib.rc("font", size=numsize)
# xwall = WallFlow["x"].values
ax2.plot(wavy1["x"], Cf, "k", linewidth=1.5)
# ax2.plot(xwall0[ind0], Cf0[ind0], "b--", linewidth=1.5)
# ax2.plot(xwall1[ind1], Cf1[ind1],
#          color='gray', linestyle=':', linewidth=1.2) #
# ax2.plot(xwall2[ind2], Cf2[ind2],
#          color='gray', linestyle=':', linewidth=1.2) #
ax2.set_xlabel(r"$x/h$", fontsize=textsize)
ax2.set_ylabel(r"$\langle C_f \rangle$", fontsize=textsize)
ax2.set_xlim([0, 400])
# ax2.set_ylim([-0.0001, 0.0003])
# ax2.set_yticks(np.arange(-0.002, 0.008, 0.002))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=-25, color="gray", linestyle=":", linewidth=1.5)
ax2.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax2.grid(visible=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
ax2.annotate("(a)", xy=(-0.09, 1.0), xycoords="axes fraction", fontsize=numsize)
plt.tick_params(labelsize=numsize)
# plt.savefig(pathF+'Cf_comp.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# %% zoom part
ax1 = fig.add_subplot(212)
matplotlib.rc("font", size=numsize)
ax1.plot(xwall[ind], Cf[ind], "k", linewidth=1.5)
ax1.plot(xwall0[ind0], Cf0[ind0], "b--", linewidth=1.5)
ax1.set_xlabel(r"$x/h$", fontsize=textsize)
ax1.set_ylabel(r"$\langle C_f \rangle$", fontsize=textsize)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.set_xlim([-1.0, 1.0])
ax1.set_ylim([-0.003, 0.005])
ax1.axvline(x=0.16, color="gray", linestyle=":", linewidth=1.5)
ax1.axvline(x=0.23, color="blue", linestyle=":", linewidth=1.5)
ax1.grid(b=True, which="both", linestyle=":")
ax1.annotate("(b)", xy=(-0.09, 0.98), xycoords="axes fraction", fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
plt.savefig(pathF + "Cf_comp.svg")

# %% pressure coefficiency
Ma = 6.0
fa = Ma * Ma * 1.4
wavy = pd.read_csv(pathM + "WallBoundary.dat", delimiter=",", skipinitialspace=True)
wavy["p"] = griddata(
    (MeanFlow.x, MeanFlow.y), MeanFlow.p, (wavy.x, wavy.y), method="cubic"
)
# %%
fig3, ax3 = plt.subplots(figsize=(15 * cm2in, 5 * cm2in))
ax3.plot(wavy["x"], wavy["p"] * fa, "k`", linewidth=1.5)
# ax3.plot(WallFlow0["x"], WallFlow0["p"] * fa, "b--", linewidth=1.5)
# p_ref = np.loadtxt(pathM + "PressureRef1.dat", skiprows=4)
# lref = 1
# ax3.scatter(p_ref[:11, 0]/lref, p_ref[:11, 1],
#             linewidth=0.8,
#             s=20.0,
#             edgecolor="gray",
#             facecolor="none")
# ax3.plot(WallFlow1['x'], WallFlow1['p']*fa,
#          color='gray', linestyle=':', linewidth=1.2) #
# ax3.plot(WallFlow2['x'], WallFlow2['p']*fa,
#          color='gray', linestyle=':', linewidth=1.2) #
ax3.set_xlabel(r"$x$", fontsize=textsize)
ax3.set_ylabel(r"$\langle p_w \rangle/p_{\infty}$", fontsize=textsize)
# ax3.set_xlim([26, 108])
# ax3.set_ylim([0.98, 0.99])
# ax3.set_yticks(np.arange(0.2, 2.6, 0.4))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axvline(x=-25, color="gray", linestyle=":", linewidth=1.5)
ax3.axvline(x=-4.2, color="blue", linestyle=":", linewidth=1.5)
ax3.grid(visible=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
# plt.savefig(pathF + "Cp_comp.svg", dpi=300)
plt.savefig(pathF + "Cp_comp.svg", dpi=300)
plt.show()

# % turbulent kinetic energy


# tke = va.tke(WallFlow).values
# ax3 = fig.add_subplot(122)
# matplotlib.rc("font", size=textsize)
# ax3.plot(xwall[ind], tke[ind], "k", linewidth=1.5)
# ax3.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
# ax3.set_ylabel(r"$k/u^2_\infty$", fontsize=textsize)
# ax3.set_xlim([-20.0, 40.0])
## ax3.set_ylim([-0.001, 0.002])
# ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# ax3.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)delta
# ax3.axvline(x=11.0, color="gray", linestyle="--", linewidth=1.0)
# ax3.grid(b=True, which="both", linestyle=":")
# ax3.yaxis.offsetText.set_fontsize(numsize)
# ax3.annotate("(b)", xy=(-0.12, 1.04), xycoords='axes fraction',
#             fontsize=numsize)
# plt.tick_params(labelsize=numsize)
# plt.subplots_adjust(wspace=0.3)
# plt.savefig(pathF+'CfTk.svg', bbox_inches='tight', pad_inches=0.1)
# plt.show()

# %%############################################################################
"""
    Intermittency
"""
# %% Load data for Computing intermittency factor
InFolder = pathSL + "TP_2D_Z_03/"
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
timezone = np.arange(975, 1064.00 + 0.25, 0.25)
dt = 0.5
for j in range(np.size(Snapshots[:, 0])):
    gamma[j] = va.intermittency(sigma, p0, Snapshots[j, :], timezone)
    alpha[j] = va.alpha3(Snapshots[j, :])

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
fig3, ax3 = plt.subplots(figsize=(6.4, 2.6))
ax3.plot(xzone, gamma, "k--")
ax3.plot(xarr, yarr, "k-")  # fitting values
ax3.scatter(x1, ga1, c="black", marker="o", s=10)  # universal distribution
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
plt.savefig(pathF + "POD/PODConvergence.svg", bbox_inches="tight", pad_inches=0.1)
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
plt.savefig(pathF + "Wavelen" + time + ".svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$u^\prime/u_\infty$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.savefig(pathF + "Wavelen" + time + ".svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.savefig(pathF + "Wavelen" + time + ".svg", bbox_inches="tight", pad_inches=0.1)
plt.show()
