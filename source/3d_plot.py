#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:39:40 2019
    3D plots use matplotlib 

@author: weibo
"""
# %% Load libraries
import pandas as pd
import os
import sys
import numpy as np
from glob import glob
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
import matplotlib
import plt2pandas as p2p

import matplotlib.ticker as ticker
from data_post import DataPost
from matplotlib.ticker import ScalarFormatter
from planar_field import PlanarField as pf
from triaxial_field import TriField as tf
import variable_analysis as va
from scipy.interpolate import griddata

get_ipython().run_line_magic("matplotlib", "qt")

# %% data path settings
path = "/media/weibo/VID21/ramp_st14/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + 'video/'
pathD = path + 'DMD/'
pathSL = path + 'Slice/'

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
tsize = textsize
nsize = numsize
lh = 3.0
# %% 3D PSD
# load data
var = 'p'
xval = np.loadtxt(pathSL + 'FWPSD_x.dat', delimiter=' ')
freq = np.loadtxt(pathSL + 'FWPSD_freq.dat', delimiter=' ')
FPSD = np.loadtxt(pathSL + var + '_FWPSD_psd.dat', delimiter=' ')
freq = freq[1:]
FPSD = FPSD[1:, :]
newx = [-10.0, 2.0, 3.0, 5.0, 9.0, 10.0]

fig = plt.figure(figsize=(7.0, 4.0))
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = 'dotted'
plt.tick_params(labelsize=numsize)
ax = fig.add_subplot(111, projection='3d')
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.8, 1.2, 1.0, 1.0]))
for i in range(np.size(newx)):
    ind = np.where(xval[:]==newx[i])[0][0]
    xloc = newx[i] * np.ones(np.shape(freq))
    ax.plot(freq, xloc, FPSD[:, ind], zdir='z', linewidth=1.5)
    
    
ax.ticklabel_format(axis="z", style="sci", scilimits=(-2, 2))
ax.zaxis.offsetText.set_fontsize(numsize)
# ax.w_xaxis.set_xscale('log')
ax.set_xscale('symlog')
# ax.set_xticks([-2, -1, 0, 0.3])
ax.set_xlabel(r'$f$', fontsize=textsize)
ax.set_ylabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_zlabel(r'$\mathcal{P}$', fontsize=textsize)
#ax.xaxis._axinfo['label']['space_factor'] = 0.1
#ax.zaxis._axinfo["grid"]['linewidth'] = 1.0
#ax.zaxis._axinfo["grid"]['color'] = "gray"
#ax.zaxis._axinfo['grid']['linstyle'] = ':'
ax.tick_params(axis='x', direction='in')
ax.tick_params(axis='y', direction='in')
ax.tick_params(axis='z', direction='in')
ax.view_init(elev=50, azim=-20)
ax.axes.xaxis.labelpad=1
ax.axes.yaxis.labelpad=6
ax.axes.zaxis.labelpad=0.1
ax.tick_params(axis='x', which='major', pad=1)
ax.tick_params(axis='y', which='major', pad=0.1)
ax.tick_params(axis='z', which='major', pad=0.1)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "3dFWPSD.svg")
plt.show()

#plt.rcParams['xtick.direction'] = 'in'

# %% 3d contour plot
# load data
var = 'p'
xval = np.loadtxt(pathSL + 'FWPSD_x.dat', delimiter=' ')
freq = np.loadtxt(pathSL + 'FWPSD_freq.dat', delimiter=' ')
FPSD = np.loadtxt(pathSL + var + '_FWPSD_psd.dat', delimiter=' ')
freq = freq[1:]
FPSD = FPSD[1:, :]


Yxval, Xfreq = np.meshgrid(xval, freq)

fig = plt.figure(figsize=(7.0, 4.0))
ax = fig.add_subplot(111, projection='3d')
# ax = fig.gca(projection='3d')
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = 'dotted'
plt.tick_params(labelsize=numsize)

ax.plot_surface(np.log10(Xfreq), Yxval, FPSD, rstride=1, cstride=1, cmap=cm.bwr)
xticks = [1e-2, 1e-1, 1e0]
ax.set_xticks(np.log10(xticks))
# ax.set_xticklabels([r'$10^{-2}$', r'$10^{-2}$', r'$10^{-2}$'])

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.8, 1.2, 1.0, 1.0]))
ax.set_xlabel(r'$f$', fontsize=tsize)
ax.set_ylabel(r'$x/\delta_0$', fontsize=tsize)
ax.set_zlabel(r'$\mathcal{P}$', fontsize=tsize)
# ax.set_powerlimits((-2, 2))
# ax.w_zaxis.set_major_formatter()
# ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="z", style="sci", scilimits=(-2, 2))
ax.view_init(elev=50, azim=-20)
ax.axes.xaxis.labelpad=1
ax.axes.yaxis.labelpad=6
ax.axes.zaxis.labelpad=0.1
ax.tick_params(axis='x', which='major', pad=1)
ax.tick_params(axis='y', which='major', pad=0.1)
ax.tick_params(axis='z', which='major', pad=0.1)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
#plt.savefig(pathF + "3dContourFWPSD.svg")
plt.show()

"""
save wall plane data and plot
"""
# %% plot Cf in x-z plane (TP_data)
time_ave = tf()
filename = glob(path + 'TP_data_1123.h5')
time_ave.load_3data(pathT, FileList=filename, NameList='h5')
# time_ave.load_3data(path+'TP_data_02087427/')
# time_ave._data_field.to_hdf(path+'TP_data_1124.h5', 'w', format='fixed')
time_ave.add_walldist(3.0)
# filename = glob(pathT + 'MeanFlow_*')
# time_ave.load_3data(pathT, FileList=filename, NameList='h5')
time_ave.add_walldist(-3.0)
df1 = time_ave.TriData.query("x>0 & walldist>0")
df1a = df1.loc[df1['walldist']==np.min(df1['walldist'])]  
# np.min(df1['walldist'])*2 # for the mean flow
df2 = time_ave.TriData.query("x<0 & y>=0 & walldist>0")
df2a = df2.loc[df2['walldist']==np.min(df2['walldist'])]  # on the wall=0.5*dy
df = pd.concat([df1a, df2a])
df.to_hdf(pathT + 'WallValue.h5', 'w', format='fixed')
del time_ave

# %%
time_ave1 = tf()
filename = '01110.00.h5'
filename1 = glob(path + 'snapshots/Y_007/TP_2D_Y_007_'+filename)
time_ave1.load_3data(pathT, FileList=filename1, NameList='h5')
time_ave1.rescale(lh)
df1 = time_ave1.TriData
df1a = df1.assign(walldist=0.003)  # Cf must be normalized by boundary layer thickness
time_ave2 = tf()
filename2 = glob(path + 'snapshots/Y_008/TP_2D_Y_008_'+filename)
time_ave2.load_3data(pathT, FileList=filename2, NameList='h5')
time_ave2.rescale(lh)
df2 = time_ave2.TriData.query("x>0")
df2a = df2.assign(walldist=0.002)
df = pd.concat([df1a, df2a])
df.to_hdf(pathT + 'WallValue.h5', 'w', format='fixed')
# del time_ave1, time_ave2
# %% skin friction
Re = 13718
df = pd.read_hdf(pathT + 'WallValue.h5')
grouped = df.groupby(['x', 'y', 'z'])
df = grouped.mean().reset_index()
mu = va.viscosity(Re, df['T'], law='POW')  # df['<T>'],
Cf = va.skinfriction(mu, df['u'], df['walldist'])  # df['<u>']
df['Cf'] = Cf
df3 = df.query("Cf < 0.015")
xx = np.unique(df['x'])  # np.linspace(-20.0, 40.0, 1201)
zz = np.unique(df['z'])  # np.linspace(-8.0, 8.0, 33) #
x, z = np.meshgrid(xx, zz)
friction = griddata((df3['x'], df3['z']), df3['Cf'], (x, z))
print("Cf_max=", np.max(df3['Cf']))
print("Cf_min=", np.min(df3['Cf']))
Cf_ft1 = np.zeros(np.shape(friction))
Cf_ft2 = np.zeros(np.shape(friction))
# %%
n_ord = 10   # digital order
fs = 1.0
Wn = 1 / fs  # cutoff frequency / (0.5 * f_sample)
for i in range(np.size(zz)):
    b, a = signal.butter(n_ord, Wn, 'lowpass', fs=32)  # Wn = 
    Cf_2 = signal.filtfilt(b, a, friction[i, :])
    Cf_ft1[i, :] = Cf_2
for i in range(np.size(xx)):
    b, a = signal.butter(n_ord, Wn, 'lowpass', fs=32)  # Wn = 
    Cf_2 = signal.filtfilt(b, a, Cf_ft1[:, i])
    Cf_ft2[:, i] = Cf_2
# %% plot skin friction
fig, ax = plt.subplots(figsize=(6.6, 2.5))
matplotlib.rc("font", size=textsize)
cx1 = -5
cx2 = 7
fa = 1000
rg1 = np.linspace(cx1, cx2, 21)
# cbar = ax.contourf(x, z, friction*fa, cmap="RdBu_r", levels=rg1, extend='both')  # jet # rainbow_r
cbar = ax.contourf(x, z, Cf_ft2*fa, cmap="RdBu_r", levels=rg1, extend='both')  # jet # rainbow_r
ax.set_xlim(-16, 5.0)
ax.set_ylim(-4, 4)
#ax.set_xlim(-7.0, 3.0)
#ax.set_ylim(-2, 2)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$z/\delta_0$", fontsize=textsize)
ax.set_yticks(np.linspace(-8.0, 8.0, 5))
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cx1, cx2, 3)
cbar = plt.colorbar(cbar, ticks=rg2, extendrect=True, fraction=0.016, pad=0.03)
cbar.ax.tick_params(labelsize=numsize)
cbar.set_label(
    r"$C_f \times 10^3$", rotation=0, fontsize=numsize, labelpad=-20, y=1.12
)
ax.axvline(x=8.9, color="k", linestyle="--", linewidth=1.2) # 10.9 # 8.9
# cbar.formatter.set_powerlimits((-2, 2))
# cbar.ax.xaxis.offsetText.set_fontsize(numsize)
# cbar.update_ticks()
plt.savefig(pathF + "skinfriction1.svg", bbox_inches="tight")
plt.show()

# %% skin friction (snapshot)
time_ave1 = tf()
filename1 = 'y3p01_1200'
lh = 3.0
# filename1 = glob(path + filename1 + '.plt')
# time_ave1.load_data(path, FileList=filename1)
# time_ave1._data_field.to_hdf(path+'y3p01_700' + '.h5', 'w', format='fixed')
filename1 = glob(path + filename1 + '.h5')
time_ave1.load_3data(path, FileList=filename1, NameList='h5')
time_ave1.rescale(lh)
df1 = time_ave1.TriData
time_ave2 = tf()
filename2 = 'y0p01_1200'
filename2 = glob(path + filename2 + '.h5')
time_ave2.load_3data(path, FileList=filename2, NameList='h5')
time_ave2.rescale(lh)
df2 = time_ave2.TriData
# %% prep-process
df3 = df1.query("x > 0")
df4 = df2.query("x < 0")
df = pd.concat([df3, df4])
xx = np.unique(df['x'])  # np.linspace(-20.0, 40.0, 1201)
zz = np.unique(df['z'])  # np.linspace(-8.0, 8.0, 33) #
x, z = np.meshgrid(xx, zz)
var = 'dudy'
vor3 = griddata((df['x'], df['z']), df[var], (x, z))
print("vor3_max=", np.max(df[var]))
print("vor3_min=", np.min(df[var]))
# %% plot x-z contour
fig, ax = plt.subplots(figsize=(7.2, 3.2))
matplotlib.rc("font", size=textsize)
cx1 = -8
cx2 = 40
fa = 1000
rg1 = np.linspace(cx1, cx2, 21)
# cbar = ax.contourf(x, z, friction*fa, cmap="RdBu_r", levels=rg1, extend='both')  # jet # rainbow_r
cbar = ax.contourf(x, z, vor3, cmap="RdBu_r", levels=rg1, extend='both')  # jet # rainbow_r
ax.set_xlim(-4.5, 2.0)
# ax.set_ylim(-4, 4)
#ax.set_xlim(-7.0, 3.0)
#ax.set_ylim(-2, 2)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/h$", fontsize=textsize)
ax.set_ylabel(r"$z/h$", fontsize=textsize)
# ax.set_yticks(np.linspace(-8.0, 8.0, 5))
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cx1, cx2, 3)
cbar = plt.colorbar(cbar, ticks=rg2, extendrect=True, fraction=0.018, pad=0.03)
cbar.ax.tick_params(labelsize=numsize)
cbar.set_label(
    r"${\partial u}/{\partial y}$", rotation=0, fontsize=numsize, labelpad=-20, y=1.1
)
ax.axvline(x=-4.3, color="k", linestyle="--", linewidth=1.2) # 10.9 # 8.9
# cbar.formatter.set_powerlimits((-2, 2))
# cbar.ax.xaxis.offsetText.set_fontsize(numsize)
# cbar.update_ticks()
#xbox = np.arange(-25, 20 + 0.125, 0.125)
#zbox = np.arange(-8.0, 8.0 + 0.0625, 0.0625)        
#xstm, zstm = np.meshgrid(xbox, zbox)
#u = griddata((df3.x, df3.z), df3['u'], (xstm, zstm))
#w = griddata((df3.x, df3.z), df3['w'], (xstm, zstm))
#ax.streamplot(
#    xstm,
#    zstm,
#    u,
#    w,
#    density=[1, 2],
#    color="k",
#    arrowsize=0.8,
#    linewidth=0.6,
#    integration_direction="both",
#)

# %% 
zprof1 = df.loc[df['x']==-5.0]
zprof2 = df.loc[df['x']==10.0] 
fig, ax = plt.subplots(2, 1, figsize=(6.4, 2.4))
matplotlib.rc("font", size=numsize)
val1 = np.sign(zprof1['u'] - np.mean(zprof1['u']))
ax[0].plot(zprof1['z'], val1, 'k-', linewidth=1.0)
# ax[0].axhline(y=np.mean(zprof1['u']), color="k", linestyle="--", linewidth=0.8)
ax[0].set_xlim([-8.0, 8.0])
ax[0].set_xticklabels('')
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
val2 = np.sign(zprof2['u'] - np.mean(zprof2['u']))
ax[1].plot(zprof2['z'], val2, 'b-', linewidth=1.0)
# ax[1].axhline(y=np.mean(zprof2['u']), color="b", linestyle="--", linewidth=0.8)
ax[1].set_xlim([-8.0, 8.0])
ax[0].set_ylabel(r"$u/u_\infty$", fontsize=textsize)
ax[1].set_xlabel(r"$z/\delta_0$", fontsize=textsize)
plt.savefig(pathF + "zprofile.svg", bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(2.4, 2.2))
freq1, FPSD1 = va.fw_psd(val1, 0.0625, 16, opt=1, seg=8, overlap=4)
freq2, FPSD2 = va.fw_psd(val2, 0.0625, 16, opt=1, seg=8, overlap=4)
matplotlib.rc("font", size=numsize)
ax.plot(freq1, FPSD1, 'k-', linewidth=1.0)
ax.plot(freq2, FPSD2, 'b-', linewidth=1.0)
ax.set_xlabel(r"$k_z$", fontsize=textsize)
ax.set_ylabel(r'$f \mathcal{P}(f)$', fontsize=textsize)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.grid(b=True, which="both", linestyle=":")
ax.set_xscale('log')
plt.savefig(pathF + "zprofile_psd.svg", bbox_inches="tight")
plt.show()
"""
calculate and save boundary layer thickness
"""
# %% calculate boundary layer thickness
StepHeight = -3.0
MeanFlow = pf()
MeanFlow.load_meanflow(path)
MeanFlow.add_walldist(StepHeight)
MeanFlow.copy_meanval()
# %%
xloc = np.linspace(-110.0, 30.0, 1121)
xloc = np.delete(xloc, np.argwhere(xloc==0.0))
dfx = MeanFlow.PlanarData
dft = dfx[dfx.x.isin(xloc)]
xnew = np.unique(dft['x'])
num = np.size(xnew)
delta = np.zeros(num)
delta_star = np.zeros(num)
theta = np.zeros(num)

boundary = np.loadtxt(pathM + "BoundaryEdgeFit.dat", skiprows=1)
y_inter = interpolate.interp1d(boundary[:, 0], boundary[:, 1])
b_arr = y_inter(xnew)

for i in range(num):
    profile = MeanFlow.yprofile("x", xnew[i])
    y0 = profile['walldist'].values
    u0 = profile['<u>'].values
    rho0 = profile['<rho>'].values
    # interpolate for velocity at boundary layer edge
    u_inter = interpolate.interp1d(y0, u0)
    blt = b_arr[i] 
    if xnew[i] > 0.0:
        blt = blt - 3.0
        if blt > np.max(y0):
            blt = np.max(y0)
        ud = u_inter(blt)
    else:
        if blt > np.max(y0):
            blt = np.max(y0)
        ud = u_inter(blt)
    delta[i] = va.bl_thickness(y0, u0, u_d=ud, up=1.0)[0]
    delta_star[i] = va.bl_thickness(y0, u0, u_d=ud, rho=rho0, up=1.0,
                                    opt='displacement')[0]
    theta[i] = va.bl_thickness(y0,u0,u_d=ud,rho=rho0,up=1,opt='momentum')[0]

# %% save boundary layer thickness
names = ['x', 'delta', 'displace', 'momentum']
res = np.vstack((xnew, delta, delta_star, theta))
frame = pd.DataFrame(data=res.T, columns=names)
frame.drop_duplicates()
frame.to_csv(pathM + 'thickness.dat', index=False,
             float_format='%1.8e', sep=' ')
# %% fit boundary layer thickness
frame = pd.read_csv(pathM + 'thickness.dat', sep=' ')
frame1 = frame.loc[frame['x'] <=0.0]
frame2 = frame.loc[frame['x'] > 0.0]
fit_frame1 = frame1.values
fit_frame2 = frame2.values
b, a = signal.butter(6, 0.05)  # 3-order low pass butter worth filter with 2*ts/sample
# %% repeat 3 times, save fit boundary layer thickness data
new_d1 = signal.filtfilt(b, a, frame1['delta']) # delta # displace # momentum 
new_d2 = signal.filtfilt(b, a, frame2['delta'])
fig, ax = plt.subplots()
ax.plot(frame['x'], frame['delta'], 'r:')
ax.plot(frame['x'], frame['displace'], 'b:')
ax.plot(frame['x'], frame['momentum'], 'g:')
ax.plot(frame1['x'], new_d1, 'k--')
ax.plot(frame2['x'], new_d2, 'k--')
ax.set_ylim(0.0, 6.0)
plt.show()
fit_frame1[:,1] = new_d1
fit_frame2[:,1] = new_d2

# %%
res = np.vstack((fit_frame1, fit_frame2))
fit_frame = pd.DataFrame(data=res, columns=names)
fit_frame.to_csv(pathM + 'fit_thickness.dat', index=False,
             float_format='%1.8e', sep=' ')

# %% validate the fit data of boundary layer thickness
# // TODO find a better way to calculate boundary layer thickness
thick = pd.read_csv(pathM + 'fit_thickness.dat', sep=' ')
thick.drop_duplicates()
fig, ax = plt.subplots()
ax.plot(thick['x'], thick['delta'], 'k--')
ax.plot(thick['x'], thick['displace'], 'b')# delta_star
ax.plot(thick['x'], thick['momentum'], 'g') # theta
ax.legend(['delta', 'delta*', 'momentum'])
plt.savefig(pathF + "boundary_layer.svg", bbox_inches="tight")
plt.show()

"""
calculate and plot for contours of Gortler number
"""
# %% calculate gortler
df = MeanFlow.PlanarData
curvature_radius = va.curvature_r(df, opt='mean')
thick = pd.read_csv(pathM + 'thickness.dat', sep=' ')
x1 = np.linspace(-10.0, 30.0, 801)
y1 = np.linspace(-3.0, 10.0, 209)
x2, y2 = np.meshgrid(x1, y1)
theta0 = np.interp(x1, thick['x'], thick['momentum'])
star0 = np.interp(x1, thick['x'], thick['displace'])
theta = np.tile(theta0, (np.size(y1), 1))
star = np.tile(star0, (np.size(y1), 1))
radius = griddata((df.x, df.y), curvature_radius, (x2, y2))
gortler = va.gortler_tur(theta, star, radius)
corner = (x2 < 0.0) & (y2 < 0.0)
gortler[corner] = np.nan
# %%
print("max=", np.max(gortler[~np.isnan(gortler)]))
print("min=", np.min(gortler[~np.isnan(gortler)]))
cval1 = 0
cval2 = 2.0
fig, ax = plt.subplots(figsize=(6.4, 2.3))
matplotlib.rc("font", size=textsize)
rg1 = np.linspace(cval1, cval2, 41)
cbar = ax.contourf(x2, y2, gortler, cmap="rainbow", levels=rg1, extend='both')  # rainbow_r
ax.set_xlim(-10.0, 30.0)
ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x/\delta_0$", fontsize=textsize)
ax.set_ylabel(r"$y/\delta_0$", fontsize=textsize)
plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(cval1, cval2, 3)
cbaxes = fig.add_axes([0.17, 0.68, 0.16, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax=cbaxes, extendrect='False',
                    orientation="horizontal", ticks=rg2)
cbar.set_label(
    r"$\langle \rho \rangle/\rho_{\infty}$", rotation=0, fontsize=textsize
)
plt.savefig(pathF + "gortler.svg", bbox_inches="tight")
plt.show()


"""
calculate and plot for lines of Gortler number
"""
# %% extract streamlines
points = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [3.25, 3.5, 3.75, 4.0, 4.25, 4.5]
    ]
)
# %%
va.streamline(pathM, MeanFlow.PlanarData, points)

# %% calculate and fit curvature radius, method 1
stream = pd.read_csv(pathM + 'streamline3.dat', sep=' ')
stream.drop_duplicates(subset=['x'], keep='first', inplace=True)
stream1 = pd.read_csv(pathM + 'streamline5.dat', sep=' ')
stream1.drop_duplicates(subset=['x'], keep='first', inplace=True)
curva1 = va.curvature(stream['x'], stream['y'], opt='internal')
fig, ax = plt.subplots(figsize=(6.6, 2.8))
matplotlib.rc("font", size=textsize)
plt.plot(stream['x'], stream['y'], 'g--',  linewidth=1.2)
plt.plot(stream1['x'], stream1['y'], 'g--',  linewidth=1.2)
ax.set_xlim(-25, 15.0)
ax.set_ylim(0.0, 5.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$y/\delta_0$', fontsize=textsize)
plt.savefig(pathF+'streamline.svg', bbox_inches='tight')
plt.show()
# filter data
# b, a = signal.butter(3, 0.1) 
# new_r1 = signal.filtfilt(b, a, curva1)

# calculate gortler number
thick = pd.read_csv(pathM + 'fit_thickness.dat', sep=' ')
theta0 = np.interp(stream['x'], thick['x'], thick['momentum'])
star0 = np.interp(stream['x'], thick['x'], thick['displace'])
gort1 = va.gortler_tur(theta0, star0, curva1, opt='curvature')
strm1 = stream['x']
# %% calculate and fit curvature radius, method 2
#df = MeanFlow.PlanarData
#curvature = va.curvature_r(df, opt='mean')
#xx, yy = np.meshgrid(stream['x'], stream['y'])
#curva = griddata((df.x, df.y), curvature, (xx, yy))
#curva1 = curva[0, :]

# %% plot gortler number
fig = plt.figure(figsize=(6.4, 2.6))
matplotlib.rc("font", size=textsize)
ax2 = fig.add_subplot(121)
matplotlib.rc("font", size=textsize)
ax2.plot(strm1/lh, curva1*lh, "k", linewidth=1.2)
# ax2.plot(strm3, curva3, "g", linewidth=1.2)
# ax2.plot(strm5, curva5, "b", linewidth=1.2)
ax2.set_xlabel(r"$x/h$", fontsize=textsize)
ax2.set_ylabel(r"$h/ R$", fontsize=textsize)
# ax2.set_xlim([-25.0, 15.0])
ax2.set_ylim([-0.5, 0.4])
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax2.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
ax2.axvline(x=-13.0/lh, color="gray", linestyle="--", linewidth=1.0)
ax2.grid(b=True, which="both", linestyle=":")
ax2.yaxis.offsetText.set_fontsize(numsize)
ax2.annotate("(a)", xy=(-0.2, 1.0), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.savefig(pathF+'curvature1.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# fig3, ax3 = plt.subplots(figsize=(5, 2.5))
ax3 = fig.add_subplot(122)
ax3.plot(strm1/lh, gort1, "k", linewidth=1.2)
# ax3.plot(strm3, gort3, "g", linewidth=1.5)
# ax3.plot(strm5, gort5, "b", linewidth=1.5)
ax3.set_xlabel(r"$x/h$", fontsize=textsize)
ax3.set_ylabel(r"$G_t$", fontsize=textsize)
# ax3.set_xlim([-25.0, 15.0])
ax3.set_ylim([-0.2, 3.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.axhline(y=0.58, color='gray', linestyle='-.', linewidth=1.0)
ax3.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.0)
ax3.axvline(x=-13.0/lh, color="gray", linestyle="--", linewidth=1.0)
ax3.grid(b=True, which="both", linestyle=":")
ax3.annotate("(b)", xy=(-0.15, 1.0), xycoords='axes fraction',
             fontsize=numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + "gortler1.svg", dpi=300)
plt.show()
