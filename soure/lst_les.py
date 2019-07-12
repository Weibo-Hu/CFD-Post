#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for comparing LST with LES

@author: weibo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:52:46 2019
    post-processing LST data

@author: weibo
"""

# %% Load necessary module
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, splev, splrep, spline
import variable_analysis as fv
from timer import timer
import os
import tecplot as tp
import plt2pandas as p2p
from glob import glob
from scipy import fftpack
import warnings
import sys
import multiprocessing

# %% data path settings
path = "/media/weibo/VID2/BFS_M1.7TS/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "Slice/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathB = path + "BaseFlow/"

# % figures properties settings
plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 12
numsize = 10

# %% load data
path1 = pathS + 'TP_2D_Z_03/'
stime = np.arange(700.0, 899.75 + 0.25, 0.25) # n*61.5
x0 = np.arange(-40.0, 0.0 + 0.5, 0.5)
newlist = ['x', 'y', 'z', 'u', 'p']
dirs = sorted(os.listdir(path1))
# fm_temp = (pd.read_hdf(path1 + f) for f in dirs)
# fm = pd.concat(fm_temp.loc[:, newlist], ignore_index=True)
df_shape = np.shape(pd.read_hdf(path1 + dirs[0]))
fm = pd.DataFrame()
for i in range(np.size(dirs)):
    fm_temp = pd.read_hdf(path1 + dirs[i])
    if df_shape[0] != np.shape(fm_temp)[0]:
        warnings.warn("Shape of" + dirs[i] + " does not match!!!",
                      UserWarning)
    fm = fm.append(fm_temp.loc[:, newlist], ignore_index=True)
# %% extract probe data with time
y0 = 0.001953125 # 0.00390625 #   0.005859375 0.0078125 0.296875
num_samp = np.size(stime)
var = np.zeros((num_samp, 2))
for j in range(np.size(x0)):
    filenm = pathP + 'timeline_' + str(x0[j]) + '.dat'
    var = fm.loc[(fm['x']==x0[j]) & (fm['y']==y0), ['u', 'p']].values
    df = pd.DataFrame(data=np.hstack((stime.reshape(-1, 1), var)), 
                      columns=['time', 'u', 'p'])
    df.to_csv(filenm, sep=' ', index=False, float_format='%1.8e')
# del fm_temp, fm   
    
# %% make base flow 
bf = pd.read_hdf(pathB + 'BaseFlow.h5')
if y0 == 0.001953125:
    file_b = 'wall'
else:
    file_b = str(y0)
baseline = np.zeros((np.size(x0), 2))
for j in range(np.size(x0)):
    basevar = bf.loc[(bf['x']==x0[j]) & (bf['y']==y0), ['u', 'p']].values
    baseline[j, :] = basevar

base = pd.DataFrame(data=np.hstack((x0.reshape(-1, 1), baseline)),
                    columns=['x', 'u', 'p'])
base.to_csv(pathB + 'baseline_' + file_b + '.dat', sep=' ',
            index=False, float_format='%1.8e')
base = pd.read_csv(pathB + 'baseline_' + file_b + '.dat', sep=' ', skiprows=0,
                   index_col=False, skipinitialspace=True) 

# %% variables with time
varnm = 'u'
if varnm == 'u':
    ylab = r"$u^\prime / u_\infty$"
    ylab_sub = r"$_{u^\prime}$"
else:
    ylab = r"$p^\prime/(\rho_\infty u_\infty ^2)$"
    ylab_sub = r"$_{p^\prime}$"
xloc = [-36.0, -24.0, -10.0]
curve= ['k-', 'k--', 'k:']
fig3, ax3 = plt.subplots(figsize=(6.0, 2.8))
matplotlib.rc("font", size=textsize)
for i in range(3):
    filenm = pathP + 'timeline_' + str(xloc[i]) + '.dat'   
    var = pd.read_csv(filenm, sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
    # val = var[varnm] - np.mean(var[varnm])
    val = var[varnm] - base.loc[base['x']==xloc[i], [varnm]].values[0]
    ax3.plot(var['time'], val, curve[i], linewidth=1.5)

ax3.set_ylabel(ylab, fontsize=textsize)
ax3.set_xlabel(r"$t u_\infty / \delta_0$", fontsize=textsize)
# ax3.set_xlim([0.0, 25.0])
ax3.set_ylim([-1.5e-4, 2.5e-4]) # ([-8.0e-4, 6.0e-4])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.grid(b=True, which="both", linestyle=":")
ax3.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + varnm + "_time.svg", dpi=300)
plt.show()

# %% Amplitude and phase with freq
if varnm == 'u':
    ylab_sub = r"$_{u^\prime}$"
else:
    ylab_sub = r"$_{p^\prime}$"
freq_samp = 4.0
num_samp = np.size(stime)
freq = np.linspace(0.0, freq_samp / 2, math.ceil(num_samp/2), endpoint=False)
fig = plt.figure(figsize=(6.4, 3.0))
matplotlib.rc("font", size=textsize)
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for i in range(3):
    filenm = pathP + 'timeline_' + str(xloc[i]) + '.dat'
    var = pd.read_csv(filenm, sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
    var_per = var[varnm] - base.loc[base['x']==xloc[i], [varnm]].values[0]
    var_per = var_per - np.mean(var_per) # make sure mean value is zero
    var_fft = np.fft.rfft(var_per)  # remove value at 0 frequency    
    amplt = np.abs(var_fft)
    phase = np.angle(var_fft, deg=False)
    ax.plot(freq, amplt, curve[i], linewidth=1.2)
    ax1.plot(freq, phase, curve[i], linewidth=1.2)

ax.set_xscale("log")
ax.set_xlabel(r"$f \delta_0 / u_\infty$", fontsize=textsize)
ax.set_ylabel(r"$A$" + ylab_sub, fontsize=textsize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.tick_params(labelsize=numsize)
ax.yaxis.offsetText.set_fontsize(numsize)
ax.grid(b=True, which="both", linestyle=":")

ax1.set_xscale("log")
ax1.set_xlabel(r"$f \delta_0 / u_\infty$", fontsize=textsize)
ax1.set_ylabel(r"$\theta$" + ylab_sub + r"[rad]", fontsize=textsize)
# ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.yaxis.offsetText.set_fontsize(numsize)
ax1.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + varnm + "_fft" + ".svg", dpi=300)
plt.show()

# %% Fourier transform of variables with time 
x0 = np.arange(-40.0, 0.0 + 0.5, 0.5)
freq_samp = 4.0
num_samp = np.size(stime) # 750
# freq = np.linspace(freq_samp / num_samp, freq_samp / 2, math.ceil(num_samp/2))
freq = np.linspace(0.0, freq_samp / 2, math.ceil(num_samp/2), endpoint=False)
freq_x = np.zeros((1, np.size(x0)))
amplt_x = np.zeros((1, np.size(x0)))
phase_x = np.zeros((1, np.size(x0)))
for i in range(np.size(x0)):
    var = pd.read_csv(pathP + 'timeline_' + str(x0[i]) + '.dat', sep=' ', 
                      skiprows=0, index_col=False, skipinitialspace=True)
    var_per = var[varnm] - base.loc[base['x']==x0[i], [varnm]].values[0]
    var_per = var_per - np.mean(var_per) # make sure mean value is zero
    # var_per = var[varnm] - base.loc[base['x']==x0[i], [varnm]].values[0]
    var_fft = np.fft.rfft(var_per)  # remove value at 0 frequency    
    amplt = np.abs(var_fft)
    phase = np.angle(var_fft, deg=False)
    ind = np.argmax(amplt)
    freq_x[0, i] = freq[ind]
    amplt_x[0, i] = amplt[ind]
    phase_x[0, i] = phase[ind]
res_fft = np.concatenate((x0.reshape(1,-1), freq_x, amplt_x, phase_x))
varlist = ['x', 'freq', 'amplt', 'phase']
df = pd.DataFrame(data=res_fft.T, columns=varlist)
df.to_csv(pathP + varnm + '_freq.dat', sep=' ',
          index=False, float_format='%1.8e')

# %% alpha with x by time sequential data
delta = 1e-3
lst = pd.read_csv(pathB + 'LST_TS.dat', sep=' ',
                  index_col=False, skipinitialspace=True)
var = pd.read_csv(pathP + varnm + '_freq.dat', sep=' ', 
                  skiprows=0, index_col=False, skipinitialspace=True)
alpha_i = fv.sec_ord_fdd(var['x'], var['amplt'])
alpha_r = -fv.sec_ord_fdd(var['x'], var['phase'])
alpha_i = alpha_i / var['amplt']
fig = plt.figure(figsize=(6.4, 3.2))
#fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc("font", size=textsize)
ax1 = fig.add_subplot(121)
ax1.plot(var['x'], alpha_r*2*np.pi, 'k-', linewidth=1.2)
ax1.plot(lst['x'], lst['alpha_r']/lst['bl'] * delta, 'k:', linewidth=1.2)
ax1.set_xlim([-40.0, -6.0])
ax1.set_ylim([0.0, 4.0])
# ax.set_xticks(np.linspace(2175, 2275, 3))
ax1.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax1.yaxis.offsetText.set_fontsize(numsize)
ax1.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax1.set_ylabel(r'$\alpha_r^* \delta_0$', fontsize=textsize)
ax1.tick_params(labelsize=numsize)
ax1.grid(b=True, which="both", linestyle=":")
ax1.annotate(r"$(a)$", xy=(-0.18, 1.00), xycoords='axes fraction',
             fontsize=numsize)

matplotlib.rc("font", size=textsize)
ax = fig.add_subplot(122)
ax.plot(var['x'], alpha_i*2*np.pi, 'k-', linewidth=1.2)
ax.plot(lst['x'], -lst['alpha_i']/lst['bl'] * delta, 'k:', linewidth=1.2)
ax.set_xlim([-40.0, -6.0])
ax.set_ylim([-0.5, 0.5])
# ax.set_xticks(np.linspace(2175, 2275, 3))
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-1, 1))
ax.yaxis.offsetText.set_fontsize(numsize)
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$-\alpha_i^* \delta_0$', fontsize=textsize)
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")
ax.annotate(r"$(b)$", xy=(-0.18, 1.00), xycoords='axes fraction',
             fontsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + varnm + "_alpha_x" + ".svg", dpi=300)
plt.show()

# %% extract probe data along spanwise
VarList = [
    'x',
    'y',
    'z',
    'u',
    'v',
    'w',
    'p',
    'vorticity_1',
    'vorticity_2',
    'vorticity_3',
    'Q-criterion',
    'L2-criterion',
    '|grad(rho)|',
    'T',
    '|gradp|'
]

path1 = path + "TP_data_01386484/"
equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2+ddz({p})**2)'
p2p.ReadINCAResults(path1, VarList, SavePath=path,
                    OutFile="SpanwiseFlow", Equ=equ)
# %% extract probe data along spanwise
pathSW = path + 'Spanwise/'
x0 = np.arange(-40.0, 0.0 + 0.5, 0.5)
y0 = 0.001953125
dirs = sorted(os.listdir(path1))
fm = pd.read_hdf(path + "SpanwiseFlow.h5")
for j in range(np.size(x0)):
    filenm = pathSW + 'spanwise_' + str(x0[j]) + '.dat'
    var = fm.loc[(fm['x']==x0[j]) & (fm['y']==y0), ['z', 'u', 'p']].values
    df = pd.DataFrame(data=var, columns=['z', 'u', 'p'])
    df.to_csv(filenm, sep=' ', index=False, float_format='%1.8e')
    
base = pd.read_csv(pathB + 'baseline.dat', sep=' ', skiprows=0,
                   index_col=False, skipinitialspace=True) 

# %% variables along spanwise direction
varnm = 'p'
if varnm == 'u':
    ylab = r"$u^\prime / u_\infty$"
else:
    ylab = r"$p^\prime/(\rho_\infty u_\infty ^2)$"
xloc = [-39.0, -30.0, -20.0, -10.0]
curve= ['k-', 'b--', 'k:', 'b-.']
fig3, ax3 = plt.subplots(figsize=(6.0, 3.0))
matplotlib.rc("font", size=textsize)
# ax3 = fig.add_subplot(122)
for i in range(4):
    filenm = pathSW + 'spanwise_' + str(xloc[i]) + '.dat'   
    var = pd.read_csv(filenm, sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
    # val = var[varnm] - np.mean(var[varnm])
    val = var[varnm] - base.loc[base['x']==xloc[i], [varnm]].values[0]
    ax3.plot(var['z'], val, curve[i], linewidth=1.5)

ax3.set_ylabel(ylab, fontsize=textsize)
ax3.set_xlabel(r"$z / \delta_0$", fontsize=textsize)
# ax3.set_xlim([0.0, 25.0])
# ax3.set_ylim([0.0, 1.0])
# ax3.set_yticks(np.arange(0.4, 1.3, 0.2))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax3.grid(b=True, which="both", linestyle=":")
ax3.yaxis.offsetText.set_fontsize(numsize)
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + varnm + "_z.svg", dpi=300)
plt.show()

# %% Fourier transform of variables with spanwise distance
x0 = np.arange(-40.0, -0.5 + 0.5, 0.5)
num_samp = 257
freq_samp = 1 / 0.0625
# freq = np.linspace(freq_samp / num_samp, freq_samp / 2, math.ceil(num_samp/2))
freq = np.linspace(0.0, freq_samp / 2, math.ceil(num_samp/2), endpoint=False)
freq_x = np.zeros((1, np.size(x0)))
amplt_x = np.zeros((1, np.size(x0)))
phase_x = np.zeros((1, np.size(x0)))
for i in range(np.size(x0)):
    var = pd.read_csv(pathSW + 'spanwise_' + str(x0[i]) + '.dat', sep=' ', 
                      skiprows=0, index_col=False, skipinitialspace=True)
    var_per = var[varnm] - base.loc[base['x']==x0[i], [varnm]].values[0]
    var_per = var_per - np.mean(var_per) # make sure mean value is zero
    # 
    var_fft = np.fft.rfft(var_per)  # remove value at 0 frequency    
    amplt = np.abs(var_fft)
    phase = np.angle(var_fft, deg=False)
    ind = np.argmax(amplt)
    freq_x[0, i] = freq[ind]
    amplt_x[0, i] = amplt[ind]
    phase_x[0, i] = phase[ind]
res_fft = np.concatenate((x0.reshape(1,-1), freq_x, amplt_x, phase_x))
varlist = ['x', 'beta', 'amplt', 'phase']
df = pd.DataFrame(data=res_fft.T, columns=varlist)
df.to_csv(pathSW + varnm + '_beta.dat', sep=' ',
          index=False, float_format='%1.8e')

# %% Amplitude and phase with spanwise distance
fig = plt.figure(figsize=(6.4, 3.0))
ax = fig.add_subplot(121)
matplotlib.rc("font", size=textsize)
ax.plot(freq, amplt, 'k-')
ax.set_xscale("log")
ax.set_xlabel(r"$\beta \delta_0 / 2\pi$", fontsize=textsize)
ax.set_ylabel(r"$A({}^\prime)$".format(varnm), fontsize=textsize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax.tick_params(labelsize=numsize)
ax.yaxis.offsetText.set_fontsize(numsize)
ax.grid(b=True, which="both", linestyle=":")

ax1 = fig.add_subplot(122)
matplotlib.rc("font", size=textsize)
ax1.plot(freq, phase, 'k-')
ax1.set_xscale("log")
ax1.set_xlabel(r"$\beta \delta_0 / 2\pi$", fontsize=textsize)
ax1.set_ylabel(r"$\theta({}^\prime)$[rad]".format(varnm), fontsize=textsize)
# ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
ax1.yaxis.offsetText.set_fontsize(numsize)
ax1.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1)
plt.savefig(pathF + varnm + "_fft_z" + str(x0[i]) + ".svg", dpi=300)
plt.show()

# %% alpha with x by spanwise profiles
delta = 1e-3
lst = pd.read_csv(pathB + 'LST_TS.dat', sep=' ',
                  index_col=False, skipinitialspace=True)
var = pd.read_csv(pathSW + 'u_beta.dat', sep=' ', 
                  skiprows=0, index_col=False, skipinitialspace=True)
alpha_i = fv.sec_ord_fdd(var['x'], var['amplt'])
alpha_r = -fv.sec_ord_fdd(var['x'], var['phase'])
alpha_i = alpha_i / var['amplt']
fig = plt.figure(figsize=(6.4, 3.2))
#fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc("font", size=textsize)
ax1 = fig.add_subplot(121)
ax1.plot(var['x'], alpha_r, 'k-', linewidth=1.2)
ax1.plot(lst['x'], lst['alpha_r']/lst['bl'] * delta, 'k:', linewidth=1.2)
ax1.set_xlim([-40.0, -6.0])
# ax1.set_ylim([0.2, 0.4])
# ax.set_xticks(np.linspace(2175, 2275, 3))
ax1.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax1.yaxis.offsetText.set_fontsize(numsize)
ax1.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax1.set_ylabel(r'$\alpha_r^* \delta_0$', fontsize=textsize)
ax1.tick_params(labelsize=numsize)
ax1.grid(b=True, which="both", linestyle=":")
ax1.annotate("(a)", xy=(-0.18, 1.00), xycoords='axes fraction',
             fontsize=numsize)

matplotlib.rc("font", size=textsize)
ax = fig.add_subplot(122)
ax.plot(var['x'], alpha_i, 'k-', linewidth=1.2)
ax.plot(lst['x'], -lst['alpha_i']/lst['bl'] * delta, 'k:', linewidth=1.2)
ax.set_xlim([-40.0, -6.0])
ax.set_ylim([-1, 1])
# ax.set_xticks(np.linspace(2175, 2275, 3))
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-1, 1))
ax.yaxis.offsetText.set_fontsize(numsize)
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$-\alpha_i^* \delta_0$', fontsize=textsize)
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")
ax.annotate("(b)", xy=(-0.18, 1.00), xycoords='axes fraction',
             fontsize=numsize)
plt.savefig(pathF + varnm + "_alpha_x-by_z" + ".svg", dpi=300)
plt.show()

# %% boundary layer profile
path = "/media/weibo/VID2/BFS_M1.7TS/"
path2 = path + "Slice/TP_2D_Z_03/"

#file = glob(pathB + '*plt')
#p2p.ReadAllINCAResults(pathB, FoldPath2=pathB, 
#                       FileName=file, SpanAve='Y', OutFile="BaseFlow")
base = pd.read_hdf(path2 + 'TP_2D_Z_03_00899.50.h5')
varlist = ['x', 'y', 'z', 'u', 'v', 'w', 'p', 'T']
x0 = np.arange(-40.0, 0.0 + 1.0, 1.0)
for i in range(np.size(xloc)):
    df = base.loc[base['x']==xloc[i], varlist]
    df.to_csv(pathSW + 'Z_03_00899.50_' + str(xloc[i]) + '.dat',
              index=False, float_format='%1.8e', sep=' ', )

# %% Save boundary layer profile at a X location
for j in range(np.size(x0)):
    filenm = pathP + 'BL_Profile_' + str(x0[j]) + '.h5'
    var = fm.loc[fm['x']==x0[j], ['y', 'u', 'p']]
    y_uniq = np.unique(var['y'])
    time = np.repeat(stime, np.size(y_uniq))
    df = pd.DataFrame(data=np.hstack((time.reshape(-1,1), var)),
                      columns=['time', 'y', 'u', 'p'])
    df.to_hdf(filenm, 'w', format='fixed')   
# %% Fourier tranform of BL profile
xloc = [-36.0, -24.0, -10.0]
num_samp = np.size(stime)
freq = np.linspace(0.0, freq_samp / 2, math.ceil(num_samp/2), endpoint=False)
bf = pd.read_hdf(pathB + 'BaseFlow.h5')
for j in range(np.size(xloc)):
    file = 'BL_Profile_' + str(xloc[j]) + '.h5'
    fm = pd.read_hdf(pathP + file)
    bf1 = bf.loc[bf['x']==xloc[j], ['y', varnm]]    
    y_uni = np.unique(fm['y'])
    profile = np.zeros( (np.size(y_uni), 3) )
    for i in range(np.size(y_uni)):
        var1 = var.loc[var['y']==y_uni[i], varnm]
        base_var = bf1.loc[bf1['y']==y_uni[i], varnm]
        var1 = var1 - base_var 
        var_per = var1 - np.mean(var1)  # make sure mean value is zero
        var_fft = np.fft.rfft(var_per)  # remove value at 0 frequency    
        amplt = np.abs(var_fft)
        phase = np.angle(var_fft, deg=False)
        ind = np.argmax(amplt)
        profile[i, 0] = freq[ind]
        profile[i, 1] = amplt[ind]
        profile[i, 2] = phase[ind]
    df = pd.DataFrame(data=np.hstack((y_uni.reshape(-1, 1), profile)),
                      columns=['y', 'freq', 'amplit', 'phase'])
    filenm = pathP + varnm + '_profile_ftt_' + str(xloc[j]) + '.dat'
    df.to_csv(filenm, sep=' ', index=False, float_format='%1.8e')

# %% plot BL profile of amplitude along streamwise
filenm = pathP + varnm + '_profile_ftt_' + str(-36.0) + '.dat'
amp_fft = pd.read_csv(filenm, sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
ts_profile = pd.read_csv(path + 'UnstableMode.inp', sep=' ',
                         index_col=False, skiprows=4,
                         skipinitialspace=True)
ts_profile['u'] = np.sqrt(ts_profile['u_r']**2+ts_profile['u_i']**2)
ts_profile['p'] = np.sqrt(ts_profile['p_r']**2+ts_profile['p_i']**2)
# normalized
ts_profile['u'] = ts_profile['u'] / np.max(ts_profile['u'])
ts_profile['p'] = ts_profile['p'] / np.max(ts_profile['p'])
# plot lines
fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc('font', size=numsize)
amp_norm = amp_fft['amplit']/np.max(amp_fft['amplit'])
ax.scatter(amp_norm, amp_fft['y'], s=12, marker='o',
           facecolors='w', edgecolors='k', linewidths=0.8)
ax.plot(ts_profile.u, ts_profile.y, 'k', linewidth=1.2)
ax.set_ylim([0, 5])
ax.set_xlabel(r'$|p^\prime|/|p^\prime|_{max}$', fontsize=textsize)
ax.set_ylabel(r'$y/\delta_0$', fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLAmplit_" + varnm + ".svg", bbox_inches="tight", pad_inches=0.1
)

# %% plot BL profile along streamwise
bf = pd.read_hdf(pathB + 'BaseFlow.h5')
varnm = 'p'
fig, ax = plt.subplots(1, 7, figsize=(6.4, 2.2))
fig.subplots_adjust(hspace=0.5, wspace=0.15)
matplotlib.rc('font', size=numsize)
title = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$']
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
xcoord = np.array([-40.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0])
for i in range(np.size(xcoord)):
    filenm = pathSW + 'Z_03_00899.50_' + str(xcoord[i]) + '.dat'
    var = pd.read_csv(filenm, sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
    basevar = bf.loc[bf['x']==xcoord[i], ['y', varnm]]   
    y0 = var['y']
    baseval = np.interp(y0, basevar['y'], basevar[varnm])
    q0 = var[varnm] - baseval
    ax[i].plot(q0, y0, "k-")
    ax[i].set_ylim([0, 3])
    if i != 0:
        ax[i].set_yticklabels('')
    # ax[i].set_xticks([0, 0.5, 1], minor=True)
    ax[i].set_title(r'$x/\delta_0={}$'.format(xcoord[i]), fontsize=numsize - 2)
    ax[i].grid(b=True, which="both", linestyle=":")
ax[0].set_ylabel(r"$\Delta y/\delta_0$", fontsize=textsize)
ax[3].set_xlabel(r'$u/u_\infty$', fontsize=textsize)
plt.tick_params(labelsize=numsize)
plt.show()
plt.savefig(
    pathF + "BLPerturb.svg", bbox_inches="tight", pad_inches=0.1
)

# %% Extract spanwise-averaged flow, obtain fluctuations
path = "/media/weibo/VID2/BFS_M1.7TS/"
path1 = path + "TP_data_01386484/"
VarList = [
    'x',
    'y',
    'z',
    'u',
    'v',
    'w',
    'p',
    'rho',
    'vorticity_1',
    'vorticity_2',
    'vorticity_3',
    'T',
]
equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2+ddz({p})**2)'
df, time = p2p.ReadINCAResults(path1, VarList)
# df.to_hdf(path + "TP_data_" + str(time) + ".h5", 'w', format='fixed')
# df = pd.read_hdf(path + "TP_data.h5")
grouped = df.groupby(['x', 'y'], as_index=False)
# df2 = grouped.mean().reset_index()
fluc = lambda x: (x-x.mean())
df1 = grouped.transform(fluc)
df2 = df.sort_values(by=['x', 'y', 'z'])
df1.insert(0, 'x', df2['x'])
df1.insert(1, 'y', df2['y'])
df1.to_hdf(path + 'ZFluctuation_' + str(time) + '.h5', 'w', format='fixed')

# %% Extract base flow, obtain fluctuations
path2 = path + "TP_data_baseflow/"
orig = pd.read_hdf(path + "ZFluctuation_899.5.h5")[VarList]
# base = p2p.ReadAllINCAResults(path2, path, SpanAve=False, OutFile='baseflow')
base = pd.read_hdf(path + "baseflow.h5")[VarList]
varname = [
    'u',
    'v',
    'w',
    'p',
    'rho',
    'vorticity_1',
    'vorticity_2',
    'vorticity_3',
    'T',
]

# %%
# df_zone = p2p.save_zone_info(path1, filename=path + "ZoneInfo.dat")
df_zone = pd.read_csv(path + "ZoneInfo.dat", sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
for i in range(10): # (np.shape(df_zone)[0]):
    file = df_zone.iloc[i]
    cube = orig.query(
        "x>={0} & x<={1} & y>={2} & y<={3}".format(
            file['x1'],
            file['x2'],
            file['y1'],
            file['y2']
        )
    )
    
    if (file['nz'] != len(np.unique(cube['z']))):
        # remove dismatch grid point on the boundary of the block
        zlist = np.linspace(file['z1'], file['z2'], int(file['nz']))
        blk1 = cube[cube['z'].isin(zlist)].reset_index(drop=True)
    else:
        blk1 = cube

    blk0 = base.query(
        "x>={0} & x<={1} & y>={2} & y<={3}".format(
            file['x1'],
            file['x2'],
            file['y1'],
            file['y2']
        )
    )
    new = blk0.loc[blk0.index.repeat(int(file['nz']))]
    new = new.reset_index(drop=True)
    flc = blk1
    flc.update(flc[varname] - new[varname])
    if (file['nx'] != len(np.unique(flc['x']))):
        xlist = np.unique(flc['x'])[0::2]
        flc = flc[flc['x'].isin(xlist)]
    if (file['ny'] != len(np.unique(flc['y']))):
        ylist = np.unique(flc['y'])[0::2]
        flc = flc[flc['y'].isin(ylist)]
    p2p.frame2tec3d(flc, path, 'fluc' + str(i), stime=899.5)
    


# %%
grouped = orig.groupby(['x', 'y'])
count = grouped.size().reset_index(name='count')
base_exts = base['u'].repeat(count['count'].values)
# %%
fm = pd.DataFrame()
for i in range(np.shape(base1)[0]):
    slc = base1.iloc[i]
    key = tuple(slc[['x', 'y']].values)
    grp = grouped.get_group(key)
    flc = grp[varname] - slc[varname]
    flc.insert(0, 'x', grp['x'])
    flc.insert(1, 'y', grp['y'])
    flc.insert(2, 'z', grp['z'])
    fm = fm.append(flc, ignore_index=True)    

# %% convert h5 to plt
df1 = pd.read_hdf(path + "ZFluctuation_899.5.h5")
df_zone = p2p.save_zone_info(path1, filename=path + "ZoneInfo.dat")
p2p.save_tec_index(df1, df_zone, filename=path + "ReadList.dat")
df_zone = pd.read_csv(path + "ZoneInfo.dat", sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
fileid = pd.read_csv(path + 'ReadList.dat', sep=' ', skiprows=0,
                     index_col=False, skipinitialspace=True)
p2p.mul_zone2tec(path, "test", df_zone, df1, stime=899.5)


