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
from scipy.interpolate import interp1d, splev, splrep #, spline
from data_post import DataPost
import variable_analysis as fv
from timer import timer
from scipy.interpolate import griddata
import plt2pandas as p2p
from glob import glob


# %% data path settings
path = "/media/weibo/Data2/BFS_M1.7TS/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathL = path + "LST/"

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
textsize = 13
numsize = 10

# %%
# path = "/media/weibo/VID2/BFS_M1.7SFD/"
# pathB = path + 'TP_data_15709808/'
# file = glob(pathB + '*plt')
# p2p.ReadAllINCAResults(pathB, FoldPath2=pathB, 
#                        FileName=file, SpanAve='Y', OutFile="BaseFlow")
# base = pd.read_hdf(pathL + 'BaseFlow.h5')
base = pd.read_hdf(pathM + 'MeanFlow.h5')
varlist = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'Mach', 'T']
base['u'] = base['<u>']
base['v'] = base['<v>']
base['w'] = base['<w>']
base['rho'] = base['<rho>']
base['p'] = base['<p>']
base['Mach'] = base['u']/np.sqrt(1.4*287*base['<T>'])
base['T'] = base['<T>']
x_loc = np.arange(-40.0, 0.0 + 1.0, 1.0)
for i in range(np.size(x_loc)):
    df = base.loc[(base['x']==x_loc[i]) & (base['z']==0.0), varlist]
    df.to_csv(pathL + 'InputProfile_' + str(x_loc[i]) + '.dat',
              index=False, float_format='%1.8e', sep=' ')

# %% Contour of TS mode
x_inlet = 0.382185449774020e3
blasius = 1.579375403141185e-04 # 1.669135812043661e-04
l_ref = 1.0e-3
rescale = blasius * np.sqrt(x_inlet/(x_inlet - 0.04)) / l_ref
ts_mode = pd.read_csv(path + 'TSMode2_-40.0.dat', sep=' ',
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
rg1 = np.linspace(-0.0028, 0.0013, 42)
x_b = x * rescale
y_b = y * rescale
growth_b = growth * rescale
cbar = ax.contourf(x_b, y_b, growth_b, cmap='rainbow_r', levels=rg1)  # rainbow_r
cs = ax.contour(x_b, y_b, growth_b, levels=[-0.0025, -0.0023, -0.0019, -0.0015],
                linewidths=1.0, colors='w')
matplotlib.rc("font", size=textsize)
ax.clabel(cs, cs.levels, inline=False, fmt=r'$%5.4f$', inline_spacing=0, 
          rightside_up=True, fontsize=numsize-1, colors='k')
ax.set_xlim(0.03, 0.12)
ax.set_ylim(0.008, 0.024)
ax.set_yticks(np.arange(0.008, 0.024 + 0.004, 0.004))
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax.yaxis.offsetText.set_fontsize(numsize)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$\beta l$", fontsize=textsize)
ax.set_ylabel(r"$\omega \l / u_\infty$", fontsize=textsize)
ax.axvline(x=0.393, ymin=0, ymax=0.5,
           color="gray", linestyle="--", linewidth=1.0)
ax.axhline(y=0.101, xmin=0, xmax=0.32,
           color="gray", linestyle="--", linewidth=1.0)
# Add colorbar
rg2 = np.linspace(-0.0028, 0.0013, 3)
cbaxes = fig.add_axes([0.65, 0.80, 0.22, 0.06])  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal", ticks=rg2)
cbar.formatter.set_powerlimits((-2, 2))
cbar.ax.xaxis.offsetText.set_fontsize(numsize)
cbar.update_ticks()
cbar.set_label(
    r"$\alpha_i \l$", rotation=0, x=-0.15,
    labelpad=-25, fontsize=textsize
)
cbaxes.tick_params(labelsize=numsize)
plt.savefig(pathF + "TS_mode_contour.pdf", bbox_inches="tight")
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
# %% BL profiles to a specific format
path = '/mnt/work/cases/flat_260/'
col = [
       'y', 'u_r', 'u_i', 'v_r', 'v_i', 'w_r', 'w_i', 
       't_r', 't_i', 'p_r', 'p_i', 'rho_r', 'rho_i'
       ]
ts_profile = pd.read_csv(path + 'profiles.dat', sep=' ',
                         names=col, header=0,
                         index_col=False, skiprows=1,
                         skipinitialspace=True)
myfile = path + 'UnstableMode.dat'
omega_r = 0.62
omega_i = -0.00309260  
alpha = 0.67465179 
beta = 0.0
with open(myfile, 'w') as file:
    # file.write(col[:11]+'\t')
    file.write('alpha = ' + str(alpha) + '\n')
    file.write('beta = ' + str(beta) + '\n')
    file.write('omega_r = ' + str(omega_r) + '\n')
    file.write('omega_i = ' + str(omega_i) + '\n')
    file.write(" ".join(col[:11]) + '\n')
    input_mode = ts_profile[col[:11]]
    input_mode.iloc[0][:7] = 0
    input_mode.to_csv(myfile, sep=' ', index=False, header=0, na_rep='0',
                      columns=col[:11], mode='a', float_format='%1.8e')
# %% Plot LST perturbations profiles
path = '/mnt/work/cases/flat_202302/export/'
ts_profile = pd.read_csv(path + 'UnstableMode.inp', sep='\t',
                         index_col=False, skiprows=4,
                         skipinitialspace=True)
ts_profile['u'] = np.sqrt(ts_profile['u_r']**2+ts_profile['u_i']**2)
ts_profile['v'] = np.sqrt(ts_profile['v_r']**2+ts_profile['v_i']**2)
ts_profile['w'] = np.sqrt(ts_profile['w_r']**2+ts_profile['w_i']**2)
ts_profile['t'] = np.sqrt(ts_profile['t_r']**2+ts_profile['t_i']**2)
ts_profile['p'] = np.sqrt(ts_profile['p_r']**2+ts_profile['p_i']**2)
# normalized
fct = 4
var_ref = np.max(ts_profile['u'])
ts_profile['u'] = ts_profile['u'] / var_ref
ts_profile['v'] = ts_profile['v'] / var_ref
ts_profile['w'] = ts_profile['w'] / var_ref
ts_profile['p'] = ts_profile['p'] / var_ref
ts_profile['t'] = ts_profile['t'] / var_ref / fct
fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc("font", size=textsize)
# plot lines
ax.plot(ts_profile.u, ts_profile.y, 'k', linewidth=1.2)
ax.plot(ts_profile.v, ts_profile.y, 'r', linewidth=1.2)
ax.plot(ts_profile.w, ts_profile.y, 'g', linewidth=1.2)
ax.plot(ts_profile.p, ts_profile.y, 'b', linewidth=1.2)
ax.plot(ts_profile.t, ts_profile.y, 'c', linewidth=1.2)
# plot scatter
#ax.scatter(ts_profile.u[0::4], ts_profile.y[0::4], s=15, marker='x',
#           facecolors='k', edgecolors='k', linewidths=0.5, label=r'$u$')
ax.set_xlim(0.0, 4.0)
ax.tick_params(axis='x', which='major', pad=5)
ax.set_ylim(0.0, 10.0)
ax.set_xlabel(r'$|\tilde{q}|/|\tilde{u}|_\mathrm{max}$', fontsize=textsize)
ax.set_ylabel(r'$y\mathrm{(mm)}$', fontsize=textsize)
# ax.set_ylabel(r'$y/\delta_0$', fontsize=textsize)
ax.tick_params(labelsize=numsize)
ax.grid(visible=True, which="both", linestyle=":")
plt.legend( (r'$u$', r'$v$', r'$w$', r'$p$', r'$T/4$'),
            fontsize=numsize, framealpha=0.5 )
plt.savefig(path + "2nd_mode_profile.png", dpi=300, bbox_inches="tight")
plt.show()

# %% omega-Rex
xloc = np.arange(-40.0, -6.0 + 1.0, 1.0)
x_inlet = 0.382185449774020e3
blasius = 1.579375403141185e-04 # 1.669135812043661e-04
delta = 1e-3
Re = 13718000
val = -0.0016
var = 'beta'
lst_var = np.zeros( (np.size(xloc), 3) )
Re_l = np.zeros(np.size(xloc))
for i in range(np.size(xloc)):
    xx = xloc[i]
    l_ref = blasius * np.sqrt((x_inlet + xx)/(x_inlet - 40.0))    
    ts_mode = pd.read_csv(pathL + 'TSMode_' + str(xx) + '_' + var + '.dat',
                          sep=' ', index_col=False, skipinitialspace=True)
    # rescaled by Blasius length
    ts_mode = ts_mode.sort_values(var).reset_index()
    ts_mode = ts_mode / delta * l_ref
    ts_mode['phi'] = ts_mode['phi'] * delta / l_ref
    # max alpha_i
    ind = ts_mode[['alpha_i']].idxmin()[0]
    # upper branch
    ts_mode1 = ts_mode.iloc[:ind]
    lst_var[i, 0] = interp1d(ts_mode1['alpha_i'], ts_mode1[var])(val)
    # ind1 = (ts_mode1['alpha_i']-val).abs().argsort()[:1]
    # lst_var[i, 0] = ts_mode[var][ind1.values].values
    # lower branch
    ts_mode2 = ts_mode.iloc[ind:]
    lst_var[i, 1] = interp1d(ts_mode2['alpha_i'], ts_mode2[var])(val)
    lst_var[i, 2] = ts_mode[var][ind]  # maximum alpha_i
    # ind2 = (ts_mode2['alpha_i']-val).abs().argsort()[:1]
    # lst_var[i, 1] = ts_mode[var][ind2.values].values
    Re_l[i] = Re * l_ref

varval = np.column_stack((Re_l, lst_var))
df = pd.DataFrame(data=varval, columns=['Re', var+'1', var+'2', var+'3'])
df.to_csv(pathL + 'LST_' + var + str(val) + '.dat',
          index=False, float_format='%1.8e', sep=' ')

# %% plot Rel-beta
beta1 = pd.read_csv(pathL + 'LST_beta-0.0014.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
beta2 = pd.read_csv(pathL + 'LST_beta-0.0015.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
beta3 = pd.read_csv(pathL + 'LST_beta-0.0016.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
Re1 = np.hstack((beta1['Re'], beta1['Re'][::-1]))
ReBeta1 = np.hstack((beta1['beta1'], beta1['beta2'][::-1]))
Re2 = np.hstack((beta2['Re'], beta2['Re'][::-1]))
ReBeta2 = np.hstack((beta2['beta1'], beta2['beta2'][::-1]))
Re3 = np.hstack((beta3['Re'], beta3['Re'][::-1]))
ReBeta3 = np.hstack((beta3['beta1'], beta3['beta2'][::-1]))
fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc("font", size=textsize)
# plot lines
y_s1 = np.linspace(ReBeta1[1:-1].min(), ReBeta1[1:-1].max(), 200)
tck = splrep(ReBeta1[1:-1], Re1[1:-1], s=0)
x_s1 = splev(y_s1, tck) # spline(ReBeta1[1:-1], Re1[1:-1], y_s1)

y_s2 = np.linspace(ReBeta2[1:-1].min(), ReBeta2[1:-1].max(), 200)
tck = splrep(ReBeta2[1:-1], Re2[1:-1], s=0)
x_s2 = splev(y_s2, tck) # x_s2 = spline(ReBeta2[1:-1], Re2[1:-1], y_s2)

y_s3 = np.linspace(ReBeta3[1:-1].min(), ReBeta3[1:-1].max(), 200)
tck = splrep(ReBeta3[1:-1], Re3[1:-1], s=0)
x_s3 = splev(y_s3, tck) # x_s3 = spline(ReBeta3[1:-1], Re3[1:-1], y_s3)
ax.plot(x_s1, y_s1, 'k-', linewidth=1.2)
ax.plot(x_s2, y_s2, 'k--', linewidth=1.2)
ax.plot(x_s3, y_s3, 'k:', linewidth=1.2)
xm_s = np.linspace(beta3['Re'][1:].min(), beta3['Re'][1:].max(), 100)
tck = splrep(beta3['Re'][1:], beta3['beta3'][1:])
ym_s = splev(xm_s, tck) # ym_s = spline(beta3['Re'][1:], beta3['beta3'][1:], xm_s, order=4)
ax.plot(beta3['Re'][1:], beta3['beta3'][1:], 'k-.', linewidth=1.2)
#ax.plot(beta1['Re'], beta1['beta1'], 'r-',
#        beta1['Re'], beta1['beta2'], 'r-', linewidth=1.2)
#ax.plot(beta2['Re'], beta2['beta1'], 'r--',
#        beta2['Re'], beta2['beta2'], 'r--', linewidth=1.2)
#ax.plot(beta3['Re'], beta3['beta1'], 'r:',
#        beta3['Re'], beta3['beta2'], 'r:', linewidth=1.2)
# ax.set_yscale('logit')
ax.set_xlim([2160, 2280])
ax.set_ylim([0.02, 0.13])
ax.set_xlabel(r'$Re_l$', fontsize=textsize)
ax.set_ylabel(r'$\beta l$', fontsize=textsize)
ax.set_xticks(np.linspace(2175, 2275, 3))
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-1, 1))
ax.yaxis.offsetText.set_fontsize(numsize)
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")
plt.savefig(pathF + "Re_beta.svg", bbox_inches="tight")
plt.show()

# %% plot Rel-omega
beta1 = pd.read_csv(pathL + 'LST_omega-0.0014.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
beta2 = pd.read_csv(pathL + 'LST_omega-0.0015.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
beta3 = pd.read_csv(pathL + 'LST_omega-0.0016.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
Re1 = np.hstack((beta1['Re'], beta1['Re'][::-1]))
ReBeta1 = np.hstack((beta1['omega1'], beta1['omega2'][::-1]))
Re2 = np.hstack((beta2['Re'], beta2['Re'][::-1]))
ReBeta2 = np.hstack((beta2['omega1'], beta2['omega2'][::-1]))
Re3 = np.hstack((beta3['Re'], beta3['Re'][::-1]))
ReBeta3 = np.hstack((beta3['omega1'], beta3['omega2'][::-1]))
fig, ax = plt.subplots(figsize=(3.2, 3.2))
matplotlib.rc("font", size=textsize)
# plot lines
y_s1 = np.linspace(ReBeta1[1:-1].min(), ReBeta1[1:-1].max(), 200)
tck = splrep(np.sort(ReBeta1[1:-1]), Re1[1:-1], s=0)
x_s1 = splev(y_s1, tck) # spline(ReBeta1[1:-1], Re1[1:-1], y_s1)

y_s2 = np.linspace(ReBeta2[1:-1].min(), ReBeta2[1:-1].max(), 200)
tck = splrep(np.sort(ReBeta2[1:-1]), Re2[1:-1], s=0)
x_s2 = splev(y_s2, tck) # x_s2 = spline(ReBeta2[1:-1], Re2[1:-1], y_s2)

y_s3 = np.linspace(ReBeta3[1:-1].min(), ReBeta3[1:-1].max(), 200)
tck = splrep(np.sort(ReBeta3[1:-1]), Re3[1:-1], s=0)
x_s3 = splev(y_s3, tck) # x_s3 = spline(ReBeta3[1:-1], Re3[1:-1], y_s3)
x_s1[0] = Re1[1]
y_s1[0] = ReBeta1[1]
x_s2[0] = Re2[1]
y_s2[0] = ReBeta2[1]
x_s3[0] = Re3[1]
y_s3[0] = ReBeta3[1]
ax.plot(x_s1, y_s1, 'k-', linewidth=1.2)
ax.plot(x_s2, y_s2, 'k--', linewidth=1.2)
ax.plot(x_s3, y_s3, 'k:', linewidth=1.2)
xm_s = np.linspace(beta1['Re'][1:].min(), beta1['Re'][1:].max(), 100)
tck = splrep(beta1['Re'][1:], beta1['omega3'][1:], s=0)
ym_s = splev(xm_s, tck)  # spline(beta1['Re'][1:], beta1['omega3'][1:], xm_s, order=4)
ax.plot(beta1['Re'][1:], beta1['omega3'][1:], 'k-.', linewidth=1.2)
# plot lines
#ax.plot(beta1['Re'], beta1['omega1'], 'r-',
#        beta1['Re'], beta1['omega2'], 'r-', linewidth=1.2)
#ax.plot(beta2['Re'], beta2['omega1'], 'r--',
#        beta2['Re'], beta2['omega2'], 'r--', linewidth=1.2)
#ax.plot(beta3['Re'], beta3['omega1'], 'r:',
#        beta3['Re'], beta3['omega2'], 'r:', linewidth=1.2)
ax.set_xlim([2160, 2280])
ax.set_ylim([0.006, 0.024])
ax.set_xticks(np.linspace(2175, 2275, 3))
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax.yaxis.offsetText.set_fontsize(numsize)
ax.set_xlabel(r'$Re_l$', fontsize=textsize)
ax.set_ylabel(r'$\omega l/u_\infty$', fontsize=textsize)
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")
plt.savefig(pathF + "Re_omega.pdf", bbox_inches="tight")
plt.show()

# %% TS-modes along streamwise
xloc = np.arange(-40.0, -1.0 + 1.0, 1.0)
x_inlet = 0.382185449774020e3
blasius = 1.579375403141185e-04 # 1.669135812043661e-04
delta = 1e-3
Re = 13718000
val = 0.06202
var = 'beta'
lst_ts = np.zeros( (np.size(xloc), 2) )
Re_l = np.zeros( (np.size(xloc), 3) )
pathL1 = pathL + 'from_baseflow/'
for i in range(np.size(xloc)):
    xx = xloc[i]
    l_ref = blasius * np.sqrt((x_inlet + xx)/(x_inlet - 40.0))    
    ts_mode = pd.read_csv(pathL1 + 'TSMode_' + var + '_' + str(xx) + '.dat',
                          sep=' ', index_col=False, skipinitialspace=True)
    # rescaled by Blasius length
    ts_mode = ts_mode.sort_values(var).reset_index()
    val1 = val / l_ref * delta
    # ts_mode = ts_mode / delta * l_ref
    # ts_mode['phi'] = ts_mode['phi'] * delta / l_ref
    lst_ts[i, 0] = interp1d(ts_mode[var], ts_mode['alpha_r'])(val1)
    lst_ts[i, 1] = interp1d(ts_mode[var], ts_mode['alpha_i'])(val1)
    Re_l[i, 0] = xx
    Re_l[i, 1] = l_ref
    Re_l[i, 2] = Re * l_ref

varval = np.column_stack((Re_l, lst_ts))
df = pd.DataFrame(data=varval, columns=['x', 'bl', 'Re', 'alpha_r', 'alpha_i'])
df.to_csv(pathL1 + 'LST_TS.dat',
          index=False, float_format='%1.8e', sep=' ')

# %% plot alpha
xloc = np.arange(-40.0, -1.0 + 1.0, 1.0)
beta1 = pd.read_csv(pathL + 'LST_TS.dat', sep=' ',
                    index_col=False, skipinitialspace=True)
fig = plt.figure(figsize=(6.4, 3.2))
matplotlib.rc("font", size=textsize)
ax = fig.add_subplot(121)
ax.plot(xloc, beta1['alpha_r'], 'k--', linewidth=1.2)
ax.set_xlim([-40.0, 0.0])
ax.set_ylim([0.16, 0.26])
# ax.set_xticks(np.linspace(2175, 2275, 3))
ax.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax.yaxis.offsetText.set_fontsize(numsize)
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$\alpha_r^* \delta_0$', fontsize=textsize)
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")
ax.annotate("(a)", xy=(-0.23, 1.00), xycoords='axes fraction',
             fontsize=numsize)

ax1 = fig.add_subplot(122)
ax1.plot(xloc, -beta1['alpha_i'], 'k--', linewidth=1.2)
ax1.set_xlim([-40.0, 0.0])
ax1.set_ylim([0.0, 0.020])
# ax1.set_xticks(np.linspace(2175, 2275, 3))
ax1.ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(-2, 2))
ax1.yaxis.offsetText.set_fontsize(numsize)
ax1.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax1.set_ylabel(r'$-\alpha_i^* \delta_0$', fontsize=textsize)
ax1.tick_params(labelsize=numsize)
ax1.grid(b=True, which="both", linestyle=":")
ax1.annotate("(b)", xy=(-0.21, 1.00), xycoords='axes fraction',
             fontsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.6, h_pad=1)
plt.savefig(pathF + "Re_alpha.svg")
plt.show()

# %% plot amplitude along x
path1 = 'E:/cases/large_oblique/'
var_x = pd.read_csv(path1 + 'var_x.dat', sep='\t', skipinitialspace=True)
path = 'E:/cases/project/LST/export/'
cols = ['x', 'omega_r', 'omega_i', 'alpha_r', 'alpha_i', 'beta_r', 'beta_i']
x_profile = pd.read_csv(path + 'x-eigenvalue.dat', sep=' ', names=cols,
                         index_col=False, skiprows=1,
                         skipinitialspace=True)
x_profile = x_profile.sort_values(by=['x'])
x1_profile = x_profile.iloc[540:941]
a0 = 1
fluc = a0*np.exp(x1_profile['alpha_i']*x1_profile['x'])
fig, ax = plt.subplots(figsize=(6.4, 6.4))
matplotlib.rc("font", size=textsize)
# plot lines
ax.scatter(x1_profile['x'], fluc, s=15, marker='o',
           facecolors='k', edgecolors='k', linewidths=0.5)
ax.plot(var_x['x']+270, np.sqrt(var_x['<T`T`>']), 'b-')
ax.set_xlim([280, 480])
# ax.set_ylim([-0.1, 0.3])
ax.set_xlabel(r'$x$', fontsize=textsize)
ax.set_ylabel(r'$q^\prime$', fontsize=textsize)
ax.tick_params(labelsize=numsize)
ax.grid(b=True, which="both", linestyle=":")
plt.savefig(path + "amplidue_x.svg", bbox_inches="tight")
plt.show()
