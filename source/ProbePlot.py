#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 18:58:21 2017
    This code for post-processing data from instantaneous/time-average plane
    data, need data file.                                                                   
@author: weibo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import griddata
from scipy.interpolate import spline
import sys
from DataPost import DataPost 
#import tqdm
#import LoadData as dt

plt.close ("All")
plt.rc('text', usetex=True)
font0 = {'family' : 'Times New Roman',
		'color' : 'k',
		'weight' : 'normal',
		'size' : 10,
}
font1 = {'family' : 'Times New Roman',
        'color' : 'k',
        'weight' : 'normal',
        'size' : 12,}

font2 = {'family' : 'Times New Roman',
         'color' : 'k',
         'weight' : 'normal',
         'size' : 14,
        }

font3 = {'family' : 'Times New Roman',
		'color' : 'k',
		'weight' : 'normal',
		'size' : 16,
}
# Get Frequency Weighted Power Spectral Density
def FW_PSD (VarZone, TimeZone, pic = None):
#    InputData ('x=50.txt')
    #InputData (xloc)
    #modify according to needs
    var = VarZone
    ave = np.mean (var)
    var_fluc = var-ave
    #    fast fourier transform and remove the half
    var_fft = np.fft.rfft (var_fluc)
    var_psd = abs(var_fft)**2
    num = np.size (var_fft)
    #    sample frequency
    fre_samp = num/(TimeZone[-1]-TimeZone[0])
    #f_var = np.linspace (0, f_samp/2, num)
    fre = np.linspace (fre_samp/2/num, fre_samp/2, num)
    fre_weighted = var_psd*fre
    if pic is not None:
        fig, ax = plt.subplots ()
        ax.semilogx (fre, fre_weighted)
        ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
        ax.set_xlabel (r'$f\delta_0/U_\infty$', fontdict = font2)
        ax.set_ylabel ('Weighted PSD, unitless', fontdict = font2)
        ax.grid (b=True, which = 'both', linestyle = '--')
        fig.savefig (pic, dpi = 600)
        #plt.show ()
    return (fre, fre_weighted)

path = "./"
path1 = "../probes/"
path2 = "../BProbes/"
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

baseflow = DataPost()
baseflow.LoadData(path+'BaseflowZ0Slice.txt', 2, 0.0)
#baseflow.unique_rows()
baseflow.AveAtSameXYZ('Part')
baseflow.GetWallDist(0.0)
t0 = 499.07144
t1 = 960 #960
t2 = 1260 #

LSTData = np.loadtxt("LST.dat", skiprows = 1)
LSTx    = LSTData[:,0]
LSTalpha_r= LSTData[:,1]
LSTalpha_i= LSTData[:,2]
LSTbeta= LSTData[:,3]
LSTomega = LSTData[:,4]

#%% Read data for Streamwise variations of frequency of a specific variable
VarName = ['itstep', 'time', 'u', 'v', 'w', 'rho', 'E', 'WallDist', 'p']
Probe0 = DataPost()
Probe0.LoadProbeData (0.0, 0.0, 0.0, path1)
Probe0.ExtraData('time', t1, t2)
#Probe0.AveAtSameXYZ('All')
#time1 = Probe0.time
Probe10 = DataPost()
Probe10.LoadProbeData (10.0, 0.0, 0.0, path1)
#Probe10.unique_rows()
Probe10.ExtraData('time', t1, t2)
#Probe10.AveAtSameXYZ('All')
Probe20 = DataPost()
Probe20.LoadProbeData (20.0, 0.0, 0.0, path1)
#Probe20.unique_rows()
Probe20.ExtraData('time', t1, t2)
#Probe20.AveAtSameXYZ('All')
#time2 = Probe20.time
Probe30 = DataPost()
Probe30.LoadProbeData (30.0, 0.0, 0.0, path1)
#Probe30.unique_rows()
Probe30.ExtraData('time', t1, t2)
#Probe30.AveAtSameXYZ('All')

Probe40 = DataPost()
Probe40.LoadProbeData (40.0, 0.0, 0.0, path1)
#Probe40.unique_rows()
Probe40.ExtraData('time', t1, t2)
#Probe40.AveAtSameXYZ('All')
time4 = Probe40.time

#%% Read Baseflow Data
BProbe0 = DataPost ()
BProbe0.LoadProbeData (0.0, 0.0, 0.0, path2)
BProbe0.ExtraData ('time', t0, t0)
#BProbe0P = BProbe0.EquValProfile(t0, BProbe0.time, BProbe0.p, 1)
BProbe10 = DataPost ()
BProbe10.LoadProbeData (10.0, 0.0, 0.0, path2)
BProbe10.ExtraData ('time', t0, t0)
#BProbe10P = BProbe10.EquValProfile(t0, BProbe10.time, BProbe10.p, 1)
BProbe20 = DataPost ()
BProbe20.LoadProbeData (20.0, 0.0, 0.0, path2)
BProbe20.ExtraData ('time', t0, t0)
#BProbe20P = BProbe20.EquValProfile(t0, BProbe20.time, BProbe20.p, 1)
BProbe30 = DataPost ()
BProbe30.LoadProbeData (30.0, 0.0, 0.0, path2)
BProbe30.ExtraData ('time', t0, t0)
#BProbe30P = BProbe30.EquValProfile(t0, BProbe30.time, BProbe30.p, 1)
BProbe40 = DataPost ()
BProbe40.LoadProbeData (40.0, 0.0, 0.0, path2)
BProbe40.ExtraData ('time', t0, t0)
#BProbe40P = BProbe40.EquValProfile(t0, BProbe40.time, BProbe40.p, 1)




#%% Streamwise variations of time evolution of a specific variable

fig = plt.figure()
ax = fig.add_subplot(411)
ax.set_title (r'$x=0$', fontdict = font1)
ax.set_xlim ([t1, t2])
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
#ax.set_xlabel (r'$t u_\infty/\delta$', fontdict = font1)
ax.set_ylabel (r'$p/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
#ax.text (850, 6.5, 'x=0', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
#Probe0P = Probe0.p-BProbe0P
#grow0, time0 = Probe0.GrowthRate(Probe0.time, Probe0P)
#ax.plot (time0, grow0, 'k', linewidth = 1.5)
ax.plot (Probe0.time, Probe0.p, 'k', linewidth = 1.5)

ax = fig.add_subplot(412)
ax.set_title (r'$x=10$', fontdict = font1)
ax.set_xlim ([t1, t2])
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_ylabel (r'$p/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
#Probe10P = Probe10.p-BProbe10P
#grow10, time10 = Probe10.GrowthRate(Probe10.time, Probe10P)
#ax.plot (time10, grow10, 'k', linewidth = 1.5)
ax.plot (Probe10.time, Probe10.p, 'k', linewidth = 1.5)

ax = fig.add_subplot(413)
ax.set_title (r'$x=20$', fontdict = font1)
ax.set_xlim ([t1, t2])
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_ylabel (r'$p/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
#Probe20P = Probe20.p-BProbe20P
#grow20, time20 = Probe20.GrowthRate(Probe20.time, Probe20P)
#ax.plot (time20, grow20, 'k', linewidth = 1.5)
ax.plot (Probe20.time, Probe20.p, 'k', linewidth = 1.5)

ax = fig.add_subplot(414)
ax.set_title (r'$x=40$', fontdict = font1)
ax.set_xlim ([t1, t2])
#ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_xlabel (r'$t u_\infty/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$p/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
#ax.text (850, 6.5, 'x=0', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
#Probe40P = Probe40.p-BProbe40P
#grow40, time40 = Probe40.GrowthRate(Probe40.time, Probe40P)
ax.plot (Probe40.time, Probe40.p, 'k', linewidth = 1.5)
#ax.plot (Probe40.time, Probe40.p, 'k', linewidth = 1.5)
plt.tight_layout (pad = 0.5, w_pad = 0.2, h_pad = 1)
plt.savefig (path+'StreawiseTimeEvolution.pdf', dpi = 300)
plt.show ()


#%% Frequency Weighted Power Spectral Density
fig = plt.figure()
ax = fig.add_subplot(221)
ax.set_title (r'$x=0$', fontdict = font1)
#ax.set_xlim ([720, 960])
#ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
#ax.set_xlabel (r'$f\delta_0/U_\infty$', fontdict = font1)
ax.set_ylabel ('Weighted PSD, unitless', fontdict = font1)
ax.grid (b=True, which = 'both', linestyle = ':')
Fre0, FPSD0 = FW_PSD (Probe0.p, Probe0.time)
ax.semilogx (Fre0, FPSD0, 'k', linewidth = 1.5)

ax = fig.add_subplot(222)
ax.set_title (r'$x=10$', fontdict = font1)
#ax.set_xlim ([720, 960])
#ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
#ax.set_xlabel (r'$f\delta_0/U_\infty$', fontdict = font1)
#ax.set_ylabel ('Weighted PSD, unitless', fontdict = font1)
ax.grid (b=True, which = 'both', linestyle = ':')
Fre10, FPSD10 = FW_PSD (Probe10.p, Probe10.time)
ax.semilogx (Fre10, FPSD10, 'k', linewidth = 1.5)

ax = fig.add_subplot(223)
ax.set_title (r'$x=20$', fontdict = font1)
#ax.set_xlim ([720, 960])
#ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_xlabel (r'$f\delta_0/U_\infty$', fontdict = font1)
ax.set_ylabel ('Weighted PSD, unitless', fontdict = font1)
ax.grid (b=True, which = 'both', linestyle = ':')
Fre20, FPSD20 = FW_PSD (Probe20.p, Probe20.time)
ax.semilogx (Fre20, FPSD20, 'k', linewidth = 1.5)

ax = fig.add_subplot(224)
ax.set_title (r'$x=40$', fontdict = font1)
#ax.set_xlim ([720, 960])
#ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_xlabel (r'$f\delta_0/U_\infty$', fontdict = font1)
#ax.set_ylabel ('Weighted PSD, unitless', fontdict = font1)
ax.grid (b=True, which = 'both', linestyle = ':')
Fre40, FPSD40 = FW_PSD (Probe40.p, Probe40.time)
ax.semilogx (Fre40, FPSD40, 'k', linewidth = 1.5)

plt.tight_layout (pad = 0.5, w_pad = 0.2, h_pad = 1)
plt.savefig (path+'StreawiseFWPSD.pdf', dpi = 300)
plt.show()

#%% Streamwise variations of time evolution of a specific variable
#t1 = 960
#t2 = 1120
fig = plt.figure()
ax = fig.add_subplot(411)
ax.set_title (r'$x=0$', fontdict = font1)
ax.set_xlim ([t1, t2])
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
#ax.set_xlabel (r'$t u_\infty/\delta$', fontdict = font1)
ax.set_ylabel (r'$p^{\prime}/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
#ax.text (850, 6.5, 'x=0', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (Probe0.time, Probe0.p-BProbe0.p, 'k', linewidth = 1.5)

ax = fig.add_subplot(412)
ax.set_title (r'$x=10$', fontdict = font1)
ax.set_xlim ([t1, t2])
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_ylabel (r'$p^{\prime}/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (Probe10.time, Probe10.p-BProbe10.p, 'k', linewidth = 1.5)

ax = fig.add_subplot(413)
ax.set_title (r'$x=20$', fontdict = font1)
ax.set_xlim ([t1, t2])
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_ylabel (r'$p^{\prime}/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (Probe20.time, Probe20.p-BProbe20.p, 'k', linewidth = 1.5)

ax = fig.add_subplot(414)
ax.set_title (r'$x=40$', fontdict = font1)
ax.set_xlim ([t1, t2])
#ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_ylabel (r'$p^{\prime}/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
#ax.text (850, 6.5, 'x=0', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (Probe40.time, Probe40.p-BProbe40.p, 'k', linewidth = 1.5)
plt.tight_layout (pad = 0.5, w_pad = 0.2, h_pad = 1)
plt.savefig (path+'StreawisePerTimeEvolution.pdf', dpi = 300)
plt.show ()

#%% Temporal growth rate
time, omega_i = Probe0.GrowthRate(Probe0.time, Probe0.p-BProbe0.p)
time1 = np.linspace(time.min(), time.max(), 100)
omega_i1 = spline(time, omega_i, time1)
#fxy = interp1d(time, omega_i, kind='cubic')
#omega_i1 = fxy(time1)
fig, ax = plt.subplots ()
ax.set_xlabel (r'$tU_{\infty}/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\omega_{i}}$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
ax.set_ylim ([-0.0001, 0.0001])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
#minorLocator = MultipleLocator (25)
#ax.xaxis.set_minor_locator (minorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
#ax.plot(time, omega_i, 'ko', time1, omega_i1, 'k', linewidth=1.5)
ax.plot(time, omega_i, 'ko', linewidth=1.5)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'TemporalGrowthRate.pdf', dpi = 300)
plt.show ()


#%% Streamwise growth rate with time
# By obtaining the maximum of time evolution curve at every x and then get dA/dx
PointNum = 41
xpoint  = np.linspace (0.0, 40.0, PointNum)
Ampli   = np.zeros(PointNum)
MaxTime = np.zeros(PointNum)
BaseVar = np.zeros(PointNum)
t3 = 960
t4 = 1260
for jj in range(PointNum):
    ProbeInd = DataPost()
    ProbeInd.LoadProbeData(xpoint[jj], 0.0, 0.0, path1)
    ProbeInd.ExtraData('time', t3, t4)
    Ampli[jj] = np.max(ProbeInd.p)
    BProbeInd = DataPost()
    BProbeInd.LoadProbeData(xpoint[jj], 0.0, 0.0, path2)
    BaseVar[jj] = BProbeInd.EquValProfile(t0, BProbeInd.time, BProbeInd.p, 1)
    del ProbeInd, BProbeInd
    
dAmpli = Probe0.SecOrdFDD(xpoint, Ampli-BaseVar)
alpha_i = dAmpli/(Ampli-BaseVar)
xval1 = np.linspace(xpoint.min(), xpoint.max(), 100)
alpha_i1 = spline(xpoint, alpha_i, xval1)
#%%
fig, ax = plt.subplots ()
ax.set_xlabel (r'$x/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\A_p}$', fontdict = font2)
ax.plot(xpoint, Ampli-BaseVar, 'ko', linewidth=1.5)
plt.show()

fig, ax = plt.subplots ()
ax.set_xlabel (r'$x/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\alpha_{i}}$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
#ax.set_ylim ([-0.001, 0.001])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
#minorLocator = MultipleLocator (25)
#ax.xaxis.set_minor_locator (minorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot(xpoint, alpha_i*-1, 'ko', LSTx, LSTalpha_i, 'k', linewidth=1.5)
ax.legend(['LES', 'LST'])
#ax.plot(xpoint, alpha_i, 'ko', xval1, alpha_i1, 'k', linewidth=1.5)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'StreamwiseGrowthRateTime.pdf', dpi = 300)
plt.show ()


#%% Streamwise growth rate with x
# By obtaining the maximum of z evolution curve at every x and then get dA/dx
PointNum = 31
xpoint  = np.linspace (0.0, 30.0, PointNum)
Ampli   = np.zeros(PointNum)
MaxTime = np.zeros(PointNum)
BaseVar = np.zeros(PointNum)
t3 = 1263.2731
for jj in range(PointNum):
    ProbeInd = DataPost()
    ProbeInd.LoadProbeData(xpoint[jj], 0.0, 0.0, path1)
    ProbeInd.ExtraData('time', t3, t3)
    Ampli[jj] = np.max(ProbeInd.p)
    BProbeInd = DataPost()
    BProbeInd.LoadProbeData(xpoint[jj], 0.0, 0.0, path2)
    BaseVar[jj] = BProbeInd.EquValProfile(t0, BProbeInd.time, BProbeInd.p, 1)
    del ProbeInd, BProbeInd
    
dAmpli = Probe0.SecOrdFDD(xpoint, Ampli-BaseVar)
alpha_i = dAmpli/(Ampli-BaseVar)
xval1 = np.linspace(xpoint.min(), xpoint.max(), 100)
alpha_i1 = spline(xpoint, alpha_i, xval1)
#%%
fig, ax = plt.subplots ()
ax.set_xlabel (r'$x/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\A_p}$', fontdict = font2)
ax.plot(xpoint, Ampli-BaseVar, 'ko', linewidth=1.5)
plt.show()

fig, ax = plt.subplots ()
ax.set_xlabel (r'$x/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\alpha_{i}}$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
#ax.set_ylim ([-0.001, 0.001])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
#minorLocator = MultipleLocator (25)
#ax.xaxis.set_minor_locator (minorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot(xpoint, alpha_i*-1, 'ko', LSTx, LSTalpha_i, 'k', linewidth=1.5)
ax.legend(['LES', 'LST'])
#ax.plot(xpoint, alpha_i, 'ko', xval1, alpha_i1, 'k', linewidth=1.5)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'StreamwiseGrowthRateTime.pdf', dpi = 300)
plt.show ()

#%% Streamwise growth rate with z
PointNum = 41
PointNumZ = 7
xpoint  = np.linspace (0.0, 40.0, PointNum)
zpoint  = np.linspace (-2.74, 2.74, PointNumZ)
var_z   = np.zeros(PointNumZ)
Ampli   = np.zeros(PointNum)
MaxTime = np.zeros(PointNum)
BaseVar = np.zeros(PointNum)

t3 = 1263.2731
for jj in range(PointNum):
    BProbeInd = DataPost()
    BProbeInd.LoadProbeData(xpoint[jj], 0.0, 0.0, path2)
    BaseVar[jj] = BProbeInd.EquValProfile(t0, BProbeInd.time, BProbeInd.p, 1)
    for kk in range(PointNumZ):
        ProbeInd = DataPost()
        ProbeInd.LoadProbeData(xpoint[jj], 0.0, zpoint[kk], path1)
        ProbeInd.ExtraData('time', t3, t3)
        var_z[kk] = ProbeInd.p
    fitfunc = DataPost.fit_sin2(zpoint, var_z-BaseVar[jj], 1.147607)
    Ampli[jj] = np.fabs(fitfunc['amp'])
    del ProbeInd, BProbeInd
#%%
dAmpli = Probe0.SecOrdFDD(xpoint, Ampli)
alpha_i = dAmpli/(Ampli)
#xval1 = np.linspace(xval.min(), xval.max(), 100)
#alpha_i1 = spline(xval, alpha_i, xval1)
#%%
fig, ax = plt.subplots ()
ax.set_xlabel (r'$x/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\alpha_{i}}$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
ax.set_ylim ([-0.1, 0.1])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
minorLocator = MultipleLocator (25)
ax.xaxis.set_minor_locator (minorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
#ax.plot(Probetest.time, Probetest.p, 'k', linewidth = 1.5)
ax.plot(xpoint, alpha_i*-1, 'ko', LSTx, LSTalpha_i, 'k', linewidth=1.5)
ax.legend(['LES', 'LST'])
#ax.plot(xval, alpha_i, 'ko', xval1, alpha_i1, 'k', linewidth=1.5)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'StreamwiseGrowthRateZ.pdf', dpi = 300)
plt.show ()