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
from scipy import signal
from scipy.interpolate import spline
import sys
from DataPost import DataPost 
#import tqdm
#import LoadData as dt

plt.close ("All")
plt.rc('text', usetex=True)
font1 = {'family' : 'Arial',
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

path = "./"
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

baseflow = DataPost()
baseflow.LoadData(path+'BaseflowZ0Slice.txt', 2, 0.0)
baseflow.unique_rows()
#baseflow.AveAtSameXYZ('Part')
baseflow.GetWallDist(0.0)
TimePoint = DataPost()
TimePoint.LoadData(path+"TimeSeriesX0Y0Z0.txt", 2, 'time')


LSTData = np.loadtxt("LST.dat", skiprows = 1)
LSTx    = LSTData[:,0]
LSTalpha_r= LSTData[:,1]
LSTalpha_i= LSTData[:,2]
LSTbeta= LSTData[:,3]
LSTomega = LSTData[:,4]


#%% Draw boundary layer profile
ZPlane = DataPost()
ZPlane.LoadData(path+"Time804Z0Slice.txt", 2, 0.0)
ZPlane.GetWallDist(0.0)
#ZPlane.unique_rows()
ZPlane.AveAtSameXYZ('All')
q0, y0 = baseflow.BLProfile(0.0, baseflow.x, baseflow.u)
q1, y1 = ZPlane.BLProfile(0.0, ZPlane.x, ZPlane.u)
f = interp1d(y0, q0, kind='slinear')
q01 = f(y1)
#q01 = np.interp(y1.astype(np.float64), y0.astype(np.float64), q0.astype(np.float64))
fig, ax = plt.subplots()
ax.plot (q1-q01, y1, 'k', linewidth = 1.5)
#ax.plot (q01, y1, 'k:', q1, y1, 'k--', q1-q01, y1, 'k', linewidth = 1.5)
#ax.legend (['base flow', 'real flow', 'perturbations'])
ax.set_xlabel (r'$u^{\prime}/u_\infty$', fontdict = font3)
#ax.set_xlabel (r'$u/u_\infty$', fontdict = font3)
#ax.set_xlabel (r'$p^{\prime}/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font3)
ax.set_ylabel (r'$y/\delta_0$', fontdict = font3)
xmajorLocator = MultipleLocator (0.2)
xminorLocator = MultipleLocator (0.1)
#ax.set_xlim ([-0.00002, 0.00018])
ax.set_ylim ([0, 5])
#ax.xaxis.set_major_locator (xmajorLocator)
#ax.xaxis.set_minor_locator (xminorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig(path + 'BLUPertProfile.pdf', dpi=300)
plt.show()

#%% Draw impose mode
ImpMode = np.loadtxt(path+"UnstableMode.inp", skiprows = 5)
y  = ImpMode[:,0]
qr = ImpMode[:,1]
qi = ImpMode[:,2]
ampl = 0.01
x = 0.0
z = 0.0
t = 804
omeg_r= 0.443259
alpha = 0.633513
beta  = 1.147607
theta = alpha*x+beta*z-omeg_r*t
q_perturb = ampl*(qr*np.cos(theta)-qi*np.sin(theta))
fig, ax = plt.subplots()
ax.plot (q_perturb, y, 'k', linewidth = 1.5)
#ax.set_xlabel (r'$p^{\prime}/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font3)
ax.set_xlabel (r'$u^{\prime}/u_{\infty}$', fontdict = font3)
ax.set_ylabel (r'$y/\delta_0$', fontdict = font3)
#ax.set_xlim ([-0.00002, 0.00018])
ax.set_ylim ([0, 5])
#ax.xaxis.set_major_locator (xmajorLocator)
#ax.xaxis.set_minor_locator (xminorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig(path + 'PertUProfile.pdf', dpi=300)
plt.show()

#%% Draw spanwise variations of a specific variable
XPlane = DataPost()
XPlane.LoadData(path+"Time804X0Slice.txt", 2, 800)
yval = 0.0
xval = 0.0
qz = XPlane.EquValProfile(yval, XPlane.y, XPlane.z, 1)
q  = XPlane.EquValProfile(yval, XPlane.y, XPlane.p, 1)
qx0 = baseflow.EquValProfile(yval, baseflow.y, baseflow.x, 1)
qp0 = baseflow.EquValProfile(yval, baseflow.y, baseflow.p, 1)
qp01 = baseflow.EquValProfile(xval, qx0, qp0, 1)
fig, ax = plt.subplots ()
ax.set_xlabel (r'$z/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$p^{\prime}/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
minorLocator = MultipleLocator (25)
ax.xaxis.set_minor_locator (minorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (qz, q-qp01, 'k', linewidth = 1.5)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'SpanwiseCurveX0.pdf', dpi = 300)
plt.show ()

#%% Streamwise growth rate with z
PointNum = 41
xpoint = np.linspace (0.0, 40.0, PointNum)
Ampli  = np.zeros (PointNum)
Ampli1  = np.zeros (PointNum)
#Cycle  = np.zeros (PointNum)
YPlane = DataPost()
YPlane.LoadData(path+'Time804Y0Slice.txt', 2, 800)
YPlane.unique_rows()
xarray = YPlane.x
qarray = YPlane.p
zarray = YPlane.z
x0 = baseflow.EquValProfile(0.0, baseflow.y, baseflow.x, 1)
p0 = baseflow.EquValProfile(0.0, baseflow.y, baseflow.p, 1)
p01 = np.interp(xpoint, x0, p0)
for j in range(np.size(xpoint)):
    VarInd = np.where (xarray == xpoint[j])
    var = qarray[VarInd]
    zval = zarray[VarInd]
    #func = DataPost.fit_sin(zval, var)
    #Ampli[j] = np.fabs(func['amp'])+func['offset']
    func = DataPost.fit_sin2(zval, var-p01[j], 1.147607)
    Ampli[j] = np.fabs(func['amp'])
    Ampli1[j] = np.max(var)

fig, ax = plt.subplots()
qval = func['fitfunc'](zval)
ax.plot(zval, var, 'ko', zval, qval, 'b-')
plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(xpoint, p01, 'ko')
plt.title('Base flow')
plt.show()
#%%
fig, ax = plt.subplots()
ax.set_xlabel (r'$x/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$A^{\prime}$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
#ax.set_ylim ([-0.001, 0.001])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot(xpoint, Ampli, 'ko', linewidth=1.5)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'StreamwiseAmpliZ.pdf', dpi = 300)
plt.show()
#%%
dAmplidx = YPlane.SecOrdFDD(xpoint, Ampli)
alpha_i  = dAmplidx/(Ampli)
xval1 = np.linspace(xpoint.min(), xpoint.max(), 100)
alpha_i1 = spline(xpoint, alpha_i, xval1)
fig, ax = plt.subplots ()
ax.set_xlabel (r'$x/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\alpha_{i}}$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
ax.set_ylim ([-0.1, 0.1])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
#minorLocator = MultipleLocator (25)
#ax.xaxis.set_minor_locator (minorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
#ax.plot(xpoint, alpha_i, 'ko', xval1, alpha_i1, 'k', linewidth=1.5)
ax.plot(xpoint, alpha_i*-1, 'ko', LSTx, LSTalpha_i, 'k', linewidth=1.5)
ax.legend(['LES', 'LST'])
fig.set_size_inches (6, 3)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'StreamwiseGrowthRateZ0.01.png', dpi = 300)
plt.show ()

#%% Spanwise growth rate with time
PointNum = 13
zpoint = np.linspace (-2.74, 2.74, PointNum)
Ampli  = np.zeros (PointNum)
#Cycle  = np.zeros (PointNum)
TSeriesXPlane = DataPost()
TSeriesXPlane.LoadData(path+'TimeSeriesX0Slice.txt', 1, 0.0)
TSeriesXPlane.GetSolutionTime(TimePoint.time)
TSeriesXPlane.GetWallDist(0.0)
zarray = TSeriesXPlane.EquValProfile(0.0, TSeriesXPlane.WallDist, TSeriesXPlane.z, 1)
qarray = TSeriesXPlane.EquValProfile(0.0, TSeriesXPlane.WallDist, TSeriesXPlane.p, 1)

z0, p0 = baseflow.EquValProfile3D(baseflow.p, x = 10.0, y = 0.0)
#z0 = baseflow.EquValProfile(0.0, baseflow.y, baseflow.z, 1)
#p0 = baseflow.EquValProfile(0.0, baseflow.y, baseflow.p, 1)
p01 = np.interp(zpoint, z0, p0)
for j in range(np.size(zpoint)):
    VarInd = np.where (np.around(zarray, 3) == np.around(zpoint[j], 3))
    var = qarray[VarInd]
    tval = qarray[VarInd]
    #func = DataPost.fit_sin2(tval, var-p01[j])
    #Ampli[j] = np.fabs(func['amp'])
    Ampli[j] = np.max(var)

dAmplidx = TSeriesXPlane.SecOrdFDD(zpoint, Ampli-p01)
beta_i  = dAmplidx/(Ampli-p01)
xval1 = np.linspace(zpoint.min(), zpoint.max(), 100)
beta_i1 = spline(zpoint, beta_i, xval1)

fig, ax = plt.subplots ()
ax.set_xlabel (r'$z/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\A_p}$', fontdict = font2)
ax.plot(zpoint, Ampli, linewidth=1.5)
plt.show ()

fig, ax = plt.subplots ()
ax.set_xlabel (r'$z/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$\beta_{i}}$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
#ax.set_ylim ([-0.001, 0.001])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
#minorLocator = MultipleLocator (25)
#ax.xaxis.set_minor_locator (minorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot(zpoint, beta_i, 'ko', xval1, beta_i1, 'k', linewidth=1.5)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'SpanwiseGrowthRateTime.pdf', dpi = 300)
plt.show ()


#%% Data for Streamwise variations of frequency of a specific variable
"""
TimeSeriesZone = DataPost()
TimeSeriesZone.LoadData(path+"TimeSeriesZ0Slice.txt", 1, 0.0)
TimeSeriesZone.GetSolutionTime(TimePoint.time)
TimeSeriesZone.GetWallDist(0.0)

#%% Streamwise variations of frequency of a specific variable
TimeSeriesZone.GetWallDist(0.0)
xpoint = np.array([0.0, 2.0, 5.0, 8.0])
xarray = TimeSeriesZone.EquValProfile(0.0, TimeSeriesZone.WallDist, TimeSeriesZone.x, 1)
qarray = TimeSeriesZone.EquValProfile(0.0, TimeSeriesZone.WallDist, TimeSeriesZone.p, 1)

fig = plt.figure()
ax = fig.add_subplot(411)
p0 = TimeSeriesZone.EquValProfile(xpoint[0], xarray, qarray, 1)
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
#ax.set_xlabel (r'$t u_\infty/\delta$', fontdict = font1)
ax.set_ylabel (r'$p/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
#ax.text (850, 6.5, 'x=0', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (TimePoint.time, p0, 'k', linewidth = 1.5)

maxind0 = DataPost.FindPeaks (p0)
maxval0 = p0[maxind0]
maxtime0= TimePoint.time[maxind0]
meanval = np.mean(p0)
fdd = DataPost.SecOrdFDD (maxtime0, maxval0)
omegi = fdd/(maxval0)

ax = fig.add_subplot(412)
p1 = TimeSeriesZone.EquValProfile(xpoint[1], xarray, qarray, 1)
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_ylabel (r'$p/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (TimePoint.time, p1, 'k', linewidth = 1.5)

maxind1 = DataPost.FindPeaks (p1)
maxval1 = p1[maxind1]
maxtime1= TimePoint.time[maxind1]


ax = fig.add_subplot(413)
p2 = TimeSeriesZone.EquValProfile(xpoint[2], xarray, qarray, 1)
ax.set_xticklabels ('')
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_ylabel (r'$p/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (TimePoint.time, p2, 'k', linewidth = 1.5)

maxind2 = DataPost.FindPeaks (p2)
maxval2 = p2[maxind2]
maxtime2= TimePoint.time[maxind2]

ax = fig.add_subplot(414)
p3 = TimeSeriesZone.EquValProfile(xpoint[3], xarray, qarray, 1)
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.set_xlabel (r'$t u_\infty/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$p/(\rho_{\infty} u_{\infty}^{2})$', fontdict = font2)
#ax.text (850, 6.5, 'x=0', fontdict = font2)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot (TimePoint.time, p3, 'k', linewidth = 1.5)
plt.tight_layout (pad = 0.5, w_pad = 0.2, h_pad = 1)
plt.savefig (path+'StreamwiseFrequencyVar.pdf', dpi = 300)
plt.show ()
maxind3 = DataPost.FindPeaks (p3)
maxval3 = p3[maxind3]
maxtime3= TimePoint.time[maxind3]

#%% Streamwise variations of amplitude of a specific variable with time
PointNum = 121
xpoint = np.linspace (0.0, 30.0, PointNum)
Ampli  = np.zeros (PointNum)
Cycle  = np.zeros (PointNum)
#xarray = TimeSeriesZone.x
#qarray = TimeSeriesZone.p
x0 = baseflow.EquValProfile(0.0, baseflow.y, baseflow.x, 1)
p0 = baseflow.EquValProfile(0.0, baseflow.y, baseflow.p, 1)
p01 = np.interp(xpoint, x0, p0)
for j in range(np.size(xpoint)):
    VarInd = np.where (xarray == xpoint[j])
    var = qarray[VarInd]
    Ampli[j] = np.max(var)
#    MaxInd  = np.argmax (var)
#    MinInd  = np.argmin (var)
#    MeanVal = np.around (np.mean (var), decimals=4)
#    MeanInd = np.where (np.around(var[:], decimals=4) == MeanVal)
#    MeanInd = np.asarray (MeanInd)
#    MeanIndDif = np.diff (MeanInd)
#    MeanValInd = np.where (MeanIndDif[:] > 10 )[1]
#    ind1 = MeanInd[0][0]
#    ind2 = MeanInd[0][MeanValInd[0]+1]
#    ind3 = MeanInd[0][MeanValInd[1]]
#    Cycle[j] = (time[ind2]-time[ind1])*2
#    Cycle[j] = np.fabs((time[MaxInd]-time[MinInd])*2)

fig, ax = plt.subplots ()
ax.set_xlabel (r'$x/\delta_0$', fontdict = font2)
ax.set_ylabel (r'$A_{p^{\prime}}$', fontdict = font2)
#ax.set_xlim ([1120, 1600])
ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
ax.ticklabel_format(useOffset=False)
minorLocator = MultipleLocator (25)
ax.xaxis.set_minor_locator (minorLocator)
ax.grid (b=True, which = 'both', linestyle = ':')
ax.plot(xpoint, Ampli-p01, 'k', linewidth = 1.5)
plt.tight_layout(pad = 0.5, w_pad = 0.2, h_pad = 0.2)
plt.savefig (path+'AllStreamwiseTimeAmplitudeVar.pdf', dpi = 300)
plt.show ()
"""

#%% Get Frequency Weighted Power Spectral Density
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
        fig.savefig (path+pic, dpi = 600)
        #plt.show ()
    return (fre, fre_weighted)

