# -*- coding: utf-8 -*-
"""
Created on Tue May 1 10:24:50 2018
    This code for reading data from specific file to post-processing data
    1. FileName (infile, VarName, (row1), (row2), (unique) ): sort data and 
    delete duplicates, SPACE is NOT allowed in VarName
    2. MergeFile (NameStr, FinalFile): NameStr-input file name to be merged
    3. GetMu (T): calculate Mu if not get from file
    4. unique_rows (infile, outfile):
@author: Weibo Hu
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import warnings
import pandas as pd
from DataPost import DataPost
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import griddata
from scipy.interpolate import spline
import scipy.optimize
from numpy import NaN, Inf, arange, isscalar, asarray, array
import sys, os, time

# Obtain intermittency factor from an undisturbed and specific wall pressure
def Intermittency(sigma, Pressure0, WallPre, TimeZone):
    #AvePre    = np.mean(WallPre)
    AvePre    = np.mean(Pressure0)
    # wall pressure standard deviation of undisturbed BL
    threshold = AvePre+3*sigma
    # Alternative approximate method
    # DynamicP = 0.5*0.371304*469.852**2, ratio = 0.006/(1+0.13*1.7**2)**0.64
    # sigma1 = DynamicP*ratio
    # threshold value for turbulence
    sign      = (WallPre-threshold)/abs(WallPre-threshold)
    sign      = np.maximum(0, sign[:])
    gamma      = np.trapz(sign, TimeZone)/(TimeZone[-1]-TimeZone[0])
    return gamma

# Obtain skewness coefficient corresponding to intermittency factor
def Alpha3(WallPre):
    AvePre = np.mean(WallPre)
    sigma  = np.std(WallPre)
    n      = np.size(WallPre)
    temp1  = np.power(WallPre-AvePre, 3)
    alpha  = np.sum(temp1)/n/np.power(sigma, 3)
    return alpha

# Obtain nondimensinal dynamic viscosity
def Viscosity(Re_delta, T): # nondimensional T
    mu = 1.0/Re_delta*np.power(T, 0.75)
    return mu

# Obtain skin friction coefficency
def SkinFriction(mu, du, dy): # all variables are nondimensional
    Cf = 2*mu*du/dy
    return Cf
# obtain Power Spectral Density
def PSD(VarZone, TimeSpan, Freq_samp):
    # interpolate data to make sure time-equaled distribution
    TotalNo = Freq_samp*(TimeSpan[-1]-TimeSpan[0])*2 #make NO of snapshot twice larger
    if (TotalNo > np.size(TimeSpan)):
        warnings.warn("PSD results are not accurate due to too few snapshots",\
                      UserWarning)
    TimeZone = np.linspace(TimeSpan[0], TimeSpan[-1], TotalNo)
    VarZone  = VarZone-np.mean(VarZone)
    Var = np.interp(TimeZone, TimeSpan, VarZone) # time space must be equal
    # POD, fast fourier transform and remove the half
    Var_fft = np.fft.rfft(Var)
    Var_psd = abs(Var_fft)**2
    num = np.size(Var_fft)
    Freq = np.linspace(Freq_samp/TotalNo, Freq_samp/2, num)
    #Freq = np.arange(Freq_samp/TotalNo, Freq_samp/2+Freq_samp/TotalNo, Freq_samp/TotalNo)
    #Freq = np.linspace(Freq_samp/2/num, Freq_samp/2, num)
    return (Freq, Var_psd)

# Obtain Frequency-Weighted Power Spectral Density
def FW_PSD (VarZone, TimeSpan, Freq_samp):
    Freq, Var_PSD = PSD(VarZone, TimeSpan, Freq_samp)
    FPSD = Var_PSD*Freq
    return (Freq, FPSD)

# Obtain the standard law of wall (turbulence)
def StdWallLaw():
    ConstK = 0.41
    ConstC = 5.2
    yplus1 = np.arange(1, 15, 0.1) # viscous sublayer velocity profile
    uplus1 = yplus1
    yplus2 = np.arange(3, 1000, 0.1) # logarithm layer velocity profile
    uplus2 = 1.0/ConstK*np.log(yplus2)+ConstC
    UPlus1 = np.column_stack((yplus1, uplus1))
    UPlus2 = np.column_stack((yplus2, uplus2))
    return(UPlus1, UPlus2)

# Draw reference experimental data of turbulence
# 0y/\delta_{99}, 1y+, 2U+, 3urms+, 4vrms+, 5wrms+, 6uv+, 7prms+, 8pu+,
# 9pv+, 10S(u), 11F(u), 12dU+/dy+, 13V+, 14omxrms^+, 15omyrms^+, 16omzrms^+
def ExpWallLaw():
    ExpData = np.loadtxt ("vel_4060_dns.prof", skiprows = 14)
    m, n = ExpData.shape
    y_delta     = ExpData[:, 0]
    y_plus      = ExpData[:, 1]
    u_plus      = ExpData[:, 2]
    urms_plus   = ExpData[:, 3]
    vrms_plus   = ExpData[:, 4]
    wrms_plus   = ExpData[:, 5]
    uv_plus     = ExpData[:, 6]
    UPlus    = np.column_stack((y_plus, u_plus))
    UVPlus   = np.column_stack((y_plus, uv_plus))
    UrmsPlus = np.column_stack((y_plus, urms_plus))
    VrmsPlus = np.column_stack((y_plus, vrms_plus))
    WrmsPlus = np.column_stack((y_plus, wrms_plus))
    return(UPlus, UVPlus, UrmsPlus, VrmsPlus, WrmsPlus)

# This code validate boundary layer profile by incompressible, Van Direst transformed
# boundary profile from mean reults
def DirestWallLaw(walldist, u, rho, mu):
    if((np.diff(walldist) < 0.0).any()):
        sys.exit("the WallDist must be in ascending order!!!")
    m = np.size(u)
    rho_wall= rho[0]
    mu_wall = mu[0]
    tau_wall  = mu_wall*u[1]/walldist[1]
    u_tau     = np.sqrt(np.abs(tau_wall/rho_wall))
    u_van  = np.zeros(m)
    for i in range(m):
        u_van[i] = np.trapz(rho[:i+1]/rho_wall, u[:i+1])
    u_plus_van = u_van/u_tau
    y_plus     = u_tau*walldist*rho_wall/mu_wall
    UPlusVan   = np.column_stack((y_plus, u_plus_van))
    return(UPlusVan)


#Fs = 1000
#t = np.arange(0.0, 1-1.0/Fs, 1/Fs)
#var = np.cos(2*3.14159265*100*t)+np.random.uniform(-1, 1, np.size(t))
#Var_fft = np.fft.fft(var)
#Var_fftr = np.fft.rfft(var)
#Var_psd = abs(Var_fft)**2
#Var_psd1 = Var_psd[:int(np.size(t)/2)]
#Var_psd2 = abs(Var_fftr)**2
#fre1, Var_psd3 = PSD(var, t, Fs)
#num = np.size(Var_psd1)
#freq = np.linspace(Fs/2/num, Fs/2, num)
#f, fpsd = FW_PSD(var, t, Fs)
##fre, psd = PSD(var, t, Fs)
##plt.plot(fre1, 10*np.log10(Var_psd3))
#fig2, ax2 = plt.subplots()
#ax2.plot(f, fpsd)
#plt.show()
#
#fig, ax = plt.subplots()
#ax.psd(var, 500, Fs)
#plt.show()
