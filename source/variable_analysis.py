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
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import warnings
import pandas as pd
import variable_analysis as fv
from scipy.interpolate import griddata  # interp1d
from scipy.integrate import trapz, dblquad  # simps,
# import scipy.optimize
from scipy.interpolate import splprep, splev, interp1d
# from numpy import NaN, Inf, arange, isscalar, asarray, array
import sys
from timer import timer
import os


# Obtain intermittency factor from an undisturbed and specific wall pressure
def intermittency(sigma, Pressure0, WallPre, TimeZone):
    # AvePre    = np.mean(WallPre)
    AvePre = np.mean(Pressure0)
    # wall pressure standard deviation of undisturbed BL
    threshold = AvePre + 3 * sigma
    # Alternative approximate method
    # DynamicP = 0.5*0.371304*469.852**2, ratio = 0.006/(1+0.13*1.7**2)**0.64
    # sigma1 = DynamicP*ratio
    # threshold value for turbulence
    sign = np.zeros(np.size(WallPre))
    ind = np.where(WallPre > threshold)[0]
    sign[ind] = 1.0
    # sign = (WallPre-threshold)/abs(WallPre-threshold)
    # sign      = np.maximum(0, sign[:])
    gamma = np.trapz(sign, TimeZone) / (TimeZone[-1] - TimeZone[0])
    return gamma


# Obtain skewness coefficient corresponding to intermittency factor
def alpha3(WallPre):
    AvePre = np.mean(WallPre)
    sigma = np.std(WallPre)
    n = np.size(WallPre)
    temp1 = np.power(WallPre - AvePre, 3)
    alpha = np.sum(temp1) / n / np.power(sigma, 3)
    return alpha


# Obtain nondimensinal dynamic viscosity
def viscosity(Re_delta, T, law='POW', T_inf=273):
    # nondimensional T
    if(law=='POW'):
        mu = 1.0 / Re_delta * np.power(T, 0.75)
    else:  # Sutherland's law, mu_ref = mu_inf
        S = 110.4 / T_inf
        mu = 1.0 * Re_delta * (1 + S) / (T + S) * np.power(T, 3/2)
    return mu


# Obtain BL thickness, momentum thickness, displacement thickness
def bl_thickness(y, u, u_d=None, rho=None, opt=None):
    ind = np.argsort(y)  # sort y from small to large
    bc = int(np.rint(np.size(y) * 0.9))
    y1 = y[ind][:bc]  # remove the part near the farfield boundary conditions
    u1 = u[ind][:bc]  # remove the part near the farfield boundary conditions
    if u_d is None:
        bl = np.where(u1[:] >= 0.99)[0][0]
    else:
        bl = np.where(u1[:] >= 0.99 * u_d)[0][0]
    if np.size(bl) == 0:
        sys.exit('This is not a complete boundary layer profile!!!')
    delta = y1[bl]
    u_d = u1[bl]
    if opt is None:
        return(delta, u_d)
    elif opt == 'displacement':
        rho1 = rho[ind][:bc]
        rho_d = np.max(rho1)
        u_d = np.max(u1)
        a1 = rho1*u1/rho_d/u_d
        var = 1-a1
        delta_star = np.trapz(var, y1)
        return(delta_star, u_d, rho_d)
    elif opt == 'momentum':
        rho1 = rho[ind][:bc]
        rho_d = np.max(rho1)
        u_d = np.max(u1)
        a1 = 1-u1/u_d
        a2 = rho1*u1/rho_d/u_d
        var = a1 * a2
        theta = np.trapz(var, y1)
        return(theta, u_d, rho_d)


# shape factor of boundary layer    
def shape_factor(y, u, rho, u_d=None):
    delta_star = bl_thickness(y, u, u_d=u_d, rho=rho, opt='displacement')[0]
    theta = bl_thickness(y, u, u_d=u_d, rho=rho, opt='momentum')[0]
    shape = delta_star / theta
    return (shape)


# radius of flow curvature
def radius(x, y):
    y1 = fv.sec_ord_fdd(x, y)
    y2 = fv.sec_ord_fdd(x, y1)
    a1 = 1+(y1)**2
    a2 = np.abs(y2)
    radi = np.power(a1, 1.5)/a2
    return radi


# Obtain G\"ortler number
def gortler(Re_inf, x, y, theta, scale=0.001):
    Re_theta = Re_inf*theta*scale
    radi = radius(x, y)
    gortler = Re_theta*np.sqrt(theta/radi)
    return gortler


def gortlerTur(theta, delta_star, radi):
    # radi = Radius(x, y)
    a1 = theta / 0.018 / delta_star
    a2 = np.sqrt(theta / np.abs(radi))
    gortler = a1 * a2 * np.sign(radi)
    return gortler


# Obtain skin friction coefficency
def skinfriction(mu, du, dy):
    # all variables are nondimensional
    Cf = 2 * mu * du / dy
    return Cf


# Obtain turbulent kinetic energy
def tke(df):
    # all variables are nondimensional
    kinetic_energy = 0.5 * (df['<u`u`>'] + df['<v`v`>'] + df['<w`w`>'])
    return kinetic_energy


# obtain Power Spectral Density
def psd(VarZone, dt, Freq_samp, opt=2, seg=8, overlap=4):
    TotalNo = np.size(VarZone)
    if np.size(dt) > 1:
        TotalNo = Freq_samp * (dt[-1] - dt[0])
        if TotalNo > np.size(dt):
            warnings.warn(
                "PSD results are not accurate as too few snapshots",
                UserWarning
            )
        TimeZone = np.linspace(dt[0], dt[-1], TotalNo)
        VarZone = VarZone - np.mean(VarZone)
        Var = np.interp(TimeZone, dt, VarZone)
    else:
        if Freq_samp > 1 / dt:
            warnings.warn(
                "PSD results are not accurate as too few snapshots",
                UserWarning
            )
            Var = VarZone - np.mean(VarZone)
        elif Freq_samp == 1 / dt:
            Var = VarZone - np.mean(VarZone)
        else:
            TimeSpan = np.arange(0, np.size(VarZone) * dt, dt)
            TotalNo = int((TimeSpan[-1] - TimeSpan[0]) * Freq_samp) + 1
            TimeZone = np.linspace(TimeSpan[0], TimeSpan[-1], TotalNo)
            VarZone = VarZone - np.mean(VarZone)
            # interpolate data to make sure time-equaled distribution
            Var = np.interp(
                TimeZone, TimeSpan, VarZone
            )  # time space must be equal
    # POD, fast fourier transform and remove the half
    if opt == 2:
        Var_fft = np.fft.rfft(Var) # [1:]  # remove value at 0 frequency
        Var_psd = np.abs(Var_fft) ** 2 / (Freq_samp * TotalNo)
        num = np.size(Var_fft)
        Freq = np.linspace(0.0, Freq_samp / 2, num)
        # Freq = np.linspace(Freq_samp / TotalNo, Freq_samp / 2, num)
    if opt == 1:
        ns = TotalNo // seg
        Freq, Var_psd = signal.welch(
            Var, fs=Freq_samp, nperseg=ns, nfft=TotalNo, noverlap=ns // overlap
        )
        # num = np.size(Var_psd)
        # Freq = np.linspace(Freq_samp / TotalNo, Freq_samp / 2, num)
        # Freq = Freq[1:]
        # Var_psd = Var_psd[1:]
    return (Freq, Var_psd)


# Obtain Frequency-Weighted Power Spectral Density
def fw_psd(VarZone, dt, Freq_samp, opt=2, seg=8, overlap=4):
    Freq, Var_PSD = psd(VarZone, dt, Freq_samp,
                        opt=opt, seg=seg, overlap=overlap)
    FPSD = Var_PSD * Freq
    return (Freq, FPSD)


def fw_psd_map(orig, xyz, var, dt, Freq_samp, opt=2, seg=8, overlap=4):
    frame1 = orig.loc[np.around(orig['x'], 5) == np.around(xyz[0], 5)]
    # frame2 = frame1.loc[np.around(frame1['y'], 5) == xyz[1]]
    frame2 = frame1.loc[np.around(frame1['y'], 5) == np.around(xyz[1], 5)]
    orig = frame2.loc[frame2['z'] == xyz[2]]
    varzone = orig[var]
    Freq, FPSD = fw_psd(varzone, dt, Freq_samp,
                        opt=opt, seg=seg, overlap=overlap)
    return (Freq, FPSD)


# Compute the RMS
def rms(dataseries):
    meanval = np.mean(dataseries)
    rmsval = np.sqrt(np.mean((dataseries - meanval) ** 2))
    return (rmsval)


# Compute the RMS
def rms_map(orig, xyz, var):
    frame1 = orig.loc[np.around(orig['x'], 6) == np.around(xyz[0], 6)]
    frame2 = frame1.loc[np.around(frame1['y'], 6) == np.around(xyz[1], 6)]
    orig = frame2.loc[np.around(frame2['z'], 6) == np.around(xyz[2], 6)]
    varzone = orig[var]
    rms_val = rms(varzone)
    return (rms_val)


# Obtain cross-power sepectral density
def cro_psd(Var1, Var2, dt, Freq_samp, opt=1):
    TotalNo = np.size(Var1)
    if np.size(Var1) != np.size(Var2):
        warnings.warn("Check the size of input varable 1 & 2", UserWarning)
    if Freq_samp > 1 / dt:
        warnings.warn(
            "PSD results are not accurate due to too few snapshots",
            UserWarning
        )
    elif Freq_samp == 1 / dt:
        NVar1 = Var1 - np.mean(Var1)
        NVar2 = Var2 - np.mean(Var2)
    else:
        TimeSpan = np.arange(0, np.size(Var1) * dt, dt)
        TotalNo = int((TimeSpan[-1] - TimeSpan[0]) * Freq_samp) + 1
        TimeZone = np.linspace(TimeSpan[0], TimeSpan[-1], TotalNo)
        VarZone1 = Var1 - np.mean(Var1)
        VarZone2 = Var2 - np.mean(Var2)
        NVar1 = np.interp(
            TimeZone, TimeSpan, VarZone1
        )  # time space must be equal
        NVar2 = np.interp(
            TimeZone, TimeSpan, VarZone2
        )  # time space must be equal
    if opt == 1:
        ns = TotalNo // 6
        Freq, Cpsd = signal.csd(
            NVar1, NVar2, Freq_samp, nperseg=ns, nfft=TotalNo, noverlap=ns // 4
        )
        Freq = Freq[1:]
        Cpsd = Cpsd[1:]
    if opt == 2:
        Var1_fft = np.fft.rfft(NVar1)[1:]
        Var2_fft = np.fft.rfft(NVar2)[1:]
        Cpsd = Var1_fft * Var2_fft
        num = np.size(Var1_fft)
        Freq = np.linspace(Freq_samp / TotalNo, Freq_samp / 2, num)
    return (Freq, Cpsd)


def coherence(Var1, Var2, dt, Freq_samp, opt=1):
    TotalNo = np.size(Var1)
    if np.size(Var1) != np.size(Var2):
        warnings.warn("Check the size of input varable 1 & 2", UserWarning)
    if Freq_samp > 1 / dt:
        warnings.warn(
            "PSD results are not accurate due to too few snapshots",
            UserWarning
        )
    elif Freq_samp == 1 / dt:
        NVar1 = Var1 - np.mean(Var1)
        NVar2 = Var2 - np.mean(Var2)
    else:
        TimeSpan = np.arange(0, np.size(Var1) * dt, dt)
        TotalNo = int((TimeSpan[-1] - TimeSpan[0]) * Freq_samp) + 1
        TimeZone = np.linspace(TimeSpan[0], TimeSpan[-1], TotalNo)
        VarZone1 = Var1 - np.mean(Var1)
        VarZone2 = Var2 - np.mean(Var2)
        NVar1 = np.interp(
            TimeZone, TimeSpan, VarZone1
        )  # time space must be equal
        NVar2 = np.interp(
            TimeZone, TimeSpan, VarZone2
        )  # time space must be equal
    if opt == 1:
        ns = TotalNo // 8  # 6-4 # 8-2
        Freq, gamma = signal.coherence(
            NVar1,
            NVar2,
            fs=Freq_samp,
            nperseg=ns,
            nfft=TotalNo,
            noverlap=ns // 2,
        )
        Freq = Freq[1:]
        gamma = gamma[1:]
    if opt == 2:
        Freq, cor = cro_psd(NVar1, NVar2, dt, Freq_samp)
        psd1 = psd(NVar1, dt, Freq_samp)[1]
        psd2 = psd(NVar2, dt, Freq_samp)[1]
        gamma = abs(cor) ** 2 / psd1 / psd2
    return (Freq, gamma)


# Obtain the standard law of wall (turbulence)
def std_wall_law():
    ConstK = 0.41
    ConstC = 5.2
    yplus1 = np.arange(1, 15, 0.1)  # viscous sublayer velocity profile
    uplus1 = yplus1
    yplus2 = np.arange(3, 1000, 0.1)  # logarithm layer velocity profile
    uplus2 = 1.0 / ConstK * np.log(yplus2) + ConstC
    UPlus1 = np.column_stack((yplus1, uplus1))
    UPlus2 = np.column_stack((yplus2, uplus2))
    return (UPlus1, UPlus2)


# Draw reference experimental data of turbulence
# 0y/\delta_{99}, 1y+, 2U+, 3urms+, 4vrms+, 5wrms+, 6uv+, 7prms+, 8pu+,
# 9pv+, 10S(u), 11F(u), 12dU+/dy+, 13V+, 14omxrms^+, 15omyrms^+, 16omzrms^+
def ref_wall_law(Re_theta):
    path = os.path.abspath('..') + '/database/'
    if Re_theta <= 830:
        file = path + 'vel_0670_dns.prof'
    elif 830 < Re_theta <= 1200:
        file = path + 'vel_1000_dns.prof'
    elif 1200 < Re_theta <= 1700:
        file = path + 'vel_1410_dns.prof'
    elif 1700 < Re_theta <= 2075:
        file = path + 'vel_2000_dns.prof'
    elif 2075 < Re_theta <= 2300:
        file = path + 'vel_2150_dns.prof'
    elif 2300 < Re_theta <= 2500:
        file = path + 'vel_2400_dns.prof'
    elif 2500 < Re_theta <= 2800:
        file = path + 'vel_2540_dns.prof'
    elif 2800 < Re_theta <= 3150:
        file = path + 'vel_3030_dns.prof'
    elif 3150 < Re_theta <= 3450:
        file = path + 'vel_3270_dns.prof'
    elif 3450 < Re_theta <= 3800:
        file = path + 'vel_3630_dns.prof'
    elif 3800 < Re_theta <= 4000:
        file = path + 'vel_3970_dns.prof'
    else:
        file = path + 'vel_4060_dns.prof'

    print('Take reference data: ' + file)
    ExpData = np.loadtxt(file, skiprows=14)
    m, n = ExpData.shape
    # y_delta = ExpData[:, 0]
    y_plus = ExpData[:, 1]
    u_plus = ExpData[:, 2]
    urms_plus = ExpData[:, 3]
    vrms_plus = ExpData[:, 4]
    wrms_plus = ExpData[:, 5]
    uv_plus = ExpData[:, 6]
    UPlus = np.column_stack((y_plus, u_plus))
    UVPlus = np.column_stack((y_plus, uv_plus))
    UrmsPlus = np.column_stack((y_plus, urms_plus))
    VrmsPlus = np.column_stack((y_plus, vrms_plus))
    WrmsPlus = np.column_stack((y_plus, wrms_plus))
    return (UPlus, UVPlus, UrmsPlus, VrmsPlus, WrmsPlus)


def u_tau(frame, option='mean'):
    """
    input
    ------
        boundary layer profile
    return
    ------
        friction/shear velocity from mean or instantaneous flow
    """
    if(frame['walldist'].values[0] != 0):
        sys.exit('Please reset wall distance/velocity from zero!!!')
    if(option == 'mean'):
        # rho_wall = frame['rho_m'].values[0]
        # mu_wall = frame['mu_m'].values[0]
        # delta_u = frame['u_m'].values[1] - frame['u_m'].values[0]
        rho_wall = frame['<rho>'].values[1]
        mu_wall = frame['<mu>'].values[1]
        delta_u = frame['<u>'].values[1] # - frame['<u>'].values[0]
        u_value = frame['<u>'].values
    else:
        rho_wall = frame['rho'].values[1]
        mu_wall = frame['mu'].values[1]
        delta_u = frame['u'].values[1] # - frame['u'].values[0]
        u_value = frame['u'].values
    walldist2 = frame['walldist'].values[1]
    
#    if(frame['walldist'].values[1] > 0.005):
#        print('Interpolate for u_wall')
#        func = interp1d(frame['walldist'].values, u_value, kind='cubic')
#        delta_u = func(0.004)
#        walldist2 = 0.004

    tau_wall = mu_wall * delta_u / walldist2
    shear_velocity = np.sqrt(np.abs(tau_wall / rho_wall))
    return (shear_velocity)


# This code validate boundary layer profile by
# incompressible, Van Direst transformed
# boundary profile from mean reults
def direst_transform(frame, option='mean'):
    """
    This code validate boundary layer profile by
    incompressible, Van Direst transformed
    boundary profile from mean reults
    """
    if option == 'mean':
        walldist = frame['walldist'].values
#       u = frame['u_m']
#       rho = frame['rho_m']
#       mu = frame['mu_m']
        u = frame['<u>']
        rho = frame['<rho>']
        mu = frame['<mu>']
    else:
        walldist = frame['walldist'].values
        u = frame['u']
        rho = frame['rho']
        mu = frame['mu']

    if (np.diff(walldist) < 0.0).any():
        sys.exit("the WallDist must be in ascending order!!!")
    if (walldist[0] != 0):
        sys.exit('Please reset wall distance from zero!!!')
    m = np.size(u)
    rho_wall = rho[1]
    mu_wall = mu[1]
    shear_velocity = u_tau(frame, option=option)
    u_van = np.zeros(m)
    dudy = sec_ord_fdd(walldist, u)
    rho_ratio = np.sqrt(rho / rho_wall)
    for i in range(m):
        # u_van[i] = np.trapz(rho_ratio[: i + 1], u[: i + 1])
        u_van[i] = np.trapz(rho_ratio[:i+1]*dudy[:i+1], walldist[:i+1])
    u_plus_van = u_van / shear_velocity
    y_plus = shear_velocity * walldist * rho_wall / mu_wall
    # return(y_plus, u_plus_van)
    y_plus[0] = 1
    u_plus_van[0] = 1
    UPlusVan = np.column_stack((y_plus, u_plus_van))
    return (UPlusVan)


def direst_wall_lawRR(walldist, u_tau, uu, rho):
    if (np.diff(walldist) < 0.0).any():
        sys.exit("the WallDist must be in ascending order!!!")
    if (walldist[0] != 0):
        sys.exit('Please reset wall distance from zero!!!')
    m = np.size(uu)
    rho_wall = rho[0]
    uu_van = np.zeros(m)
    for i in range(m):
        uu_van[i] = np.trapz(np.sqrt(rho[: i + 1] / rho_wall), uu[: i + 1])
    uu_plus_van = uu_van / u_tau
    # y_plus = u_tau * walldist * rho_wall / mu_wall
    # return(y_plus, u_plus_van)
    # y_plus[0] = 1
    # u_plus_van[0] = 1
    # UPlusVan = np.column_stack((y_plus, u_plus_van))
    return uu_plus_van


# Obtain reattachment location with time
def reattach_loc(InFolder, OutFolder, timezone, skip=1, opt=2):
    dirs = sorted(os.listdir(InFolder))
    xarr = np.zeros(np.size(timezone))
    j = 0
    if opt == 1:
        data = pd.read_hdf(InFolder + dirs[0])
        # NewFrame = data.query("x>=9.0 & x<=13.0 & y==-2.99703717231750488")
        NewFrame = data.query("x>=8.0 & x<=13.0")
        TemFrame = NewFrame.loc[NewFrame["y"] == -2.99703717231750488]
        ind = TemFrame.index.values
        for i in range(np.size(dirs)):
            if i % skip == 0:
                with timer("Computing reattaching point " + dirs[i]):
                    frame = pd.read_hdf(InFolder + dirs[i])
                    frame = frame.iloc[ind]
                    xarr[j] = frame.loc[frame["u"] >= 0.0, "x"].head(1)
                    j = j + 1
    else:
        for i in range(np.size(dirs)):
            if i % skip == 0:
                with timer("Computing reattaching point " + dirs[i]):
                    frame = pd.read_hdf(InFolder + dirs[i])
                    xy = dividing_line(frame)
                    xarr[j] = np.max(xy[:, 0])
                    j = j + 1
    reatt = np.vstack((timezone, xarr)).T
    np.savetxt(
        OutFolder + "Reattach.dat",
        reatt,
        fmt="%.8e",
        delimiter="  ",
        header="t, x",
    )
    return (reatt)


def extract_point(InFolder, OutFolder, timezone, xy, skip=1, col=None):
    if col is None:
        col = ["u", "v", "w", "p", "vorticity_1", "vorticity_2", "vorticity_3"]
    dirs = sorted(os.listdir(InFolder))
    data = pd.read_hdf(InFolder + dirs[0])
    NewFrame = data.loc[data["x"] == xy[0]]
    TemFrame = NewFrame.loc[NewFrame["y"] == xy[1]]
    ind = TemFrame.index.values[0]
    xarr = np.zeros((np.size(timezone), np.size(col)))
    j = 0
    for i in range(np.size(dirs)):
        if i % skip == 0:
            with timer("Extracting probes"):
                frame = pd.read_hdf(InFolder + dirs[i])
                frame = frame.iloc[ind]
                xarr[j, :] = frame[col].values
                j = j + 1
    probe = np.hstack((timezone.reshape(-1, 1), xarr))
    col.insert(0, "t")
    np.savetxt(
        OutFolder + "x" + str(xy[0]) + "y" + str(xy[1]) + ".dat",
        probe,
        fmt="%.8e",
        delimiter="  ",
        header=", ".join(col),
    )


# Obtain shock location inside the boudary layer with time
def shock_foot(InFolder, OutFolder, timepoints, yval, var, skip=1):
    dirs = sorted(os.listdir(InFolder))
    xarr = np.zeros(np.size(timepoints))
    j = 0
    # if np.size(timepoints) != np.size(dirs):
    #    sys.exit("The input snapshots does not match!!!")
    for i in range(np.size(dirs)):
        if i % skip == 0:
            with timer("Computing shock foot location " + dirs[i]):
                frame = pd.read_hdf(InFolder + dirs[i])
                NewFrame = frame.loc[frame["y"] == yval]
                temp = NewFrame.loc[NewFrame["u"] >= var, "x"]
                xarr[j] = temp.head(1)
                j = j + 1
    foot = np.vstack((timepoints, xarr)).T
    np.savetxt(
        OutFolder + "ShockFoot.dat",
        foot,
        fmt="%.8e",
        delimiter="  ",
        header="t, x",
    )


# Obtain shock location outside boundary layer with time
def shock_loc(InFolder, OutFolder, timepoints, skip=1):
    dirs = sorted(os.listdir(InFolder))
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_xlim([0.0, 30.0])
    ax1.set_ylim([-3.0, 10.0])
    matplotlib.rc("font", size=18)
    data = pd.read_hdf(InFolder + dirs[0])
    x0 = np.unique(data["x"])
    x1 = x0[x0 > 10.0]
    x1 = x1[x1 <= 30.0]
    y0 = np.unique(data["y"])
    y1 = y0[y0 > -2.5]
    xini, yini = np.meshgrid(x1, y1)
    corner = (xini < 0.0) & (yini < 0.0)
    shock1 = np.empty(shape=[0, 3])
    shock2 = np.empty(shape=[0, 3])
    ys1 = 0.5
    ys2 = 5.0
    j = 0
    # if np.size(timepoints) != np.size(dirs):
    #    sys.exit("The input snapshots does not match!!!")
    for i in range(np.size(dirs)):
        if i % skip == 0:
            with timer("Shock position at " + dirs[i]):
                frame = pd.read_hdf(InFolder + dirs[i])
                gradp = griddata(
                        (frame["x"], frame["y"]),
                        frame["|gradp|"],
                        (xini, yini)
                        )
                gradp[corner] = np.nan
                cs = ax1.contour(xini, yini, gradp, levels=[0.06],
                                 linewidths=1.2, colors="gray")
                xycor = np.empty(shape=[0, 2])
                x1 = np.empty(shape=[0, 1])
                x2 = np.empty(shape=[0, 1])
                for isoline in cs.collections[0].get_paths():
                    xy = isoline.vertices
                    xycor = np.append(xycor, xy, axis=0)
                    ax1.plot(xy[:, 0], xy[:, 1], "r:")
                    ind1 = np.where(np.around(xycor[:, 1], 8) == ys1)[0]
                    x1 = np.append(x1, np.mean(xycor[ind1, 0]))
                    ind2 = np.where(np.around(xycor[:, 1], 8) == ys2)[0]
                    x2 = np.append(x2, np.mean(xycor[ind2, 0]))
                x1 = x1[~np.isnan(x1)]
                if np.size(x1) == 0:
                    x1 = 0.0
                else:
                    x1 = x1[0]
                x2 = x2[~np.isnan(x2)]
                if np.size(x2) == 0:
                    x2 = 0.0
                else:
                    x2 = x2[0]
                ax1.plot(x1, ys1, "g*")
                ax1.axhline(y=ys1)
                ax1.plot(x2, ys2, "b^")
                ax1.axhline(y=ys2)
                shock1 = np.append(shock1, [[timepoints[j], x1, ys1]], axis=0)
                shock2 = np.append(shock2, [[timepoints[j], x2, ys2]], axis=0)
                j = j + 1
    np.savetxt(
        OutFolder + "ShockA.dat",
        shock1,
        fmt="%.8e",
        delimiter="  ",
        header="t, x, y",
    )
    np.savetxt(
        OutFolder + "ShockB.dat",
        shock2,
        fmt="%.8e",
        delimiter="  ",
        header="t, x, y",
    )


# Save shock isoline
def shock_line(dataframe, path):
    x0 = np.unique(dataframe["x"])
    x1 = x0[x0 > 10.0]
    x1 = x1[x1 <= 30.0]
    y0 = np.unique(dataframe["y"])
    y1 = y0[y0 > -2.5]
    xini, yini = np.meshgrid(x1, y1)
    corner = (xini < 0.0) & (yini < 0.0)
    gradp = griddata((dataframe["x"], dataframe["y"]), dataframe["|gradp|"],
                     (xini, yini))
    gradp[corner] = np.nan
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(
         xini, yini, gradp, levels=[0.06], linewidths=1.2, colors="gray"
    )
    plt.close()
    header = "x, y"
    xycor = np.empty(shape=[0, 2])
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor = np.vstack((xycor, xy))
        # ax1.scatter(xy[:, 0], xy[:, 1], "r:")
    tck, yval = splprep(xycor.T, s=1.0, per=1)
    x_new, y_new = splev(yval, tck, der=0)
    xy_fit = np.vstack((x_new, y_new))
    np.savetxt(
        path + "ShockLine.dat",
        xycor,
        fmt="%.8e",
        delimiter="  ",
        header=header
    )
    np.savetxt(
        path + "ShockLineFit.dat",
        xy_fit.T,
        fmt="%.8e",
        delimiter="  ",
        header=header
    )


def sonic_line(dataframe, path, option='Mach', Ma_inf=1.7):
    # NewFrame = dataframe.query("x>=0.0 & x<=15.0 & y<=0.0")
    x, y = np.meshgrid(np.unique(dataframe.x), np.unique(dataframe.y))
    if 'u' not in dataframe.columns:
        dataframe['u'] = dataframe['<u>']
        dataframe['v'] = dataframe['<v>']
        dataframe['w'] = dataframe['<w>']
        dataframe['T'] = dataframe['<T>']
    if option == 'velocity':
        if 'w' in dataframe.columns:
            c = np.sqrt(dataframe['u'] ** 2 + dataframe['v'] ** 2
                        + dataframe['w'] ** 2)
        else:
            c = np.sqrt(dataframe['u'] ** 2 + dataframe['v'] ** 2)
        dataframe['Mach'] = c / np.sqrt(dataframe['T']) * Ma_inf
        Ma = griddata((dataframe.x, dataframe.y), dataframe.Mach, (x, y))
    else:
        Ma = griddata((dataframe.x, dataframe.y), dataframe.Mach, (x, y))
        # sys.exit("Mach number is not in the dataframe")
    corner = (x < 0.0) & (y < 0.0)
    Ma[corner] = np.nan
    header = "x, y"
    xycor = np.empty(shape=[0, 2])
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(x, y, Ma, levels=[1.0], linewidths=1.5, colors='k')
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor = np.vstack((xycor, xy))
    np.savetxt(path + "SonicLine.dat", xycor, fmt='%.8e',
               delimiter='  ', header=header)


def dividing_line(dataframe, path=None):
    NewFrame = dataframe.query("x>=0.0 & x<=15.0 & y<=0.0")
    x, y = np.meshgrid(np.unique(NewFrame.x), np.unique(NewFrame.y))
    if 'u' not in NewFrame.columns:
        NewFrame['u'] = NewFrame['<u>']
    u = griddata((NewFrame.x, NewFrame.y), NewFrame.u, (x, y))
    fig, ax = plt.subplots(figsize=(10, 4))
    cs1 = ax.contour(
        x, y, u, levels=[0.0], linewidths=1.5, linestyles="--", colors="k"
    )
    plt.close()
    header = "x, y"
    xycor = np.empty(shape=[0, 2])
    xylist = []
    for i, isoline in enumerate(cs1.collections[0].get_paths()):
        xy = isoline.vertices
        if np.any(xy[:, 1] == -0.015625):  # pick the bubble line
            ind = i
        xylist.append(xy)
        xycor = np.vstack((xycor, xy))
    xy = xylist[ind]
    fig1, ax1 = plt.subplots(figsize=(10, 4))  # plot only bubble
    ax1.scatter(xy[:, 0], xy[:, 1])
    plt.close()
    if path is not None:
        np.savetxt(
            path + "DividingLine.dat", xycor, fmt="%.8e", delimiter="  ",
            header=header
        )
        np.savetxt(
            path + "BubbleLine.dat", xy, fmt="%.8e", delimiter="  ",
            header=header
        )
    return xy


def boundary_edge(dataframe, path, jump1=None, jump2=None):  # jump = reattachment location
    # dataframe = dataframe.query("x<=30.0 & y<=3.0")
    x, y = np.meshgrid(np.unique(dataframe.x), np.unique(dataframe.y))
    if 'u' not in dataframe.columns:
        dataframe.loc[:, 'u'] = dataframe['<u>']
        # dataframe['u'] = dataframe['<u>']

    u = griddata((dataframe.x, dataframe.y), dataframe.u, (x, y))
    if (jump1==None):
        expand = 0.0  # reattchament location
    else:
        expand = jump1
    if (jump2==None):
        shock = 10.375  # reattchament location
    else:
        shock = jump2
    umax = u[-1, :]
    rg1 = (x[1, :] < expand)  # in front of the shock
    umax[rg1] = 1.0
    rg2 = (x[1, :] >= shock)  # behind the shock
    umax[rg2] = 0.98
    u = u / (np.transpose(umax))
    corner = (x < 0.0) & (y < 0.0)
    u[corner] = np.nan
    header = 'x, y'
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(
        x, y, u, levels=[0.99], linewidths=1.5, linestyles="--", colors="k"
    )
    xycor = np.empty(shape=[0, 2])
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor = np.vstack((xycor, xy))
    np.savetxt(
        path + "BoundaryEdge.dat", xycor, fmt="%.8e", delimiter="  ",
        header=header
    )


def bubble_area(InFolder, OutFolder, timezone, step=3.0, skip=1):
    dirs = sorted(os.listdir(InFolder))
    area = np.zeros(np.size(timezone))
    j = 0
    for i in range(np.size(dirs)):
        if i % skip == 0:
            with timer("Bubble area at " + dirs[i]):
                dataframe = pd.read_hdf(InFolder + dirs[i])
                xy = dividing_line(dataframe)
                area[j] = trapz(xy[:, 1] + step, xy[:, 0])
            j = j + 1
    area_arr = np.vstack((timezone, area)).T
    np.savetxt(
        OutFolder + "BubbleArea.dat",
        area_arr,
        fmt="%.8e",
        delimiter="  ",
        header="area",
    )
    return area


def correlate(x, y, method="Sample"):
    if np.size(x) != np.size(y):
        sys.exit("The size of two datasets do not match!!!")
    if method == "Population":
        sigma1 = np.std(x, ddof=0)
        sigma2 = np.std(y, ddof=0)  # default population standard deviation
        sigma12 = np.cov(x, y, ddof=0)[0][
            1
        ]  # default sample standard deviation
        correlation = sigma12 / sigma1 / sigma2
    else:
        sigma1 = np.std(x, ddof=1)
        sigma2 = np.std(y, ddof=1)  # default Sample standard deviation
        sigma12 = np.cov(x, y, ddof=1)[0][
            1
        ]  # default sample standard deviation
        correlation = sigma12 / sigma1 / sigma2
    return correlation


def delay_correlate(x, y, dt, delay, method="Sample"):
    if delay == 0.0:
        correlation = correlate(x, y, method=method)
    elif delay < 0.0:
        delay = np.abs(delay)
        num = int(delay // dt)
        y1 = y[:-num]
        x1 = x[num:]
        correlation = correlate(x1, y1, method=method)
    else:
        num = int(delay // dt)
        x1 = x[:-num]
        y1 = y[num:]
        correlation = correlate(x1, y1, method=method)
    return correlation


def perturbations(orig, mean):
    grouped = orig.groupby(['x', 'y', 'z'])
    frame1 = grouped.mean().reset_index()
    grouped = mean.groupby(['x', 'y', 'z'])
    frame2 = grouped.mean().reset_index()
    if (np.shape(frame1)[0] != np.shape(frame2)[0]):
        sys.exit("The size of two datasets do not match!!!")
    pert = frame1 - frame2
    return pert


def pert_at_loc(orig, var, loc, val, mean=None):
    frame1 = orig.loc[np.around(orig[loc[0]], 5) == np.around(val[0], 5)]
    frame1 = frame1.loc[np.around(frame1[loc[1]], 5) == np.around(val[1], 5)]
    grouped = frame1.groupby(['x', 'y', 'z'])
    frame1 = grouped.mean().reset_index()
    print(np.shape(frame1)[0])
    frame = frame1
    if mean is not None:
        frame2 = mean.loc[mean[loc[0]] == val[0]]
        frame2 = frame2.loc[np.around(frame2[loc[1]], 5) == val[1]]
        grouped = frame2.groupby(['x', 'y', 'z'])
        frame2 = grouped.mean().reset_index()
        print(np.shape(frame2)[0])
        if (np.shape(frame1)[0] != np.shape(frame2)[0]):
            sys.exit("The size of two datasets do not match!!!")
        # z value in frame1 & frame2 is equal or not ???
        frame[var] = frame1[var] - frame2[var]
    else:
        frame[var] = frame1[var]
    return frame


def max_pert_along_y(orig, var, val, mean=None):
    frame1 = orig.loc[orig['x'] == val[0]]
    frame1 = frame1.loc[np.around(frame1['z'], 5) == val[1]]
    grouped = frame1.groupby(['x', 'y', 'z'])
    frame1 = grouped.mean().reset_index()
    print(np.shape(frame1)[0])
    if mean is not None:
        frame2 = mean.loc[mean['x'] == val[0]]
        frame2 = frame2.loc[np.around(frame2['z'], 5) == val[1]]
        grouped = frame2.groupby(['x', 'y', 'z'])
        frame2 = grouped.mean().reset_index()
        print(np.shape(frame2)[0])
        if (np.shape(frame1)[0] != np.shape(frame2)[0]):
            sys.exit("The size of two datasets do not match!!!")
        # z value in frame1 & frame2 is equal or not ???
        frame1[var] = frame1[var] - frame2[var]
    frame = frame1.loc[frame1[var].idxmax()]
    return frame


def amplit(orig, xyz, var, mean=None):
    frame1 = orig.loc[np.around(orig['x'], 6) == np.around(xyz[0], 6)]
    # frame2 = frame1.loc[np.around(frame1['y'], 5) == xyz[1]]
    frame2 = frame1.loc[np.around(frame1['y'], 6) == np.around(xyz[1], 6)]
    orig = frame2.loc[frame2['z'] == xyz[2]]
    if mean is None:
        grouped = orig.groupby(['x', 'y', 'z'])
        mean = grouped.mean().reset_index()
    else:
        frame1 = mean.loc[mean['x'] == xyz[0]]
        frame2 = frame1.loc[frame1['y'] == xyz[1]]
        mean = mean.loc[frame2['z'] == xyz[2], var]
    pert = orig[var].values - mean[var].values
    amplit = np.max(np.abs(pert))
    return amplit


def growth_rate(xarr, var):
    dAmpli = sec_ord_fdd(xarr, var)
    growthrate = dAmpli/var
    return growthrate


#   Obtain finite differential derivatives of a variable (2nd order)
def sec_ord_fdd(xarr, var):
    dvar = np.zeros(np.size(xarr))
    for jj in range(1, np.size(xarr)):
        if jj == 1:
            dvar[jj-1] = (var[jj] - var[jj-1]) / (xarr[jj] - xarr[jj-1])
        elif jj == np.size(xarr):
            dvar[jj-1] = (var[jj-1] - var[jj-2]) / (xarr[jj-1] - xarr[jj-2])
        else:
            dy12 = xarr[jj-1] - xarr[jj-2]
            dy23 = xarr[jj] - xarr[jj-1]
            dvar1 = -dy23 / dy12 / (dy23 + dy12) * var[jj-2]
            dvar2 = (dy23 - dy12) / dy23 / dy12 * var[jj-1]
            dvar3 = dy12 / dy23 / (dy23 + dy12) * var[jj]
            dvar[jj-1] = dvar1 + dvar2 + dvar3
    return (dvar)
# Vorticity: omega=delta*v
# omega1 = dw/dy-dv/dz; omega2 = du/dz-dw/dx, omega3 = dv/dx-du/dy


# double integral
def integral_db(x, y, val, range1=None, range2=None, opt=2):
    if range1 is None:
        min1 = np.min(x)
        max1 = np.max(x)
        n1 = np.size(np.unique(x))
        range1 = np.linspace(min1, max1, n1)
    else:
        n1 = np.size(range1)
        min1 = np.min(range1)
        max1 = np.max(range1)
    if range2 is None:
        min2 = np.min(y)
        max2 = np.max(y)
        n2 = np.size(np.unique(y))
        range2 = np.linspace(min2, max2, n2)
    else:
        n2 = np.size(range2)
        min2 = np.min(range2)
        max2 = np.max(range2)
    if opt == 1:
        def func(var1, var2):
            vorticity = griddata((x, y), val, (var1, var2))
            return vorticity
        results = dblquad(func, min1, max1, lambda x: min2, lambda y: max2)
    elif opt == 2:
        ms1, ms2 = np.meshgrid(range1, range2)
        val_intp = griddata((x, y), val, (ms1, ms2))
        Iy = np.zeros(n2)
        for i in range(n2):
            Iy[i] = np.trapz(val_intp[i, :], range1)
        # print("finish integral over x-axis")
        results = np.trapz(Iy, range2)
    elif opt == 3:
        ms1, ms2 = np.meshgrid(range1, range2)
        val_intp = griddata((x, y), val, (ms1, ms2))
        val_intp_p = np.where(val_intp > 0.0, val_intp, 0.0)  # posivive values
        val_intp_n = np.where(val_intp < 0.0, val_intp, 0.0)  # negative values
        Iy = np.zeros((n2, 2))
        for i in range(n2):
            Iy[i, 0] = np.trapz(val_intp_p[i, :], range1)
            Iy[i, 1] = np.trapz(val_intp_n[i, :], range1)
        # print("finish integral over x-axis")
        res1 = np.trapz(Iy[:, 0], range2)
        res2 = np.trapz(Iy[:, 1], range2)
        results = (res1, res2)          
    else:
        pass
    return results


def vorticity_abs(df, mode=None):
    if mode is None:
        vorticity = df['vorticity_1'].values**2 \
            + df['vorticity_2'].values**2 \
            + df['vorticity_3'].values**2
    elif mode == 'x':
        vorticity = df['vorticity_1'].values**2
    elif mode == 'y':
        vorticity = df['vorticity_2'].values**2
    elif mode == 'z':
        vorticity = df['vorticity_3'].values**2
    rst = vorticity * 0.5
    return rst


def enstrophy(df, type='x', mode=None, rg1=None, rg2=None, opt=2):
    if type == 'x':
        x1 = df.y
        x2 = df.z
    if type == 'y':
        x1 = df.x
        x2 = df.z
    if type == 'z':
        x1 = df.x
        x2 = df.y
    vorticity = vorticity_abs(df, mode=mode)
    ens = integral_db(x1, x2, vorticity, range1=rg1,
                      range2=rg2, opt=opt)
    return ens


def vortex_dyna(df, type='z', opt=1):
    delta_u = df['ux'] + df['vy'] + df['wz']
    rho_inv = 1 / df['rho'] ** 2
    if type == 'z':
        vorticity = df['vorticity_3']
        tilting1 = df['vorticity_1'] * df['wx']
        tilting2 = df['vorticity_2'] * df['wy']
        stretching = vorticity * df['wz']
        dilatation = -vorticity * delta_u
        torque = rho_inv * (df['rhox'] * df['py'] - df['rhoy'] * df['px'])
    if type == 'x':
        vorticity = df['vorticity_1']
        stretching = vorticity * df['ux']
        tilting1 = df['vorticity_2'] * df['uy']
        tilting2 = df['vorticity_3'] * df['uz']
        dilatation = -df['vorticity_1'] * delta_u
        torque = rho_inv * (df['rhoy'] * df['pz'] - df['rhoz'] * df['py'])
    if type == 'y':
        vorticity = df['vorticity_2']
        tilting1 = df['vorticity_1'] * df['vx']
        stretching = vorticity * df['vy']
        tilting2 = df['vorticity_3'] * df['vz']
        dilatation = -vorticity * delta_u
        torque = rho_inv * (df['rhoz'] * df['px'] - df['rhox'] * df['pz'])
    if opt == 2:
        tilting1 = tilting1 * vorticity
        tilting2 = tilting2 * vorticity
        stretching = stretching * vorticity
        dilatation = -dilatation * vorticity
        torque = torque * vorticity
    df = df.assign(tilt1=tilting1)
    df = df.assign(tilt2=tilting2)
    df = df.assign(stretch=stretching)
    df = df.assign(dilate=dilatation)
    df = df.assign(bar_tor=torque)
    return df


if __name__ == "__main__":
    """
    InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots1/"
    OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/Data/"
    timezone = np.arange(600, 999.5 + 0.5, 0.5)
    ShockFoot(InFolder, OutFolder, timezone, -1.875, 0.8)

    InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots1/"
    OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/Data/"
    path2 = "/media/weibo/Data1/BFS_M1.7L_0505/"
    numsize = 14
    textsize = 18
    #%%  Plot reattachment point with time
    #spl = splrep(timezone, xarr, s=0.35)
    #xarr1 = splev(timezone[0::5], spl)
    reatt = np.loadtxt(OutFolder+"Reattach.dat", skiprows=1)
    timezone = reatt[:, 0]
    Xr = reatt[:, 1]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
    ax.set_ylabel('Weighted PSD, unitless', fontsize=textsize)
    ax.grid(b=True, which='both', linestyle=':')
    Fre, FPSD = FW_PSD(Xr, 0.5, 2.0)
    Fre1, psd1 = PSD(Xr, 0.5, 2.0, opt=2)
    ax.semilogx(Fre, FPSD, 'k', linewidth=1.0)
    ax.yaxis.offsetText.set_fontsize(numsize)
    plt.tick_params(labelsize=numsize)
    plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
    plt.savefig(path2+'XrFWPSD.svg', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    shock1 = np.loadtxt(OutFolder+"Shock1.dat", skiprows=1)
    shock2 = np.loadtxt(OutFolder+"Shock2.dat", skiprows=1)
    angle = np.arctan(5/(shock2[:, 1]-shock1[:, 1]))
    foot = shock2[:, 1] - 8.0/np.tan(angle)
    Xs = foot
    #Xs = angle*180/np.pi #shock2[:, 1]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.yaxis.major.formatter.set_powerlimits((0, 3))
    ax.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
    ax.set_ylabel ('Weighted PSD, unitless', fontsize=textsize)
    ax.grid(b=True, which='both', linestyle=':')
    Fre1, FPSD1 = FW_PSD(Xs, 0.5, 2.0)
    Fre2, psd2 = PSD(Xs, 0.5, 2.0, opt=2)
    ax.semilogx(Fre, FPSD1, 'k', linewidth=1.0)
    ax.yaxis.offsetText.set_fontsize(numsize)
    plt.tick_params(labelsize=numsize)
    plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
    plt.savefig(path2+'ShockangleFWPSD.svg',
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    fre, cor = Cro_PSD(Xr, Xs, 0.5, 2.0)
    phase = np.arctan(cor.imag, cor.real)
    fre, gam = Coherence(Xr, Xs, 0.5, 2.0)
    fre1, gam1 = Coherence(Xr, Xs, 0.5, 2.0, opt=2)

    fre1 = fre1[1:]
    gam1 = gam1[1:]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.yaxis.major.formatter.set_powerlimits((0, 3))
    ax.set_xlabel(r'$f\delta_0/u_\infty$', fontsize=textsize)
    ax.set_ylabel ('Weighted PSD, unitless', fontsize=textsize)
    ax.grid(b=True, which='both', linestyle=':')
    # Fre3, PSD3 = signal.welch(Xs, fs=2.0, nperseg=800, scaling='density')
    Fre3, PSD3 = PSD(Xr, 0.5, 2.0, opt=1)
    FPSD3 = Fre3*PSD3
    ax.semilogx(Fre3, FPSD3, 'k', linewidth=1.0)
    ax.yaxis.offsetText.set_fontsize(numsize)
    plt.tick_params(labelsize=numsize)
    plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1)
    plt.savefig(path2+'test.svg', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    """
