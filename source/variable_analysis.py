# -*- coding: utf-8 -*-
"""
Created on Tue May 1 10:24:50 2018

@author: Weibo Hu
@license: OpenSource

This code for reading data from specific file to post-processing data.

Overview:

- basic_var(opt)
- mean_var(opt)
- intermittency(sigma, pressure, wall_pres, timezone)
- alpha3(wall_pres)
- viscosity(re_delta, temp, law, tmep_inf)
- bl_thickness(y, u, u_d, rho, opt, up)
- shape_factor(y, u, rho, u_d)
- radius(x, y, opt)
- curvature(x, y, opt)
- gortler(re_inf, x, y, theta, scale, rad)
- gortler_tur(theta, delta_star, rad, opt)
- curvature_r(df, opt)
- skinfriction(mu, du, dy)
- tke(df)
- psd(varzone, dt, freq_samp, opt, seg, overlap)
- fw_psd(varzone, dt, freq_samp, opt, seg, overlap)
- fw_psd_map(orig, xyz, var, dt, freq_samp, opt, seg, overlap)
- rms(dataseries)
- rms_map(orig, xyz, var)
- cro_psd(var1, var2, dt, freq_samp, opt, seg, overlap)
- coherence(var1, var2, dt, freq_samp, opt, seg, overlap)
- std_wall_law()
- ref_wall_law(Re_theta)
- u_tau(frame, option)
- direst_transform(frame, option)
- direst_wall_lawRR(walldist, u_tau, uu, rho)
- reattach_loc(InFolder, OutFolder, timezone, loc, skip, opt)
- separate_loc(InFolder, OutFolder, timezone, loc, skip, opt)
- extract_point(InFolder, OutFolder, timezone, xy, skip, col)
- shock_foot(InFolder, OutFolder, timepoints, yval, var, skip)
- shock_loc(InFolder, OutFolder, timepoints, skip, opt, val)
- shock_line(dataframe, path)
- shock_line_ffs(dataframe, path, val)
- sonic_line(dataframe, path, option, Ma_inf)
- dividing_line(dataframe, path, loc)
- boundary_edge(dataframe, path, jump0, jump1, jump2, val1, val2)
- bubble_area(InFolder, OutFolder, timezone, loc, step, skip, cutoff)
- streamline(InFolder, df, seeds, OutFile, partition=None, opt):
- correlate(x, y, method)
- delay_correlate(x, y, dt, delay, method)
- perturbations(orig, mean)
- pert_at_loc(orig, var, loc, val, mean)
- max_pert_along_y(orig, var, val, mean)
- amplit(orig, xyz, var, mean)
- growth_rate(xarr, var)
- sec_ord_fdd(xarr, var)
- integral_db(x, y, val, range1, range2, opt)
- vorticity_abs(df, mode)
- enstrophy(df, type, mode, rg1, rag2, opt)
- vortex_dyna(df, type, opt)
- grs(r, s, bs)
- mixing(r, s, Cd, phi, opt)
- stat2tot(Ma, Ts, opt, mode)

"""
# %%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import warnings
import pandas as pd
import variable_analysis as fv
from scipy import interpolate  # scipy.optimize
from scipy.interpolate import griddata  # interp1d
from scipy.interpolate import splprep, splev, interp1d, UnivariateSpline
from scipy.integrate import trapz, dblquad  # simps,
import sys
from timer import timer
import os

# from numpy import NaN, Inf, arange, isscalar, asarray, array

#%%
def basic_var(opt):
    """generate name list for the dataframe

       Args:
        opt: if include vorticity or walldist

       Return:
        namelist and the corresponding equations
    """
    varlist = ["x", "y", "z", "u", "v", "w", "p", "rho", "T", "|gradp|"]
    equ = ["{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2+ddz({p})**2)"]
    if opt == "vorticity":
        varlist.extend(
            ["vorticity_1", "vorticity_2", "vorticity_3",
                "Q-criterion", "L2-criterion"]
        )
    elif opt == "walldist":
        varlist.append("walldist")
    return (varlist, equ)


def mean_var(opt):
    """generat name list for the meanflow dataframe

       Args:
        opt: if include vorticity, velocity gradient or walldist

       Return:
        namelist and the corresponding equations
    """
    varlist = ["x", "y", "z", "<u>", "<v>", "<w>",
               "<p>", "<rho>", "<T>", "|grad(<p>)|"]
    equ = ["{|gradp|}=sqrt(ddx({<p>})**2+ddy({<p>})**2+ddz({<p>})**2)"]
    if opt == "vorticity":
        varlist.extend(
            [
                "<vorticity_1>",
                "<vorticity_2>",
                "<vorticity_3>",
                "<Q-criterion>",
                "<lambda_2>",
            ]
        )
    elif opt == "walldist":
        varlist.append("walldist")
    elif opt == "gradient":
        varlist.extend(
            ["dudx", "dudy", "dudz", "dvdx", "dvdy", "dvdz", "dwdx", "dwdy", "dwdz"]
        )
        equ.extend(
            [
                "{dudx}=ddx({<u>})",
                "{dudy}=ddy({<u>})",
                "{dudz}=ddz({<u>})",
                "{dvdx}=ddx({<v>})",
                "{dvdy}=ddy({<v>})",
                "{dvdz}=ddz({<v>})",
                "{dwdx}=ddx({<w>})",
                "{dwdy}=ddy({<w>})",
                "{dwdz}=ddz({<w>})",
            ]
        )
    return (varlist, equ)


def add_variable(df, wavy, nms=None):
    if nms is None:
        nms = ["p", "T", "u", "v", "walldist"]
    for i in range(np.size(nms)):
        wavy[nms[i]] = griddata(
            (df.x, df.y), df[nms[i]],
            (wavy.x, wavy.y),
            method="linear",
        )     
    return(wavy)


def intermittency(sigma, Pressure0, WallPre, TimeZone):
    """Obtain intermittency factor from pressure

       Args:
        sigma: standard deviation of undisturbed wall pressure
        pressure0: undisturbed wall pressure
        wallpres: wall pressure
        timezone: time periods

       Return:
        intermittency factor
    """
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
def viscosity(Re_ref, T, law="POW", T_inf=45):
    # nondimensional T
    # Re based on reference length instead of delta
    if law == "POW":
        mu = 1.0 / Re_ref * np.power(T, 0.75)
    elif law == "Suther":  # Sutherland's law, mu_ref = mu_inf
        # mu_ref = 0.00001716
        # T_ref = 273.15 / T_inf
        S = 110.4 / T_inf
        a = (1 + S) / (T + S)
        mu = 1.0 / Re_ref * a * np.power(T, 3 / 2)
    elif law == "dim":
        mu_ref = 0.00001716
        T_ref = 273.15
        S = 110.4
        mu = mu_ref * np.power(T / T_ref, 1.5) * (T_ref + S) / (T + S)
    return(mu)


# Obtain the thermal conductivity
def thermal(viscosity, Pr=0.72, opt=None):
    gammar = 1.4
    R = 287.05
    if opt is None:  # nondimensional
        k_thermal = viscosity / Pr
    else:  # dimensional
        a = gammar * R / (gammar-1) / Pr
        k_thermal = a * viscosity        
    return(k_thermal)
        

# Obtain BL thickness, momentum thickness, displacement thickness
def bl_thickness(y, u, u_d=None, rho=None, opt=None, up=0.95):
    if isinstance(y, np.ndarray):
        pass
    else:
        y = y.values
    if isinstance(u, np.ndarray):
        pass
    else:
        u = u.values
    # ind = np.argsort(y)  # sort y from small to large
    if np.any(np.diff(y) <= 0):
        sys.exit("The boundary layer is not sorted in y-direction!!!")
    bc = int(np.rint(np.size(y) * up))
    y1 = y[:bc]  # remove the part near the farfield boundary conditions
    u1 = u[:bc]  # remove the part near the farfield boundary conditions
    if u_d is None:
        bl = np.where(u1[:] >= 0.99)[0][0]
    else:
        bl = np.where(u1[:] >= 0.99 * u_d)[0][0]
    if np.size(bl) == 0:
        sys.exit("This is not a complete boundary layer profile!!!")
    delta = y1[bl]
    u_d = u1[bl]
    if opt is None:
        return (delta, u_d)
    elif opt == "displacement":
        rho1 = rho[:bc]
        rho_d = np.max(rho1)  # rho1[bl] #
        u_d = np.max(u1)  # u1[bl] #
        a1 = rho1 * u1 / rho_d / u_d
        var = 1 - a1
        delta_star = np.trapz(var, y1)
        return (delta_star, u_d, rho_d)
    elif opt == "momentum":
        rho1 = rho[:bc]
        rho_d = np.max(rho1)  # rho1[bl]  #
        u_d = np.max(u1)  # u1[bl]  #
        a1 = 1 - u1 / u_d
        a2 = rho1 * u1 / rho_d / u_d
        var = a1 * a2
        theta = np.trapz(var, y1)
        return (theta, u_d, rho_d)


# shape factor of boundary layer
def shape_factor(y, u, rho, u_d=None):
    delta_star = bl_thickness(y, u, u_d=u_d, rho=rho, opt="displacement")[0]
    theta = bl_thickness(y, u, u_d=u_d, rho=rho, opt="momentum")[0]
    shape = delta_star / theta
    return shape


# radius of curve curvature
def radius(x, y, opt="external"):
    curva = curvature(x, y, opt=opt)
    radi = 1 / curva
    return radi


def curvature(x, y, opt="external"):
    if opt == "internal":
        dydx = np.gradient(y, x)
        ddyddx = np.gradient(dydx, x)
    if opt == "external":
        dydx = fv.sec_ord_fdd(x, y)  # y1, dydx; y2, ddyddx
        ddyddx = fv.sec_ord_fdd(x, dydx)
    a1 = 1 + (dydx) ** 2
    a2 = ddyddx
    curv = a2 / np.power(a1, 1.5)
    return curv


# Obtain G\"ortler number
def gortler(Re_inf, x, y, theta, scale=0.001, radi=None):
    Re_theta = Re_inf * theta * scale
    if radi is None:
        radi = radius(x, y)
    gortler = Re_theta * np.sqrt(theta / radi)
    return gortler


def gortler_tur(theta, delta_star, radi, opt="radius"):
    # radi = Radius(x, y)
    a1 = theta / 0.018 / delta_star
    if opt == "radius":
        a2 = np.sqrt(theta / np.abs(radi))
    elif opt == "curvature":
        a2 = np.sqrt(theta * np.abs(radi))
    gortler = a1 * a2  # * np.sign(radi)
    return gortler


def curvature_r(df, opt="mean"):
    if opt == "mean":
        u = df["<u>"]
        v = df["<v>"]
    elif opt == "inst":
        u = df["u"]
        v = df["v"]
    dudx = df["dudx"]
    dudy = df["dudy"]
    dvdx = df["dvdx"]
    dvdy = df["dvdy"]
    numerator = u ** 2 * dvdx - v ** 2 * dudy + u * v * (dvdy - dudx)
    denominator = np.power(u ** 2 + v ** 2, 3.0 / 2.0)
    radius = numerator / denominator
    return radius


def dist(wav_seg, path):
    dxdy = np.gradient(wav_seg['y'], wav_seg['x'])
    ang = np.arctan(dxdy)
    wav_seg['ut'] = wav_seg['u'] * np.cos(ang) + wav_seg['v'] * np.sin(ang)
    # normal distance
    wall = pd.read_csv(path + "WallBoundary.dat", skipinitialspace=True)
    wavval = wav_seg[['x', 'y']].values
    dim = np.shape(wav_seg)[0]
    dist = np.zeros(dim)
    for i in range(dim):
        dist[i] = np.min(np.linalg.norm(wavval[i] - wall, axis=1)) 
    return(wav_seg, dist)   


# Obtain skin friction coefficency
def skinfriction(mu, du, dy, factor=1):
    # all variables are nondimensional
    if isinstance(dy, np.ndarray):
        if(np.size(np.where(dy == 0.0)) > 0):
            dy[np.where(dy == 0.0)] = 1e-8
            print('Warning: there is zero value for dy!!!')
    Cf = 2 * mu * du / dy * factor
    return Cf


def skinfric_wavy(path, wavy, Re, T_inf, wall_val):
    # wavy = pd.read_csv(path + "FirstLev.dat", skipinitialspace=True)
    # nms = ["p", "T", "u", "v"]
    # for i in range(np.size(nms)):
    #     wavy[nms[i]] = griddata(
    #         (df.x, df.y), df[nms[i]],
    #         (wavy.x, wavy.y),
    #         method="cubic",
    #     )
    # skin friction for a flat plate    
    mu = viscosity(Re, wavy["T"], law="Suther", T_inf=T_inf)
    Cf = skinfriction(mu, wavy["u"].values, wavy["y"].values)
    # skin friction for a wavy wall
    ind = wavy.index[wavy["y"] < wall_val]
    wav_seg = wavy.iloc[ind]
    # tangential velocity
    dxdy = np.gradient(wav_seg['y'], wav_seg['x'])
    ang = np.arctan(dxdy)
    wav_seg['ut'] = wav_seg['u'] * np.cos(ang) + wav_seg['v'] * np.sin(ang)
    
    # normal distance
    wall = pd.read_csv(path + "WallBoundary.dat", skipinitialspace=True)
    wavval = wav_seg[['x', 'y']].values
    dim = np.shape(wav_seg)[0]
    dist = np.zeros(dim)
    for i in range(dim):
        dist[i] = np.min(np.linalg.norm(wavval[i] - wall, axis=1))
    mu = viscosity(Re, wav_seg["T"], law="Suther", T_inf=T_inf)
    Cf2 = skinfriction(mu, wav_seg["ut"], dist).values
    Cf[ind] = Cf2
    return(wavy.x.values, Cf.values)


# obtain Stanton number 
def Stanton(dT, Tw, dy, Re, Ma, T_inf, factor=1):
    if isinstance(dy, np.ndarray):
        if(np.size(np.where(dy == 0.0)) > 0):
            dy[np.where(dy == 0.0)] = 1e-8
            print('Warning: there is zero value for dy!!!')
    Tt = stat2tot(Ma, T_inf, opt='t')
    mu = viscosity(Re, dT, T_inf=T_inf, law="Suther")
    kt = thermal(mu, 0.72)
    St = - kt * (dT - Tw) / dy / (Tt - Tw) * factor
    return(St)


def Stanton_wavy(path, wavy, Re, Ma, T_inf, T_wall, wall_val):
    # wavy = pd.read_csv(path + "FirstLev.dat", skipinitialspace=True)
    # nms = ["p", "T", "u", "v", "walldist"]
    # for i in range(np.size(nms)):
    #     wavy[nms[i]] = griddata(
    #         (df.x, df.y), df[nms[i]],
    #         (wavy.x, wavy.y),
    #         method="cubic",
    #     )
    # skin friction for a flat plate   
    Tt = stat2tot(Ma, Ts=T_inf, opt="t") / 45
    mu = viscosity(Re, wavy["T"], T_inf=T_inf, law="Suther")
    kt = thermal(mu, 0.72)
    Cs = Stanton(
        kt,
        wavy["T"].values,
        T_wall,
        wavy["walldist"].values,
        Re,
        Ma,
        T_inf
    )

    # skin friction for a wavy wall
    ind = wavy.index[wavy["y"] < wall_val]
    wav_seg = wavy.iloc[ind]
    wav_seg, wall_dist = dist(wav_seg, path)
    mu = viscosity(10000, wav_seg["T"], T_inf=T_inf, law="Suther")
    kt = thermal(mu, 0.72)
    Cs2 = Stanton(
        kt,
        wav_seg["T"].values,
        6.66,
        wall_dist,
        Tt,
    ) 
    Cs[ind] = Cs2
    return(wavy.x.values, Cs.values)

# Obtain turbulent kinetic energy
def tke(df):
    # all variables are nondimensional
    kinetic_energy = 0.5 * (df["<u`u`>"] + df["<v`v`>"] + df["<w`w`>"])
    return kinetic_energy


# obtain Power Spectral Density
def psd(VarZone, dt, Freq_samp, opt=2, seg=8, overlap=4):
    TotalNo = np.size(VarZone)
    if np.size(dt) > 1:
        TotalNo = int(Freq_samp * (dt[-1] - dt[0]))
        if TotalNo > np.size(dt):
            warnings.warn(
                "PSD results are not accurate as too few snapshots", UserWarning
            )
        TimeZone = np.linspace(dt[0], dt[-1], TotalNo)
        VarZone = VarZone - np.mean(VarZone)
        Var = np.interp(TimeZone, dt, VarZone)
    else:
        if Freq_samp > 1 / dt:
            warnings.warn(
                "PSD results are not accurate as too few snapshots", UserWarning
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
            # time space must be equal
            Var = np.interp(TimeZone, TimeSpan, VarZone)
    # POD, fast fourier transform and remove the half
    if opt == 2:
        Var_fft = np.fft.rfft(Var)  # [1:]  # remove value at 0 frequency
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
    Freq, Var_PSD = psd(VarZone, dt, Freq_samp, opt=opt,
                        seg=seg, overlap=overlap)
    FPSD = Var_PSD * Freq
    return (Freq, FPSD)


def fw_psd_map(orig, xyz, var, dt, Freq_samp, opt=2, seg=8, overlap=4):
    frame1 = orig.loc[np.around(orig["x"], 5) == np.around(xyz[0], 5)]
    # frame2 = frame1.loc[np.around(frame1['y'], 5) == xyz[1]]
    frame2 = frame1.loc[np.around(frame1["y"], 5) == np.around(xyz[1], 5)]
    orig = frame2.loc[frame2["z"] == xyz[2]]
    varzone = orig[var]
    Freq, FPSD = fw_psd(varzone, dt, Freq_samp, opt=opt,
                        seg=seg, overlap=overlap)
    return (Freq, FPSD)


# Compute the RMS
def rms(dataseries):
    meanval = np.mean(dataseries)
    rmsval = np.sqrt(np.mean((dataseries - meanval) ** 2))
    return rmsval


# Compute the RMS
def rms_map(orig, xyz, var):
    frame1 = orig.loc[np.around(orig["x"], 6) == np.around(xyz[0], 6)]
    frame2 = frame1.loc[np.around(frame1["y"], 6) == np.around(xyz[1], 6)]
    orig = frame2.loc[np.around(frame2["z"], 6) == np.around(xyz[2], 6)]
    varzone = orig[var]
    rms_val = rms(varzone)
    return rms_val


# Obtain cross-power sepectral density
def cro_psd(Var1, Var2, dt, Freq_samp, opt=1, seg=8, overlap=4):
    TotalNo = np.size(Var1)
    if np.size(Var1) != np.size(Var2):
        warnings.warn("Check the size of input varable 1 & 2", UserWarning)
    if Freq_samp > 1 / dt:
        warnings.warn(
            "PSD results are not accurate due to too few snapshots", UserWarning
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
        # time space must be equal
        NVar1 = np.interp(TimeZone, TimeSpan, VarZone1)
        # time space must be equal
        NVar2 = np.interp(TimeZone, TimeSpan, VarZone2)
    if opt == 1:
        ns = TotalNo // seg
        Freq, Cpsd = signal.csd(
            NVar1, NVar2, Freq_samp, nperseg=ns, nfft=TotalNo, noverlap=ns // overlap
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


def coherence(Var1, Var2, dt, Freq_samp, opt=1, seg=8, overlap=4):
    TotalNo = np.size(Var1)
    if np.size(Var1) != np.size(Var2):
        warnings.warn("Check the size of input varable 1 & 2", UserWarning)
    if Freq_samp > 1 / dt:
        warnings.warn(
            "PSD results are not accurate due to too few snapshots", UserWarning
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
        # time space must be equal
        NVar1 = np.interp(TimeZone, TimeSpan, VarZone1)
        # time space must be equal
        NVar2 = np.interp(TimeZone, TimeSpan, VarZone2)
    if opt == 1:
        ns = TotalNo // seg  # 6-4 # 8-2
        Freq, gamma = signal.coherence(
            NVar1,
            NVar2,
            fs=Freq_samp,
            nperseg=ns,
            nfft=TotalNo,
            noverlap=ns // overlap,
        )
        Freq = Freq[1:]
        gamma = gamma[1:]
    if opt == 2:
        Freq, cor = cro_psd(NVar1, NVar2, dt, Freq_samp, opt=1)
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
    path = os.path.abspath("..") + "/database/"
    if Re_theta <= 830:
        file = path + "vel_0670_dns.prof"
    elif 830 < Re_theta <= 1200:
        file = path + "vel_1000_dns.prof"
    elif 1200 < Re_theta <= 1700:
        file = path + "vel_1410_dns.prof"
    elif 1700 < Re_theta <= 2050:
        file = path + "vel_2000_dns.prof"
    elif 2050 < Re_theta <= 2100:
        file = path + "vel_2080_dns.prof"
    elif 2100 < Re_theta <= 2300:
        file = path + "vel_2150_dns.prof"
    elif 2050 < Re_theta <= 2120:
        file = path + "vel_2080_dns.prof"
    elif 2120 < Re_theta <= 2300:
        file = path + "vel_2150_dns.prof"
    elif 2300 < Re_theta <= 2500:
        file = path + "vel_2400_dns.prof"
    elif 2500 < Re_theta <= 2800:
        file = path + "vel_2540_dns.prof"
    elif 2800 < Re_theta <= 3150:
        file = path + "vel_3030_dns.prof"
    elif 3150 < Re_theta <= 3450:
        file = path + "vel_3270_dns.prof"
    elif 3450 < Re_theta <= 3800:
        file = path + "vel_3630_dns.prof"
    elif 3800 < Re_theta <= 4000:
        file = path + "vel_3970_dns.prof"
    else:
        file = path + "vel_4060_dns.prof"

    print("Take reference data: " + file)
    ExpData = np.loadtxt(file, skiprows=14)
    m, n = ExpData.shape
    # y_delta = ExpData[:, 0]
    y_plus = ExpData[:, 1]
    u_plus = ExpData[:, 2]
    urms_plus = ExpData[:, 3]
    vrms_plus = ExpData[:, 4]
    wrms_plus = ExpData[:, 5]
    uv_plus = ExpData[:, 6]
    if n > 7:
        Xi = ExpData[:, 7]
    else:
        Xi = np.ones(np.shape(y_plus))
    UPlus = np.column_stack((y_plus, u_plus))
    UVPlus = np.column_stack((y_plus, uv_plus))
    UrmsPlus = np.column_stack((y_plus, urms_plus))
    VrmsPlus = np.column_stack((y_plus, vrms_plus))
    WrmsPlus = np.column_stack((y_plus, wrms_plus))
    return (UPlus, UVPlus, UrmsPlus, VrmsPlus, WrmsPlus, Xi)


def u_tau(frame, option="mean", grad=False):
    """
    input
    ------
        boundary layer profile
    return
    ------
        friction/shear velocity from mean or instantaneous flow
    """
    if frame["walldist"].values[0] != 0:
        sys.exit("Please reset wall distance/velocity from zero!!!")
    if option == "mean":
        # rho_wall = frame['rho_m'].values[0]
        # mu_wall = frame['mu_m'].values[0]
        # delta_u = frame['u_m'].values[1] - frame['u_m'].values[0]
        rho_wall = frame["<rho>"].values[0]
        mu_wall = frame["<mu>"].values[0]
        delta_u = frame["<u>"].values[1] - frame["<u>"].values[0]
        u_grad = np.gradient(
            frame["<u>"].values, frame["walldist"].values, edge_order=2
        )
    else:
        rho_wall = frame["rho"].values[0]
        mu_wall = frame["mu"].values[0]
        delta_u = frame["u"].values[1] - frame["u"].values[0]
        u_grad = np.gradient(
            frame["u"].values, frame["walldist"].values, edge_order=2)
    walldist2 = frame["walldist"].values[1]

    #    if(frame['walldist'].values[1] > 0.005):
    #        print('Interpolate for u_wall')
    #        func = interp1d(frame['walldist'].values, frame['u'].values, kind='cubic')
    #        delta_u = func(0.004)
    #        walldist2 = 0.004
    if grad == True:
        tau_wall = mu_wall * u_grad[1]
    else:
        tau_wall = mu_wall * delta_u / walldist2
    shear_velocity = np.sqrt(np.abs(tau_wall / rho_wall))
    return shear_velocity


# This code validate boundary layer profile by
# incompressible, Van Direst transformed
# boundary profile from mean reults
def direst_transform(frame, option="mean", grad=False):
    """
    This code validate boundary layer profile by
    incompressible, Van Direst transformed
    boundary profile from mean reults
    """
    if option == "mean":
        walldist = frame["walldist"].values
        #       u = frame['u_m']
        #       rho = frame['rho_m']
        #       mu = frame['mu_m']
        u = frame["<u>"].values
        rho = frame["<rho>"].values
        mu = frame["<mu>"].values
    else:
        walldist = frame["walldist"].values
        u = frame["u"].values
        rho = frame["rho"].values
        mu = frame["mu"].values

    if (np.diff(walldist) < 0.0).any():
        sys.exit("the WallDist must be in ascending order!!!")
    if walldist[0] != 0:
        sys.exit("Please reset wall distance from zero!!!")
    m = np.size(u)
    rho_wall = rho[0]
    mu_wall = mu[0]
    shear_velocity = u_tau(frame, option=option, grad=grad)
    u_van = np.zeros(m)
    dudy = sec_ord_fdd(walldist, u)
    rho_ratio = np.sqrt(rho / rho_wall)
    for i in range(m):
        # u_van[i] = np.trapz(rho_ratio[: i + 1], u[: i + 1])
        u_van[i] = np.trapz(rho_ratio[: i + 1] *
                            dudy[: i + 1], walldist[: i + 1])
    u_plus_van = u_van / shear_velocity
    y_plus = shear_velocity * walldist * rho_wall / mu_wall
    # return(y_plus, u_plus_van)
    y_plus = y_plus[1:]  # y_plus[0] = 1 #
    u_plus_van = u_plus_van[1:]  # u_plus_van[0] = 1 #
    UPlusVan = np.column_stack((y_plus, u_plus_van))
    return UPlusVan


def direst_wall_lawRR(walldist, u_tau, uu, rho):
    if (np.diff(walldist) < 0.0).any():
        sys.exit("the WallDist must be in ascending order!!!")
    if walldist[0] != 0:
        sys.exit("Please reset wall distance from zero!!!")
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
def reattach_loc(InFolder, OutFolder, timezone, loc=-0.015625, skip=1, opt=2):
    dirs = sorted(os.listdir(InFolder))
    xarr = np.zeros(np.size(timezone))
    j = 0
    if opt == 1:
        data = pd.read_hdf(InFolder + dirs[0])
        grouped = data.groupby(["x", "y"])
        data = grouped.mean().reset_index()
        # NewFrame = data.query("x>=9.0 & x<=13.0 & y==-2.99703717231750488")
        NewFrame = data.query("x>=7.5 & x<=13.0")
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
                    grouped = frame.groupby(["x", "y"])
                    frame = grouped.mean().reset_index()
                    xy = dividing_line(frame, loc=loc)
                    xarr[j] = np.max(xy[:, 0])
                    j = j + 1
    reatt = np.vstack((timezone, xarr)).T
    np.savetxt(
        OutFolder + "Reattach.dat", reatt, fmt="%.8e", delimiter="  ", header="t, x",
    )
    return reatt


# obtain separation location with time
def separate_loc(InFolder, OutFolder, timezone, loc=-0.015625, skip=1, opt=2):
    dirs = sorted(os.listdir(InFolder))
    xarr = np.zeros(np.size(timezone))
    yarr = np.zeros(np.size(timezone))
    j = 0
    if opt == 1:
        data = pd.read_hdf(InFolder + dirs[0])
        grouped = data.groupby(["x", "y"])
        data = grouped.mean().reset_index()
        # NewFrame = data.query("x>=9.0 & x<=13.0 & y==-2.99703717231750488")
        NewFrame = data.query("x>=-30.0 & x<=-5.0")
        ywall = np.unique(NewFrame["y"])[1]
        TemFrame = NewFrame.loc[NewFrame["y"] == ywall]
        ind = TemFrame.index.values

        NewFrame1 = data.query("y<=3.0 & x<=0")
        xwall = np.unique(NewFrame1["x"])[-2]
        TemFrame1 = NewFrame1.loc[NewFrame1["x"] == xwall]
        ind1 = TemFrame1.index.values
        for i in range(np.size(dirs)):
            if i % skip == 0:
                with timer("Computing reattaching point " + dirs[i]):
                    frame = pd.read_hdf(InFolder + dirs[i])
                    frame0 = frame.iloc[ind]
                    # aa = frame0.iloc[frame0['Cf'].abs().argsort()]
                    aa = frame0.loc[frame0["u"] < 0.0, "x"].head(16)
                    xarr[j] = aa.values[-1]

                    frame1 = frame.iloc[ind1]
                    yarr[j] = frame1.loc[frame1["u"] < 0.0, "y"].tail(1)
                    j = j + 1
    else:
        for i in range(np.size(dirs)):
            if i % skip == 0:
                with timer("Computing reattaching point " + dirs[i]):
                    frame = pd.read_hdf(InFolder + dirs[i])
                    grouped = frame.groupby(["x", "y"])
                    frame = grouped.mean().reset_index()
                    xy = dividing_line(frame, loc=loc)
                    xarr[j] = np.min(xy[:, 0])
                    ind = np.argwhere(xy[:, 0] == 0.0)[0]
                    yarr[j] = xy[ind, 1]
                    j = j + 1
    separate = np.vstack((timezone, xarr)).T
    reattach = np.vstack((timezone, yarr)).T
    np.savetxt(
        OutFolder + "Separate.dat", separate, fmt="%.8e", delimiter="  ", header="t, x",
    )
    np.savetxt(
        OutFolder + "Reattach.dat", reattach, fmt="%.8e", delimiter="  ", header="t, y",
    )
    return separate


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
                grouped = frame.groupby(["x", "y"])
                frame = grouped.mean().reset_index()
                NewFrame = frame.loc[frame["y"] == yval]
                temp = NewFrame.loc[NewFrame["u"] <= var, "x"]
                xarr[j] = temp.head(1)
                j = j + 1
    foot = np.vstack((timepoints, xarr)).T
    np.savetxt(
        OutFolder + "ShockFoot.dat", foot, fmt="%.8e", delimiter="  ", header="t, x",
    )


# Obtain shock location outside boundary layer with time
def shock_loc(
    InFolder,
    OutFolder,
    timepoints,
    skip=1,
    opt=1,
    var="|gradp|",
    lev=0.065,
    val=[0.91, 0.92],
):
    dirs = sorted(os.listdir(InFolder))
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_xlim([-30.0, 5.0])
    ax1.set_ylim([0.0, 10.0])
    matplotlib.rc("font", size=18)
    data = pd.read_hdf(InFolder + dirs[0])
    x0 = np.unique(data["x"])
    x1 = x0[x0 > -25]
    x1 = x1[x1 <= -4.0]
    y0 = np.unique(data["y"])
    y1 = y0[y0 > 0.5]
    xini, yini = np.meshgrid(x1, y1)
    corner = (xini > 0.0) & (yini < 3.0)
    shock1 = np.empty(shape=[0, 3])
    shock2 = np.empty(shape=[0, 3])
    ys1 = 3.0
    ys2 = 6.0
    j = 0
    # if np.size(timepoints) != np.size(dirs):
    #    sys.exit("The input snapshots does not match!!!")
    if opt == 1:
        for i in range(np.size(dirs)):
            if i % skip == 0:
                with timer("Shock position at " + dirs[i]):
                    frame = pd.read_hdf(InFolder + dirs[i])
                    grouped = frame.groupby(["x", "y"])
                    frame = grouped.mean().reset_index()
                    gradp = griddata(
                        (frame["x"], frame["y"]), frame[var], (xini, yini))
                    gradp[corner] = np.nan
                    cs = ax1.contour(
                        xini, yini, gradp, levels=[lev], linewidths=1.2, colors="gray"
                    )
                    xycor = np.empty(shape=[0, 2])
                    x1 = np.empty(shape=[0, 1])
                    x2 = np.empty(shape=[0, 1])
                    for isoline in cs.collections[0].get_paths():
                        xy = isoline.vertices
                        xycor = np.append(xycor, xy, axis=0)
                        ax1.plot(xy[:, 0], xy[:, 1], ":")
                        # yarr = np.ones(np.shape(xycor)[0]) * ys1
                        # ydif = xycor[:, 1] - yarr
                    yarr1 = np.ones(np.shape(xycor)[0]) * ys1
                    ydif1 = xycor[:, 1] - yarr1
                    ind0 = np.where(ydif1[:] >= 0.0)[0]  # upper half
                    xy_n0 = xycor[ind0, :]
                    ind1 = (xy_n0[:, 0] >= -15.5) & (xy_n0[:, 0] <= -13.5)
                    xy_n1 = xy_n0[ind1, :]
                    xy_n1 = xy_n1[xy_n1[:, 1].argsort()]  # most close two
                    xtm1 = (xy_n1[0, 0] + xy_n1[1, 0]) / 2
                    x1 = np.append(x1, xtm1)

                    yarr2 = np.ones(np.shape(xycor)[0]) * ys2
                    ydif2 = xycor[:, 1] - yarr2
                    ind2 = np.where(ydif2[:] >= 0.0)[0]
                    xy_n02 = xycor[ind2, :]
                    ind01 = (xy_n02[:, 0] >= -13) & (xy_n02[:, 0] <= -10)
                    xy_n2 = xy_n02[ind01, :]
                    xy_n2 = xy_n2[xy_n2[:, 1].argsort()]
                    xtm2 = (xy_n2[0, 0] + xy_n2[1, 0]) / 2
                    x2 = np.append(x2, xtm2)

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
                    shock1 = np.append(
                        shock1, [[timepoints[j], x1, ys1]], axis=0)
                    shock2 = np.append(
                        shock2, [[timepoints[j], x2, ys2]], axis=0)
                    j = j + 1
                    plt.show()
                    # plt.close()
    elif opt == 2:
        for i in range(np.size(dirs)):
            if i % skip == 0:
                with timer("Shock position at " + dirs[i]):
                    frame = pd.read_hdf(InFolder + dirs[i])
                    grouped = frame.groupby(["x", "y"])
                    frame = grouped.mean().reset_index()
                    NewFrame1 = frame.loc[frame["y"] == ys1]
                    temp1 = NewFrame1.loc[NewFrame1["u"] <= val[0], "x"]
                    x1 = temp1.head(1)

                    NewFrame2 = frame.loc[frame["y"] == ys2]
                    temp2 = NewFrame2.loc[NewFrame2["u"] <= val[1], "x"]
                    x2 = temp2.head(1)
                    shock1 = np.append(
                        shock1, [[timepoints[j], x1, ys1]], axis=0)
                    shock2 = np.append(
                        shock2, [[timepoints[j], x2, ys2]], axis=0)
                    j = j + 1

    np.savetxt(
        OutFolder + "ShockA.dat",
        shock1,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header="t, x, y",
    )
    np.savetxt(
        OutFolder + "ShockB.dat",
        shock2,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header="t, x, y",
    )


# Save shock isoline
def shock_line_old(dataframe, path, var="|gradp|", val=0.06):
    grouped = dataframe.groupby(["x", "y"])
    dataframe = grouped.mean().reset_index()
    x0 = np.unique(dataframe["x"])
    x1 = x0[x0 > 10.0]
    x1 = x1[x1 <= 30.0]
    y0 = np.unique(dataframe["y"])
    y1 = y0[y0 > -2.5]
    xini, yini = np.meshgrid(x1, y1)
    corner = (xini < 0.0) & (yini < 0.0)
    if var=="|gradp|":
        gradp = griddata(
            (dataframe["x"],
             dataframe["y"]),
             dataframe["|gradp|"],
             (xini, yini)
        )
    else:
        gradp = griddata(
            (dataframe["x"],
             dataframe["y"]),
             dataframe[var],
             (xini, yini)
        )
    gradp[corner] = np.nan
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(xini, yini, gradp, levels=[
                    val], linewidths=1.2, colors="gray")
    plt.show()
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
        delimiter=", ",
        comments="",
        header=header,
    )
    np.savetxt(
        path + "ShockLineFit.dat",
        xy_fit.T,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header=header,
    )

# Save shock isoline
def shock_line(dataframe, path, var="|gradp|", val=0.06, mask=False):
    grouped = dataframe.groupby(["x", "y"])
    dataframe = grouped.mean().reset_index()
    x0 = np.unique(dataframe["x"])
    y0 = np.unique(dataframe["y"])
    xini, yini = np.meshgrid(x0, y0)
    if var == "|gradp|":
        gradp = griddata(
            (dataframe["x"],
             dataframe["y"]),
            dataframe["|gradp|"],
            (xini, yini)
        )
    else:
        gradp = griddata(
            (dataframe["x"],
             dataframe["y"]),
            dataframe[var],
            (xini, yini)
        )
    if mask is True:
        walldist = griddata(
            (dataframe["x"],
             dataframe["y"]),
            dataframe["walldist"],
            (xini, yini)
        )
        corner = (walldist < 0.0)
        gradp[corner] = np.nan
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(xini, yini, gradp, levels=[
                    val], linewidths=1.2, colors="gray")
    plt.show()
    header = "x, y"
    xycor = np.empty(shape=[0, 2])
    ii = 0
    nolist = []
    if not os.path.exists(path + 'Shock'):
        os.mkdir(path + 'Shock')
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        lnsize = np.shape(xy)[0]
        nolist.append(lnsize)
        np.savetxt(
            path + "Shock/ShockLine{:03d}.dat".format(ii),
            xy,
            fmt="%.8e",
            delimiter=", ",
            comments="",
            header=header,
        )
        ii = ii + 1
        xycor = np.vstack((xycor, xy))
        
    ind = np.argmax(nolist)
    print("pick the line with id of " + str(ind))
    xypick = pd.read_csv(
        path + "Shock/ShockLine{:03d}.dat".format(ind),
        skipinitialspace=True
        )
    ax.scatter(xypick.x, xypick.y, s=0.5, c='r')
    np.savetxt(
        path + "ShockLineAll.dat",
        xycor,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header=header,
    )
    tck, yval = splprep(xycor.T, s=1.0, per=1)
    x_new, y_new = splev(yval, tck, der=0)
    xy_fit = np.vstack((x_new, y_new))
    np.savetxt(
        path + "ShockLineFit.dat",
        xy_fit.T,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header=header,
    )


# Save shock isoline
def shock_line_ffs(dataframe, path, val=[0.06], show=False):
    if not isinstance(val, list):
        print("input value must be an array!")
        exit
    grouped = dataframe.groupby(["x", "y"])
    dataframe = grouped.mean().reset_index()
    x0 = np.unique(dataframe["x"])
    y0 = np.unique(dataframe["y"])
    x1 = x0[x0 > -70]
    y1 = y0[(y0 > 0.5) & (y0 < 30.0)]
    xini, yini = np.meshgrid(x1, y1)
    corner = (xini > 0.0) & (yini < 3.0)
    gradp = griddata(
        (dataframe["x"], dataframe["y"]), dataframe["|gradp|"], (xini, yini)
    )
    gradp[corner] = np.nan
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(xini, yini, gradp, levels=val,
                    linewidths=1.2, colors="gray")
    ax.set_xlim([-30.0, 10.0])
    ax.set_ylim([0.0, 12.0])
    if show == False:
        plt.close()
    header = "x, y"
    xycor1 = np.empty(shape=[0, 2])
    # xycor2 = np.empty(shape=[0, 2])
    xylist = []
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor1 = np.vstack((xycor1, xy))
        # if np.min(xy[:, 0]) < -10.0:
        #     xycor1 = np.vstack((xycor1, xy))
        # elif np.min(xy[:, 0]) > -5.0:
        #     xycor2 = np.vstack((xycor2, xy))
        xylist.append(xy)
        # ax1.scatter(xy[:, 0], xy[:, 1], "r:")
    tck, yval = splprep(xycor1.T, s=1.0, per=1)
    x_new, y_new = splev(yval, tck, der=0)
    xy_fit = np.vstack((x_new, y_new))
    np.savetxt(
        path + "ShockLine1.dat",
        xycor1,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header=header,
    )
    np.savetxt(
        path + "ShockLineFit.dat",
        xy_fit.T,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header=header,
    )
    # np.savetxt(
    #     path + "ShockLine2.dat",
    #     xycor2,
    #     fmt="%.8e",
    #     delimiter=", ",
    #     comments="",
    #     header=header,
    # )


def sonic_line(dataframe, path, option="Mach", Ma_inf=1.7, mask=None):
    # NewFrame = dataframe.query("x>=0.0 & x<=15.0 & y<=0.0")
    grouped = dataframe.groupby(["x", "y"])
    dataframe = grouped.mean().reset_index()
    x, y = np.meshgrid(np.unique(dataframe.x), np.unique(dataframe.y))
    if "u" not in dataframe.columns:
        dataframe["u"] = dataframe["<u>"]
        dataframe["v"] = dataframe["<v>"]
        dataframe["w"] = dataframe["<w>"]
        dataframe["T"] = dataframe["<T>"]
    if option == "velocity":
        if "w" in dataframe.columns:
            c = np.sqrt(dataframe["u"] ** 2 +
                        dataframe["v"] ** 2 + dataframe["w"] ** 2)
        else:
            c = np.sqrt(dataframe["u"] ** 2 + dataframe["v"] ** 2)
        dataframe["Mach"] = c / np.sqrt(dataframe["T"]) * Ma_inf
        Ma = griddata((dataframe.x, dataframe.y), dataframe.Mach, (x, y))
    else:
        Ma = griddata((dataframe.x, dataframe.y), dataframe.Mach, (x, y))
        # sys.exit("Mach number is not in the dataframe")
    if mask is not None:
        Ma = np.ma.array(Ma, mask=mask)
    header = "x, y"
    xycor = np.empty(shape=[0, 2])
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(x, y, Ma, levels=[1.0], linewidths=1.5, colors="k")
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor = np.vstack((xycor, xy))
    np.savetxt(
        path + "SonicLine.dat",
        xycor,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header=header,
    )


def wall_line(dataframe, path, mask=None, val=0.0):
    # NewFrame = dataframe.query("x>=0.0 & x<=15.0 & y<=0.0")
    grouped = dataframe.groupby(["x", "y"])
    dataframe = grouped.mean().reset_index()
    x, y = np.meshgrid(np.unique(dataframe.x), np.unique(dataframe.y))
    try:
        "walldist" in dataframe.columns
    except:
        sys.exit("there is no walldist!!!")

    walldist = griddata(
        (dataframe.x, dataframe.y), dataframe.walldist, (x, y), method="cubic"
    )

    if mask is not None:
        # corner = (x < 0.0) & (y < 0.0)
        # walldist[corner] = np.nan
        # cover1 = walldist < -0.003
        walldist = np.ma.array(walldist, mask=mask)  # mask=cover
    # header = "x, y"
    xycor = np.empty(shape=[0, 2])
    fig, ax = plt.subplots(figsize=(10, 4))
    # cs = ax.contourf(x, y, walldist, extend='min')
    cs = ax.contour(x, y, walldist, levels=[val], linewidths=1.5, colors="k")
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor = np.vstack((xycor, xy))
    onelev = pd.DataFrame(data=xycor, columns=["x", "y"])
    onelev.drop_duplicates(subset='x', keep='first', inplace=True)
    onelev.to_csv(path + "WallBoundary.dat", index=False, float_format="%9.8e")
    return(onelev.values)


def dividing_line(dataframe, path=None, show=False, mask=None):
    """Obtain dividing line

       Args:
        dataframe: dataframe
        path: path of saving data
        loc: point passing through the dividing line

       Return:
        array of coordinates

       Raises:
        data error: find no bubble line
    """
    grouped = dataframe.groupby(["x", "y"])
    NewFrame = grouped.mean().reset_index()
    # NewFrame = dataframe.query("x>=-70.0 & x<=0.0 & y<=10.0")
    x, y = np.meshgrid(np.unique(NewFrame.x), np.unique(NewFrame.y))
    if "u" not in NewFrame.columns:
        NewFrame["u"] = NewFrame["<u>"]
    u = griddata((NewFrame.x, NewFrame.y), NewFrame.u, (x, y))
    if mask is not None:
        u = np.ma.array(u, mask=mask)  # mask=cover
    fig, ax = plt.subplots(figsize=(10, 4))
    cs1 = ax.contour(x, y, u, levels=[
                     0.0], linewidths=1.5, linestyles="--", colors="k")
    if show == False:
        plt.close()
    header = "x, y, zone"
    xycor = np.empty(shape=[0, 2])
    xylist = []
    nolist = []
    ind = 0
    for i, isoline in enumerate(cs1.collections[0].get_paths()):
        xy = isoline.vertices
        nrows = np.shape(xy)[0]
        nolist.append(nrows)
        xylist.append(xy)
        xycor = np.vstack((xycor, xy))
        key_id = "zone_" + "%03d" % i
        df = pd.DataFrame(data=xy, columns=["x", "y"])
        df['zone'] = key_id
        if i == 0:
            save_df = df
        else:
            save_df = pd.concat([save_df, df])
    save_df.to_hdf(
        path + "DividingLine.h5",
        key='data', mode="w",
        format="fixed"
        )
    ind = np.argmax(nolist)
    print("pick the line with id of " + str(ind))
    xy = xylist[ind]
    fig1, ax1 = plt.subplots(figsize=(10, 4))  # plot only bubble
    ax1.scatter(xy[:, 0], xy[:, 1])
    if show is False:
        plt.close()
    else:
        plt.show()
    if path is not None:
        np.savetxt(
            path + "DividingLine.dat",
            xycor,
            fmt="%.8e",
            delimiter=", ",
            comments="",
            header=header,
        )
        np.savetxt(
            path + "BubbleLine.dat",
            xy,
            fmt="%.8e",
            delimiter=", ",
            comments="",
            header=header,
        )
    return xy


def boundary_edge(
    dataframe,
    path,
    shock=True,
    mask=True,
    jump0=-18,
    jump1=-15.0,
    jump3=16.0,
    val1=0.81,
    val3=0.98,
):  # jump = reattachment location
    # dataframe = dataframe.query("x<=30.0 & y<=3.0")
    grouped = dataframe.groupby(["x", "y"])
    dataframe = grouped.mean().reset_index()
    x, y = np.meshgrid(np.unique(dataframe.x), np.unique(dataframe.y))
    if "u" not in dataframe.columns:
        dataframe.loc[:, "u"] = dataframe["<u>"]
        # dataframe['u'] = dataframe['<u>']
    u = griddata((dataframe.x, dataframe.y), dataframe.u, (x, y))
    umax = u[-1, :]
    umax[:] = 1.0
    if shock is True:
        # range1
        rg1 = (x[1, :] <= jump1) & (x[1, :] >= jump0)  # between two shocks
        uinterp = np.interp(x[1, rg1], [jump0, jump1], [0.99, val1 + 0.000])
        umax[rg1] = uinterp
        # range2
        rg2 = (x[1, :] > jump1) & (x[1, :] < -0.5)
        umax[rg2] = val1
        # range3
        rg3 = (x[1, :] >= -0.5) & (x[1, :] < jump3)
        uinterp1 = np.interp(x[1, rg3], [-0.5, jump3], [val1, val3 + 0.003])
        umax[rg3] = uinterp1
        # range4
        rg4 = x[1, :] >= jump3
        umax[rg4] = val3
    u = u / (np.transpose(umax))
    if mask is True:
        corner = (x > 0.0) & (y < 3.0)
        u[corner] = np.nan
    # header = "x, y"
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(x, y, u, levels=[0.99],
                    linewidths=1.5, linestyles="--", colors="k")
    plt.show()
    xycor = np.empty(shape=[0, 2])
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor = np.vstack((xycor, xy))

    wall = pd.DataFrame(data=xycor, columns=["x", "y"])
    wall.drop_duplicates(subset='x', keep='first', inplace=True)
    wall.to_csv(path + "BoundaryEdge.dat", index=False, float_format="%9.8e")
    return(wall.values)


def enthalpy_boundary_edge(
    dataframe,
    path,
    Ma_inf,
    crit=1.005,
    corner=None,
):  # jump = reattachment location
    # dataframe = dataframe.query("x<=30.0 & y<=3.0")
    gamma = 1.4
    grouped = dataframe.groupby(["x", "y"])
    dataframe = grouped.mean().reset_index()
    x, y = np.meshgrid(np.unique(dataframe.x), np.unique(dataframe.y))
    a0 = 1/Ma_inf**2/(gamma-1)
    if "u" not in dataframe.columns:
        dataframe["u"] = dataframe["<u>"]
        dataframe["rho"] = dataframe["<rho>"]
        dataframe["T"] = dataframe["<T>"]
    if "H" not in dataframe.columns:
        a1 = dataframe["rho"]*dataframe["T"]*a0
        a2 = 0.5*dataframe["rho"]*dataframe["u"]**2
        dataframe.loc[:, "H"] = a1 + a2
        
    enthalpy = griddata((dataframe.x, dataframe.y), dataframe.H, (x, y))
    H_inf = a0 + 0.5
    val = crit * H_inf
    if corner is not None:
        enthalpy[corner] = np.nan
    # header = "x, y"
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(x, y, enthalpy, levels=[val],
                    linewidths=1.5, linestyles="--", colors="k")
    plt.show()
    xycor = np.empty(shape=[0, 2])
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor = np.vstack((xycor, xy))

    wall = pd.DataFrame(data=xycor, columns=["x", "y"])
    wall.drop_duplicates(subset='x', keep='first', inplace=True)
    wall.to_csv(path + "EnthalpyBoundaryEdge.dat", index=False, float_format="%9.8e")
    return(wall.values)


def thermal_boundary_edge(
    dataframe,
    path,
    T_wall,
    corner=None,
):  # jump = reattachment location
    # dataframe = dataframe.query("x<=30.0 & y<=3.0")
    grouped = dataframe.groupby(["x", "y"])
    dataframe = grouped.mean().reset_index()
    x, y = np.meshgrid(np.unique(dataframe.x), np.unique(dataframe.y))
    a0 = 0.99 * (T_wall - 1.0)
    if "T" not in dataframe.columns:
        dataframe.loc[:, "T"] = dataframe["<T>"]       
    Temp = griddata((dataframe.x, dataframe.y), dataframe['T'], (x, y))
    val = T_wall - a0
    if corner is not None:
        Temp[corner] = np.nan
    # header = "x, y"
    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contour(x, y, Temp, levels=[val],
                    linewidths=1.5, linestyles="--", colors="k")
    plt.show()
    xycor = np.empty(shape=[0, 2])
    for isoline in cs.collections[0].get_paths():
        xy = isoline.vertices
        xycor = np.vstack((xycor, xy))

    wall = pd.DataFrame(data=xycor, columns=["x", "y"])
    wall.drop_duplicates(subset='x', keep='first', inplace=True)
    wall.to_csv(path + "ThermalpyBoundaryEdge.dat", index=False, float_format="%9.8e")
    return(wall.values)



def bubble_area(
    InFolder, OutFolder, timezone, loc=-0.015625, step=3.0, skip=1, cutoff=None
):
    dirs = sorted(os.listdir(InFolder))
    area = np.zeros(np.size(timezone))
    j = 0
    if cutoff is None:
        cutoff = -0.015625
    else:
        assert isinstance(cutoff, float), "cutoff:{} is not a float".format(
            cutoff.__class__.__name__
        )
    plt.close()
    fig1, ax1 = plt.subplots(figsize=(10, 4))  # plot only bubble
    for i in range(np.size(dirs)):
        if i % skip == 0:
            with timer("Bubble area at " + dirs[i]):
                dataframe = pd.read_hdf(InFolder + dirs[i])
                grouped = dataframe.groupby(["x", "y"])
                dataframe = grouped.mean().reset_index()
                xy_org = dividing_line(dataframe, loc=loc)
                if np.max(xy_org[:, 1]) < cutoff:
                    ind = np.argmax(xy_org[:, 1])
                    xy = xy_org[ind:, :]
                    area[j] = trapz(xy[:, 1] + step, xy[:, 0])
                    area[j] = area[j] + 0.5 * xy[0, 0] * (0 - xy[0, 1])
                else:
                    xy = xy_org
                    area[j] = trapz(xy[:, 1] + step, xy[:, 0])
                ax1.plot(xy[:, 0], xy[:, 1])
                # ind = np.argmax(xy_new[:, 1])
                # if xy[ind, 1] < cutoff:
                #    area[j] = area[j] + 0.5 * xy[ind, 0] * (0 - xy[ind, 1])
            j = j + 1
    plt.show()
    area_arr = np.vstack((timezone, area)).T
    np.savetxt(
        OutFolder + "BubbleArea.dat",
        area_arr,
        fmt="%.8e",
        delimiter=", ",
        comments="",
        header="area",
    )
    return area


def streamline(InFolder, df, seeds, OutFile=None, partition=None, opt="both"):
    if partition is None:
        xa1 = np.arange(-40.0, 0.0 + 0.03125, 0.03125)
        ya1 = np.arange(0.0, 5.0 + 0.03125, 0.03125)
        xa2 = np.arange(0.0, 40.0 + 0.03125, 0.03125)
        ya2 = np.arange(3.0, 5.0 + 0.03125, 0.03125)
    else:
        if np.shape(partition) != (2, 3):
            sys.exit("the shape of partition does not match (2,3)!")
        xa1 = np.arange(partition[0, 0], partition[0, 1] + 0.0625, 0.0625)
        xa2 = np.arange(partition[0, 1], partition[0, 2] + 0.0625, 0.0625)
        ya1 = np.arange(partition[1, 0], partition[1, 1] + 0.015625, 0.015625)
        ya2 = np.arange(partition[1, 1], partition[1, 2] + 0.015625, 0.015625)

    xb1, yb1 = np.meshgrid(xa1, ya1)
    xb2, yb2 = np.meshgrid(xa2, ya2)
    u1 = griddata((df.x, df.y), df.u, (xb1, yb1))
    v1 = griddata((df.x, df.y), df.v, (xb1, yb1))
    u2 = griddata((df.x, df.y), df.u, (xb2, yb2))
    v2 = griddata((df.x, df.y), df.v, (xb2, yb2))
    strm1 = np.empty([0, 2])
    strm2 = np.empty([0, 2])
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for j in range(np.shape(seeds)[1]):
        point = np.reshape(seeds[:, j], (1, 2))
        # upstream the step
        if (opt == "up") or (opt == "both"):
            stream1 = ax.streamplot(
                xb1,
                yb1,
                u1,
                v1,
                color="k",
                density=[3.0, 2.0],
                arrowsize=0.7,
                start_points=point,
                maxlength=30.0,
                linewidth=1.0,
            )
            seg = stream1.lines.get_segments()
            strm1 = np.asarray([i[0] for i in seg])
            ax.plot(strm1[:, 0], strm1[:, 1], "b:")
        # downstream the step
        if (opt == "down") or (opt == "both"):
            stream2 = ax.streamplot(
                xb2,
                yb2,
                u2,
                v2,
                color="g",
                density=[3.0, 2.0],
                arrowsize=0.7,
                start_points=point,
                maxlength=30.0,
                linewidth=0.8,
            )
            seg = stream2.lines.get_segments()
            strm2 = np.asarray([i[0] for i in seg])
            ax.plot(strm2[:, 0], strm2[:, 1], "r--")
        # save data
        frame = pd.DataFrame(data=np.vstack(
            (strm1, strm2)), columns=["x", "y"])
        frame.drop_duplicates(subset=["x"], keep="first", inplace=True)
        if OutFile is None:
            frame.to_csv(
                InFolder + "streamline" + str(j + 1) + ".dat",
                index=False,
                float_format="%1.8e",
                sep=" ",
            )
        else:
            frame.to_csv(
                InFolder + OutFile + str(j + 1) + ".dat",
                index=False,
                float_format="%1.8e",
                sep=" ",
            )
        plt.savefig(InFolder + "streamline" + str(j + 1) +
                    ".svg", bbox_inches="tight")

    return frame


def correlate(x, y, method="Sample"):
    if np.size(x) != np.size(y):
        sys.exit("The size of two datasets do not match!!!")
    if method == "Population":
        sigma1 = np.std(x, ddof=0)
        sigma2 = np.std(y, ddof=0)  # default population standard deviation
        # default sample standard deviation
        sigma12 = np.cov(x, y, ddof=0)[0][1]
        correlation = sigma12 / sigma1 / sigma2
    else:
        sigma1 = np.std(x, ddof=1)
        sigma2 = np.std(y, ddof=1)  # default Sample standard deviation
        # default sample standard deviation
        sigma12 = np.cov(x, y, ddof=1)[0][1]
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
    grouped = orig.groupby(["x", "y", "z"])
    frame1 = grouped.mean().reset_index()
    grouped = mean.groupby(["x", "y", "z"])
    frame2 = grouped.mean().reset_index()
    if np.shape(frame1)[0] != np.shape(frame2)[0]:
        sys.exit("The size of two datasets do not match!!!")
    pert = frame1 - frame2
    return pert


def pert_at_loc(orig, var, loc, val, mean=None):
    frame1 = orig.loc[np.around(orig[loc[0]], 5) == np.around(val[0], 5)]
    frame1 = frame1.loc[np.around(frame1[loc[1]], 5) == np.around(val[1], 5)]
    grouped = frame1.groupby(["x", "y", "z"])
    frame1 = grouped.mean().reset_index()
    print(np.shape(frame1)[0])
    frame = frame1
    if mean is not None:
        frame2 = mean.loc[mean[loc[0]] == val[0]]
        frame2 = frame2.loc[np.around(frame2[loc[1]], 5) == val[1]]
        grouped = frame2.groupby(["x", "y", "z"])
        frame2 = grouped.mean().reset_index()
        print(np.shape(frame2)[0])
        if np.shape(frame1)[0] != np.shape(frame2)[0]:
            sys.exit("The size of two datasets do not match!!!")
        # z value in frame1 & frame2 is equal or not ???
        frame[var] = frame1[var] - frame2[var]
    else:
        frame[var] = frame1[var]
    return frame


def max_pert_along_y(orig, var, val, mean=None):
    frame1 = orig.loc[orig["x"] == val[0]]
    frame1 = frame1.loc[np.around(frame1["z"], 5) == val[1]]
    grouped = frame1.groupby(["x", "y", "z"])
    frame1 = grouped.mean().reset_index()
    print(np.shape(frame1)[0])
    if mean is not None:
        frame2 = mean.loc[mean["x"] == val[0]]
        frame2 = frame2.loc[np.around(frame2["z"], 5) == val[1]]
        grouped = frame2.groupby(["x", "y", "z"])
        frame2 = grouped.mean().reset_index()
        print(np.shape(frame2)[0])
        if np.shape(frame1)[0] != np.shape(frame2)[0]:
            sys.exit("The size of two datasets do not match!!!")
        # z value in frame1 & frame2 is equal or not ???
        frame1[var] = frame1[var] - frame2[var]
    frame = frame1.loc[frame1[var].idxmax()]
    return frame


def amplit(orig, xyz, var, mean=None):
    frame1 = orig.loc[np.around(orig["x"], 6) == np.around(xyz[0], 6)]
    # frame2 = frame1.loc[np.around(frame1['y'], 5) == xyz[1]]
    frame2 = frame1.loc[np.around(frame1["y"], 6) == np.around(xyz[1], 6)]
    orig = frame2.loc[frame2["z"] == xyz[2]]
    if mean is None:
        grouped = orig.groupby(["x", "y", "z"])
        mean = grouped.mean().reset_index()
    else:
        frame1 = mean.loc[mean["x"] == xyz[0]]
        frame2 = frame1.loc[frame1["y"] == xyz[1]]
        mean = mean.loc[frame2["z"] == xyz[2], var]
    pert = orig[var].values - mean[var].values
    amplit = np.max(np.abs(pert))
    return amplit


def growth_rate(xarr, var):
    dAmpli = sec_ord_fdd(xarr, var)
    growthrate = dAmpli / var
    return growthrate


#   Obtain finite differential derivatives of a variable (2nd order)
def sec_ord_fdd(xarr, var):
    dvar = np.zeros(np.size(xarr))
    for jj in range(1, np.size(xarr)):
        if jj == 1:
            dvar[jj - 1] = (var[jj] - var[jj - 1]) / (xarr[jj] - xarr[jj - 1])
        elif jj == np.size(xarr):
            dvar[jj - 1] = (var[jj - 1] - var[jj - 2]) / \
                (xarr[jj - 1] - xarr[jj - 2])
        else:
            dy12 = xarr[jj - 1] - xarr[jj - 2]
            dy23 = xarr[jj] - xarr[jj - 1]
            dvar1 = -dy23 / dy12 / (dy23 + dy12) * var[jj - 2]
            dvar2 = (dy23 - dy12) / dy23 / dy12 * var[jj - 1]
            dvar3 = dy12 / dy23 / (dy23 + dy12) * var[jj]
            dvar[jj - 1] = dvar1 + dvar2 + dvar3
    return dvar


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
    if opt == 1:  # use built-in functions

        def func(var1, var2):
            vorticity = griddata((x, y), val, (var1, var2))
            return vorticity

        results = dblquad(func, min1, max1, lambda x: min2, lambda y: max2)
    elif opt == 2:  # integral over x and then over y
        ms1, ms2 = np.meshgrid(range1, range2)
        val_intp = griddata((x, y), val, (ms1, ms2))
        Iy = np.zeros(n2)
        for i in range(n2):
            Iy[i] = np.trapz(val_intp[i, :], range1)
        # print("finish integral over x-axis")
        results = np.trapz(Iy, range2)
    elif opt == 3:  # separate positive and negative values
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
        vorticity = (
            df["vorticity_1"].values ** 2
            + df["vorticity_2"].values ** 2
            + df["vorticity_3"].values ** 2
        )
    elif mode == "x":
        vorticity = df["vorticity_1"].values ** 2
    elif mode == "y":
        vorticity = df["vorticity_2"].values ** 2
    elif mode == "z":
        vorticity = df["vorticity_3"].values ** 2
    rst = vorticity * 0.5
    return rst


def enstrophy(df, type="x", mode=None, rg1=None, rg2=None, opt=2):
    if type == "x":
        x1 = df.y
        x2 = df.z
    if type == "y":
        x1 = df.x
        x2 = df.z
    if type == "z":
        x1 = df.x
        x2 = df.y
    vorticity = vorticity_abs(df, mode=mode)
    ens = integral_db(x1, x2, vorticity, range1=rg1, range2=rg2, opt=opt)
    return ens


def vortex_dyna(df, type="z", opt=1):
    delta_u = df["ux"] + df["vy"] + df["wz"]
    rho_inv = 1 / df["rho"] ** 2
    if type == "z":
        vorticity = df["vorticity_3"]
        tilting1 = df["vorticity_1"] * df["wx"]
        tilting2 = df["vorticity_2"] * df["wy"]
        stretching = vorticity * df["wz"]
        dilatation = -vorticity * delta_u
        torque = rho_inv * (df["rhox"] * df["py"] - df["rhoy"] * df["px"])
    if type == "x":
        vorticity = df["vorticity_1"]
        stretching = vorticity * df["ux"]
        tilting1 = df["vorticity_2"] * df["uy"]
        tilting2 = df["vorticity_3"] * df["uz"]
        dilatation = -df["vorticity_1"] * delta_u
        torque = rho_inv * (df["rhoy"] * df["pz"] - df["rhoz"] * df["py"])
    if type == "y":
        vorticity = df["vorticity_2"]
        tilting1 = df["vorticity_1"] * df["vx"]
        stretching = vorticity * df["vy"]
        tilting2 = df["vorticity_3"] * df["vz"]
        dilatation = -vorticity * delta_u
        torque = rho_inv * (df["rhoz"] * df["px"] - df["rhox"] * df["pz"])
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


# calculate g(r,s) for the mixing layer
def grs(r, s, bs):
    Const = 0.14
    eq1 = (1 - r) * (1 + np.sqrt(s))
    eq2 = 1 + r * np.sqrt(s)
    eq3 = (1 - r) * Const + 0.5 * r
    ans = 0.5 * bs * eq1 * eq3 / eq2
    return ans


# calculate thickness of the mixing layer
def mixing(r, s, Cd, phi, opt=1):
    if opt == 1:
        eq1 = (1 - r) * (1 + np.sqrt(s))
        eq2 = 1 / (1 + r * np.sqrt(s))
    elif opt == 2:
        a1 = (1 - r) / (1 + r * np.sqrt(s))
        b1 = 0.5 * (1 + np.sqrt(s))
        eq1 = a1 * b1
        a2 = (1 - np.sqrt(s)) / (1 + np.sqrt(s))
        b2 = 1 + 1.29 * (1 + r) / (1 - r)
        eq2 = 1 - a2 / b2
    ans = Cd * phi * eq1 * eq2
    return ans


def stat2tot(Ma, Ts, opt, mode="total"):
    gamma = 1.4
    aa = 1 + 0.5 * (gamma - 1) * Ma ** 2
    if opt == "t":
        if mode == "total":
            Tt = Ts * aa
        else:
            Tt = Ts / aa
    elif opt == "p":
        if mode == "total":
            Tt = Ts * np.power(aa, gamma / (gamma - 1))
        else:
            Tt = Ts / np.power(aa, gamma / (gamma - 1))
    elif opt == "rho":
        if mode == "total":
            Tt = Ts * np.power(aa, 1 / (gamma - 1))
        else:
            Tt = Ts / np.power(aa, 1 / (gamma - 1))
    return Tt


def temp_span(infile, xloc, vname):
    df = pd.read_hdf(infile)
    df1 = df.loc[np.round(df["x"], 4) == xloc]
    vname = ["x", "z", "u", "v", "w", "p", "T"]
    df2 = df1[vname]
    return df2


def curve_fit(xval, yval, xsegs, deg=None):
    """fitting a curve smoothly

       Args:
        opt: xval, yval for the curves;
        xsegs: the number of segments to fit the curve
        deg: the degree of splines 

       Return:
        the fitting curves with x and y values
    """
    if np.size(xsegs) < 2:
        sys.exit("the number of segments is too small!")
    for i in range(np.size(xsegs) + 1):
        if i == 0:
            rg = (xval <= xsegs[i])
        elif i == np.size(xsegs):
            rg = (xval >= xsegs[i-1])
        else:
            rg = (xval >= xsegs[i-1]) & (xval <= xsegs[i])
        xrg = xval[rg]
        yrg = yval[rg]
        if deg is None:
            spl = UnivariateSpline(xrg, yrg, k=3, ext='extrapolate')
        else:
            if np.size(xsegs) + 1 != np.size(deg):
                sys.exit("the number of segments doesn't match!")
            spl = UnivariateSpline(xrg, yrg, k=deg[i], ext='extrapolate')
        ytem = spl(xrg)
        
        if i == 0:
            xfit = xrg
            yfit = ytem
        else:
            xfit = np.hstack((xfit, xrg[1:]))
            yfit = np.hstack((yfit, ytem[1:]))
            
    return(xfit, yfit)

# %%
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
