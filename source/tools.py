#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:04:17 2020
    calculate basic aerodynamics parameters
@author: weibo
"""
# %%
import numpy as np
import math

def sutherland(Ts):
    T_ref = 273.15
    miu_ref = 17.16e-6
    S_ref = 110.4
    a1 = (T_ref + S_ref) / (Ts + S_ref)
    b1 = np.power(Ts/T_ref, 1.5)
    miu_s = miu_ref * a1 * b1
    return (miu_s)
    
def total_pressure(ps, Ma, gamma=1.4):
    a1 = 1 + (gamma -1)/2*Ma**2
    b1 = gamma/(gamma-1)
    p_t = ps*np.power(a1, b1)
    return (p_t)


def static_pressure(pt, Ma, gamma=1.4):
    a1 = 1 + (gamma - 1)/2*Ma**2
    b1 = gamma/(gamma-1)
    p_s = pt/np.power(a1, b1)
    return(p_s)


def state_equation(rho, Ts, R_c=287):
    """
    state equation for ideal gas
    p = rho * R_c * Ts
    """
    p_s = rho * R_c * Ts
    return (p_s)


def total_temperature(Ts, Ma, gamma=1.4):
    a1 = 1 + (gamma -1)/2*Ma**2
    # b1 = gamma/(gamma-1)
    T_t = Ts*a1
    return (T_t)

def static_temperature(Tt, Ma, gamma=1.4):
    a1 = 1 + (gamma -1)/2*Ma**2
    # b1 = gamma/(gamma-1)
    T_s = Tt/a1
    return (T_s)

def velocity(Ma, Ts, gamma=1.4, R_c=287):
    u_inf = Ma*np.sqrt(gamma*R_c*Ts)
    return (u_inf)


def rho(Re, Ma, Ts, gamma=1.4, R_c=287):
    miu_s = sutherland(Ts)
    print(miu_s)
    u_inf = Ma*np.sqrt(gamma*R_c*Ts)
    print(u_inf)
    rho = Re*miu_s/u_inf
    return(rho)


def x_inlet(delta, opt='tur'):
    if opt == 'tur':
        a1 = 0.385/np.power(Re_inf, 0.2)
        a2 = delta / a1
        x_in = np.power(a2, 1/0.8)
    if opt == 'lam':
        a1 = 5.0/np.power(Re_inf, 0.5)
        a2 = delta / a1
        x_in = a2
    return(x_in)


def C_f(Re_inf, delta, opt='tur'):
    x_in = x_inlet(delta, opt=opt)
    Rex = Re_inf * x_in
    if opt == 'tur':
        C_f = 0.026 / np.power(Rex, 1/7)
        # C_f = 0.0594 / np.power(Rex, 0.2)
    if opt == 'lam':
        C_f = 0.664 / np.sqrt(Rex)
    return(C_f)


def u_tau(Cf, rho_inf, u_inf, rho_w):
    tau_w = Cf * rho_inf * u_inf**2 / 2
    u_t = np.sqrt(tau_w / rho_w)
    return(u_t)


def Delta_y(Cf, t_inf, rho_inf, u_inf, rho_w, y_plus=1):
    miu_inf = sutherland(t_inf)
    niu_inf = miu_inf / rho_inf
    u_t = u_tau(Cf, rho_inf, u_inf, rho_w)
    dy = y_plus * niu_inf / u_t
    return(dy)


def omega2fre(Ma, Ts, omega, l_ref):
    c = velocity(Ma, Ts, gamma=1.4, R_c=287)
    fre = omega * c / 2 / np.pi / l_ref
    return (fre)


def fre2omega(Ma, Ts, fre, l_ref):
    c = velocity(Ma, Ts, gamma=1.4, R_c=287)
    omega = 2 * np.pi * fre * l_ref / c
    return (omega)


def F2omega(F, Re_inf, x_in_or_d, opt='blasius'):
    """
    convert frequency to omega
    f: frequency in Hz
    Blasius length: l_B=sqrt(x*miu_inf/(rho_inf*u_inf))=sqrt(x/Re_inf)
    ReL=l_B*Re_inf=sqrt(x*Re_inf)
    F=2*pi*f*miu_inf/(rho_inf*u_inf**2)= 2*pi*f/(Re_inf*u_inf)
    omega_blasius = 2*pi*f*l_B/u_inf=F*ReL

    """
    if opt == 'blasius':
        l_b = np.sqrt(x_in_or_d/Re_inf)  # Blasius length
        omega = F * Re_inf * l_b
    elif opt == 'delta':
        omega = F * Re_inf * x_in_or_d
    return (omega)


def freeinter(Cf0, Ma, Fx=6.0, opt=1):
    """
    compute pressure behind reattachment based on free interaction theory
    pressure based on p0 (initial pressure)
    """
    if opt == 1:
        v1 = np.sqrt(2*Cf0/np.sqrt(Ma**2-1))
        v2 = 0.5 * 1.4 * Ma**2 * Fx * v1
        Cpr = v2 + 1
    elif opt == 2:
        Cpr = 1 + 0.5 * Ma
    else:
        v1 = 1.88*Ma - 1
        Cpr = np.power(v1, 0.64)
    return (Cpr)

# %%
if __name__ == '__main__':
    Ma = 6.0
    Cf0 = 5.4e-4
    p0 = 0.019869
    Cp = freeinter(Cf0, Ma, opt=1)
    Re_inf = 7.736*1e6
    Ts = 86.6

    delta = 1.0  # unit: mm
    Tw = 303.1

    R_c = 287
    x_in = x_inlet(delta=delta/1000)  # unit: m
    Re_x = x_in * Re_inf

    omega2fre(6.0, 45, 0.7, 0.001)

    Tt = total_temperature(Ts, Ma, gamma=1.4)

    miu = sutherland(Ts)
    rho_inf = rho(Re_inf, Ma, Ts)
    u_inf = velocity(Ma, Ts)
    ps = static_pressure(rho_inf, Ts)
    pt = total_pressure(ps, Ma)

    rho_w = ps/R_c/Tw
    Cf =  C_f(Re_inf, delta, opt='tur')
    u_t = u_tau(Cf, rho_inf, u_inf, rho_w)
    dy = Delta_y(Cf, Ts, rho_inf, u_inf, rho_w, y_plus=1)
    print('dy=',dy)
    # %%
    import pandas as pd
    import numpy as np
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    path = '/mnt/work/cases/flat_260/'
    data = pd.read_csv(path + 'myfile.txt', sep=' ', skipinitialspace=True)
    omega = np.linspace(0.4, 1.2, 81)
    x1 = np.linspace(10, 360, 701)
    xx, yy = np.meshgrid(x1, omega)
    growth = griddata((data['X'], data['Y']),
                      data['F1V1'], (xx, yy), fill_value=0)
    nval_mat = np.zeros(np.shape(growth))
    nval = np.zeros(np.size(x1))
    nval_max = np.zeros(np.size(omega))

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for i in range(np.size(omega)):
        temp = growth[i,:] * 0.5
        for j in range(np.size(nval)):
            nval[j] = np.sum(temp[:j]) 
        nval_mat[i, :] = nval
        nval_max[i] = nval[-1]
        ax.plot(x1, nval)
    plt.savefig(path+ "Nval.svg", bbox_inches="tight")
    plt.show()

    xf = xx.flatten()
    yf = yy.flatten()
    nf = nval_mat.flatten()
    mat = np.vstack((xf, yf, nf))
    df = pd.DataFrame(data=np.transpose(mat), columns=['x', 'omega', 'nval'])
    df.to_csv(path+'Nval.data', sep=' ', index=False, float_format='%1.8e')

    # %%
    import pandas as pd
    df = pd.read_csv('/home/weibo/code/CFD-Post/database/new/cZPGTBL_M2.00_Retau450.    txt', sep=' ', skipinitialspace=True, skiprows=56)
    df['uu'] = (df['u_rms+'] * 0.04744)**2
    df['vv'] = (df['v_rms+'] * 0.04744)**2
    df['ww'] = (df['w_rms+'] * 0.04744)**2
    df['uv'] = df['uv+'] * 0.04744 * 0.04744
    df['uw'] = 0.0
    df['vw'] = 0.0
    df['w/u_e'] = 0.0
    cols = ['y/delta_99', 'u+', 'u/u_e', 'v/u_e', 'w/u_e', 'r/r_e', 'T/T_e', 'uu',  'vv', 'ww', 'uv', 'uw', 'vw']
    df.to_csv('/home/weibo/code/CFD-Post/database/new/M_2.0_REd_23917.dat',     sep='\t', index=False, float_format='%1.8e', columns=cols)