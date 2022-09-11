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
    return(miu_s)
    
def total_pressure(ps, Ma, gamma=1.4):
    a1 = 1 + (gamma -1)/2*Ma**2
    b1 = gamma/(gamma-1)
    p_t = ps*np.power(a1, b1)
    return(p_t)

def total_temperature(Ts, Ma, gamma=1.4):
    a1 = 1 + (gamma -1)/2*Ma**2
    # b1 = gamma/(gamma-1)
    T_t = Ts*a1
    return(T_t)

def velocity(Ma, Ts, gamma=1.4, R_c=287):
    u_inf = Ma*np.sqrt(gamma*R_c*Ts)
    return(u_inf)


def rho(Re, Ma, Ts, gamma=1.4, R_c=287):
    miu_s = sutherland(Ts)
    print(miu_s)
    u_inf = Ma*np.sqrt(gamma*R_c*Ts)
    print(u_inf)
    rho = Re*miu_s/u_inf
    return(rho)


def static_pressure(rho, Ts, R_c=287):
    p_s = rho*R_c*Ts
    return(p_s)


def x_inlet(delta, opt='tur'):
    if opt == 'tur':
        a1 = 0.385/np.power(Re_inf, 0.2)
        a2 = delta / a1
        x_in = np.power(a2, 1/0.8)
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



Ma = 6.0
Re_inf = 1*1e7
delta = 2  # unit: mm
Tw = 292.15
Ts = 45
R_c = 287
x_in = x_inlet(delta=2/1000)  # unit: m
Re_x = x_in * Re_inf

miu = sutherland(Ts)
rho_inf = rho(Re_inf, Ma, Ts)
u_inf = velocity(Ma, Ts)
ps = static_pressure(rho_inf, Ts)
pt = total_pressure(ps, Ma)

rho_w = ps/R_c/Tw
Cf =  C_f(Re_inf, delta, opt='tur')
u_t = u_tau(Cf, rho_inf, u_inf, rho_w)
dy = Delta_y(Cf, Ts, rho_inf, u_inf, rho_w, y_plus=1)
