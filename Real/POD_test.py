#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:48:31 2018
    This code for validating the POD.
@author: weibo
"""
import matplotlib.pyplot as plt
import numpy as np
import ReducedModel as rm
from timer import timer

#%% define the problem for DMD
def f1(x,t):
    return 1./np.cosh(x+3)*np.exp(2.3j*t)

def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)


x = np.linspace(-5, 5, 256)
t = np.linspace(0, 4*np.pi, 128)
tgrid, xgrid = np.meshgrid(t, x)

x1 = f1(xgrid, tgrid)
x2 = f2(xgrid, tgrid)
XX = x1+x2

titles = ['$f_1(x,t)$', '$f_2(x,t)$', '$f$']
data   = [x1, x2, XX]
fig = plt.figure(figsize=(17,6))
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title)
plt.colorbar()
plt.show()

#%% DMD
with timer("POD test case computing"):
    eigval, phi, coeff= \
        rm.POD(XX, './', fluc = None)
    
#%% Eigvalue Spectrum
N_modes = 50
xaxis = np.arange(1, N_modes+1)
fig1, ax1 = plt.subplots()
ax1.plot(xaxis, eigval[:N_modes], color='black', marker='o', markersize=4,)
ax1.set_ylim(bottom=-5)
ax1.set_xlabel(r'$i$')
ax1.set_ylabel(r'$E_i$')
ax1.grid(b=True, which = 'both', linestyle = ':')

Cumulation = np.cumsum(eigval/np.sum(eigval)*100)
ax2 = ax1.twinx()
ax2.fill_between(xaxis, Cumulation[:N_modes], color='grey', alpha=0.5)
ax2.set_ylim([-5, 100])
ax2.set_ylabel(r'$ES_i$')
fig1.set_size_inches(5, 4, forward=True)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.show()

#%% Modes in space
plt.figure(figsize=(10, 5))
i = 0
modes = phi.T
for i in range(2):
    plt.plot(x, phi[:,i].real)
    plt.xlabel('x')
    plt.ylabel('f')
    plt.title('Modes')
plt.plot(x, f1(x, 0.0)/5.0, ':') # exact results
plt.plot(x, f2(x, 0.0)/-6.0, '--')
plt.show()

#%% Time evolution of each mode
plt.figure(figsize=(10, 5))
i = 0
for i in range(2):
    plt.plot(t, coeff[i,:].real)
    plt.title('Dynamics')
    plt.xlabel('t')
    plt.ylabel('f')
plt.plot(t, f1(0.1, t), ':') # exact results
plt.plot(t, f2(0.1, t), '--')
plt.show()    

#%% reconstruct flow field
reconstruct = rm.DMD_Reconstruct(phi, dynamics)
titles = [r'$f_1^{\prime}(x,t)$', r'$f_2^{\prime}(x,t)$']
fig = plt.figure(figsize=(17,6))
for i in range(2):
    n = '13'+str(i+1)
    plt.subplot(int(n))
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(titles[i])
    phi1 = phi[:,i].reshape(np.size(x), 1)
    dyn1 = dynamics[i,:].reshape(1, np.size(t))
    plt.pcolor(xgrid, tgrid, (phi1@dyn1).real)
    
plt.subplot(133)
plt.pcolor(xgrid, tgrid, reconstruct.real)
plt.title(r'$f^{\prime}$')
plt.xlabel('x')
plt.ylabel('t')
plt.show()
#%% abosulte error between exact and reconstruct resutls
err = XX-reconstruct
print("Errors of DMD: ", np.linalg.norm(err))
fig = plt.figure()
plt.pcolor(xgrid, tgrid, (err).real)
fig = plt.colorbar()
plt.show()
