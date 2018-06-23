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
from numpy.core.umath_tests import inner1d
import sys, os
from timer import timer

plt.close("all")
plt.rc('text', usetex=True)
font0 = {
    'family': 'Times New Roman',
    'color': 'k',
    'weight': 'normal',
    'size': 10,
}
font1 = {
    'family': 'Times New Roman',
    'color': 'k',
    'weight': 'normal',
    'size': 12,
}

font2 = {
    'family': 'Times New Roman',
    'color': 'k',
    'weight': 'normal',
    'size': 14,
}

font3 = {
    'family': 'Times New Roman',
    'color': 'k',
    'weight': 'normal',
    'size': 16,
}

# Proper Orthogonal Decomposition, equal time space
# Input: the variable of POD (fluctuations)
# phi-each column is a mode structure
# eigval-the amount of enery in each mode
# alpha -time amplitude, each mode varies in time   
def POD(var, outfile, fluc = None, method = None):
    m, n = np.shape(var) # n: the number of snapshots, m: dimensions
    if(n > m):
        sys.exit("NO of snapshots had better be smaller than NO of grid points!!!")
    if fluc is not None:
        var  = var - np.transpose(np.tile(np.mean(var, axis=1), (n, 1)))
    if method is not None: # eigvalue problem method
        CorrMat = np.matmul(np.transpose(var), var) # correlation matrix,
        eigval, eigvec = np.linalg.eig(CorrMat)  # right eigvalues
        eigval = np.absolute(eigval)
        eigvec = np.absolute(eigvec)
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:,idx] # in descending order if necessary
        phi = np.matmul(var, eigvec) # POD basis function, only real part(m*n)
        norm2 = np.sqrt(np.sum(phi*phi, axis=0)) # normlized by norm2
        phi   = phi/norm2 # nomalized POD modes
        #phi = var@eigvec*np.sqrt(np.reciprocal(eigval))
        alpha = np.diag(np.sqrt(eigval))@eigvec.T
    else: # svd method
        phi, sigma, VH = sp.linalg.svd(var, full_matrices=False)
        eigval = sigma*sigma
        alpha  = np.diag(sigma)@VH
#    np.savetxt(outfile+'EIGVAL.dat', eigval, fmt='%1.8e', \
#               delimiter='\t', header = 'Eigvalue')
#    np.savetxt(outfile + 'COEFF.dat', alpha, fmt='%1.8e', \
#               delimiter = "\t", header = 'Coeff')
#    np.savetxt(outfile + 'MODE.dat', phi, fmt='%1.8e', \
#               delimiter = "\t", header = 'mode')
    #coeff: (row: different times, column: different modes)
    return (eigval, phi, alpha)

def POD_Reconstruct(Percent, eigval, phi, coeff):
    EnergyFrac       = eigval/np.sum(eigval)*100
    EnergyCumulation = np.cumsum(EnergyFrac)
    index = np.where(EnergyCumulation >= Percent)
    num = index[0][0]
    #NewFlow = inner1d(coeff[:,:num], phi[:,:num])
    return EnergyFrac, EnergyCumulation, num#NewFlow

# Standard Dynamic Mode Decompostion, equal time space
# Ref: Jonathan H. T., et. al.-On dynamic mode decomposition: theory and application
# x/y, phi: construct flow field for each mode (phi:row-coordinates, column-time)
def DMD_Standard(var, timepoints, outfolder, fluc = None): # scaled method
    period = np.round(np.diff(timepoints), 6)
    if(np.size(np.unique(period)) != 1):
        sys.exit("Time period is not equal!!!")
    m, n = np.shape(var) # n: the number of snapshots, m: dimensions
    if fluc is not None:
        var  = var - np.transpose(np.tile(np.mean(var, axis=1), (n, 1)))
    V1   = var[:, :-1]
    V2   = var[:, 1:]
    U, D, VH = sp.linalg.svd(V1, full_matrices=False) # do not perform tlsq
    V = VH.conj().T
    S = U.T.conj()@V2@V*np.reciprocal(D)
    eigval, eigvec = sp.linalg.eig(S)
    phi    = U@eigvec # projected dynamic modes
    coeff  = np.linalg.lstsq(phi, var.T[0], rcond=-1)[0] # least-square???
    #resid  = np.linalg.norm(V2-np.matmul(S, V1))/n
    #resid  = np.linalg.norm(V2-np.matmul(V1, S))/n
    return (eigval, phi, coeff)

def DMD_Dynamics(eigval, coeff, timepoints):
    t_samp = timepoints[1]-timepoints[0]
    lamb = np.log(eigval)/t_samp # growth rate(real part), frequency(imaginary part)
    m = np.size(lamb)
    n = np.size(timepoints)
    TimeGrid = np.transpose(np.tile(timepoints, (m, 1)))
    LambGrid = np.tile(lamb, (n,1))
    OmegaT   = LambGrid*TimeGrid # element wise multiply
    dynamics = np.transpose(np.exp(OmegaT)*coeff)
    return (dynamics)

def DMD_Reconstruct(phi, dynamics):
    reconstruct = phi@dynamics
    return(reconstruct)    
    
# Exact Dynamic Mode Decompostion
# Ref: Jonathan H. T., et. al.-On dynamic mode decomposition: theory and application
def DMD_Exact(): # scaled method
    m, n = np.shape(var) # n: the number of snapshots, m: dimensions
    if fluc is not None:
        var  = var - np.transpose(np.tile(np.mean(var, axis=1), (n, 1)))
    V1   = var[:, :-1]
    V2   = var[:, 1:]

    U, D, VH = np.linalg.svd(V1, full_matrices=False) # do not perform tlsq
    # V = VH.conj().T = VH.H
    S = U.conj().T@V2@VH.conj().T*np.reciprocal(D) # or ##.dot(np.diag(D)), @=np.matmul=np.dot
    eigval, eigvec = np.linalg.eig(S)
    eigvec = U.dot(eigvec) # dynamic modes
    coeff  = np.linalg.lstsq(eigvec, var.T[0])[0]
    return (coeff, eigval, eigvec)


#%% load data
InFolder  = "/media/weibo/Data1/BFS_M1.7L_0419/SpanAve/1/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0419/SpanAve/Test"
path  = "/media/weibo/Data1/BFS_M1.7L_0419/DataPost/"
dirs = os.listdir(InFolder)
DataFrame = pd.read_hdf(InFolder+dirs[0])
Snapshots = DataFrame['u']
xval = DataFrame['x']
yval = DataFrame['y']
del DataFrame
for jj in range(np.size(dirs)-1):
    DataFrame = pd.read_hdf(InFolder+dirs[jj+1])
    VarVal    = DataFrame['u']
    Snapshots = np.column_stack((Snapshots,VarVal))
    del DataFrame
#Snapshots = Snapshots.astype(complex)
#%% POD
with timer("POD computing"):
    eigval, phi, coeff = POD(Snapshots, OutFolder, fluc = 'Yes')
with timer("POD computing"):
    eigval1, phi1, coeff1 = POD(Snapshots, OutFolder, fluc = 'Yes', method = 1)
#Frac, Cumulation, NewFlow = FlowReproduce(99.99, eigval, phi, coeff)

#%% DMD
#timepoints = np.linspace(200, 439, 240)
timepoints = np.linspace(200, 399, 200)
#with timer("DMD computing"):
#    eigval, phi, coeff = \
#    DMD_Standard(Snapshots, timepoints, OutFolder, fluc = 'Yes')

#%%    
plt.figure(figsize=(10, 10))
plt.gcf()
ax = plt.gca()
points, = ax.plot(eigval.real, eigval.imag, 'bo', label='eigvalues')
limit = np.max(np.ceil(np.absolute(eigval)))
ax.set_xlim((-limit, limit))
ax.set_ylim((-limit, limit))
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
unit_circle = plt.Circle((0., 0.), 1., color='black', fill=False, \
                         label='unit circle', linestyle='--')
ax.add_artist(unit_circle)
ax.grid(b=True, which = 'both', linestyle = ':')
plt.show()

#%% Modes in space
uval = phi[:,0].real
umax = np.max(uval)
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
u    = griddata((xval, yval), uval, (x, y))
corner = (x<0.0) & (y<0.0)
u[corner] = np.nan
fig, ax = plt.subplots()
#rg1 = np.linspace(-0.025, 0.025, 11)
cbar = ax.contourf(x, y, u, cmap = 'rainbow') #, levels = rg1)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_xlabel(r'$x/\delta_0$', fontdict = font3)
ax.set_ylabel(r'$y/\delta_0$', fontdict = font3)
#ax.grid (b=True, which = 'both', linestyle = ':')
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
#rg2 = np.linspace(0.000001, 0.001, 20)
cbaxes = fig.add_axes([0.5, 0.75, 0.35, 0.07]) # x, y, width, height
#cbar = plt.colorbar(cbar, cax = cbaxes, orientation="horizontal", ticks=rg2)
#cbaxes.set_ylabel(r'$\rho/\rho_{\infty}$', \
#                  rotation = 0, labelpad = 20)
plt.colorbar(cbar, cax = cbaxes, orientation='horizontal')
fig.set_size_inches(12, 4, forward=True)
#plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(path+'DMDMode1.svg', dpi=300)
plt.show()
    
#%%
    
"""
#%% POD Energy Spectrum
N_modes = 50
xaxis = np.arange(1, N_modes+1)
fig1, ax1 = plt.subplots()
ax1.plot(xaxis, Frac[:N_modes], color='black', marker='o', markersize=4,)
ax1.set_ylim(bottom=-5)
ax1.set_xlabel(r'$i$', fontdict=font3)
ax1.set_ylabel(r'$E_i$', fontdict=font3)
ax1.grid(b=True, which = 'both', linestyle = ':')

ax2 = ax1.twinx()
#ax2.plot(xaxis, Cumulation, color='grey', marker='o',  markersize=4, alpha=0.5)
ax2.fill_between(xaxis, Cumulation[:N_modes], color='grey', alpha=0.5)
ax2.set_ylim([-5, 100])
ax2.set_ylabel(r'$ES_i$', fontdict=font3)
fig1.set_size_inches(5, 4, forward=True)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(path+'POD-EnergySpectrumMeanflow.svg', dpi=300)
plt.show()

#%% POD Eigenfunction of a specific mode
uval = phi[:,0]
umax = np.max(uval)
x, y = np.meshgrid(np.unique(xval), np.unique(yval))
u    = griddata((xval, yval), uval, (x, y))
corner = (x<0.0) & (y<0.0)
u[corner] = np.nan
fig, ax = plt.subplots()
#rg1 = np.linspace(0.25, 1.06, 13)
cbar = ax.contourf(x, y, u, cmap = 'rainbow')
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_xlabel(r'$x/\delta_0$', fontdict = font3)
ax.set_ylabel(r'$y/\delta_0$', fontdict = font3)
#ax.grid (b=True, which = 'both', linestyle = ':')
plt.gca().set_aspect('equal', adjustable='box')
# Add colorbar
#rg2 = np.linspace(0.000001, 0.001, 20)
cbaxes = fig.add_axes([0.68, 0.7, 0.2, 0.07]) # x, y, width, height
#cbar = plt.colorbar(cbar, cax = cbaxes, orientation="horizontal", ticks=rg2)
#cbaxes.set_ylabel(r'$\rho/\rho_{\infty}$', \
#                  rotation = 0, labelpad = 20)
plt.colorbar(cbar, cax = cbaxes, orientation='horizontal')
fig.set_size_inches(12, 4, forward=True)
#plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(path+'PODMeanflow.svg', dpi=300)
plt.show()

#%%
fig, ax = plt.subplots()
levels = [-0.002, 0.002]
ax.contourf(x, y, u, levels, colors=('#66ccff', '#e6e6e6', '#ff4d4d'), \
            extend='both') #blue, grey, red
#ax.contour(x, y, u, levels = -0.003, linewidths = 1.2, colors='r')
ax.plot([-40.0, 0.0], [0.0, 0.0], 'k-')
ax.plot([0.0, 0.0], [-3.0, 0.0], 'k-')
ax.grid(b=True, which = 'both', linestyle = ':')
ax.set_xlabel(r'$x/\delta_0$', fontdict = font3)
ax.set_ylabel(r'$y/\delta_0$', fontdict = font3)
fig.set_size_inches(12, 4, forward=True)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(path+'PODIsosurfaceMeanflow.svg', dpi=300)
plt.show()
"""
"""
path = "/media/weibo/Data1/BFS_M1.7L_0419/DataPost/"
MeanFlow = DataPost()
MeanFlow.LoadData(path+'MeanSlice141.dat', Sep = '\t')
print(np.size(MeanFlow.p))
x = MeanFlow.x
y = MeanFlow.y
p = MeanFlow.p
MeanFlow.LoadData(path+'MeanSlice163.dat', Sep = '\t')
p = np.column_stack((p, MeanFlow.p))
MeanFlow.LoadData(path+'MeanSlice199.dat', Sep = '\t')
p = np.column_stack((p, MeanFlow.p))
MeanFlow.LoadData(path+'MeanSlice236.dat', Sep = '\t')
p = np.column_stack((p, MeanFlow.p))
MeanFlow.LoadData(path+'MeanSlice260.dat', Sep = '\t')
p = np.column_stack((p, MeanFlow.p))
coeff, phi, eigval, eigvec = POD(p, path+'test0528', 1)
"""
