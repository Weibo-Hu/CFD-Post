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
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from DataPost import DataPost
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import griddata
from scipy.interpolate import spline
import scipy.optimize
from numpy import NaN, Inf, arange, isscalar, asarray, array
import time
import sys

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

# Obtain Frequency-Weighted Power Spectral Density
def FW_PSD (VarZone, TimeZone):
    Ave = np.mean (VarZone)
    Var_fluc = VarZone-Ave
    # time space must be equal
    # fast fourier transform and remove the half
    Var_fft = np.fft.rfft (Var_fluc)
    Var_psd = abs(Var_fft)**2
    num = np.size (Var_fft)
    # sample frequency
    Freq_samp = num/(TimeZone[-1]-TimeZone[0])
    # f_var = np.linspace (0, f_samp/2, num)
    Freq = np.linspace (Freq_samp/2/num, Freq_samp/2, num)
    Freq_weighted = Var_psd*Freq
    return (Freq, Freq_weighted)

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

# Proper Orthogonal Decomposition, equal time space
# Input: the variable of POD (fluctuations)
def POD(var, outfile, fluc = None):
    start_time = time.clock()
    m, n = np.shape(var) # n: the number of snapshots, m: dimensions
    if fluc is not None:
        var  = var - np.transpose(np.tile(np.mean(var, axis=1), (n, 1))) # for fluctuations
    # less time by this way in that m<n
    CorrMat = np.matmul(np.transpose(var), var)/n # correlation matrix,
    #CorMat = np.matmul(var, np.transpose(var))/n # correlation matrix
    eigval, eigvec = np.linalg.eig(CorrMat)  # original eigval(n), eigvec(n*n)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx] # in descending order if necessary
    # eigvec, eigval, coeff = LA.svd(CorrMat)
    phi = np.matmul(var, eigvec.real) # POD basis function, only real part(m*n)
    norm2 = np.sqrt(np.sum(phi*phi, axis=0)) # normlized by norm2
    phi   = phi/norm2 # nomalized POD modes
    coeff = np.matmul(np.transpose(var), phi) # coefficiency of the corresponding POD modes
    # save data as text, be sure the file is closed after writting.
#    np.savetxt(outfile+'EIG.dat', np.column_stack((eigval, eigvec)), \
#            fmt='%1.9e', delimiter = "\t", header = 'eigval eigvec')
    with open(outfile+'EIG.dat', 'wb') as f:
        np.savetxt(f, np.column_stack((eigval, eigvec)), \
            fmt='%1.9e', delimiter = "\t", header = 'eigval eigvec')
    with open(outfile + 'COEFF.dat', 'wb') as f:
        np.savetxt(f, coeff, \
            fmt='%1.9e', delimiter = "\t", header = 'coeff')
    with open(outfile + 'MODE.dat', 'wb') as f:
        np.savetxt(f, phi, \
            fmt='%1.9e', delimiter = "\t", header = 'mode')
    print("The computational time is ", time.clock()-start_time)
    return (coeff, phi, eigval, eigvec)

# Standard Dynamic Mode Decompostion, equal time space
# Ref: Jonathan H. T., et. al.-On dynamic mode decomposition: theory and application
def DMD_Standard(var, t_samp, outfile, fluc = None): # scaled method
    start_time = time.clock()
    m, n = np.shape(var) # n: the number of snapshots, m: dimensions
    if fluc is not None:
        var  = var - np.tile(np.mean(var, axis=1), (m, 1)).T # for fluctuations
    V1   = var[:, :-1]
    V2   = var[:, 1:]

    U, D, VH = np.linalg.svd(V1, full_matrices=False) # do not perform tlsq
    # V = VH.conj().T = VH.H
    S = U.conj().T@V2@VH.conj().T*np.reciprocal(D) # or ##.dot(np.diag(D)), @=np.matmul=np.dot 
    eigval, eigvec = np.linalg.eig(S)
    eigvec = U.dot(eigvec) # dynamic modes
    lamb   = np.log(eigval)/t_samp
    coeff  = np.linalg.lstsq(eigvec, var.T[0])[0] # least-square?
    print("The computational tiem is ", time.clock()-start_time)
    return (coeff, eigval, eigvec, lamb)

# Exact Dynamic Mode Decompostion
# Ref: Jonathan H. T., et. al.-On dynamic mode decomposition: theory and application
def DMD_Exact(): # scaled method
    start_time = time.clock()
    m, n = np.shape(var) # n: the number of snapshots, m: dimensions
    if fluc is not None:
        var  = var - np.transpose(np.tile(np.mean(var, axis=1), (m, 1))) # for fluctuations
    V1   = var[:, :-1]
    V2   = var[:, 1:]

    U, D, VH = np.linalg.svd(V1, full_matrices=False) # do not perform tlsq
    # V = VH.conj().T = VH.H
    S = U.conj().T@V2@VH.conj().T*np.reciprocal(D) # or ##.dot(np.diag(D)), @=np.matmul=np.dot 
    eigval, eigvec = np.linalg.eig(S)
    eigvec = U.dot(eigvec) # dynamic modes
    coeff  = np.linalg.lstsq(eigvec, var.T[0])[0]
    print("The computational tiem is ", time.clock()-start_time)
    return (coeff, eigval, eigvec)
#uu = np.linspace(1, 50, 50)
#uu = np.reshape(uu, [5,10])
#uu = np.transpose(uu)
#coeff, phi, eigval, eigvec = POD(uu, '/media/weibo/Data1/'+'0527', 1)
#
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