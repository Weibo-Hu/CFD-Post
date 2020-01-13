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
import warnings
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy.optimize
from numpy import NaN, Inf, arange, isscalar, asarray, array
from numpy.core.umath_tests import inner1d
import sys, os
from timer import timer


# Proper Orthogonal Decomposition (snapshots method), equal time space
# Input: the variable of POD (fluctuations)
# phi-each column is a mode structure
# eigval-the amount of enery in each mode
# alpha -time amplitude, each mode varies in time
def pod(var, fluc=False, method=None):
    m, n = np.shape(var) # n: the number of snapshots, m: dimensions
    if(n > m):
        sys.exit(
            "NO of snapshots had better be smaller than NO of grid points!!!")
    if fluc == True:
        var = var - np.transpose(np.tile(np.mean(var, axis=1), (n, 1)))
    if method == 'svd':  # svd method
        phi, sigma, VH = sp.linalg.svd(var, full_matrices=False)
        eigval = sigma*sigma
        alpha  = np.diag(sigma)@VH  # row: mode; every column:coordinates
        # alpha  = phi.T.conj()@var
        eigvec = VH.T.conj()
    else:  # eigvalue problem method
        CorrMat = np.matmul(np.transpose(var), var)  # correlation matrix,
        eigval, eigvec = np.linalg.eig(CorrMat)  # right eigvalues
        eigval = np.absolute(eigval)
        eigvec = np.absolute(eigvec)
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]  # in descending order if necessary
        phi = np.matmul(var, eigvec)  # POD basis function, only real part(m*n)
        norm2 = np.sum(phi*phi, axis=0)  # normlized by norm2
        phi = phi/np.sqrt(norm2)  # nomalized POD modes, not orthoganal!!!
        # phi = var@eigvec@np.sqrt(np.diag(np.reciprocal(eigval)))
        alpha = np.diag(np.sqrt(eigval))@eigvec.T  # alpha = phi.T@var

        # np.savetxt(outfile+'EIGVAL.dat', eigval, fmt='%1.8e', \
        #            delimiter='\t', header = 'Eigvalue')
        # np.savetxt(outfile + 'COEFF.dat', alpha, fmt='%1.8e', \
        #            delimiter = "\t", header = 'Coeff')
        # np.savetxt(outfile + 'MODE.dat', phi, fmt='%1.8e', \
        #            delimiter = "\t", header = 'mode')
    return (eigval, eigvec, phi, alpha)

def pod_eigspectrum(Percent, eigval):
    EnergyFrac       = eigval/np.sum(eigval)*100
    EnergyCumulation = np.cumsum(EnergyFrac)
    index = np.where(EnergyCumulation >= Percent)
    num = np.size(eigval) - np.size(index[0]) + 1
    #NewFlow = inner1d(coeff[:,:num], phi[:,:num])
    return EnergyFrac, EnergyCumulation, num#NewFlow

def pod_reconstruct(num, Snapshots, eigval, phi, alpha):
    # SVD
    NewFlow = phi[:,:num]*alpha[:num,:]
    # eigval problem
    phi = Snapshots@eigvec
    phi = phi[:,:num]@np.sqrt(np.diag(np.reciprocal(eigval[:num])))
    alpha = phi.T@Snapshots
    NewFlow[:,0] = alpha[:,0]*phi[:,num]
    NewFlow[:,1] = alpha[:,1]*phi[:,num]

# Standard Dynamic Mode Decompostion, equal time space
# Ref: Jonathan H. T., et. al.-On dynamic mode decomposition: theory and application
# x/y, phi: construct flow field for each mode (phi:row-coordinates, column-time)
def DMD_Standard(var, outfolder, fluc = None): # scaled method
    n = np.shape(var)[1]  # n: the number of snapshots, m: dimensions
    if fluc == 'True':
        var  = var - np.transpose(np.tile(np.mean(var, axis=1), (n, 1)))
    V1 = var[:, :-1]
    V2 = var[:, 1:]
    U, D, VH = sp.linalg.svd(V1, full_matrices=False)  # do not perform tlsq
    V = VH.conj().T
    ind = np.where(np.round(D, 1) == 0.0)
    if np.size(ind) != 0:
        print("There are "+str(np.size(ind))+" zero-value singular value!!!")
        nonzero = n-1-np.size(ind)
        V = V[:, :nonzero]
        D = D[:nonzero]
        U = U[:, :nonzero]
    S = U.T.conj()@V2@V*np.reciprocal(D)
    eigval, eigvec = sp.linalg.eig(S)
    phi    = U@eigvec  # projected dynamic modes
    if np.size(ind) != 0:
        residual = np.linalg.norm(V2[:, :nonzero]-V1[:, :nonzero]@S)/n
    else:
        residual = np.linalg.norm(V2-V1@S)/n
    return (eigval, phi, U, eigvec, residual)


# convex optimization problem
def OldDMD_Amplitude(var, phi, eigval, lstsq=None):
    n = np.shape(var)[1]
    if lstsq == 'False':
        coeff  = np.linalg.lstsq(phi, var.T[0], rcond=-1)[0]  # modes coeff
    else:  # min(var-phi@coeff@Vand)-->min(UT@V1-UT@U@eigvec@coeff@Vand)
        A = phi@np.diag(eigval**0)
        for i in range(n-1):
            a2 = phi@np.diag(eigval**(i+1))
            A = np.vstack([A, a2])
        b = np.reshape(var, (-1, ), order='F')
        coeff = np.linalg.lstsq(A, b, rcond=-1)[0]
    return coeff

def DMD_Amplitude(var, U, eigvec, phi, eigval, lstsq=None):
    m, n = np.shape(var)
    # min(var-phi@coeff@Vand)-->min(UT@V1-UT@U@eigvec@coeff@Vand)
    if lstsq == 'False':
        coeff  = np.linalg.lstsq(phi, var.T[0], rcond=-1)[0]
    else:  # min(var-phi@coeff@Vand)-->min(UT@V1-UT@U@eigvec@coeff@Vand)
        A = eigvec@np.diag(eigval**0)
        for i in range(n-2):
            a2 = eigvec@np.diag(eigval**(i+1))
            A = np.vstack([A, a2])
        b = np.reshape(U.T.conj()@var[:, :-1], (-1, ), order='F')
        coeff = np.linalg.lstsq(A, b, rcond=-1)[0]
    return coeff

def SPDMD_J(var, eigval, eigvec, D, V, coeff):
    n = np.shape(var)[1]
    Vand = np.hstack([eigval.reshape(-1, 1)**i for i in range(n-1)])
    p1 = (eigvec.T.conj() @ eigvec)
    p2 = (Vand.T.conj() @ Vand).conj()
    P = p1 * p2
    q1 = Vand @ V @ np.diag(D).T.conj() @ eigvec
    q = np.diag(q1).conj()
    ss = (np.linalg.norm(D))**2
    Ja = coeff.T.conj@P@coeff-q.T.conj()@coeff-coeff.T.conj@q+ss
    return Ja

def DMD_Dynamics(eigval, coeff, timepoints):
    period = np.round(np.diff(timepoints), 6)
    if(np.size(np.unique(period)) != 1 or \
        np.size(timepoints) != np.size(coeff) + 1):
        sys.exit("Time period is not equal!!!")
    t_samp = timepoints[1]-timepoints[0]
    lamb = np.log(eigval)/t_samp  # growth rate(real part),frequency(imaginary)
    m = np.size(lamb)
    n = np.size(timepoints)
    TimeGrid = np.transpose(np.tile(timepoints, (m, 1)))
    LambGrid = np.tile(lamb, (n,1))
    OmegaT = LambGrid*TimeGrid # element wise multiply
    dynamics = np.transpose(np.exp(OmegaT)*coeff)
    return (dynamics)

def DMD_Reconstruct(phi, dynamics):
    reconstruct = phi@dynamics
    return(reconstruct)

# Exact Dynamic Mode Decompostion
# Ref: Jonathan H. T., et. al.-On dynamic mode decomposition: theory and application
def DMD_Exact(var): # scaled method
    n = np.shape(var)[1]  # n: the number of snapshots, m: dimensions
    if fluc is not None:
        var  = var - np.transpose(np.tile(np.mean(var, axis=1), (n, 1)))
    V1 = var[:, :-1]
    V2 = var[:, 1:]

    U, D, VH = np.linalg.svd(V1, full_matrices=False)  # do not perform tlsq
    # V = VH.conj().T = VH.H
    S = U.conj().T@V2@VH.conj().T*np.reciprocal(D)  # or ##.dot(np.diag(D))
    eigval, eigvec = np.linalg.eig(S)
    eigvec = U.dot(eigvec) # dynamic modes
    coeff = np.linalg.lstsq(eigvec, var.T[0])[0]
    return (coeff, eigval, eigvec)


