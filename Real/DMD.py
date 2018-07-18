# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:24:50 2018
    This code for Dynamic Mode decomposition
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


class DMD(object):
    def __init__(self):
        pass

    @property
    def eigval(self):
        return self._eigval

    @property
    def eigvec(self):
        return self._eigvec

    @property
    def modes(self):
        return self._modes

    @property
    def snapshots(self):
        return self._snapshots

    @property
    def shape(self):
        return self._snapshots.shape()

    @staticmethod
    def dmd_standard(fluc=None):
        m, n = np.shape(snapshots)
        if fluc == 'True':
            snapshots = snapshots - np.transpose(
                np.mean(snapshots, axis=1), (n, 1))
        V1 = snapshots[:, :-1]
        V2 = snapshots[:, 1:]
        U, D, VH = sp.linalg.svd(V1, full_matrices=False) # use min(m, n)
        V = VH.conj().T
        ind = np.where(np.round(D, 1) == 0.0)
        if np.size(ind) != 0:
            print("There are " + str(np.size(ind)) +
                  " zero-value singular value!!!")
            nonzero = n - 1 - np.size(ind)
            V = V[:, :nonzero]
            D = D[:nonzero]
            U = U[:, :nonzero]
        S = U.T.conj() @ V2 @ V * np.reciprocal(D)
        eigval, eigvec = sp.linalg.eig(S)
        modes = U @ eigvec  # projected dynamic modes
        if np.size(ind) != 0:
            residual = np.linalg.norm(V2[:, :nonzero] - V1[:, :nonzero] @ S) / n
        else:
            residual = np.linalg.norm(V2 - V1 @ S) / n
        return (eigval, modes, U, eigvec, residual)

    # convex optimization problem
    @staticmethod
    def dmd_amplitude(U, eigvec, modes, eigval, lstsq=None):
        m, r = np.shape(modes)  # n: the number of snapshots, m: dimensions
        m, n = np.shape(snapshots)
        if lstsq == 'False':  # min(var-phi@coeff@Vand)-->min(UT@V1-UT@U@eigvec@coeff@Vand)
            coeff = np.linalg.lstsq(modes, snapshots.T[0], rcond=-1)[0]
        else:  # min(var-phi@coeff@Vand)-->min(UT@V1-UT@U@eigvec@coeff@Vand)
            A = eigvec @ np.diag(eigval**0)
            for i in range(n - 2):
                a2 = eigvec @ np.diag(eigval**(i + 1))
                A = np.vstack([A, a2])
            b = np.reshape(U.T.conj() @ snapshots[:, :-1], (-1, ), order='F')
            coeff = np.linalg.lstsq(A, b, rcond=-1)[0]
        return coeff

    @staticmethod
    def dmd_dynamics(eigval, coeff, timepoints):
        period = np.round(np.diff(timepoints), 6)
        if (np.size(np.unique(period)) != 1):
            sys.exit("Time period is not equal!!!")
        t_samp = timepoints[1] - timepoints[0]
        lamb = np.log(
            eigval) / t_samp  # growth rate(real part), frequency(imaginary part)
        m = np.size(lamb)
        n = np.size(timepoints)
        TimeGrid = np.transpose(np.tile(timepoints, (m, 1)))
        LambGrid = np.tile(lamb, (n, 1))
        OmegaT = LambGrid * TimeGrid  # element wise multiply
        dynamics = np.transpose(np.exp(OmegaT) * coeff)
        return (dynamics)

    @staticmethod
    def DMD_Reconstruct(modes, dynamics):
        reconstruct = modes @ dynamics
        return (reconstruct)

    # Exact Dynamic Mode Decompostion
    # Ref: Jonathan H. T., et. al.-On dynamic mode decomposition: theory and application
    @staticmethod
    def DMD_Exact():  # scaled method
        m, n = np.shape(snapshots)  # n: the number of snapshots, m: dimensions
        if fluc is not None:
            snapshots = snapshots - np.transpose(
                np.tile(np.mean(snapshots, axis=1), (n, 1)))
        V1 = snapshots[:, :-1]
        V2 = snapshots[:, 1:]

        U, D, VH = np.linalg.svd(V1, full_matrices=False)  # do not perform tlsq
        # V = VH.conj().T = VH.H
        S = U.conj().T @ V2 @ VH.conj().T * np.reciprocal(
            D)  # or ##.dot(np.diag(D)), @=np.matmul=np.dot
        eigval, eigvec = np.linalg.eig(S)
        eigvec = U.dot(eigvec)  # dynamic modes
        coeff = np.linalg.lstsq(eigvec, snapshots.T[0])[0]
        return (coeff, eigval, eigvec)
