# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:24:50 2018
    This code for Dynamic Mode decomposition
@author: Weibo Hu
"""

import numpy as np
import scipy as sp
import warnings
import pandas as pd
from numpy.core.umath_tests import inner1d
import sys


class DMD(object):
    def __init__(self, snapshots):
        self.eigval = None
        self.eigvec = None
        self.modes = None
        self.snapshots = snapshots
        self.amplit = None
        self.dynamics = None
        self.residual = 0
        self.Ja = None
        self._dim = 0
        self._tn = 0
        self._U = None
        self._V = None
        self._D = None

    @property
    def dim(self):
        return np.shape(self.snapshots)[0]

    @property
    def tn(self):
        return np.shape(self.snapshots)[1]

    @property
    def U(self):
        return self._U

    def dmd_standard(self, fluc=None):
        snapshots = self.snapshots
        tn = self.tn
        if fluc == 'True':
            snapshots = snapshots - np.transpose(
                np.tile(np.mean(snapshots, axis=1), (tn, 1)))
        V1 = snapshots[:, :-1]
        V2 = snapshots[:, 1:]
        U, D, VH = sp.linalg.svd(V1, full_matrices=False)  # use min(m, n)
        V = VH.conj().T
        ind = np.where(np.round(D, 2) == 0.0)
        if np.size(ind) != 0:
            print("There are " + str(np.size(ind)) +
                  " zero-value singular value!!!")
            nonzero = tn - 1 - np.size(ind)
            V = V[:, :nonzero]
            D = D[:nonzero]
            U = U[:, :nonzero]
        S = U.T.conj() @ V2 @ V * np.reciprocal(D)
        self.eigval, self.eigvec = sp.linalg.eig(S)
        self.modes = U @ self.eigvec  # projected dynamic modes
        if np.size(ind) != 0:
            self.residual = np.linalg.norm(
                V2[:, :nonzero] - V1[:, :nonzero] @ S) / tn
        else:
            self.residual = np.linalg.norm(V2 - V1 @ S) / tn
        self._U = U
        self._D = D
        self._V = V
        return (self.eigval, self.modes, self.residual)

    # convex optimization problem
    def dmd_amplitude(self, opt=None):
        snapshots = self.snapshots
        tn = self.tn
        if opt == 'False':
            self.amplit = np.linalg.lstsq(
                self.modes, snapshots.T[0], rcond=-1)[0]
        else:  # min(var-phi@coeff@Vand)-->min(UT@V1-UT@U@eigvec@coeff@Vand)
            A = self.eigvec @ np.diag(self.eigval**0)
            for i in range(tn - 2):
                a2 = self.eigvec @ np.diag(self.eigval**(i + 1))
                A = np.vstack([A, a2])
            b = np.reshape(
                self._U.T.conj() @ snapshots[:, :-1], (-1, ), order='F')
            self.amplit = np.linalg.lstsq(A, b, rcond=-1)[0]
        return self.amplit

    def dmd_dynamics(self, timepoints):
        period = np.round(np.diff(timepoints), 6)
        if (np.size(np.unique(period)) != 1):
            sys.exit("Time period is not equal!!!")
        t_samp = timepoints[1] - timepoints[0]
        # growth rate(real part), frequency(imaginary part)
        lamb = np.log(
            self.eigval) / t_samp
        m = np.size(lamb)
        n = np.size(timepoints)
        TimeGrid = np.transpose(np.tile(timepoints, (m, 1)))
        LambGrid = np.tile(lamb, (n, 1))
        OmegaT = LambGrid * TimeGrid  # element wise multiply
        self.dynamics = np.transpose(np.exp(OmegaT) * self.amplit)
        return self.dynamics

    def SPDMD_J(self):
        tn = self.tn
        coeff = self.amplit
        Vand = np.hstack([self.eigval.reshape(-1,1)**i for i in range(tn-1)])
        p1 = (self.eigvec.T.conj() @ self.eigvec)
        p2 = (Vand.T.conj() @ Vand).conj()
        P = p1 * p2
        q1 = Vand @ self._V @ np.diag(self._D).T.conj() @ self.eigvec
        q = np.diag(q1).conj()
        ss = (np.linalg.norm(self._D))**2
        self.Ja = coeff.T.conj@P@coeff-q.T.conj()@coeff-coeff.T.conj@q+ss
        return self.Ja

    @staticmethod
    def dmd_reconstruct(modes, dynamics):
        reconstruct = modes @ dynamics
        return (reconstruct)

    # Exact Dynamic Mode Decompostion
    # Ref: Jonathan H. T., et. al.-On dynamic mode decomposition:
    # theory and application
    def dmd_Exact(self, snapshots, fluc):  # scaled method
        n = np.shape(snapshots)[1]  # n: the number of snapshots
        if fluc is not None:
            snapshots = snapshots - np.transpose(
                np.tile(np.mean(snapshots, axis=1), (n, 1)))
        V1 = snapshots[:, :-1]
        V2 = snapshots[:, 1:]
        # do not perform tlsq
        U, D, VH = np.linalg.svd(V1, full_matrices=False)
        # V = VH.conj().T = VH.H
        S = U.conj().T @ V2 @ VH.conj().T * np.reciprocal(
            D)  # or ##.dot(np.diag(D)), @=np.matmul=np.dot
        eigval, eigvec = np.linalg.eig(S)
        eigvec = U.dot(eigvec)  # dynamic modes
        coeff = np.linalg.lstsq(eigvec, snapshots.T[0])[0]
        return (coeff, eigval, eigvec)
