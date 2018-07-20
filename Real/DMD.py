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
        self.s = 0.0
        self.Ja = None
        self.rho = 1.0
        self.maxiter = 10000
        self.Prho = None
        self.eps_abs = 1.e-6
        self.eps_rel = 1.e-4
        self.gamma = np.logspace(-2, 6, 100)
        self.r = 0.0
        self.ng = 0.0
        self._dim = 0
        self._tn = 0
        self._U = None
        self._V = None
        self._VH = None
        self._D = None
        self._DM = None

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
        """
        U D VH = V1
        U: the left singular vectors
        D: the sigular values, it is a 1d array in python
        V: the right singular vectors, VH = V.T.conj()
        """
        snapshots = self.snapshots
        tn = self.tn
        if fluc == 'True':
            snapshots = snapshots - np.transpose(
                np.tile(np.mean(snapshots, axis=1), (tn, 1)))
        V1 = snapshots[:, :-1]
        V2 = snapshots[:, 1:]
        U, D, VH = sp.linalg.svd(V1, full_matrices=False)  # use min(m, n)
        V = VH.conj().T
        DM = np.diag(D)
        rank = np.linalg.matrix_rank(DM)
        if np.size(D) != rank:
            print("There are " + str(np.size(D) - rank) +
                  " zero-value singular value!!!")
            V = V[:, :rank]
            D = D[:rank]
            U = U[:, :rank]
        S = U.T.conj() @ V2 @ V * np.reciprocal(D)
        self.eigval, self.eigvec = sp.linalg.eig(S)
        self.modes = U @ self.eigvec  # projected dynamic modes
        if np.size(D) != rank:
            self.residual = np.linalg.norm(
                V2[:, :rank] - V1[:, :rank] @ S) / tn
        else:
            self.residual = np.linalg.norm(V2 - V1 @ S) / tn
        self._U = U
        self._D = D
        self._V = V
        self._VH = V.T.conj()
        self._DM = DM
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

    def spdmd_j(self):
        tn = self.tn
        # coeff = self.amplit
        # Vand = np.hstack([self.eigval.reshape(-1,1)**i for i in range(tn-1)])
        Vand = np.vander(self.eigval, tn-1, increasing=True)
        # Determine optimal vector of amplitudes (amplit)
        # Objective: minimize the least-squares deviation between
        # the matrix of snapshots V1 and the linear combination of
        # the dmd modes
        # Can be formulated : min ||sigma Vh - eigvec diag(amplit) Vand||_F^2
        p1 = (self.eigvec.T.conj() @ self.eigvec)
        p2 = (Vand.T.conj() @ Vand).conj()
        P = p1 * p2
        q = np.diag(Vand @ self._V @ self._DM.T.conj() @ self.eigvec).conj()
        ss = (np.linalg.norm(self._D))**2
        # ss = np.trace(self._DM.T.conj() @ self._DM)
        # self.Ja = coeff.T.conj@P@coeff-q.T.conj()@coeff-coeff.T.conj@q+ss
        self.amplit = sp.linalg.cho_solve(sp.linalg.cho_factor(P), q)
        self.q = q
        self.P = P
        self.s = ss
        return self.amplit

    def spdmd(self, maxiter=None, gamma=None, rho=None,
              eps_abs=None, eps_rel=None):
        if eps_abs is not None:
            self.eps_abs = eps_abs
        if eps_rel is not None:
            self.eps_rel = eps_rel
        if rho is not None:
            self.rho = rho
        if gamma is not None:
            self.gamma = gamma
        if maxiter is not None:
            self.maxiter = maxiter

        self.r = len(self.q)
        self.ng = len(gamma)
        self.Prho = self.Prho = self.P + (self.rho / 2.0) * np.identity(self.r)
        self.maxiter = maxiter
        answer = SparseAnswer(self.r, self.ng)
        answer.gamma = gamma

        for i, gammaval in enumerate(gamma):
            ret = self.optimize_gamma(gammaval)
            answer.xsp[:, i] = ret['xsp']
            answer.xpol[:, i] = ret['xpol']
            answer.Nz[i] = ret['Nz']
            answer.Jsp[i] = ret['Jsp']
            answer.Jpol[i] = ret['Jpol']
            answer.Ploss[i] = ret['Ploss']
        answer.nonzero[:] = answer.xsp != 0

        return answer

    def optimize_gamma(self, gamma):
        """
        minimize J(a), subject to E^T a = 0
        This amounts to finding the optimal amplitudes for a given
        sparsity. Sparsity is encoded in the structure of E.
        The first step is solved using ADMM.
        The second constraint is satisfied using KKT_solve.
        """
        # 1. use ADMM to solve the gamma-parameterized problem
        # minimiszing J with initial conditions z0, y0
        y0 = np.zeros(self.r)  # Lagrange multiplier
        z0 = np.zeros(self.r)  # initial amplitudes
        z = self.admm(z0, y0, gamma)
        # 2. use the minimized amplitudes as the input to the sparsity
        # contraint to create a vector of polished (optimal) amplitudes
        xpol = self.KKT_solve(z)[:self.r]
        # outputs that we are intrested
        sparse_amplitudes = z  # vector of amplitudes
        num_nonzero = (z != 0).sum()  # number of non-zero amplitudes
        residuals = self.spdmd_resid(z)  # least squares residual

        # Vector of polished (optimal) amplitudes
        polished_amplitudes = xpol
        # Polished (optimal) least-squares residual
        polished_residual = self.spdmd_resid(xpol)
        # Polished (optimal) performance loss
        polished_performance_loss = 100 * \
            np.sqrt(polished_residual / self.s)

        return {
            'xsp': sparse_amplitudes,
            'Nz': num_nonzero,
            'Jsp': residuals,
            'xpol': polished_amplitudes,
            'Jpol': polished_residual,
            'Ploss': polished_performance_loss,
        }

    def admm(self, z, y, gamma):
        """Alternating direction method of multipliers."""
        # There are two complexity sources:
        # 1. the matrix solver. I can't see how this can get any
        #    faster (tested with Intel MKL on Canopy).
        # 2. the test for convergence. This is the dominant source
        #    now (~3x the time of the solver)

        # Further avenues for optimization:
        # - write in cython and import as compiled module, e.g.
        #   http://docs.cython.org/src/userguide/numpy_tutorial.html
        # - use two cores, with one core performing the admm and
        #   the other watching for convergence.

        # %% square root outside of the loop
        root_r = np.sqrt(self.r)
        for ADMMstep in range(self.maxiter):
            # %% x-minimization step (alpha minimisation)
            u = z - (1. / self.rho) * y
            qs = self.q + (self.rho / 2.) * u
            # Solve Prho x = qs (x = Prho^-1 qs), using fact that Prho
            # is hermitian and positive definite and
            # assuming Prho is well behaved (no inf or nan).
            xnew = sp.linalg.inv(self.Prho) @ qs

            # %%z-minimization step (beta minimisation)
            v = xnew + (1 / self.rho) * y
            # Soft-thresholding of v
            # zero for |v| < k
            # v - k for v > k
            # v + k for v < -k
            k = (gamma / self.rho)
            abs_v = np.abs(v)
            znew = ((1 - k / abs_v) * v) * (abs_v > k)

            # %% Lagrange multiplier update step
            y = y + self.rho * (xnew - znew)
            # %% Test convergence of admm
            res_prim = np.linalg.norm(xnew - znew)  # Primal residuals
            res_dual = self.rho * np.linalg.norm(znew - z)  # dual residuals
            # Stopping criteria
            eps_prim = root_r * self.eps_abs \
                + self.eps_rel * max(np.linalg.norm(xnew),
                                     np.linalg.norm(znew))
            eps_dual = root_r * self.eps_abs + self.eps_rel * np.linalg.norm(y)
            if (res_prim < eps_prim) & (res_dual < eps_dual):
                return z
            else:
                z = znew
        return z

    def KKT_solve(self, z):
        """
        Polish of the sparse vector z , seek solution to E^T z = 0
        """
        ind_zero = abs(z) < 1.e-12  # ignore zero elements of z
        m = ind_zero.sum()
        E = np.identity(self.r)[:, ind_zero]
        # form KKT system for the optimality conditions
        KKT = np.vstack((np.hstack((self.P, E)),
                         np.hstack((E.T.conj(), np.zeros((m, m))))
                         ))
        rhs = np.hstack((self.q, np.zeros(m)))
        # solve KKT system
        return sp.linalg.solve(KKT, rhs)

    def spdmd_resid(self, x):
        """Calculate the residuals from a minimised
        vector of amplitudes x.
        """
        # conjugate transpose
        x_ = x.T.conj()
        q_ = self.q.T.conj()

        x_P = np.dot(x_, self.P)
        x_Px = np.dot(x_P, x)
        q_x = np.dot(q_, x)

        return x_Px.real - 2 * q_x.real + self.s

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


class SparseAnswer(object):
    """A set of results from sparse dmd optimisation.

    Attributes:
    gamma     the parameter vector
    nz        number of non-zero amplitudes
    nonzero   where modes are nonzero
    jsp       square of frobenius norm (before polishing)
    jpol      square of frobenius norm (after polishing)
    ploss     optimal performance loss (after polishing)
    xsp       vector of sparse amplitudes (before polishing)
    xpol      vector of amplitudes (after polishing)
    """

    def __init__(self, n, ng):
        """Create an empty sparse dmd answer.

        n - number of optimization variables
        ng - length of parameter vector
        """
        # the parameter vector
        self.gamma = np.zeros(ng)
        # number of non-zero amplitudes
        self.Nz = np.zeros(ng)
        # square of frobenius norm (before polishing)
        self.Jsp = np.zeros(ng, dtype=np.complex)
        # square of frobenius norm (after polishing)
        self.Jpol = np.zeros(ng, dtype=np.complex)
        # optimal performance loss (after polishing)
        self.Ploss = np.zeros(ng, dtype=np.complex)
        # vector of amplitudes (before polishing)
        self.xsp = np.zeros((n, ng), dtype=np.complex)
        # vector of amplitudes (after polishing)
        self.xpol = np.zeros((n, ng), dtype=np.complex)

    @property
    def nonzero(self):
        """where amplitudes are nonzero"""
        return self.xsp != 0
