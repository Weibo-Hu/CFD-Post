# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:24:50 2018
    This code for Dynamic Mode decomposition
@author: Weibo Hu
"""

import numpy as np
import scipy as sp
import warnings
import sys
from progressbar import ProgressBar, Percentage, Bar


class DMD(object):
    def __init__(self, snapshots):
        self.eigval = None
        self.eigvec = None
        self.modes = None
        self.snapshots = snapshots
        self.amplit = None
        self.Vand = None
        self.dynamics = None
        self.residual = 0
        self.beta = None
        self.omega = None
        self.s = 0.0
        self.Ja = None
        self.rho = 1.0
        self.maxiter = 10000
        self.Prho = None
        self.eps_abs = 1.e-6
        self.eps_rel = 1.e-4
        self.gamma = np.logspace(-2, 6, 50)
        self.spdmd = None
        self.spdmd_nmodes = None
        self.spdmd_modes = None
        self.spdmd_amplit = None
        self.spdmd_Vand = None
        self.spdmd_reconstruct = None
        self.spdmd_loss = None
        self.r = 0.0
        self.ng = 0.0
        self.U = None
        self.V = None
        self.VH = None
        self.D = None
        self.DM = None

    @property
    def dim(self):
        return np.shape(self.snapshots)[0]

    @property
    def tn(self):
        return np.shape(self.snapshots)[1]

    @property
    def dmd_residual(self):
        V1 = self.modes @ self.dynamics
        resid = V1 - self.snapshots[:, :-1]
        return (np.linalg.norm(resid))

    def dmd_standard(self, fluc=None):
        """
        U D VH = V1
        U: the left singular vectors
        D: the sigular values, it is a 1d array in python
        V: the right singular vectors, VH = V.T.conj()
        """
        tn = self.tn
        if fluc == True:
            self.snapshots = self.snapshots - np.transpose(
                np.tile(np.mean(self.snapshots, axis=1), (tn, 1)))
        snapshots = self.snapshots
        V1 = np.float64(snapshots[:, :-1])
        V2 = np.float64(snapshots[:, 1:])
        #U, D, VH = sp.linalg.svd(V1, full_matrices=False, lapack_driver='gesvd')  # use min(m, n)
        U, D, VH = np.linalg.svd(V1, full_matrices=False)  # use min(m, n)
        V = VH.conj().T
        DM = np.diag(D)
        rank = np.linalg.matrix_rank(DM)
        if np.size(D) != rank:
            warnings.warn("There are " + str(np.size(D) - rank) +
                          " zero-value singular value!!!")
            V = V[:, :rank]
            D = D[:rank]
            U = U[:, :rank]
        S = U.T.conj() @ V2 @ V * np.reciprocal(D)
        self.eigval, self.eigvec = np.linalg.eig(S)
        self.modes = U @ self.eigvec  # projected dynamic modes

        self.U = U
        self.D = D
        self.V = V
        self.VH = V.T.conj()
        self.DM = DM
        return (self.eigval, self.modes)

    # convex optimization problem
    def dmd_amplitude(self, opt=None):
        snapshots = self.snapshots
        tn = self.tn
        if opt == False:
            self.amplit = np.linalg.lstsq(
                self.modes, snapshots.T[0], rcond=-1)[0]
        elif opt == 'spdmd':
            self.amplit = self.spdmd_amplitude()
        else:  # min(var-phi@coeff@Vand)-->min(UT@V1-UT@U@eigvec@coeff@Vand)
            A = self.eigvec @ np.diag(self.eigval**0)
            for i in range(tn - 2):
                a2 = self.eigvec @ np.diag(self.eigval**(i + 1))
                A = np.vstack([A, a2])
            b = np.reshape(
                self.U.T.conj() @ snapshots[:, :-1], (-1, ), order='F')
            #b = np.reshape(self.D * self.VH, (-1, ), order='F')
            self.amplit = np.linalg.lstsq(A, b, rcond=-1)[0]
        return self.amplit


    def dmd_dynamics(self, timepoints):
        period = np.round(np.diff(timepoints), 6)
        if (np.size(np.unique(period)) != 1):
            sys.exit("Time period is not equal!!!")
        t_samp = timepoints[1] - timepoints[0]
        # # growth rate(real part), frequency(imaginary part)
        lamb = np.log(
            self.eigval) / t_samp
        self.beta = lamb.real
        self.omega = lamb.imag
        # m = np.size(lamb)
        # n = np.size(timepoints)
        # TimeGrid = np.transpose(np.tile(timepoints[:-1], (m, 1)))
        # LambGrid = np.tile(lamb, (n-1, 1))
        # OmegaT = LambGrid * TimeGrid  # element wise multiply
        # self.dynamics = np.transpose(np.exp(OmegaT) * self.amplit)
        self.Vand = np.vander(self.eigval, self.tn - 1, increasing=True)
        self.dynamics = np.diag(self.amplit) @ self.Vand
        return self.dynamics

    def spdmd_amplitude(self):
        # coeff = self.amplit
        # Vand = np.hstack([self.eigval.reshape(-1,1)**i for i in range(tn-1)])
        self.Vand = np.vander(self.eigval, self.tn - 1, increasing=True)
        Vand = self.Vand
        # Determine optimal vector of amplitudes (amplit)
        # Objective: minimize the least-squares deviation between
        # the matrix of snapshots V1 and the linear combination of
        # the dmd modes
        # Can be formulated : min ||sigma Vh - eigvec diag(amplit) Vand||_F^2
        p1 = (self.eigvec.T.conj() @ self.eigvec)
        p2 = (Vand @ Vand.T.conj()).conj()
        P = p1 * p2
        q = np.diag(Vand @ self.V @ self.DM.T.conj() @ self.eigvec).conj()
        ss = (np.linalg.norm(self.D))**2
        # ss = np.trace(self.DM.T.conj() @ self.DM)
        # Opitmal vector of amplitudes (amplit)
        # amplit = P^-1 q (P amplit = q)
        amplit = sp.linalg.cho_solve(sp.linalg.cho_factor(P), q)
        # amplit = sp.linalg.solve(P, q)
        self.q = q
        self.P = P
        self.s = ss
        # self.amplit = amplit
        return amplit


    # Ref: Jovanovic H. T., et. al.-2014
    # Sparsity-promoting dynamic mode decomposition
    def compute_spdmd(self, maxiter=None, gamma=None, rho=None,
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
        # Appendix B
        self.r = len(self.q)
        self.ng = len(self.gamma)
        self.Prho = self.P + (self.rho / 2.0) * np.identity(self.r)
        answer = SparseAnswer(self.r, self.ng)
        answer.gamma = self.gamma
        widgets = ['SPDMD iteration on gamma:', Percentage(), ' ',
                    Bar(marker='>', left='[', right=']')]
        pbar = ProgressBar(widgets=widgets, maxval=np.size(self.gamma))
        pbar.start()
        for i, gammaval in enumerate(self.gamma):
            # find a sparisty structure which achieves a tradeoff between
            # the number of the modes and the approximation error
            ret = self.optimize_gamma(gammaval)
            answer.amplit[:, i] = ret['xsp']
            answer.PolAmplit[:, i] = ret['xpol']
            answer.Nz[i] = ret['Nz']
            answer.Ja[i] = ret['Jsp']
            answer.PolJa[i] = ret['Jpol']
            answer.PolLoss[i] = ret['Ploss']
            pbar.update(i)
        pbar.finish()
        answer.nonzero[:] = answer.amplit != 0
        self.spdmd = answer

        return self.spdmd

    def optimize_gamma(self, gamma):
        """
        minimize J(a), subject to E^T a = 0
        This amounts to finding the optimal amplitudes for a given
        sparsity. Sparsity is encoded in the structure of E.
        The first step is solved using ADMM.
        The second constraint is satisfied using KKT_solve.
        """
        # 1. use ADMM to solve the gamma-parameterized problem
        # minimiszing J with initial conditions z0,y0 (seek sparsity structure)
        lambda0 = np.zeros(self.r)  # Lagrange multiplier
        beta0 = np.zeros(self.r)  # initial amplitudes
        beta = self.admm(beta0, lambda0, gamma)  # (alpha = beta)
        # 2. use the minimized amplitudes as the input to the sparsity
        # contraint to create a vector of polished (optimal) amplitudes (fix)
        alpha = self.KKT_solve(beta)[:self.r]
        # outputs that we are intrested
        sparse_amplitudes = beta  # vector of amplitudes
        num_nonzero = (beta != 0).sum()  # number of non-zero amplitudes
        residuals = self.spdmd_residual(beta)  # least squares residual

        # Vector of polished (optimal) amplitudes
        polished_amplitudes = alpha
        # Polished (optimal) least-squares residual
        polished_residual = self.spdmd_residual(alpha)
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

    def admm(self, z, y, gamma, opt=None):
        """Alternating direction method of multipliers. 
        Sec. III A, Appendix B"""
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
        k = (gamma / self.rho)
        if opt is None:
            # precompute cholesky decomposition
            C = sp.linalg.cholesky(self.Prho, lower=False)
            # link directly to LAPACK fortran solver for positive
            # definite symmetric system with precomputed cholesky decomp:
            potrs, = sp.linalg.get_lapack_funcs(('potrs', ), arrays=(C, self.q))
            # simple norm of a 1d vector, called directly from BLAS
            norm, = sp.linalg.get_blas_funcs(('nrm2', ), arrays=(self.q, ))
        # %% square root outside of the loop
        root_r = np.sqrt(self.r)
        for ADMMstep in range(self.maxiter):
            # %% x-minimization step (alpha minimisation)
            u = z - (1. / self.rho) * y
            qs = self.q + (self.rho / 2.) * u
            # Solve Prho x = qs (x = Prho^-1 qs), using fact that Prho
            # is hermitian and positive definite and
            # assuming Prho is well behaved (no inf or nan).
            if opt is not None:
                xnew = sp.linalg.inv(self.Prho) @ qs
            else:
                xnew = potrs(C, qs, lower=False, overwrite_b=False)[0]
            # %%z-minimization step (beta minimisation)
            v = xnew + (1 / self.rho) * y
            # Soft-thresholding of v
            # zero for |v| < k
            # v - k for v > k
            # v + k for v < -k
            abs_v = np.abs(v)
            znew = ((1 - k / abs_v) * v) * (abs_v > k)

            # %% Lagrange multiplier update step
            y = y + self.rho * (xnew - znew)
            # %% Test convergence of admm
            if opt is not None:
                res_prim = np.linalg.norm(xnew - znew)  # Primal residuals
                res_dual = self.rho * np.linalg.norm(znew - z)  # dual residuals
                # Stopping criteria
                eps_prim = root_r * self.eps_abs \
                           + self.eps_rel * max(np.linalg.norm(xnew),
                                                np.linalg.norm(znew))
                eps_dual = root_r * self.eps_abs + self.eps_rel \
                           * np.linalg.norm(y)
            else:
                res_prim = norm(xnew - znew)
                res_dual = self.rho * norm(znew - z)
                eps_prim = root_r * self.eps_abs \
                           + self.eps_rel * max(norm(xnew), norm(znew))
                eps_dual = root_r * self.eps_abs + self.eps_rel * norm(y)
            if (res_prim < eps_prim) & (res_dual < eps_dual):
                return z
            else:
                z = znew
        return z

    def KKT_solve(self, z):
        """
        Fix the sparsity structure 
        Polish of the sparse vector z , seek solution to E^T z = 0
        Appendix C
        """
        ind_zero = abs(z) < 1.e-12  # ignore zero elements of z
        m = ind_zero.sum()
        E = np.identity(self.r)[:, ind_zero]
        # form KKT system for the optimality conditions
        KKT = np.vstack((np.hstack((self.P, E)),
                         np.hstack((E.T.conj(), np.zeros((m, m))))
                         ))
        rhs = np.hstack((self.q, np.zeros(m)))
        # solve KKT system (Appendix C)
        # res = sp.linalg.cho_solve(sp.linalg.cholesky(KKT), rhs)
        res = sp.linalg.solve(KKT, rhs)
        return res


    def spdmd_residual(self, x):
        """Calculate the residuals from a minimised
        vector of amplitudes x.
        """
        # qstar @ x = xstar @ q
        xstar = x.T.conj()
        Ja = xstar @ self.P @ x - 2 * xstar @ self.q + self.s

        return Ja

    @staticmethod
    def reconstruct(modes, amplitude, Vand):
        reconstruct = modes @ np.diag(amplitude) @ Vand
        # dynamics = amplitude @ Vand
        return (reconstruct)

    def spdmd_reconstruct(self, ind):
        """Reconstruction of the snapshots based on the selected 
        number of modes, given the index of gamma array
        """
        # the reconstructed data via spdmd
        Damplit = np.diag(self.spdmd.PolAmplit[:, ind])
        sp_reconst = self.modes @ Damplit @ self.Vand
        # the selected modes and corresponding parameters
        # the index of non zero amplitudes
        nonzero = self.spdmd.nonzero[:, ind]
        self.spdmd_nmodes = self.spdmd.Nz[ind]
        self.spdmd_modes = self.modes[:, nonzero]
        self.spdmd_amplit = self.spdmd.PolAmplit[nonzero, ind]
        self.spdmd_Vand = self.Vand[nonzero, :]
        self.spdmd_loss = self.spdmd.PolLoss[ind]
        # actually, self.spdmd_reconstruct == sp_reconst
        # zero value amplitudes almost still remain zero after polished
        self.spdmd_reconstruct = \
            self.spdmd_modes @ np.diag(self.spdmd_amplit) @ self.spdmd_Vand

        return (sp_reconst)

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
        self.Ja = np.zeros(ng, dtype=np.complex)
        # square of frobenius norm (after polishing)
        self.PolJa = np.zeros(ng, dtype=np.complex)
        # optimal performance loss (after polishing)
        self.PolLoss = np.zeros(ng, dtype=np.complex)
        # vector of amplitudes (before polishing)
        self.amplit = np.zeros((n, ng), dtype=np.complex)
        # vector of amplitudes (after polishing)
        self.PolAmplit = np.zeros((n, ng), dtype=np.complex)

    @property
    def nonzero(self):
        """where amplitudes are nonzero"""
        return self.amplit != 0
