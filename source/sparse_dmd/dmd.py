# %%
from collections import namedtuple
import numpy as np
import scipy.linalg as linalg
import warnings
import sys

from . import util


class DMD(object):
    def __init__(self, snapshots=None, dt=1, fluc=False, axis=-1):
        """Dynamic Mode Decomposition with optimal amplitudes.

        Arguments
            snapshots - the matrix of snapshots / multi-dimensional data

            dt - time interval between snapshots

            axis - the data axis along which to perform the
                   decomposition. Defaults to last axis, i.e.
                   you can supply the matrix of snapshots as
                   data.

        Methods
            compute - perform the decomposition on the snapshots and
                      calculate the modes, frequencies and amplitudes
            dmd_reduction - compute the matrices U*X1, S and V that
                            represent the input snapshots
            init - perform the dmd computation given a reduction

        Attributes
            modes - the dmd modes
            amplitudes - corresponding optimal amplitudes
            ritz_values - corresponding ritz values
            frequencies - corresponding frequencies
        """
        if snapshots is not None:
            self.data_shape = snapshots.shape
            self.snapshots = snapshots
            # self.snapshots = util.to_snaps(snapshots, axis=axis)
        if self.data_shape[0] < self.data_shape[1]:
            sys.exit("Input snapshots error: time series must be in column!")

        self.fluc = fluc
        if self.fluc is True:
            self.snapshots = self.snapshots - np.transpose(
                np.tile(
                    np.mean(self.snapshots, axis=1), (self.data_shape[1], 1)))

        self.axis = axis
        self.dt = dt
        self.computed = False

        self.keep_reduction = True

    def compute(self):
        reduction = self.reduction
        self.init(reduction.UstarX1, reduction.S, reduction.V)

        self.modes = np.dot(reduction.U, self.Ydmd)
        self.eigval = self.Edmd
        self.frequencies = np.log(self.eigval) / self.dt
        self.beta = np.real(self.frequencies)
        self.omega = np.imag(self.frequencies)
        # the optimal amplitudes
        self.amplitudes = self.xdmd
        self.computed = True

    @property
    def reduction(self):
        """Compute the reduced form of the data snapshots"""
        if not hasattr(self, '_reduction') and self.keep_reduction:
            self._reduction = self.dmd_reduction(self.snapshots)
            return self._reduction
        elif hasattr(self, '_reduction'):
            return self._reduction
        else:
            return self.dmd_reduction(self.snapshots)

    @reduction.deleter
    def reduction(self):
        if hasattr(self, '_reduction'):
            del self._reduction

    @staticmethod
    def dmd_reduction(snapshots):
        """Takes a series of snapshots and splits into two subsequent
        series, X0, X1, where

            snapshots = [X0, X1[-1]]

        then computes the (economy) single value decomposition

            X0 = U S V*

        returns an object with attributes

            UstarX1 - U* X1, the left singular vectors (POD modes)
                      projected onto the data
            U       - U, the left singular vectors
            S       - S, the singular values
            V       - V, the right singular vectors
            X0      - X0, snapshots[:, :-1]
            X1      - X1, snapshots[:. 1:]
        """
        X0 = snapshots[:, :-1]
        X1 = snapshots[:, 1:]

        # economy size SVD of X0
        U, S, Vh = linalg.svd(X0, full_matrices=False, lapack_driver='gesvd')
        # n.b. differences with matlab svd:
        # 1. S is a 1d array of diagonal elements
        # 2. Vh == V': the matlab version returns V for X = U S V',
        #    whereas python returns V'

        S = np.diag(S)
        V = Vh.T.conj()
        # truncate zero values from svd
        r = np.linalg.matrix_rank(S)

        if np.shape(S)[0] != r:
            warnings.warn("There are " + str(np.shape(S)[0] - r) +
                          " zero-value singular value!!!")
        U = U[:, :r]
        S = S[:r, :r]
        V = V[:, :r]

        # Determine matrix UstarX1
        UstarX1 = np.dot(U.T.conj(), X1)   # conjugate transpose (U' in matlab)

        reduction_keys = ('UstarX1', 'S', 'V', 'U', 'X0', 'X1')
        Reduction = namedtuple('DMDReduction', reduction_keys)

        return Reduction(UstarX1=UstarX1, S=S, V=V, U=U, X0=X0, X1=X1)

    def init(self, UstarX1=None, S=None, V=None):
        """Calculate the DMD matrix from the data reduction."""
        if UstarX1 is not None:
            self.UstarX1 = UstarX1
        if S is not None:
            self.S = S
        if V is not None:
            self.V = V

        self.UstarX1 = UstarX1
        self.S = S
        self.V = V

        Vstar = self.V.T.conj()

        # The number of snapshots
        N = Vstar.shape[1]

        # Optimal DMD matrix resulting from Schmid's 2010 algorithm
        # Fdmd = U'*X1*V*inv(S)
        self.Fdmd = np.dot(np.dot(self.UstarX1, self.V),
                           linalg.inv(self.S))

        # eigenvalue decomposition of Fdmd
        self.Edmd, self.Ydmd = linalg.eig(self.Fdmd)

        # Form Vandermonde matrix
        # Vand = Edmd ** np.arange(N)[None].T
        self.Vand = np.vander(self.Edmd, N).T[::-1].T

        # Determine optimal vector of amplitudes xdmd
        # Objective: minimize the least-squares deviation between
        # the matrix of snapshots X0 and the linear combination of
        # the dmd modes
        # Can be formulated as:
        # minimize || G - L*diag(xdmd)*R ||_F^2
        L = self.Ydmd
        R = self.Vand
        G = np.dot(self.S, Vstar)

        # Form matrix P, vector q, and scalar s, where
        # J = x'*P*x - q'*x - x'*q + s
        # with x as the optimization variable (i.e., the unknown
        # vector of amplitudes). We seek to minimize J to find the
        # optimal amplitudes.
        self.P = np.dot(L.T.conj(), L) * np.dot(R, R.T.conj()).conj()
        self.q = np.diagonal(np.dot(np.dot(R, G.T.conj()), L)).conj()
        self.s = np.trace(np.dot(G.T.conj(), G))

        # Optimal vector of amplitudes xdmd
        # computed by cholesky factorization of P (more efficient
        # way of computing x = P^(-1) q, as P is hermitian positive
        # definite matrix) === solve(Pl.T.conj(), solve(Pl, self.q))
        self.xdmd = linalg.cho_solve(linalg.cho_factor(self.P), self.q)

    def reduce(self, percent):
        ind = np.where(np.abs(self.eigval) > percent)[0]
        num = np.shape(self.modes)[0]
        reduced = ReducedDMD(num, np.size(ind))
        reduced.ind = ind
        reduced.eigval = self.eigval[ind]
        reduced.modes = self.modes[:, ind]
        reduced.omega =  self.omega[ind]
        reduced.beta = self.beta[ind]
        reduced.amplitudes = self.amplitudes[ind]
        return reduced

    @property
    def dmodes(self):
        """Return modes reshaped into original data shape."""
        return util.to_data(self.modes, shape=self.data_shape, axis=self.axis)

    @property
    def dynamics(self):
        dynamics = np.diag(self.amplitudes) @ self.Vand
        return dynamics

    @property
    def residuals(self):
        V1 = self.modes @ self.dynamics
        resid = V1 - self.snapshots[:, :-1]
        return (np.linalg.norm(resid))


class ReducedDMD(object):

    def __init__(self, num1, num2):
        self.eigval = np.zeros(num2, dtype=np.complex)
        self.modes = np.zeros((num1, num2), dtype=np.complex)
        self.omega = np.zeros(num2)
        self.beta = np.zeros(num2)
        self.amplitudes = np.zeros(num2)
        self.ind = np.zeros(num2)


    def reduce(self, rr):
        reduction = self.reduction
        UstarX1 = np.dot(reduction.U[:, :rr].T.conj(), reduction.X1)
        S = reduction.S[:rr, :rr]
        V = reduction.V[:, :rr]
        self.init(UstarX1, S, V)
        self.modes = np.dot(reduction.U[:, :rr], self.Ydmd)
        self.eigval = self.Edmd
        self.frequencies = np.log(self.eigval) / self.dt
        self.beta = np.real(self.frequencies)
        self.omega = np.imag(self.frequencies)
        # the optimal amplitudes
        self.amplitudes = self.xdmd
        self.computed = True


