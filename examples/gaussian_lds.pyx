# distutils: extra_compile_args = -O3
# cython: wraparound=False 
# cython: boundscheck=False
# cython: nonecheck=False
## cython: cdivision=True
"""
A simple demo of a particle MCMC implementation with cython
"""
from hips.inference.particle_mcmc cimport InitialDistribution, Proposal, Likelihood, ParticleGibbsAncestorSampling

import numpy as np
cimport numpy as np

import matplotlib.pyplot as plt

class GaussianInitialDistribution(InitialDistribution):

    def __init__(self, mu, sigma):
        # Check sizes
        if np.isscalar(mu) and np.isscalar(sigma):
            self.D = 1
            mu = np.atleast_2d(mu)
            sigma = np.atleast_2d(sigma)

        elif mu.ndim == 1 and sigma.ndim == 2:
            assert mu.shape[0] == sigma.shape[0] == sigma.shape[1]
            self.D = mu.shape[0]
            mu = mu.reshape((1,self.D))

        elif mu.ndim == 2 and sigma.ndim == 2:
            assert mu.shape[1] == sigma.shape[0] == sigma.shape[1] and mu.shape[0] == 1
            self.D = mu.shape[1]
        else:
            raise Exception('Invalid shape for mu and sigma')

        self.mu = mu
        self.sigma = sigma
        self.chol = np.linalg.cholesky(self.sigma)

    def sample(self, N=1):
        smpls = np.tile(self.mu, (N,1))
        smpls += np.dot(np.random.randn(N,self.D), self.chol)
        return smpls

cdef class LinearGaussianDynamicalSystemProposal(Proposal):
    # Transition matrix
    cdef double[:,::1] A
    cdef int D

    # Transition noise
    cdef double[::1] sigma

    def __init__(self, double[:,::1] A, double[::1] sigma):
        self.A = A
        self.D = A.shape[0]
        assert A.shape[1] == self.D, "Transition matrix A must be square!"
        self.sigma = sigma

    cpdef set_sigma(self, double[::1] sigma):
        self.sigma = sigma

    cpdef predict(self, double[:,:,::1] zpred, double[:,:,::1] z, int[::1] ts ):
        cdef int T = z.shape[0]
        cdef int N = z.shape[1]
        cdef int D = z.shape[2]

        cdef int S = ts.shape[0]
        cdef int s, t, n, d1, d2

        for s in range(S):
            t = ts[s]
            for n in range(N):
                # TODO: Use a real matrix multiplication library
                for d1 in range(D):
                    # z_t = A*z_{t-1} + noise
                    zpred[t+1, n, d1] = 0
                    for d2 in range(D):
                        zpred[t+1, n, d1] += self.A[d1,d2] * z[t,n,d2]


    cpdef sample_next(self, double[:,:,::1] z, int i_prev, int[::1] ancestors):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        cdef int N = z.shape[1]
        cdef int D = z.shape[2]
        cdef int n, d1, d2

        # Preallocate random variables
        cdef double[:,::1] rands = np.random.randn(N,D)

        for n in range(N):
            # TODO: Use a real matrix multiplication library
            for d1 in range(D):
                # z_t = A*z_{t-1} + noise
                z[i_prev+1, n, d1] = 0
                for d2 in range(D):
                    z[i_prev+1, n, d1] += self.A[d1,d2] * z[i_prev,ancestors[n],d2]

                # Add noise
                z[i_prev+1, n, d1] += self.sigma[d1] * rands[n,d1]


    cpdef logp(self, double[:,::1] z_prev, int i_prev, double[::1] z_curr, double[::1] lp):
        """ Compute the log probability of transitioning from z_prev to z_curr
            at time self.ts[i_prev] to self.ts[i_prev+1]

            :param z_prev:  NxD buffer of particle states at the i_prev-th time index
            :param i_prev:  Time index into self.ts
            :param z_curr:  D buffer of particle states at the (i_prev+1)-th time index
            :param lp:      NxM buffer in which to store the probability of each transition

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        cdef int N = z_prev.shape[0]
        cdef int D = z_prev.shape[1]
        cdef int n, d, d1, d2
        cdef double[::1] z_mean = np.zeros((D,))
        cdef double sqerr
        for n in range(N):
            # Compute z_mean = dot(self.A, z_prev[n,:]
            for d1 in range(D):
                z_mean[d1] = 0
                for d2 in range(D):
                    z_mean[d1] += self.A[d1,d2] * z_prev[n,d2]

            sqerr = 0
            for d in range(D):
                sqerr += (z_curr[d] - z_mean[d])**2 / self.sigma[d]**2
            lp[n] = -0.5 * sqerr


cdef class LinearGaussianLikelihood(Likelihood):
    """
    General wrapper for a likelihood object.
    It must support efficient calculation of the log
    likelihood given a set of particles and observations.
    Extend this for the likelihood of interest
    """
    cdef double[:,::1] C
    cdef int D
    cdef int O
    cdef double eta
    def __init__(self, double[:,::1] C, double eta):

        self.C = C
        self.D = C.shape[0]
        self.O = C.shape[1]
        self.eta = eta

    cpdef logp(self, double[:,:,::1] z, double[:,::1] x, int i, double[::1] ll):
        """ Compute the log likelihood, log p(x|z), at time index i and put the
            output in the buffer ll.

            :param z:   TxNxD buffer of latent states
            :param x:   TxO buffer of observations
            :param i:   Time index at which to compute the log likelihood
            :param ll:  N buffer to populate with log likelihoods

            :return     Buffer ll should be populated with the log likelihood of
                        each particle.
        """
        cdef int N = z.shape[1]
        cdef int n, o, o1, d2
        cdef double[::1] x_mean = np.zeros(self.O)
        for n in range(N):
            for o1 in range(self.O):
                x_mean[o1] = 0
                for d2 in range(self.D):
                    x_mean[o1] += self.C[o1,d2] * z[i,n,d2]

            sqerr = 0
            for o in range(self.O):
                sqerr += (x[i,o] - x_mean[o])**2

            ll[n] = -0.5/self.eta**2 * sqerr

    cpdef sample(self, double[:,:,::1] z, double[:,::1] x, int i, int n):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        cdef int o
        cdef np.ndarray[double, ndim=1] x_mean, noise
        x_mean = np.dot(self.C, z[i,n,:])
        noise = self.eta * np.random.randn(self.O)

        for o in range(self.O):
            x[i,o] = x_mean[o] + noise[o]
