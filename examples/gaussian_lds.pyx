# cython: profile=True

"""
A simple demo of a particle MCMC implementation
"""
# from hips.inference.particle_mcmc import InitialDistribution
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
    cdef double sigma

    def __init__(self, double[:,::1] A, double sigma):
        self.A = A
        self.D = A.shape[0]
        assert A.shape[1] == self.D, "Transition matrix A must be square!"
        self.sigma = sigma

    cpdef sample_next(self, double[:,:,::1] z, int i_prev):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        cdef int N = z.shape[1]
        cdef int D = z.shape[2]
        cdef int n, d, d1, d2

        # Preallocate random variables
        cdef np.ndarray[np.float_t,
                        ndim=2,
                        negative_indices=False,
                        mode='c'] rands = \
            np.random.randn(N, D)

        for n in range(N):
            # TODO: Use a real matrix multiplication library
            for d1 in range(D):
                z[i_prev+1, n, d1] = 0
                for d2 in range(D):
                    z[i_prev+1, n, d1] += self.A[d1,d2] * z[i_prev,n,d2]

            # TODO: Debug compiler issue with inplace operators
            # z[i_prev+1, n, :] = z[i_prev+1, n, :] + self.sigma * np.random.randn(self.D)
            # tmp = self.sigma * np.random.randn(self.D)
            for d in range(D):
                # z[i_prev+1, n, d] = z[i_prev+1, n, d] + tmp[d]
                # z[i_prev+1, n, d] += self.sigma * np.random.randn()
                z[i_prev+1, n, d] += self.sigma * rands[n,d]


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
                for d2 in range(D):
                    z_mean[d1] += self.A[d1,d2] * z_prev[n,d2]

            # z_mean = np.dot(self.A, z_prev[n,:])
            sqerr = 0
            for d in range(D):
                sqerr += (z_curr[d] - z_mean[d])**2
            lp[n] = -0.5/self.sigma**2 * sqerr


cdef class GaussianLikelihood(Likelihood):
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

            # x_mean = np.dot(self.C, z[i,n,:])
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

# def create_gaussian_lds(D, dt, sig_init,  sig_trans, sig_obs):
#     if D == 1:
#         A = 0.75
#     elif D == 2:
#         th = np.pi/3.0
#         A = np.array([[np.cos(th), -np.sin(th)],
#                       [np.sin(th),  np.cos(th)]])
#     else:
#         assert np.mod(D, 2) == 0
#
#         ths = np.random.rand(D/2) * 2*np.pi
#
#         A = np.zeros((D,D))
#         for i in range(D/2):
#             th = ths[i]
#             A[i*2:(i+1)*2] = np.array([[np.cos(th), -np.sin(th)],
#                                        [np.sin(th),  np.cos(th)]])
#
#
#     # Try to fit it with a particle filter
#     p_initial = SphericalGaussianDistribution(D, sig=sig_init)
#     lkhd = NoiseLikelihood(SphericalGaussianDistribution(D, sig=sig_obs))
#     prop = DynamicalSystemProposal(lambda t,z: np.dot(A-np.eye(D),z)/dt,
#                                    SphericalGaussianDistribution(D, sig=sig_trans))
#
#     return A, p_initial, lkhd, prop
#
# def sample_x_given_z(z, lkhd):
#     x = lkhd.sample(z)
#     return x
#
# def sample_z_given_x(x, z_curr, dt,
#                      p_initial, lkhd, prop,
#                      N_particles=100):
#     D,T = x.shape
#     t = dt*np.arange(T)
#
#     pf = ConditionalParticleFilterWithAncestorSampling(D, T, t[0],
#                                                        x[:,0],
#                                                        prop, lkhd, p_initial, z_curr,
#                                                        Np=N_particles)
#
#     # pf = ConditionalParticleFilter(D, T, t[0],
#     #                                x[:,0],
#     #                                prop, lkhd, p_initial, z_curr,
#     #                                Np=N_particles)
#
#     for ind in np.arange(1,T):
#         t_curr = t[ind]
#         x_curr = x[:,ind]
#         pf.filter(t_curr, x_curr)
#
#     # Sample a particular weight trace given the particle weights at time T
#     i = np.sum(np.cumsum(pf.trajectory_weights) < np.random.rand()) - 1
#
#     trajs = pf.trajectories
#     z = trajs[:,i,:].reshape((D,T))
#
#     # plot_geweke_single_iter(trajs, z, x)
#
#     return z
#
# def sample_z_prior(D, T, dt,  p_initial, prop):
#     # Sample from prior
#     t = dt * np.arange(T)
#     z = np.zeros((D,T))
#     z[:,0] = p_initial.sample()
#     for i in np.arange(1,T):
#         z[:,i] = prop.sample_next(t[i-1], z[:,i-1], t[i])
#     return z
