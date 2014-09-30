# encoding: utf-8
# cython: profile=True

import numpy as np
from log_sum_exp import log_sum_exp_sample, log_sum_exp_normalize, discrete_sample

import numpy as np
cimport numpy as np

from libc.math cimport log

cdef class InitialDistribution(object):
    """
    Simple python class for the initial distribution. This doesn't get
    called nearly as frequently as the proposal and likelihood, so we
    can afford to have it in Python.
    """
    cpdef double[:,::1] sample(self, int N=1):
        raise NotImplementedError("Initial Distribution is a base class")


cdef class Proposal:
    """
    General wrapper for a proposal distribution.
    It must support efficient log likelihood calculations and sampling.
    Extend this for the proposal of interest
    """
    cpdef sample_next(self, double[:,:,::1] z, int i_prev, int[::1] ancestors):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts
            :param ancestors: The indices in z[i_prev,:,:] of the particles to propagate

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        pass

    cpdef logp(self, double[:,::1] z_prev, int i_prev, double[::1] z_curr, double[::1] lp):
        """ Compute the log probability of transitioning from z_prev to z_curr
            at time self.ts[i_prev] to self.ts[i_prev+1]

            :param z_prev:  NxD buffer of particle states at the i_prev-th time index
            :param i_prev:  Time index into self.ts
            :param z_curr:  MxD buffer of particle states at the (i_prev+1)-th time index
            :param lp:      NxM buffer in which to store the probability of each transition

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        pass

cdef class Likelihood:
    """
    General wrapper for a likelihood object.
    It must support efficient calculation of the log
    likelihood given a set of particles and observations.
    Extend this for the likelihood of interest
    """
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
        pass


cdef class ParticleGibbsAncestorSampling(object):
    """
    A conditional particle filter with ancestor sampling
    """
    # Class variables are defined in the pxd file
    def __init__(self, int T, int N, int D):
        """
        Initialize the particle filter with:
        T:           Number of time steps
        N:           The number of paricles
        D:           Dimensionality of latent state space
        """
        # Initialize data structures for particles and weights
        self.T = T
        self.N = N
        self.D = D

        self.z = np.zeros((T,N,D), dtype=np.float)
        self.ancestors = np.zeros((T,N), dtype=np.int32)
        self.weights = np.zeros((T,N), dtype=np.float)


    cpdef initialize(self,
                     InitialDistribution init,
                     Proposal prop,
                     Likelihood lkhd,
                     double[:,::1] x,
                     double[:,::1] fixed_particle):
        """

        proposal:    A proposal distribution for new particles given old
        lkhd:        An observation likelihood model for particles
        p_initial:   Distribution over initial states
        z_fixed:     Particle representing current state of Markov chain
        """
        # Set the observations
        assert x.shape[0] == self.T and x.ndim == 2, "Invalid observation shape"
        self.x = x
        self.init = init
        self.prop = prop
        self.lkhd = lkhd

        # Store the fixed particle
        assert fixed_particle.shape[0] == self.T
        assert fixed_particle.shape[1] == self.D
        self.fixed_particle = fixed_particle

        # Let the first particle correspond to the fixed particle
        self.z[:,0,:] = self.fixed_particle
        # Sample the initial state
        self.z[0,1:,:] = init.sample(N=self.N-1)

        # TODO: Should also factor in probability under the initial distribution
        # Initialize weights according to observation likelihood
        cdef double[::1] ll0 = np.zeros(self.N)
        self.lkhd.logp(self.z, self.x, 0, ll0)
        log_sum_exp_normalize(ll0, self.weights[0,:])

    cpdef double[:,::1] sample(self):
        """
        Sample a new set of particle trajectories
        """

        # Allocate memory
        cdef double[::1] lp_trans = np.zeros(self.N)
        cdef double[::1] ll_obs = np.zeros(self.N)
        cdef int t, n

        for t in range(1, self.T):
            # First, resample the previous parents
            self.systematic_resampling(t)

            # Move each particle forward according to the proposal distribution
            self.prop.sample_next(self.z, t-1, self.ancestors[t,:])

            # Override the first particle with the fixed particle
            self.z[t,0,:] = self.fixed_particle[t,:]

            # Resample the parent index of the fixed particle
            # TODO: Does the parent get to choose from all particles or only the resampled particles?
            self.prop.logp(self.z[t-1,:,:], t-1, self.z[t,0,:], lp_trans)
            for n in range(self.N):
                lp_trans[n] += log(self.weights[t-1,n])

            self.ancestors[t,0] = log_sum_exp_sample(lp_trans)

            # Update the weights. Since we sample from the prior,
            # the particle weights are always just a function of the likelihood
            self.lkhd.logp(self.z, self.x, t, ll_obs)
            log_sum_exp_normalize(ll_obs, self.weights[t,:])

        # Sample a trajectory according to the final weights
        n = discrete_sample(self.weights[self.T-1,:])
        return self.get_trajectory(n)

    cpdef double[:,::1] get_trajectory(self, int n):
        # Compute the i-th trajectory from the particles and ancestors
        cdef int T = self.T
        cdef double[:,::1] traj = np.zeros((T, self.D))
        traj[T-1,:] = self.z[T-1,n,:]
        cdef int curr_ancestor = self.ancestors[T-1,n]

        for t in range(T-2,-1,-1):
            traj[t,:] = self.z[t,curr_ancestor,:]
            curr_ancestor = self.ancestors[t,curr_ancestor]
        return traj

    cdef systematic_resampling(self, int t):
        """
        Resample particles using the "systematic resampling" method.
        We sample a single random number, u, as a shift in the CDF
        Of a multinomial distribution. That is, our ancestors 'a' will
        be resampled according to:
            a_n = F^{-1} [ u_n]
        where:
            F^{-1} is the inverse CDF of the distribution defined by
            the normalized particle weights

            u_n is a random number in [0,1] drawn from one of the resampling schemes

        For systematic resampling, we let
            u_shift ~ U[0,1]
            u_n = (n + u_shift)/N
        """
        # Compute the CDF of the weights at the previous time bin
        cdef double[::1] cdf = np.empty(self.N)
        cdef int n
        cdf[0] = self.weights[t-1,0]
        for n in range(1,self.N):
            cdf[n] = cdf[n-1] + self.weights[t-1,n]

        # First sample a global u_shift
        cdef double u_shift = np.random.rand()

        # Now go through each ancestor. Since the u_n's are necessarily increasing
        # in the systematic resampling method, we can perform this with a simple loop
        cdef int offset = 0
        cdef double u_n
        for n in range(self.N):
            u_n = (n + u_shift)/self.N

            # Loop until we find an offset where the CDF exceeds u_n
            # u_n may be larger than 1, in which case we will put it
            # in the last bin. That is why we break when offset >= N-1,
            # but by design this means offset = N-1
            while offset < self.N-1 and cdf[offset] < u_n:
                offset += 1
            self.ancestors[t,n] = offset
