# encoding: utf-8
# cython: profile=True

import numpy as np
from scipy.misc import logsumexp

from log_sum_exp_sample import log_sum_exp_sample, log_sum_exp_normalize

import numpy as np
cimport numpy as np

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

    # ts:   Length T vector of times at which the proposals will be sampled
    # cdef public double[::1] ts

    cpdef sample_next(self, double[:,:,::1] z, int i_prev):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts

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

# class DynamicalSystemProposal(Proposal):
#     def __init__(self, dzdt, noiseclass):
#         self.dzdt = dzdt
#         self.noisesampler = noiseclass
#
#     def sample_next(self, t_prev, Z_prev, t_next):
#         assert t_next >= t_prev
#         D,Np = Z_prev.shape
#         z = Z_prev + self.dzdt(t_prev, Z_prev) * (t_next-t_prev)
#
#         # Scale down the noise to account for time delay
#         dt = self.t[t_next] - self.t[t_prev]
#         sig = self.sigma * dt
#         noise = self.noisesampler.sample(Np=Np, sigma=sig)
#         logp = self.noisesampler.logp(noise)
#
#         state = {'z' : z}
#         return z + noise, logp, state
#
#     def logp(self, t_prev, Z_prev, t_next, Z_next, state=None):
#         if state is not None and 'z' in state:
#             z = state['z']
#         else:
#             # Recompute
#             z = Z_prev + self.dzdt(t_prev, Z_prev) * (t_next - t_prev)
#         return self.noisesampler.logp((Z_next-z)/np.sqrt(t_next-t_prev))


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
    # TxNxD buffer of latent state particles to be filled in
    # T:    number of time bins
    # N:    number of particles
    # D:    dimensionality of each particle
    # cdef public double[:,:,::1] z
    #
    # cdef public double[:,::1] fixed_particle
    #
    # # TxO buffer of observations
    # # O:    dimensionality of each observation
    # cdef public double[:,::1] x
    #
    # # TxN buffer of "ancestors" of each particle
    # cdef public int[:,::1] ancestors
    #
    # # TxN buffer of "weights" of each particle
    # cdef public double[:,::1] weights
    #
    # cdef Proposal prop
    # cdef Likelihood lkhd

    def __init__(self, int T, int N, int D):
        """
        Initialize the particle filter with:
        T:           Number of time steps
        N:           The number of paricles
        D:           Dimensionality of latent state space
        """
        # Initialize data structures for particles and weights
        self.D = D
        self.T = T
        self.N = N

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
        # cdef np.ndarray[double, ndim=2] tmp = init.sample(N=self.N-1)
        # cdef int n, d
        # for n in range(0,self.N-1):
        #     for d in range(self.D):
        #         self.z[0,1+n,d] = tmp[n,d]

        # Initialize weights according to observation likelihood
        cdef np.ndarray[double, ndim=1] ll0 = np.zeros(self.N, dtype=np.float)
        self.lkhd.logp(self.z, self.x, 0, ll0)
        cdef double lse_ll0 = logsumexp(ll0)

        for n in range(self.N):
            self.weights[0,n] = np.exp(ll0[n] - lse_ll0)

        cdef double w_tot = np.sum(self.weights[0,:])
        for n in range(self.N):
            self.weights[0,n] = self.weights[0,n] / w_tot


    cpdef double[:,::1] sample(self):
        """
        Sample a new set of particle trajectories
        """

        # Allocate memory
        cdef double[::1] lp_trans = np.zeros((self.N))
        cdef double[::1] ll_obs = np.zeros((self.N,))
        cdef double[::1] w_as
        cdef double[::1] w = np.empty((self.N,))
        cdef int t, n

        # DEBUG: This should eventually be set to T
        cdef int N_steps = self.T
        # for t in range(1, self.T):
        for t in range(1, N_steps):
            # First, resample the previous parents
            self._resample_particles(t-1)

            # Move each particle forward according to the proposal distribution
            self.prop.sample_next(self.z, t-1)

            # Override the first particle with the fixed particle
            self.z[t,0,:] = self.fixed_particle[t,:]

            # Resample the parent index of the fixed particle
            self.prop.logp(self.z[t-1,:,:], t-1, self.z[t,0,:], lp_trans)
            lp_trans += np.log(self.weights[t-1,:])

            self.ancestors[t,0] = log_sum_exp_sample(lp_trans)

            # Update the weights. Since we sample from the prior,
            # the particle weights are always just a function of the likelihood
            self.lkhd.logp(self.z, self.x, t, ll_obs)
            log_sum_exp_normalize(ll_obs, w)
            # w = np.exp(ll_obs - logsumexp(ll_obs))
            # w /= np.sum(w)
            self.weights[t,:] = w

        # Sample a trajectory according to the final weights
        n = np.random.choice(np.arange(self.N), p=self.weights[N_steps-1,:])
        return self.get_trajectory(n)

    cpdef double[:,::1] get_trajectory(self, int n):
        # Compute the i-th trajectory from the particles and ancestors
        cdef int T = self.T
        cdef double[:,::1] traj = np.zeros((T, self.D))
        # traj = np.zeros((T, self.D))
        traj[T-1,:] = self.z[T-1,n,:]
        cdef int curr_ancestor = self.ancestors[T-1,n]

        for t in range(T-1,-1,-1):
            traj[t,:] = self.z[t,curr_ancestor,:]
            curr_ancestor = self.ancestors[t,curr_ancestor]
        return traj

    cdef _resample_particles(self, int t):
        # Get a set of resampled parents, or sources
        cdef int[::1] ancestors = np.empty(self.N, dtype=np.int32)
        # cdef int[::1] sources = self._lowvariance_sources(self.weights[t,:], self.N)
        self.systematic_resampling(t, ancestors)

        # First pass: copy the parents into a buffer
        cdef double[:,::1] buffer = np.empty((self.N, self.D))
        cdef int n, d
        for n in range(self.N):
            for d in range(self.D):
                buffer[n,d] = self.z[t,ancestors[n],d]

        # Second pass: put the resampled parents back
        for n in range(self.N):
            for d in range(self.D):
                self.z[t,n,d] = buffer[n,d]

    cdef systematic_resampling(self, int t, int[::1] ancestors):
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
        # First sample a global u_shift
        cdef double u_shift = np.random.rand()

        # Compute the CDF of the weights
        cdef double[::1] cdf = np.cumsum(self.weights[t,:])

        # Now go through each ancestor. Since the u_n's are necessarily increasing
        # in the systematic resampling method, we can perform this with a simple loop
        cdef int offset = 0
        cdef int n
        cdef double u_n
        for n in range(self.N):
            u_n = (n + u_shift)/self.N

            # Loop until we find an offset where the CDF exceeds u_n
            # u_n may be larger than 1, in which case we will put it
            # in the last bin. That is why we break when offset >= N-1,
            # but by design this means offset = N-1
            while offset < self.N-1 and cdf[offset] < u_n:
                offset += 1
            ancestors[n] = offset
