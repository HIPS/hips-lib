import numpy as np
from scipy.misc import logsumexp

import numpy as np
cimport numpy as np

cdef class Proposal:
    """
    General wrapper for a proposal distribution.
    It must support efficient log likelihood calculations and sampling.
    Extend this for the proposal of interest
    """

    # ts:   Length T vector of times at which the proposals will be sampled
    cdef public double[::1] ts

    cpdef sample_next(self, double[:,:,::1] z, int i_prev):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        pass

    cpdef logp(self, double[:,::1] z_prev, int i_prev, double[:,::1] z_curr, double[:,::1] lp):
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
    # TxNxD buffer of latent state particles to be filled in
    # T:    number of time bins
    # N:    number of particles
    # D:    dimensionality of each particle
    cdef public double[:,:,::1] z

    cdef public double[:,::1] fixed_particle

    # TxO buffer of observations
    # O:    dimensionality of each observation
    cdef public double[:,::1] x

    # TxN buffer of "ancestors" of each particle
    cdef public int[:,::1] ancestors

    # TxN buffer of "weights" of each particle
    cdef public double[:,::1] weights

    cdef Proposal prop
    cdef Likelihood lkhd

    def __init__(self,
                 T, N, D
                 ):
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

        # We should probably do this the other way around since
        # T > Np > D in most cases!
        self.z = np.zeros((T,N,D), dtype=np.float)
        self.ancestors = np.zeros((T,N), dtype=np.int)
        self.weights = np.zeros((T,N), dtype=np.float)


    cpdef initialize(self,
                     Proposal proposal,
                     Likelihood lkhd,
                     double[:,::1] x,
                     p_initial,
                     fixed_particle):
        """

        proposal:    A proposal distribution for new particles given old
        lkhd:        An observation likelihood model for particles
        p_initial:   Distribution over initial states
        z_fixed:     Particle representing current state of Markov chain
        """
        # Set the observations
        assert x.shape[0] == self.T and x.ndim == 2, "Invalid observation shape"
        self.x = x
        self.proposal = proposal
        self.lkhd = lkhd

        # Store the fixed particle
        assert fixed_particle.shape == (self.T,self.D)
        self.fixed_particle = fixed_particle

        # Let the first particle correspond to the fixed particle
        self.z[:,0,:] = self.fixed_particle
        # Sample the initial state
        self.z[0,1:,:] = p_initial.sample(Np=self.N-1)

        # Initialize weights according to observation likelihood
        ll0 = np.zeros(self.N, dtpye=np.float)
        self.lkhd.logp(self.z, self.x, 0, ll0)
        self.weights[0,:] = np.exp(ll0 - logsumexp(ll0))
        self.weights[0,:] = self.weights[:,0] / np.sum(self.weights[0,:])


    cpdef double[:,::1] sample(self):
        """
        Sample a new set of particle trajectories
        """

        # Allocate memory
        cdef double[:,::1] lp_trans = np.zeros((self.N,1))
        cdef double[::1] ll_obs = np.zeros((self.N,))
        cdef double[::1] w_as
        cdef int t, n

        for t in range(1, self.T):

            # TODO: Implement resampling!
            # First, resample the previous parents
            # self._resample_particles(t)

            # Move each particle forward according to the proposal distribution
            self.proposal.sample_next(self.z, t-1)

            # Override the first particle with the fixed particle
            self.z[t,0,:] = self.fixed_particle[t,:]

            # Resample the parent index of the fixed particle
            self.proposal.logp(self.z[t-1,:,:], t-1, self.z[t,:1,:], lp_trans)
            lp_trans += np.log(self.weights[t-1,:])


            w_as = np.exp(lp_trans - logsumexp(lp_trans))
            w_as /= np.sum(w_as)
            self.ancestors[t,0] = np.random.choice(np.arange(self.N), p=w_as)

            # Update the weights. Since we sample from the prior,
            # the particle weights are always just a function of the likelihood
            self.lkhd.logp(self.z, self.x, t, ll_obs)
            w = np.exp(ll_obs - logsumexp(ll_obs))
            w /= w.sum()
            self.weights[t,:] = w

        # Sample a trajectory according to the final weights
        n = np.random.choice(np.arange(self.N), p=self.weights[-1,:])
        return self.get_trajectory(n)

    cpdef double[:,::1] get_trajectory(self, int n):
        # Compute the i-th trajectory from the particles and ancestors
        cdef int T = self.T
        cdef double[:,::1] traj = np.zeros((T, self.D))
        # traj = np.zeros((T, self.D))
        traj[T-1,:] = self.z[T-1,n,:]
        curr_ancestor = self.ancestors[T-1,n]

        for t in range(T-1,-1,-1):
            traj[t,:] = self.z[t,curr_ancestor,:]
            curr_ancestor = self.ancestors[t,curr_ancestor]
        return traj
    #
    # # @property
    # # def trajectories(self):
    # #     # Compute trajectories from the particles and ancestors
    # #     T = self.offset
    # #
    # #     if not np.allclose(self.particles[:,0,:T], self.fixed_particle[:,:T]):
    # #         import pdb; pdb.set_trace()
    # #
    # #     x = np.zeros((self.D, self.Np, T))
    # #     x[:,:,T-1] = self.particles[:,:,T-1]
    # #     curr_ancestors = self.ancestors[:,T-1]
    # #
    # #     for t in np.arange(T-1)[::-1]:
    # #         x[:,:,t] = self.particles[:,curr_ancestors,t]
    # #         curr_ancestors = self.ancestors[curr_ancestors, t]
    # #     return x
    #
    # @property
    # def trajectory_weights(self):
    #     return self.weights[:, -1]
    #
    # def sample_trajectory(self):
    #     # Sample a particular weight trace given the particle weights at time T
    #     # i = np.sum(np.cumsum(self.trajectory_weights) < np.random.rand()) - 1
    #     n = np.random.choice(np.arange(self.N), p=self.trajectory_weights)
    #     return self.trajectories[:,i,:]
    #
    # # def _resample(self, w, method='lowvariance'):
    # #     # Resample all but the fixed particle
    # #     assert method in ['lowvariance','independent']
    # #     if method is 'lowvariance':
    # #         sources = self._lowvariance_sources(w, self.Np)
    # #     if method is 'independent':
    # #         sources = self._independent_sources(w, self.Np)
    # #
    # #     return sources
    # #
    # # def _independent_sources(self, w, num):
    # #     # Return an ordered array of source indices from source counts
    # #     # e.g. if the sources are 3x'0', 2x'1', 0x'2', and 1x'3', as specified
    # #     # by the vector [3,2,0,1], then the output will be
    # #     # [0, 0, 0, 1, 1, 3]
    # #     return ibincount(np.random.multinomial(num,w))
    # #
    # # def _lowvariance_sources(self, w, num):
    # #     r = np.random.rand()/num
    # #     bins = np.concatenate(((0,),np.cumsum(w)))
    # #     return ibincount(np.histogram(r+np.linspace(0,1,num,endpoint=False), bins)[0])
    #
