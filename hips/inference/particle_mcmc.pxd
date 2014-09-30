cdef class InitialDistribution(object):
    cpdef double[:,::1] sample(self, int N=?)


cdef class Proposal:
    cdef public double[::1] ts

    cpdef sample_next(self, double[:,:,::1] z, int i_prev, int[::1] ancestors)
    cpdef logp(self, double[:,::1] z_prev, int i_prev, double[::1] z_curr, double[::1] lp)


cdef class Likelihood:
    cpdef logp(self, double[:,:,::1] z, double[:,::1] x, int i, double[::1] ll)


cdef class ParticleGibbsAncestorSampling(object):
    # Class variables are defined in the pxd file

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

    # T:    number of time bins
    # N:    number of particles
    # D:    dimensionality of each particle
    cdef public int T, N, D

    # z:    TxNxD array of latent state particles to be filled in
    cdef public double[:,:,::1] z

    # The particle representing the previous MCMC sample
    cdef public double[:,::1] fixed_particle

    # x:    TxO array of observatoins
    # O:    dimensionality of each observation
    cdef public double[:,::1] x

    # ancs  TxN array of ancestor indices
    #       ancestor[t,n]=j means that the n-th particle at time t
    #       was parented by the j-th particle at time t-1
    cdef public int[:,::1] ancestors

    # wghts TxN array of normalized particle weights
    cdef public double[:,::1] weights

    # init  InitialDistribution object to sample z0
    cdef InitialDistribution init

    # prop  Proposal object for advancing particles
    cdef Proposal prop

    # lkhd  Likelihood object for evaluating likelihood of observations
    cdef Likelihood lkhd

    # Initialize the particle Gibbs with ancestor sampling object
    # This could really happen in the constructor, but it seems to
    # make sense to keep one object and reinitialize with updated
    # Proposals and Likelihoods at each MCMC iteration
    cpdef initialize(self,
                     InitialDistribution init,
                     Proposal prop,
                     Likelihood lkhd,
                     double[:,::1] x,
                     double[:,::1] fixed_particle)

    # Take one step of the MCMC chain with the PGAS
    cpdef double[:,::1] sample(self)

    # Get a specific trajectory (mostly for visualization)
    cpdef double[:,::1] get_trajectory(self, int n)

    # Resample particles and set the ancestors for time t
    cdef systematic_resampling(self, int t)
