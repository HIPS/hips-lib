cdef class InitialDistribution(object):
    cpdef double[:,::1] sample(self, int N=?)


cdef class Proposal:
    cdef public double[::1] ts

    cpdef sample_next(self, double[:,:,::1] z, int i_prev, int[::1] ancestors)
    cpdef logp(self, double[:,::1] z_prev, int i_prev, double[::1] z_curr, double[::1] lp)


cdef class Likelihood:
    cpdef logp(self, double[:,:,::1] z, double[:,::1] x, int i, double[::1] ll)


cdef class ParticleGibbsAncestorSampling(object):
    cdef public int T, N, D
    cdef public double[:,:,::1] z
    cdef public double[:,::1] fixed_particle
    cdef public double[:,::1] x
    cdef public int[:,::1] ancestors
    cdef public double[:,::1] weights

    cdef Proposal prop
    cdef Likelihood lkhd

    cpdef initialize(self,
                     InitialDistribution init,
                     Proposal prop,
                     Likelihood lkhd,
                     double[:,::1] x,
                     double[:,::1] fixed_particle)

    cpdef double[:,::1] sample(self)

    cpdef double[:,::1] get_trajectory(self, int n)

    cdef systematic_resampling(self, int t, int[::1] ancestors)
