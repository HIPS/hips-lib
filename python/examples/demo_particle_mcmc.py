"""
A simple demo of a particle MCMC implementation
"""
from hips.inference.particle_mcmc cimport Proposal, Likelihood, ParticleGibbsAncestorSampling

class LinearDynamicalSystemProposal(Proposal):
    def __init__(self, dzdt, noiseclass):
        self.dzdt = dzdt
        self.noisesampler = noiseclass

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