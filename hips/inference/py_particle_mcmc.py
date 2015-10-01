import abc
import numpy as np


from pybasicbayes.utils.general import ibincount


###############################################################################
# Particle filtering code
###############################################################################
class InitialDistribution(object):
    """
    Abstract base class for distributions
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample(self, Np=1):
        """
        sample the initial particle distribution
        """
        pass

class Proposal(object):
    """
    General wrapper for a proposal distribution. It must support efficient
    log likelihood calculations and sampling.
    Extend this for the proposal of interest
    """
    __metaclass__ == abc.ABCMeta

    @abc.abstractmethod
    def sample_next(self, t_prev, Z_prev, t_next):
        """ Sample the next state Z given Z_prev
        """
        pass

    @abc.abstractmethod
    def logp(self, t_prev, z_prev, t_next, z_next):
        pass


class Likelihood(object):
    """
    General wrapper for a proposal distribution. It must support efficient
    log likelihood calculations and sampling.
    Extend this for the proposal of interest
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def logp(self, X, Z):
        """ Compute the log probability of X given Z
        """
        pass

class ConditionalParticleFilterWithAncestorSampling(object):
    """
    A conditional particle filter with ancestor sampling
    """
    def __init__(self,
                 D,
                 T,
                 t0,
                 X0,
                 proposal,
                 lkhd,
                 p_initial,
                 fixed_particle,
                 Np=100):
        """
        Initialize the particle filter with:
        D:           Dimensionality of latent state space
        T:           Number of time steps
        proposal:    A proposal distribution for new particles given old
        lkhd:        An observation likelihood model for particles
        p_initial:   Distribution over initial states
        fixed_particle: Particle representing current state of Markov chain
        Np:          The number of paricles
        """
        # Initialize data structures for particles and weights
        self.D = D
        self.T = T
        self.Np = Np
        self.proposal = proposal
        self.lkhd = lkhd

        # Store the particles in a (D x Np) matrix for fast, vectorized
        # proposals and likelihood evaluations
        self.times = np.zeros(T)
        self.particles = np.zeros((T,Np,D))
        self.ancestors = np.zeros((T,Np), dtype=np.int)
        self.weights = np.zeros((T,Np))
        self.logweights = np.zeros((T,Np))

        # Keep track of the times when the filter has been called
        self.times[0] = t0

        # Store the fixed particle
        if not fixed_particle.shape == (T,D):
            raise Exception("fixed particle shape is not equal to %s" % str((T,D)))
        self.fixed_particle = fixed_particle

        # Let the first particle correspond to the fixed particle
        # Sample the initial state
        self.particles[0,0] = self.fixed_particle[0]
        self.particles[0,1:] = p_initial.sample(Np=self.Np-1)

        # Initialize weights according to observation likelihood
        log_W = self.lkhd.logp(X0, self.particles[0,:,:])
        self.logweights[0] = log_W
        self.weights[0] = np.exp(log_W - np.logaddexp.reduce(log_W))
        self.weights[0] /= self.weights[0].sum()

        # Increment the offset to point to the next particle slot
        self.offset = 1

    def filter(self,
               t_next,
               X_next,
               resample_method='independent'):
        """
        Filter a given observation sequence to get a sequence of latent states, Z.
        """
        # Save the current filter time
        self.times[self.offset] = t_next

        # First, ressample the previous parents
        curr_ancestors = self._resample(self.weights[self.offset-1,:],
                                        resample_method)

        # print "Num ancestors: %d" % len(np.unique(curr_ancestors))
        # TODO Is permutation necessary?
        curr_ancestors = np.random.permutation(curr_ancestors)

        # Move each particle forward according to the proposal distribution
        self.particles[self.offset] = self.proposal.sample_next(self.times[self.offset-1],
                                                                self.particles[self.offset-1,curr_ancestors],
                                                                t_next)

        # Override the first particle with the fixed particle
        self.particles[self.offset, 0] = self.fixed_particle[self.offset]

        # Resample the parent index of the fixed particle
        logw_as = np.log(self.weights[self.offset-1]) + \
                  self.proposal.logp(self.times[self.offset-1],
                                     self.particles[self.offset-1],
                                     t_next,
                                     self.fixed_particle[self.offset].reshape(self.D,1)
                                     )

        logw_as = logw_as.ravel()
        assert logw_as.ndim == 1
        w_as = np.exp(logw_as - np.logaddexp.reduce(logw_as))
        w_as /= w_as.sum()

        if not np.allclose(w_as.sum(), 1.0):
            import pdb; pdb.set_trace()

        curr_ancestors[0] = np.random.choice(np.arange(self.Np), p=w_as)
        # curr_ancestors[0] = np.sum(np.cumsum(w_as) < np.random.rand()) - 1

        # print "Average ancestor %d: %.3f" % (self.offset, np.mean(self.particles[:,curr_ancestors,self.offset-1]))
        # print "Std ancestor %d: %.3f" % (self.offset, np.std(self.particles[:,curr_ancestors,self.offset-1]))

        # Save the ancestors
        self.ancestors[self.offset] = curr_ancestors

        # if np.any(np.abs(self.particles[:,:,self.offset] - self.particles[:,curr_ancestors, self.offset-1]) > 0.2):
        #     import pdb; pdb.set_trace()

        # Update the weights. Since we sample from the prior,
        # the particle weights are always just a function of the likelihood
        log_W = self.lkhd.logp(X_next, self.particles[self.offset])
        self.logweights[self.offset] = log_W
        
        if np.all(np.isinf(log_W)):
            w = np.ones_like(log_W)
        else:
            w = np.exp(log_W - np.logaddexp.reduce(log_W))
        w /= w.sum()
        self.weights[self.offset] = w

        # Increment the offset
        self.offset += 1

    @property
    def marginal_log_likelihood(self):
        """
        Compute the marginal log likelihood of the data by summing the
        particle weights
        """
        # First compute the marginal log likelihood of each time bin
        # by taking the average of the unnormalized weights
        # p(y_t | y_{1:t-1}) = 1/Np \sum_{p=1}^Np w_t(X_{1:t}^p)
        # Do log sum exp manually
        maxlw = np.amax(self.logweights, axis=0)
        mll = -np.log(self.Np) + maxlw + np.log(np.sum(np.exp(self.logweights-maxlw[None,:]), axis=0))

        # Now sum the log likelihood from each time bin
        return mll.sum()

    @property
    def trajectories(self):
        # Compute trajectories from the particles and ancestors
        T = self.offset

        if not np.allclose(self.particles[:T,0], self.fixed_particle[:T]):
            import pdb; pdb.set_trace()

        x = np.zeros((self.D, self.Np, T))
        x[T-1] = self.particles[T-1]
        curr_ancestors = self.ancestors[T-1]

        for t in np.arange(T-1)[::-1]:
            x[t] = self.particles[t,curr_ancestors]
            curr_ancestors = self.ancestors[t,curr_ancestors]

        return x

    @property
    def trajectory_weights(self):
        return self.weights[self.offset-1]

    def sample_trajectory(self):
        # Sample a particular weight trace given the particle weights at time T
        # i = np.sum(np.cumsum(self.trajectory_weights) < np.random.rand()) - 1
        i = np.random.choice(np.arange(self.Np), p=self.trajectory_weights)
        # i = np.argmax(self.trajectory_weights)
        # print "Sampled trajectory %d with weight %.3f" % (i, self.trajectory_weights[i])
        return self.trajectories[:self.offset,i].reshape((self.offset, self.D))

    def _resample(self, w, method='lowvariance'):
        # Resample all but the fixed particle
        assert method in ['lowvariance','independent']
        if method is 'lowvariance':
            sources = self._lowvariance_sources(w, self.Np)
        if method is 'independent':
            try:
                sources = self._independent_sources(w, self.Np)
            except:
                import pdb; pdb.set_trace()

        return sources

    def _independent_sources(self, w, num):
        # Return an ordered array of source indices from source counts
        # e.g. if the sources are 3x'0', 2x'1', 0x'2', and 1x'3', as specified
        # by the vector [3,2,0,1], then the output will be
        # [0, 0, 0, 1, 1, 3]
        return ibincount(np.random.multinomial(num,w))

    def _lowvariance_sources(self, w, num):
        r = np.random.rand()/num
        bins = np.concatenate(((0,),np.cumsum(w)))
        return ibincount(np.histogram(r+np.linspace(0,1,num,endpoint=False),bins)[0])

