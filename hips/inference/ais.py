"""
Simple demo to play around with annealed importance sampling. Probably not much
here to be used directly since you'll need to rewrite the samplers for each model,
but perhaps still a useful reference.
"""
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt

from mh import mh
from hmc import hmc

# Let's work with a Gaussian-Gaussian model. Say the mean of the Gaussian is
# distributed according to a mean zero Gaussian, and the likelihood is a
# product of Gaussian terms arising from datapoints drawn from a Gaussian with
# true mean 1 and (known) standard deviation 0.1.
#
# For model comparison, we'll investigate the model evidence for a range of
# variances in the prior for mu.

# Define the prior distribution p_n to be \mu ~ Gaussian(0,eta**2)
# With normalizing constant Z_n = 1/sqrt(2 * pi * 1**2)
log_prior = lambda mu, eta: -0.5*np.log(2*np.pi*eta**2)  +  -0.5/eta**2 * mu**2
g_log_prior = lambda mu, eta: -mu/eta**2
# Define a likelihood to be a product of Gaussian observations
# This must be normalized in order for the estimate of the marginal log
# likelihood to be correct.
sigma = 0.1
# x = np.array([0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1]) - 0.9
x = np.array([0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1])
# x = np.array([1.0])
log_lkhd = lambda mu: x.size * -0.5*np.log(2*np.pi*sigma**2) + \
                      reduce(lambda ll,xn: ll + -0.5/sigma**2 *(xn-mu)**2, x, 0)
g_log_lkhd = lambda mu: reduce(lambda ll,xn: ll + -1.0/sigma**2 *(xn-mu), x, 0)

# Log posterior (that we sample from) is the sum of the prior and the lkhd
log_posterior = lambda mu, eta: log_lkhd(mu) + log_prior(mu, eta)
g_log_posterior = lambda mu, eta: g_log_lkhd(mu) + g_log_prior(mu, eta)

# Now compute the evidence with annealed importance sampling.
# First we need to specify a set of betas, i.e. mixing weights that
# we use to geometrically weight the prior and the posterior. The idea
# is that we'll start by sampling from the prior and then "anneal" our
# way toward the posterior. We'll do this a number of times to generate
# samples from the posterior and then use the importance weights of each
# sample to approximate the model evidence. Hopefully it will look like
# the evidence that we manually computed!

# As mentioned, our intermediate distributions will be weighted combos
# of the prior and the posterior:
# f_j \propto (posterior)^\beta_j * (prior)^(1-\beta_j)
#           = (lkhd)^\beta_j * prior
# \log f_j  = \beta_j * log_lkhd + log_prior
N = 100
betas = np.linspace(0,1,N)

# We'll use Metropolis Hastings to sample the intermediate distributions
# Our proposal will be a Gaussian with std 0.05.
q_std = 0.05
q_sample = lambda mu: mu + q_std * np.random.randn()
q = lambda x0, xf: 1.0

def ais(eta):
    # Run the annealed importance sampler to generate M samples
    M = 100
    log_weights = np.zeros(M)

    # Make a sequence of intermediate functions
    f = lambda beta, mu: beta * log_posterior(mu, eta) + (1.0-beta) * log_prior(mu, eta)

    # Sample m points
    for m in range(M):
        print "M: %d" % m
        # Sample mus from each of the intermediate distributions,
        # starting with a draw from the prior.
        mus = np.zeros(N)
        mus[0] = eta*np.random.randn()

        # Ratios correspond to the 'f_{n-1}(x_{n-1})/f_{n}(x_{n-1})' values in Neal's paper
        ratios = np.zeros(N-1)

        # Sample the intermediate distributions
        for (n,beta) in zip(range(1,N), betas[1:]):
            # print "M:%d\tN%d" % (m,n)
            # Sample from the intermediate distribution with mixing weight beta
            # fj = lambda mu: beta * log_posterior(mu, eta) + (1.0-beta) * log_prior(mu, eta)
            fj = lambda mu: f(beta, mu)
            gfj = lambda mu: beta * g_log_posterior(mu, eta) + (1.0-beta) * g_log_prior(mu, eta)

            # We can either use Metropolis Hastings or HMC to sample mu
            mus[n] = mh(mus[n-1], fj, q, q_sample, steps=10)[-1]
            # mus[n] = hmc(lambda mu: -1.0*fj(mu),
            #              lambda mu: -1.0*gfj(mu),
            #              0.01, 10,
            #              np.atleast_1d(mus[n-1]))

            # Compute the ratio of this sample under this distribution and the previous distribution
            fjm1 = lambda mu: f(betas[n-1], mu)
            ratios[n-1] = fj(mus[n]) - fjm1(mus[n])

        # Compute the log weight of this sample
        log_weights[m] = np.sum(ratios)

    # Compute the mean of the weights to get an estimate of the normalization constant
    log_Z = -np.log(M) + logsumexp(log_weights)
    return log_Z

# Compute the posterior by hand for this model
# Remember the prior mean is zero so it does not show up in the posterior mean
posterior_var = lambda eta: 1.0/(1.0/eta**2 + x.size/sigma**2)
posterior_mean = lambda eta: (x.sum() / sigma**2) * posterior_var(eta)
true_log_posterior = lambda mu, eta: -0.5*np.log(2*np.pi*posterior_var(eta)) \
                                -0.5/posterior_var(eta) * (mu-posterior_mean(eta))**2


# Compute the true normalization constant (i.e. model evidence) for a variety
# of prior variances, eta.
etas = np.linspace(0.25, 2, 10)
true_evidence = np.zeros_like(etas)
ais_evidence = np.zeros_like(etas)
for j,eta in enumerate(etas):
    # Compute true evidence. We can calculate it in closed form or by plugging in mu's.
    # Since the evidence is not a function of mu, it should be the same for all mus.
    print 'eta: %f' % eta
    mu = 0.0
    true_evidence[j] = log_prior(mu, eta) + log_lkhd(mu) - true_log_posterior(mu, eta)

    # Compute the evidence with Annealed Importance Sampling (AIS)
    ais_evidence[j] = ais(eta)

plt.figure()
plt.plot(etas, true_evidence, label='True')
plt.plot(etas, ais_evidence, label='AIS')
plt.xlabel('$\\eta$')
plt.ylabel('Evidence')
plt.legend(loc='lower right')
plt.show()

