import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from gaussian_lds import *
from hips.inference.particle_mcmc import *

def sample_gaussian_lds(plot=False):
    T = 1000
    N = 1
    D = 2
    dt = 1
    t = dt * np.arange(T)

    # Rotational dynamics matrix
    th = np.pi/12.0
    A = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    sigma = 0.1

    # Direct observation matrix
    C = np.eye(D)
    eta = 0.2

    init = GaussianInitialDistribution(np.ones(D), 1*np.eye(D))
    prop = LinearGaussianDynamicalSystemProposal(A, sigma)
    lkhd = LinearGaussianLikelihood(C, eta)
    z = np.zeros((T,N,D))
    z[0,0,:] = init.sample()
    x = np.zeros((T,D))

    # Sample the latent state sequence
    for i in np.arange(0,T-1):
        # The interface kinda sucks. We have to tell it that
        # the first particle is always its ancestor
        prop.sample_next(z, i, np.array([0], dtype=np.int32))

    # Sample observations
    for i in np.arange(0,T):
        lkhd.sample(z,x,i,0)

    # Extract the first (and in this case only) particle
    z = z[:,0,:]

    # Plot the first particle trajectory
    if plot:
        plt.ion()
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        plt.plot(z[:,0], z[:,1],'k')
        plt.plot(x[:,0], x[:,1], 'ro')
        plt.plot(z[0,0], z[0,1], 'ko', markersize=12, markerfacecolor='none')
        plt.plot(z[-1,0], z[-1,1], 'x', markersize=12, markerfacecolor='none')

    return z, x, init, prop, lkhd

def sample_z_given_x(z_curr, x,
                     init, prop, lkhd,
                     N_particles=100,
                     plot=False):

    T,D = z_curr.shape
    T,O = x.shape
    # import pdb; pdb.set_trace()
    pf = ParticleGibbsAncestorSampling(T, N_particles, D)
    pf.initialize(init, prop, lkhd, x, z_curr)

    S = 500
    z_smpls = np.zeros((S,T,D))
    for s in range(S):
        # Reinitialize with the previous particle
        pf.initialize(init, prop, lkhd, x, z_smpls[s,:,:])
        z_smpls[s,:,:] = pf.sample()

    z_mean = z_smpls.mean(axis=0)
    z_std = z_smpls.std(axis=0)
    z_env = np.zeros((T*2,2))

    z_env[:,0] = np.concatenate((z_mean[:,0] + z_std[:,0], z_mean[::-1,0] - z_std[::-1,0]))
    z_env[:,1] = np.concatenate((z_mean[:,1] + z_std[:,1], z_mean[::-1,1] - z_std[::-1,1]))

    # import pdb; pdb.set_trace()
    if plot:
        plt.gca().add_patch(Polygon(z_env, facecolor='b', alpha=0.25, edgecolor='none'))
        plt.plot(z_mean[:,0], z_mean[:,1], 'b', lw=2)


        # Plot a few random samples
        for s in range(10):
            si = np.random.randint(S)
            plt.plot(z_smpls[si,:,0], z_smpls[si,:,1], '-b', lw=0.5)

        plt.ioff()
        plt.show()

    return z_smpls


def demo():
    z, x, init, prop, lkhd = sample_gaussian_lds(plot=True)
    sample_z_given_x(z, x, init, prop, lkhd, plot=True)

def profile_demo():
    import pstats, cProfile
    z, x, init, prop, lkhd = sample_gaussian_lds()

    cProfile.runctx("sample_z_given_x(z, x, init, prop, lkhd)", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

profile_demo()
demo()