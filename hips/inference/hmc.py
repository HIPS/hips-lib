"""
Implementation of Hybrid Monte Carlo (HMC) sampling algorithm following Neal (2010).
Use the log probability and the gradient of the log prob to navigate the distribution.

Scott Linderman
slinderman@seas.harvard.edu
2012-2014
"""
import numpy as np

def hmc(U, 
        grad_U, 
        step_sz, 
        n_steps, 
        q_curr, 
        adaptive_step_sz=False,
        tgt_accept_rate=0.9,
        avg_accept_time_const=0.95,
        avg_accept_rate=0.9,
        min_step_sz=0.001,
        max_step_sz=1.0,
        negative_log_prob=True):
    """
    U       - function handle to compute log probability we are sampling
    grad_U  - function handle to compute the gradient of the density with respect 
              to relevant params
    step_sz - step size
    n_steps       - number of steps to take
    q_curr  - current state

    negative_log_prob   - If True, assume U is the negative log prob
    
    """
    # Start at current state
    q = np.copy(q_curr)
    # Momentum is simplest for a normal rv
    p = np.random.randn(*np.shape(q))
    p_curr = np.copy(p)

    # Set a prefactor of -1 if given log prob instead of negative log prob
    pre = 1.0 if negative_log_prob else -1.0

    # Evaluate potential and kinetic energies at start of trajectory
    U_curr = pre * U(q_curr)
    K_curr = np.sum(p_curr**2)/2.0

    # Make a half step in the momentum variable
    p -= step_sz * pre * grad_U(q)/2.0
    
    # Alternate L full steps for position and momentum
    for i in np.arange(n_steps):
        q += step_sz*p
        
        # Full step for momentum except for last iteration
        if i < n_steps-1:
            p -= step_sz * pre * grad_U(q)
        else:
            p -= step_sz * pre * grad_U(q)/2.0
    
    # Negate the momentum at the end of the trajectory to make proposal symmetric?
    p = -p
    
    # Evaluate potential and kinetic energies at end of trajectory
    U_prop = pre * U(q)
    K_prop = np.sum(p**2)/2.0
    
    # Accept or reject new state with probability proportional to change in energy.
    # Ideally this will be nearly 0, but forward Euler integration introduced errors.
    # Exponentiate a value near zero and get nearly 100% chance of acceptance.
    accept = np.log(np.random.rand()) < U_curr-U_prop + K_curr-K_prop
    if accept:
        q_next = q
    else:
        q_next = q_curr

    q_next = q_next.reshape(q.shape)
        
    # Do adaptive step size updates if requested
    if adaptive_step_sz:
        new_accept_rate = avg_accept_time_const * avg_accept_rate + \
                          (1.0-avg_accept_time_const) * accept
        if avg_accept_rate > tgt_accept_rate:
            new_step_sz = step_sz * 1.02
        else:
            new_step_sz = step_sz * 0.98

        new_step_sz = np.clip(new_step_sz, min_step_sz, max_step_sz)

        return (q_next, new_step_sz, new_accept_rate)
    else:
        return q_next

def test_hmc():
    """
    Test HMC on a Gaussian distribution
    """
    from scipy.stats import norm
    mu = 0
    sig = 1
    p = norm(mu, sig).pdf

    f = lambda x: -0.5*x**2
    grad_f = lambda x: -x

    N_samples = 10000
    smpls = np.zeros(N_samples)
    for s in np.arange(1,N_samples):
        smpls[s] = hmc(lambda x: -1.0*f(x),
                       lambda x: -1.0*grad_f(x),
                       0.1, 10,
                       np.atleast_1d(smpls[s-1]),
                       negative_log_prob=True)
    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(smpls, 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p(bincenters), 'r--', linewidth=1)
    plt.show()


def test_gamma_linear_regression_hmc():
    """
    Test ARS on a gamma distributed coefficient for a gaussian noise model
    y = c*x + N(0,1)
    c ~ gamma(2,2)
    """
    a = 6.
    b = 1.
    x = 1
    sig = 1.0
    avg_accept_rate = 0.9
    stepsz = 0.01
    nsteps = 10
    N_samples = 10000

    from scipy.stats import gamma, norm
    g = gamma(a, scale=1./b)
    prior = lambda logc: a * logc -b*np.exp(logc)
    dprior = lambda logc: a -b*np.exp(logc)
    lkhd = lambda logc,y: -0.5/sig**2 * (y-np.exp(logc)*x)**2
    dlkhd = lambda logc,y: 1.0/sig**2 * (y-np.exp(logc)*x) * np.exp(logc)*x
    posterior = lambda logc,y: prior(logc) + lkhd(logc,y)
    dposterior = lambda logc,y: dprior(logc) + dlkhd(logc,y)

    logc_smpls = np.zeros(N_samples)
    y_smpls = np.zeros(N_samples)
    logc_smpls[0] = np.log(g.rvs(1))
    y_smpls[0] = np.exp(logc_smpls[0]*x) + sig*np.random.randn()

    for s in np.arange(1,N_samples):
        # Sample y given c
        y_smpls[s] = np.exp(logc_smpls[s-1])*x + sig*np.random.randn()

        # Sample c given y
        logc_smpls[s], stepsz, avg_accept_rate =  \
            hmc(lambda logc: -1.0*posterior(logc, y_smpls[s]),
                lambda logc: -1.0*dposterior(logc, y_smpls[s]),
                stepsz, nsteps,
                logc_smpls[s-1].reshape((1,)),
                avg_accept_rate=avg_accept_rate,
                adaptive_step_sz=True)

    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(np.exp(logc_smpls), 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, g.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

if __name__ == '__main__':
    test_hmc()
    # test_gamma_linear_regression_hmc()
