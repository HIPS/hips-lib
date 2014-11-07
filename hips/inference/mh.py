import numpy as np

def mh(x0, p, q, sample_q, steps=1):
    """
    Metropolis Hastings sampling from p with proposal distribution q and initial
    state x0.

    :param x0:           initial state
    :param p(x):         log probability density to sample from
    :param q(x0,xf):     log proposal density of going x0->xf
    :param sample_q(x0): function to propose xf given x0
    :param steps:        num MH steps to take
    :return:             list of samples of length 'steps'
    """
    xs = [] * steps
    x = x0
    for step in range(steps):
        # Make a proposal
        p0 = p(x)
        xf = sample_q(x)
        pf = p(xf)
        qf, qr = q(x, xf), q(xf, x)

        # Compute acceptance ratio and accept or reject
        odds = pf + qr - p0 - qf
        if np.log(np.random.rand()) < odds:
            x = xf

        xs.append(x)
    if steps > 1:
        return xs
    else:
        return xs[0]

def test_mh():
    """
    Test ARS on a Gaussian distribution
    """
    from scipy.stats import norm
    mu = 0
    sig = 1
    p = norm(mu, sig).pdf

    f = lambda x: -0.5*x**2

    # Define a spherical proposal function
    q = lambda x0, xf: 1.0
    sample_q = lambda x: x + 1.*np.random.randn()


    N_samples = 10000
    smpls = np.zeros(N_samples)
    smpls[1:] = mh(smpls[0], f, q, sample_q, N_samples-1)
    # for s in np.arange(1,N_samples):
    #     smpls[s] = mh(smpls[s-1], f, q, sample_q)

    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(smpls, 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p(bincenters), 'r--', linewidth=1)
    plt.show()

if __name__ == '__main__':
    test_mh()
