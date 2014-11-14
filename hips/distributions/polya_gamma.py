import numpy as np
import numpy.random as npr

def polya_gamma(a, c, trunc=500, biased=False):
    """
    Sample a Polya-gamma distribution. Ported from Mingyuan Zhou's code at:
    http://mingyuanzhou.github.io/Code.html

    Details are described in the Appendix for
     [1] M. Zhou, L. Li, D. Dunson and L. Carin, "Lognormal and
         Gamma Mixed Negative Binomial Regression," in ICML 2012.


    :param a: shape of the gamma distributions
    :param c: something like a scale parameter
    :param trunc: truncation level
    :param biased: whether or not to accept biased samples
    :return: a sample from the Polya-gamma distribution
    """

    # Make sure a and c are vectors of the same shape
    assert (np.isscalar(a) and np.isscalar(c)) or \
           (a.ndim == 1 and c.ndim==1 and a.shape == c.shape)

    # Compute the effective size for sampling
    aa = np.asarray(a)
    cc = np.asarray(c)
    N = a.size

    # Precompute fixed values
    Ksq = (np.arange(trunc)+0.5)**2

    # Iterative implementation
    # for i in range(eff_size):
    #     A = npr.gamma(aa[i] * np.ones(K),1.0)
    #     D = Ksq + cc[i]**2/4.0/np.pi**2
    #     x[i] = 0.5/np.pi**2*np.sum(A/D)
    #
    #     # Bias correction
    #     if not biased:
    #         temp = max(abs(cc[i]/2.0), 1e-8)
    #         xmeanfull = np.tanh(temp)/(temp)/4.0
    #         xmeantruncate = 0.5/np.pi**2*np.sum(1./D)
    #         x[i] = x[i] * xmeanfull / xmeantruncate

    # Vectorized implementation
    A = npr.gamma(aa[:, None] * np.ones((1,trunc)),1.0)
    D = Ksq[None,:] + cc[:,None]**2/4.0/np.pi**2
    x = 0.5/np.pi**2*np.sum(A/D, axis=1)

    assert x.shape == (N,)

    # Bias correction
    if not biased:
        temp = np.maximum(abs(cc/2.0), 1e-8)
        xmeanfull = np.tanh(temp)/(temp)/4.0
        xmeantruncate = 0.5/np.pi**2*np.sum(1./D, axis=1)
        x = x * xmeanfull / xmeantruncate

    # Convert to scalar if necessary
    if np.isscalar(a):
        return np.asscalar(x)
    else:
        return x


def test_polya_gamma():
    a = 1.0 * np.ones(1000)
    c = 0.0 * np.ones(1000)

    import cProfile, StringIO, pstats
    pr = cProfile.Profile()
    pr.enable()
    x = polya_gamma(a, c, 10)
    pr.disable()

    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.hist(x, 20, normed=True)
    # plt.show()

# test_polya_gamma()