"""
Slice sampling implementation from Ryan Adams
http://machinelearningmisc.blogspot.com/

Scott Linderman
slinderman@seas.harvard.edu
2013-2014
"""
import numpy as np

def slicesample(xx, llh_func, last_llh=None, step=1, step_out=True, x_l=None, x_r=None, lb=-np.Inf, ub=np.Inf):
    xx = np.atleast_1d(xx)
    dims = xx.shape[0]
    perm = range(dims)
    np.random.shuffle(perm)

    assert xx < ub
    assert xx > lb

    if isinstance(step, int) or isinstance(step, float) or \
        isinstance(step, np.int) or isinstance(step, np.float):
        step = np.tile(step, dims)
    elif isinstance(step, tuple) or isinstance(step, list):
        step = np.array(step)
 
    if last_llh is None:
        last_llh = llh_func(xx)
 
    for d in perm:
        llh0 = last_llh + np.log(np.random.rand())
        rr = np.random.rand()
        if x_l is None:
            x_l    = xx.copy()
            x_l[d] = max(x_l[d] - rr*step[d], lb)
        else:
            x_l = np.atleast_1d(x_l)
            assert x_l.shape == xx.shape
            assert np.all(x_l <= xx)
        if x_r is None:
            x_r    = xx.copy()
            x_r[d] = min(x_r[d] + (1-rr)*step[d], ub)
        else:
            x_r = np.atleast_1d(x_r)
            assert x_r.shape == xx.shape
            assert np.all(x_r >= xx)
         
        if step_out:
            llh_l = llh_func(x_l)
            while llh_l > llh0 and x_l[d] > lb:
                x_l[d] = max(x_l[d] - step[d], lb)
                llh_l  = llh_func(x_l)
            llh_r = llh_func(x_r)
            while llh_r > llh0 and x_r[d] < ub:
                x_r[d] = min(x_r[d] + step[d], ub)
                llh_r  = llh_func(x_r)


        assert np.isfinite(llh0)

        x_cur = xx.copy()
        n_steps = 0
        while True:
            xd = np.random.rand()*(x_r[d] - x_l[d]) + x_l[d]
            x_cur[d] = xd
            last_llh = llh_func(x_cur)
            if last_llh > llh0:
                xx[d] = xd
                break
            elif xd > xx[d]:
                x_r[d] = xd
            elif xd < xx[d]:
                x_l[d] = xd
            else:
                raise Exception("Slice sampler shrank too far.")
            n_steps += 1


    if not np.isfinite(last_llh):
        raise Exception("Likelihood is not finite at sampled point")

    return xx, last_llh


def test_slicesample():
    from scipy.stats import gamma
    import matplotlib.pyplot as plt

    n_iter = 1000

    # Gamma distribution (bounded on left)
    print "Gamma test"
    g = gamma(2.0, loc=0., scale=2.0)

    smpls = np.zeros(n_iter)
    smpls[0] = g.rvs(1)
    for n in np.arange(1,n_iter):
        sn, _ = slicesample(smpls[n-1], g.logpdf, lb=1e-5)
        smpls[n] = sn

    print "Expected gamma mean: ", g.mean()
    print "Inferred gamma mean: ", smpls.mean()
    print "Expected gamma std:  ", g.std()
    print "Inferred gamma std:  ", smpls.std()

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(1e-5, g.mean() + 4*g.std(), 1000)
    ax.plot(x, g.pdf(x), 'k-', lw=2, label='true pdf')
    ax.hist(smpls, 25, normed=True, alpha=0.2)
    ax.legend(loc='best', frameon=False)
    plt.show()

