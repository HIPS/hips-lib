"""
Adaptive rejection sampling
"""
import numpy as np
from scipy.misc import logsumexp


class Hull:
    def __init__(self):
        self.m = None
        self.b = None
        self.left = None
        self.right = None
        self.pr = None
        self.lpr = None

class AdaptiveRejectionSampler:
    """
    Class to perform adaptive rejection sampling and keep track of a hull of points.
    """
    def __init__(self, func, lb, ub, initial_points=None, initial_values=None, stepsz=1.0):
        """
        Initialize the sampler
        """
        self.lb = lb
        self.ub = ub
        self.func = func

        if lb >= ub:
            raise Exception('invalid domain')

        if len(initial_points) < 2:
            raise Exception('ARS must be given at least 2 points to start with')

        self.hull_points = []
        self.hull_values = []
        for x,v in zip(initial_points, initial_values):
            if not np.isfinite(x) or not np.isfinite(v):
               raise Exception('Initial points and values must be finite')
            if x < lb or x > ub:
                raise Exception('Invalid initial points')
            self._add_hull_point(x, v)

        self.initialize_hull_points_and_domain(stepsz)

        # initialize a mesh on which to create upper & lower hulls with at least 5 points

        if self.num_hull_points < 5:
            a = self.hull_points[0]
            b = self.hull_points[-1]
            x_prop = a + (b-a)*np.random.rand(5-self.num_hull_points)
            v_prop = func(x_prop)
            for x,v in zip(x_prop, v_prop):
                self._add_hull_point(x, v)

        # Compute the hull object
        self.compute_hulls()

    @property
    def num_hull_points(self):
        return len(self.hull_points)

    def initialize_hull_points_and_domain(self, stepsz=1.0):
        """
        Initialize the hull points and the domain (as necessary)
        """
        left_initialized = False
        while not left_initialized:

            # Fix up the left bound
            if self.lb == -np.Inf:
                # ensure the derivative on the leftmost hull point is positive
                left_hull_pnt = self.hull_points[0]
                left_hull_val = self.hull_values[0]
                lgrad, xp, vxp = self._check_grad(left_hull_pnt, left_hull_val)

                assert np.isfinite(lgrad)
                assert np.isfinite(vxp)
                self._add_hull_point(xp, vxp)

                # If the gradient is not positive, try to step left
                if lgrad < 0:
                    xl = left_hull_pnt - stepsz

                    # Before updating the hull, check that the function is valid here
                    vxl = self.func(xl)
                    # If the function is finite at the new point, add it to the hull and
                    # then continue to check the gradient again
                    if np.isfinite(vxl):
                        self._add_hull_point(xl, vxl)
                        continue
                    # If the function is not finite, move our domain in
                    else:
                        print "Found left bound at ", left_hull_pnt
                        self.lb = left_hull_pnt
            left_initialized = True

        # Initialize the right bounds
        right_initialized = False
        while not right_initialized:
            # Fix up the right bound
            if self.ub == np.Inf:
                # ensure the derivative on the leftmost hull point is positive
                right_hull_pnt = self.hull_points[-1]
                right_hull_val = self.hull_values[-1]
                rgrad, xp, vxp = self._check_grad(right_hull_pnt, right_hull_val)

                assert np.isfinite(rgrad)
                assert np.isfinite(vxp)
                self._add_hull_point(xp, vxp)

                # If the gradient is not negative, try to step right
                if rgrad > 0:
                    xr = right_hull_pnt + stepsz

                    # Before updating the hull, check that the function is valid here
                    vxr = self.func(xr)
                    # If the function is finite at the new point, add it to the hull and
                    # then continue to check the gradient again
                    if np.isfinite(vxr):
                        self._add_hull_point(xr, vxr)
                        continue
                    # If the function is not finite, move our domain in
                    else:
                        self.ub = right_hull_pnt

            right_initialized = True

    def sample(self, debug=False):
        """
        Sample the logpdf
        """
        rejects = 0
        while True:

            # sample x from Hull
            x = self.sample_upper_hull()
            # Evaluate upper and lower hull at x
            lhVal, uhVal = self.eval_hulls(x)

            # Sample under the upper hull at x and see if we accept or reject
            u = np.log(np.random.rand())
            # Three cases for acception/rejection
            if u <= lhVal - uhVal:
                # accept, u is below lower bound
                if debug:
                    print "Sample found after %d rejects" % rejects
                return x

            # Otherwise we must compute the actual function
            vx = self.func(x)
            if u <= vx - uhVal:
                # accept, u is between lower bound and f
                if debug:
                    print "Sample found after %d rejects" % rejects
                return x

            # If we made it this far, we rejected.
            # Now we have another evaluation that we can add to our hull though.
            if np.isfinite(vx):
                self._add_hull_point(x, vx)
            elif x < self.hull_points[0]:
                self.lb = x
            elif x > self.hull_points[-1]:
                self.ub = x
            else:
                # This should never happen for concave functions that are > -inf at the left and right
                # unless it diverges to +inf somewhere in the middle...
                raise Exception("Found invalid point in the middle of the domain!")

            # Recompute the hulls
            self.compute_hulls()

            if debug:
                print 'reject %d' % rejects

            rejects += 1


    def _add_hull_point(self, xp, vxp):
        assert np.isfinite(vxp)
        xs = np.concatenate((self.hull_points, [xp]))
        vxs = np.concatenate((self.hull_values, [vxp]))
        # Sort the points
        perm = np.argsort(xs)
        self.hull_points = xs[perm]
        self.hull_values = vxs[perm]

        if len(self.hull_points) > 40:
            import pdb; pdb.set_trace()



    def _check_grad(self, x, vx=None, deriv_step=1e-3):
        # Check gradients at left and right boundary of domain
        dx = deriv_step + 1e-8*np.random.randn()
        xp = x + dx

        if vx is None:
            vx_given = False
            vx = self.func(x)
        else:
            vx_given = True

        vxp = self.func(xp)

        grad = (vxp-vx)/dx

        if vx_given:
            return grad, xp, vxp
        else:
            return grad, vx, xp, vxp


    def compute_hulls(self):
        # compute lower piecewise-linear hull
        # if the domain of func is unbounded to the left or right, then the lower
        # hull takes on -inf to the left or right of the end points of S
        S = self.hull_points
        fS = self.hull_values
        lowerHull = []
        for li in np.arange(len(S)-1):
            h = Hull()
            h.m = (fS[li+1]-fS[li])/(S[li+1]-S[li])
            h.b = fS[li] - h.m*S[li]
            h.left = S[li]
            h.right = S[li+1]
            lowerHull.append(h)

        # compute upper piecewise-linear hull
        upperHull = []

        if np.isinf(self.lb):
            # first line (from -infinity)
            m = (fS[1]-fS[0])/(S[1]-S[0])
            b = fS[0] - m*S[0]
            # pro = np.exp(b)/m * ( np.exp(m*S[0]) - 0 ) # integrating in from -infinity
            lnpr = b - np.log(m) + m*S[0]
            h = Hull()
            h.m = m
            h.b = b
            h.lnpr = lnpr
            h.left = -np.Inf
            h.right = S[0]
            upperHull.append(h)

        # second line
        m = (fS[2]-fS[1])/(S[2]-S[1])
        b = fS[1] - m*S[1]
        # pro = np.exp(b)/m * ( np.exp(m*S[1]) - np.exp(m*S[0]) )
        lnpr = _signed_lse(m, b, S[1], S[0])
        # Append upper hull for second line
        h = Hull()
        h.m = m
        h.b = b
        h.lnpr = lnpr
        h.left = S[0]
        h.right = S[1]
        upperHull.append(h)

        # interior lines
        # there are two lines between each abscissa
        for li in np.arange(1,len(S)-2):

            m1 = (fS[li]-fS[li-1])/(S[li]-S[li-1])
            b1 = fS[li] - m1*S[li]

            m2 = (fS[li+2]-fS[li+1])/(S[li+2]-S[li+1])
            b2 = fS[li+1] - m2*S[li+1]

            # compute the two lines' intersection
            # Make sure it's in the valid range
            ix = (b1-b2)/(m2-m1)
            if not (ix >= S[li] and ix <= S[li+1]):
                import pdb; pdb.set_trace()
                _ars_plot(upperHull, lowerHull, self.lb, self.ub, S, fS, self.func)

            # pro = np.exp(b1)/m1 * ( np.exp(m1*ix) - np.exp(m1*S[li]) )
            lnpr1 = _signed_lse(m1, b1, ix, S[li])
            h = Hull()
            h.m = m1
            h.b = b1
            h.lnpr = lnpr1
            h.left = S[li]
            h.right = ix
            upperHull.append(h)

            # pro = np.exp(b2)/m2 * ( np.exp(m2*S[li+1]) - np.exp(m2*ix) )
            lnpr2 = _signed_lse(m2, b2, S[li+1], ix)
            h = Hull()
            h.m = m2
            h.b = b2
            h.lnpr = lnpr2
            h.left = ix
            h.right = S[li+1]
            upperHull.append(h)

        # second to last line (m<0)
        m = (fS[-2]-fS[-3])/(S[-2]-S[-3])
        b = fS[-2] - m*S[-2]
        # pro = np.exp(b)/m * ( np.exp(m*S[-1]) - np.exp(m*S[-2]) )
        lnpr = _signed_lse(m, b, S[-1], S[-2])
        h = Hull()
        h.m = m
        h.b = b
        h.lnpr = lnpr
        h.left = S[-2]
        h.right = S[-1]
        upperHull.append(h)

        if np.isinf(self.ub):
            # last line (to infinity)
            m = (fS[-1]-fS[-2])/(S[-1]-S[-2])
            b = fS[-1] - m*S[-1]
            # pro = np.exp(b)/m * ( 0 - np.exp(m*S[-1]) )
            lnpr = b - np.log(np.abs(m)) + m*S[-1]
            h = Hull()
            h.m = m
            h.b = b
            h.lnpr = lnpr
            h.left = S[-1]
            h.right = np.Inf
            upperHull.append(h)


        lnprs = np.array([h.lnpr for h in upperHull])
        lnZ = logsumexp(lnprs)
        prs = np.exp(lnprs - lnZ)
        for (i,h) in enumerate(upperHull):
            h.pr = prs[i]

        if not np.all(np.isfinite(prs)):
            print "ARS prs contains Inf or NaN"
            print lnprs
            print lnZ
            print prs
            import pdb; pdb.set_trace()
            raise Exception("ARS prs contains Inf or NaN")

        self.lower_hull = lowerHull
        self.upper_hull = upperHull

    def sample_upper_hull(self):
        upperHull = self.upper_hull
        prs = np.array([h.pr for h in upperHull])
        if not np.all(np.isfinite(prs)):
            print prs
            raise Exception("ARS prs contains Inf or NaN")

        cdf = np.cumsum(prs)
        if not np.all(np.isfinite(cdf)):
            print cdf
            raise Exception("ARS cumsum Inf or NaN")

        # randomly choose a line segment
        U = np.random.rand()
        for li in np.arange(len(upperHull)):
            if U < cdf[li]:
                break

        # sample along that line segment
        U = np.random.rand()

        m = upperHull[li].m
        b = upperHull[li].b
        left = upperHull[li].left
        right = upperHull[li].right

        # x = np.log( U*(np.exp(m*right) - np.exp(m*left)) + np.exp(m*left) ) / m
        # If we sampled an interior edge then we can do the sampling in a more
        # stable manner.
        if np.isfinite(left) and np.isfinite(right):
            # x = np.log( U*(np.exp(m*right) - np.exp(m*left)) + np.exp(m*left) ) / m
            # x = left + np.log(U*(np.exp(m*(right-left))-1) + 1) / m
            # x = left + np.log(np.exp(np.log(U) + m*(right-left))-np.exp(np.log(U)) + np.exp(0))/ m
            # lnu = np.log(U)
            # v = lnu + m*(right-left)
            # x = left + np.log(np.exp(v) - np.exp(lnu) + np.exp(0))/ m
            # vmax = np.amax([v, lnu, 0])
            # x = left + np.log(np.exp(vmax)*(np.exp(v-vmax) - np.exp(lnu-vmax) + np.exp(-vmax)))/m
            # x = left + vmax + np.log(np.exp(v-vmax) - np.exp(lnu-vmax) + np.exp(-vmax))/m

            # First check if the slope is zero, because then our integrals are undefined
            if np.allclose(m, 0.0):
                # If so, the pdf is flat in this interval so sample uniformly
                x = left + np.random.rand() * (right-left)
            else:
                # Otherwise, inverse sample an exponential distribution
                lnu = np.log(U)
                v = lnu + m*(right-left)
                vmax = np.amax([v, lnu, 0])
                x = left + vmax/m + np.log(np.exp(v-vmax) - np.exp(lnu-vmax) + np.exp(-vmax))/m

        # If the left edge is -Inf, we need to be smarter
        elif np.isinf(left):
            assert m > 0
            x = right + np.log(U) / m
        # Same for the right edge being +Inf
        else:
            # x = np.log( U*(- np.exp(m*left)) + np.exp(m*left) ) / m
            # x = np.log( np.exp(m*left)*(1-U)) / m
            assert m < 0
            x = left + np.log(1-U) / m

        if np.isinf(x) or np.isnan(x):
            # import pdb; pdb.set_trace()

            raise Exception('sampled an infinite or NaN x. Left=%.3f. Right=%.3f. m=%.3f. b=%.3f. cdf=%s. U=%.3f' % (left, right, m, b, str(cdf), U))

        return x

    def eval_hulls(self, x):
        lowerHull = self.lower_hull
        upperHull = self.upper_hull
        # lower bound
        lhVal = None
        if x< np.amin(np.array([h.left for h in lowerHull])):
            lhVal = -np.Inf
        elif x>np.amax(np.array([h.right for h in lowerHull])):
            lhVal = -np.Inf
        else:
            for h in lowerHull:
                left = h.left
                right = h.right

                if x>=left and x<=right:
                    lhVal = h.m*x + h.b
                    break

        # upper bound
        uhVal = None
        for h in upperHull:
            left = h.left
            right = h.right

            if x>=left and x<=right:
                uhVal = h.m*x + h.b
                break

        return lhVal, uhVal

def _signed_lse(m, b, a1, a0):
    """
    Compute log[ e^{b}/m * (e^{m*a1} - e^{m*a2}) ]
    """
    # Make sure that the term inside the log is postiive.
    # If m>0: m*a1 > m*a2, aka a1 > a2
    # if m<0: m*a1 < m*a2, aka a1 > a2
    sgn = np.sign(m)
    assert a1 > a0, "a1 must be greater than a2!"

    if np.allclose(m,0.0):
        return b + np.log(a1-a0)

    # Now we can work with absolute value of m and e^{a1} - e^{a2}
    am = np.maximum(m*a1, m*a0)
    se = np.exp(m*a1-am)-np.exp(m*a0-am)
    lse = b - np.log(m*sgn) + am + np.log(se*sgn)

    if not np.isfinite(lse):
        print "LSE is not finite"
        print "lse: %f" % lse
        print "m: %f" % m
        print "b: %f" % b
        print "a1: %f" % a1
        print "a2: %f" % a0
        print "am: %f" % am
        print "se: %f" % se
        raise Exception("LSE is not finite!")

    return lse

def _ars_plot(upperHull, lowerHull, lb, ub, S, fS, func):
    import matplotlib.pyplot as plt

    Swidth = S[-1]-S[0]
    plotStep = Swidth/1000.0
    ext = 0.15*Swidth; # plot this much before a and past b, if the domain is infinite

    left = S[0]; right = S[-1]
    if np.isinf(lb):
        left -= ext
    if np.isinf(ub):
        right += ext

    x = np.arange(left, right, plotStep)
    fx = func(x)


    plt.plot(x,fx, 'k-')
    plt.plot(S, fS, 'ko')
    plt.title('ARS')

    # plot lower hull
    for h in lowerHull[:-1]:
        m = h.m
        b = h.b

        x = np.arange(h.left, h.right, plotStep)
        plt.plot( x, m*x+b, 'b-' )

    # plot upper bound

    # first line (from -infinity)
    if np.isinf(lb):
        x = np.arange(upperHull[0].right-ext, upperHull[0].right, plotStep)
        m = upperHull[0].m
        b = upperHull[0].b
        plt.plot( x, x*m+b, 'r-')

    # middle lines
    for li in np.arange(1, len(upperHull)-1):

        x = np.arange(upperHull[li].left, upperHull[li].right, plotStep)
        m = upperHull[li].m
        b = upperHull[li].b
        plt.plot( x, x*m+b, 'r-')

    # last line (to infinity)
    if np.isinf(ub):
        x = np.arange(upperHull[-1].left, (upperHull[-1].left+ext), plotStep)
        m = upperHull[-1].m
        b = upperHull[-1].b
        plt.plot( x, x*m+b, 'r-')

    plt.show()

def test_ars_gaussian():
    """
    Test ARS on a Gaussian distribution
    """
    from scipy.stats import norm
    mu = 0
    sig = 1
    p = norm(mu, sig).pdf

    f = lambda x: -0.5*x**2
    x_init = np.array([-2, -0.995, 0, 0.995, 2.0])
    v_init = f(x_init)

    N_samples = 10000
    smpls = np.zeros(N_samples)
    for s in np.arange(N_samples):
        ars = AdaptiveRejectionSampler(f, -np.inf, np.inf, x_init, v_init)
        smpls[s] =  ars.sample()

    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(smpls, 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p(bincenters), 'r--', linewidth=1)
    plt.show()

def test_ars_trunc_gaussian():
    """
    Test ARS on a truncated Gaussian distribution
    """
    from scipy.stats import norm
    mu = 0
    sig = 1
    lb = 0.5
    ub = np.inf
    p = lambda x: norm(mu, sig).pdf(x) / (norm.cdf(ub) - norm.cdf(lb))

    # Truncated Gaussian log pdf
    def f(x):
        x1d = np.atleast_1d(x)
        lp = np.empty_like(x1d)
        oob = (x1d < lb) | (x1d > ub)
        lp[oob] = -np.inf
        lp[~oob] = -0.5 * x1d[~oob]**2
        return lp.reshape(x.shape)

    x_init = np.array([ 0.5, 2.0])
    v_init = f(x_init)

    N_samples = 10000
    smpls = np.zeros(N_samples)
    ars = AdaptiveRejectionSampler(f, lb, ub, x_init, v_init)
    for s in np.arange(N_samples):
        smpls[s] =  ars.sample()

    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(smpls, 25, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p(bincenters), 'r--', linewidth=1)
    plt.show()

if __name__ == '__main__':
    # test_ars_gaussian()
    test_ars_trunc_gaussian()
