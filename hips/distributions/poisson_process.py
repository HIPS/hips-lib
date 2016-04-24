"""
Basic functionality for Poisson processes
"""
import numpy as np

class PoissonProcess(object):

    def __init__(self, lam, t):
        """
        Initialize a Poisson process with a rate and a set of time points
        """
        self.t = t
        self.N_t = len(t)

        # Check that t is sorted
        assert all(t[i] <= t[i+1] for i in xrange(self.N_t-1))

        self.T_start = t[0]
        self.T_stop = t[-1]
        self.lam = lam

    def sample(self):
        """
        Sample an inhomogeneous Poisson process with rate lam and times t using
        the time-rescaling theorem. Integrate the rate from [0,t) to get the cumulative
        rate. Spikes are uniformly drawn from this cumulative rate as in a
        homogeneous process. The spike times in the homogeneous rate can be inverted
        to get spikes from the inhomogeneuos rate.
        """
        t = self.t
        N_t = self.N_t
        lam = self.lam

        # Numerically integrate (using trapezoidal quadrature) lam with respect to tt
        dt = np.diff(t)
        lam = np.squeeze(lam)
        dlam = np.diff(lam)

        # Trapezoidal rule: int(x,t) ~= 0.5 * sum_n=1^N-1 (t[n+1]-t[n])*(x[n+1]+x[n])
        #                             = 0.5 * sum_n=1^N-1 dt[n+1] * (2x[n+1]-dx[n+1])
        # where dt[n+1] = t[n+1]-t[n], dx[n+1] = x[n+1]-x[n]
        trapLam = 0.5*dt*(2*lam[1:]-dlam)
        cumLam = np.ravel(np.cumsum(trapLam))
        cumLam = np.hstack((np.array([0.0]), cumLam))

        # Total area under lam is the last entry in cumLam
        intLam = cumLam[-1]

        # Spikes are drawn uniformly on interval [0,intLam] with rate 1
        # Therefore the number of spikes is Poisson with rate intLam
        N = np.random.poisson(intLam)
        # Call the transformed spike times Q
        Q = np.random.uniform(0, intLam, size=N)
        Q = np.sort(Q)

        # Invert the transformed spike times
        S = np.zeros(N)
        tt_off = 0
        for (n,q) in enumerate(Q):
            while q > cumLam[tt_off]:
                tt_off += 1
                assert tt_off < N_t, "ERROR: inverted spike time exceeds time limit!"

            # q lies in the time between tt[tt_off-1] and tt[tt_off]. Linearly interpolate
            # to find exact time
            q_lb = cumLam[tt_off-1]
            q_ub = cumLam[tt_off]
            q_frac = (q-q_lb)/(q_ub-q_lb)
            assert q_frac >= 0.0 and q_frac <= 1.0, "ERROR: invalid spike index"

            tt_lb = t[tt_off-1]
            tt_ub = t[tt_off]
            S[n] = tt_lb + q_frac*(tt_ub-tt_lb)

        return S

    def _check_spikes_in_range(self, S):
        T_start = self.t[0]
        T_stop = self.t[-1]
        assert np.all(S >= T_start), "ERROR: S lies below range [%f, %f]" % (T_start, T_stop)
        assert np.all(S <= T_stop), "ERROR: S lies above range [%f, %f]" % (T_start, T_stop)

    def nonparametric_fit(self, S, N_bins):
        """
        Approximate the firing rate of a set of spikes by binning into equispaced
        bins and dividing by the bin width. Smooth with a Gaussian kernel.

        TODO: This could be improved with equimass bins as opposed to equispaced bins.
        """
        self._check_spikes_in_range(S)
        T_start = self.T_start
        T_stop = self.T_stop

        # Bin predicted spikes to achieve firing rate
        bin_edges = np.linspace(T_start,T_stop,N_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.0
        bin_width = (T_stop-T_start)/N_bins

        # Approximate firing rate. Histogram returns bin counts and edges as tuple.
        # Take only the bin counts
        fr = np.histogram(S, bin_edges, range=(T_start, T_stop))[0] / bin_width

        # Number of bins to eval kernel at (must be odd to have point at 0)
        N_smoothing_kernel_bins = 9
        smoothing_kernel_bin_centers = np.linspace(-(N_smoothing_kernel_bins-1)/2*bin_width,
                                                   (N_smoothing_kernel_bins-1)/2*bin_width,
                                                   N_smoothing_kernel_bins
                                                   )
        # Evaluate a standard normal pdf at the bin centers. Normalize to one
        # so that we don't change the total area under the firing rate
        smoothing_kernel = np.exp(-0.5*smoothing_kernel_bin_centers**2)
        smoothing_kernel = smoothing_kernel / np.sum(smoothing_kernel)

        # Since the kernel is symmetric we don't have to worry about flipping the kernel left/right
        # Before smoothing, pad fr to minimize boundary effects
        l_pad = np.mean(fr[:N_smoothing_kernel_bins])
        r_pad = np.mean(fr[-N_smoothing_kernel_bins:])
        fr_pad = np.hstack((l_pad*np.ones(N_smoothing_kernel_bins),fr,r_pad*np.ones(N_smoothing_kernel_bins)))
        fr_smooth = np.convolve(fr_pad, smoothing_kernel, "same")

        # Drop the pad components and keep the center of the convolution
        fr_smooth = fr_smooth[N_smoothing_kernel_bins:-N_smoothing_kernel_bins]

        assert np.size(fr_smooth) == N_bins, "ERROR: approximation returned invalid length firing rate"

        # Interpolate to get firing rate at t
        self.lam = np.interp(self.t, bin_centers, fr_smooth)

    def ks_test(self, S):
        """
        KS test to check how well the firing rate explains the observed spikes
        """
        raise NotImplemented('I have code for KS tests but I haven\'t copied it over yet')

class HomogeneousPoissonProcess(PoissonProcess):
    """
    Subclass specifically for homogeneous poisson processes
    """
    def __init__(self, homogeneous_lam, t):
        """
        Initialize a Poisson process with a rate and a set of time points
        """
        lam = self._homogeneous_lam_to_array(homogeneous_lam, t)
        super(HomogeneousPoissonProcess, self).__init__(lam, t)

    def _homogeneous_lam_to_array(self, homogeneous_lam, t):
        # Convert lambda to 2d
        N_t = len(t)
        if np.isscalar(homogeneous_lam) or homogeneous_lam.ndim == 0:
            lam = homogeneous_lam * np.ones(N_t)
        elif homogeneous_lam.ndim == 1:
            assert homogeneous_lam.size == N_t, "Homogeneous rate must be either scalar or length T"
            lam = homogeneous_lam
        else:
            raise Exception("Homogeneous rate must be either scalar or length T")

        return lam

    def mle(self, S):
        """
        Compute the maximum likelihood firing rate (number of spikes / total time)
        """
        homogeneous_lam = len(S)/(self.T_stop - self.T_start)
        self.lam = self._homogeneous_lam_to_array(homogeneous_lam, self.t)

# Helper functions
def bin_spike_train(S, T_start, T_stop, nbins=None, binw=None):
    if nbins is None and binw is None:
        raise Exception("Either nbins or binw must be specified")
    elif nbins is not None:
        bins = np.linspace(T_start, T_stop, nbins)
    else:
        bins = np.arange(T_start, T_stop, binw)

    return np.histogram(S, bins)


# Demos
def demo_homogeneous_pp():
    # Demo 1: Independent homogeneous Poisson process neurons
    N = 10                                          # Number of neurons
    T_start = 0                                     # Start of simulation
    T_stop = 30                                     # Stop time of simulation (seconds)
    f = 2                                        # sampling frequency (Hz)
    t = np.arange(T_start, T_stop, step=1.0/f)      # time points at which firing rate is sampled
    lams = np.random.gamma(10.0, 1.0, (N,))

    hpps = [HomogeneousPoissonProcess(lams[n], t) for n in range(N)]
    Ss = map(lambda hpp: hpp.sample(), hpps)
    Sbinned = np.array(map(lambda S: np.histogram(S, t)[0], Ss))

    # Plot the spike trains
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(Sbinned)
    plt.colorbar()
    plt.show()

    return Sbinned, t, lams