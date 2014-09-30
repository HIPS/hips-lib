"""
An empirical distribution of points in a 2D circle

Scott Linderman
2014
"""
import numpy as np
import scipy.interpolate
import matplotlib.patches
import matplotlib.cm
import matplotlib.pyplot as plt

class CircularDistribution(object):
    def __init__(self, center, radius, rbins=11, thbins=11, pdf=None):
        self.center = np.array(center)
        assert self.center.size == 2
        self.radius = radius

        # Create a grid of r's and thetas
        self.rbins = rbins
        self.thbins = thbins
        self.rs = np.sqrt(np.linspace(0, radius**2, rbins+1))
        self.ths = np.linspace(-np.pi, np.pi, thbins+1)

        self.areas = np.zeros((rbins, thbins))
        for i,rl in enumerate(self.rs[:-1]):
            for j,thl in enumerate(self.ths[:-1]):
                dr = self.rs[i+1] - self.rs[i]
                rcent = (self.rs[i+1] + self.rs[i]) / 2.0
                dth = self.ths[j+1] - self.ths[j]
                self.areas[i,j] = rcent * dr * dth
        assert np.allclose(self.areas.sum(), np.pi*radius**2, 1e-8,1e-8)

        # Initialize the data pointer
        self.data_xy = None
        self.data_polar = None

        # Initialize the pmf
        if pdf is not None:
            assert pdf.shape == self.areas.shape, "PDF does not match the number of bins!"
            self.pdf = pdf
        else:
            # Initialize to a uniform distribution
            self.pdf = 1.0/self.areas.sum() * np.ones_like(self.areas)

        # assert np.allclose((self.pdf * self.areas).sum(), 1.0)

    @property
    def normalizer(self):
        return (self.pdf * self.areas).sum()

    def normalize(self):
        self.pdf /= self.normalizer

    def fit_xy(self, x, y):
        """
        Fit to a dataset of x,y locations
        """
        N = len(x)
        assert len(y)==N, "x and y must be of the same length"

        if N == 0:
            # print "WARNING: Attempting to fit to zero datapoints"
            return

        self.data_xy = (x,y)

        xc = x - self.center[0]
        yc = y - self.center[1]

        pos_r = np.sqrt(xc**2 + yc**2)

        if np.any(pos_r > self.radius):
            # print "WARNING: Points found outside limits of circle! Truncating..."
            pos_r = np.clip(pos_r, 0, self.radius)

        pos_th = np.arctan2(yc, xc)

        self.fit_polar(pos_r, pos_th)

    def fit_polar(self, pos_r, pos_th):
        N = len(pos_r)
        assert len(pos_th)==N, "r and th must be of the same length"

        self.data_polar = (pos_r, pos_th)

        bins = np.zeros_like(self.areas)
        for i,rl in enumerate(self.rs[:-1]):
            for j,thl in enumerate(self.ths[:-1]):
                ru = self.rs[i+1]
                thu = self.ths[j+1]
                bins[i,j] = ((pos_r > rl) * (pos_r <= ru) *
                             (pos_th > thl) * (pos_th <= thu)).sum()

        # Make sure we counted all the points
        assert bins.sum() == N

        # Normalize probabilities by the area of the bins
        self.pdf = bins/float(N)/self.areas

        assert np.allclose((self.pdf * self.areas).sum(), 1.0)

    @property
    def mean(self):
        """
        Convert rs and ths to complex form in order to take the mean
        Basic directional statistics
        """
        thc = (self.ths[:-1] + self.ths[1:])/2.0
        rc = (self.rs[:-1] + self.rs[1:])/2.0

        thcg, rcg = np.meshgrid(thc, rc)
        zs = rcg * np.exp(1j * thcg)
        m = (self.pdf * self.areas * zs).sum()
        rm = np.abs(m)
        thm = np.angle(m)

        # xm = self.center[0] + rm*np.cos(thm)
        # ym = self.center[1] + rm*np.sin(thm)
        #
        # return xm, ym

        return rm, thm


    def __add__(self, other):
        """
        Add the pdfs of two distributions (may not be normalized)
        """
        assert isinstance(other, CircularDistribution)
        assert np.allclose(self.rs, other.rs)
        assert np.allclose(self.ths, other.ths)
        return CircularDistribution(self.center, self.radius, self.rbins, self.thbins,
                                    self.pdf + other.pdf)

    def __mul__(self, other):
        """
        Support multiplication by a scalar
        """
        assert np.isscalar(other)
        return CircularDistribution(self.center, self.radius, self.rbins, self.thbins,
                                    self.pdf * other)

    def plot(self, show=False, plot_data=True, ax=None, cmap=None, alpha=None, lw=1, N_pts=300, plot_colorbar=True):
        # Plot the results at a fine grid of N points
        # rc = (self.rs[:-1] + self.rs[1:])/2.0
        thc = (self.ths[:-1] + self.ths[1:])/2.0
        rc = self.rs[1:]
        xs = (self.center[0] + rc[:,None] * np.cos(thc[None,:])).ravel()
        ys = (self.center[1] + rc[:,None] * np.sin(thc[None,:])).ravel()
        zs = self.pdf.ravel()

        # Manually add the center point
        xs = np.concatenate((xs, [self.center[0]]))
        ys = np.concatenate((ys, [self.center[1]]))
        zs = np.concatenate((zs, [self.pdf[0,:].mean()]))

        xi = np.linspace(self.center[0]-self.radius, self.center[0]+self.radius, N_pts)
        yi = np.linspace(self.center[1]-self.radius, self.center[1]+self.radius, N_pts)
        zi = scipy.interpolate.griddata((xs, ys), zs,
                                        (xi[None,:], yi[:,None]),
                                        method='linear',
                                        fill_value=0.0)

        zi = np.clip(zi, 0, np.inf)

        # set points > radius to not-a-number. They will not be plotted.
        # the dr/2 makes the edges a bit smoother
        for i in range(N_pts):
            for j in range(N_pts):
                r = np.sqrt((xi[i]-self.center[0])**2 + (yi[j]-self.center[1])**2)
                if (r) > self.radius:
                    zi[j,i] = "nan"

        if ax is None:
            # make figure
            fig = plt.figure(figsize=(3,3))
            # set aspect = 1 to make it a circle
            ax = fig.add_subplot(111, aspect=1)

        # use different number of levels for the fill and the lines
        ncontours = 60
        cs = ax.contourf(xi, yi, zi, ncontours, cmap=cmap, zorder=1, alpha=alpha, vmin=0)

        if plot_colorbar:
            plt.colorbar(cs)

        # DEBUG
        # ax.plot(xs, ys, 'r.', linestyle='none')
        # grid_x, grid_y = np.meshgrid(xi, yi)
        # ax.plot(grid_x, grid_y, 'b.', linestyle='none')

        circle = matplotlib.patches.Circle(xy=self.center,
                                           radius= self.radius,
                                           linewidth=lw,
                                           edgecolor="k",
                                           facecolor="none")
        ax.add_patch(circle)

        if self.data_xy is not None and plot_data:
            x,y = self.data_xy
            ax.scatter(x,y, s=1, marker='.', c='k')

        ax.set_xlim(self.center[0]-self.radius, self.center[0]+self.radius)
        ax.set_ylim(self.center[1]-self.radius, self.center[1]+self.radius)

        if show:
            plt.show()

        return ax, cs


def test_circular_distribution():
    center = (0,0)
    radius = 1

    # Rejection sample uniform points inside the circle
    N = 20000
    allpts = -1 + 2*np.random.rand(N,2)
    r = np.sqrt(allpts[:,0]**2 + allpts[:,1]**2)
    inside = r <= 1
    # inside = np.bitwise_and(r <= 1, allpts[:,1] > 0)
    pts = allpts[inside,:]

    # Fit the distribution
    cd = CircularDistribution(center, radius)
    cd.fit_xy(pts[:,0], pts[:,1])

    # Plot the results
    cmap = matplotlib.cm.get_cmap('Greys')
    cd.plot(show=True, plot_data=False, cmap=cmap)

    print np.mean(cd.areas.ravel())
    print np.std(cd.areas.ravel())