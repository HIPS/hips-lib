# This example uses a MovieWriter directly to grab individual frames and
# write them to a file. This avoids any event loop integration, but has
# the advantage of working with even the Agg backend. This is not recommended
# for use in an interactive setting.
# -*- noplot -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.cm import get_cmap

class MovieEffect(object):
    def update(self, data):
        pass

class BlinkyCircleEffect(MovieEffect):

    def __init__(self, ax, data, names=None, pos=None, cmap=None,
                 size=0.4, vmin=0.0, vmax=1.0,
                 make_patch=lambda x,y,sz: Circle((x,y),sz/0.2)):
        """ Initialize the frame for
            Create a polygon for each stock
        """
        self.data = data
        self.N, self.T = self.data.shape
        self.offset = 0

        if pos is None:
            # Initial setup: Just place the nodes randomly
            pos = np.zeros((self.N,2))
            pos[:,0] = np.mod(np.arange(self.N), 10)
            pos[:,1] = np.arange(self.N) / 10
            pos += 0.1*np.random.randn(self.N,2)

        # Set axis limits
        ax.set_xlim([np.amin(pos[:,0])-1,np.amax(pos[:,0])+1])
        ax.set_ylim([np.amin(pos[:,1])-1,np.amax(pos[:,1])+1])

        patches = []
        for n in np.arange(self.N):
            patch = make_patch(pos[n,0], pos[n,1], size)
            patches.append(patch)
        p = PatchCollection(patches, cmap=cmap, alpha=1.0)

        # Set the values of the patches
        p.set_array(self.data[:,0])
        p.set_clim(vmin, vmax)
        ax.add_collection(p)

        # Plot names above a few circles
        if names is not None:
            assert isinstance(names, list)
            assert len(names) == self.N
            for n,name in enumerate(names):
                if name is not None:
                    ax.text(pos[n,0], pos[n,1], name,
                            horizontalalignment='left',
                            verticalalignment='top')

        self.p = p

    def update(self):
        """ Plot a frame of the financial data
        """
        self.offset += 1
        if self.offset > self.T:
            print "WARNING: BlinkyCircleEffect has passed the length of the data!"

        self.p.set_array(self.data[:,self.offset])


class SlidingMatrixEffect(MovieEffect):
    def __init__(self, ax, M,
                 window=100,
                 cmap=get_cmap('Greys'),
                 pix_per_y=1, pix_per_x=1,
                 aspect='auto',
                 vmax=None):
        """ Initialize the frame for a spike train raster.
        """
        self.M = M
        if vmax is None:
            vmax = np.amax(M)
        self.N, self.T = self.M.shape
        self.window = window
        self.pix_per_y = pix_per_y
        self.pix_per_x = pix_per_x
        self.offset = 0

        self.st = ax.imshow(self._kron_matrix(self._pad_matrix(0)),
                            vmin=0,vmax=vmax,
                            interpolation='nearest',
                            aspect=aspect,
                            cmap=cmap)

        ax.set_xticks([])

    def _pad_matrix(self, offset):
        pre_pad = self.window - offset
        post_pad = self.offset + self.window - self.T

        if pre_pad > 0:
            return np.concatenate((np.nan*np.ones((self.N, pre_pad)),
                                   self.M[:, :offset]),
                                  axis=1)

        elif post_pad > 0:
            return np.concatenate((self.M[:, offset:],
                                   np.nan*np.ones((self.N, post_pad))),
                                  axis=1)
        else:
            return self.M[:,(offset-self.window):offset]


    def _kron_matrix(self, M):
        return np.kron(M, np.ones((self.pix_per_y, self.pix_per_x)))

    def update(self):
        self.offset += 1
        self.st.set_data(self._kron_matrix(self._pad_matrix(self.offset)))

        if self.offset > self.T:
            print "WARNING: SlidingMatrixEffect has passed the length of the data!"


class DisappearingTraceEffect(MovieEffect):
    """
    Disappearing trace object. Eg. for moving rat positions
    """
    def __init__(self, ax, pos, window=5,
                 cmap=get_cmap('Greys'), sz=5, sz_decay=1):
        """ Initialize the frame for
            Create a polygon for each stock
        """
        self.pos = pos
        self.window = window
        self.colors_inds = cmap(np.linspace(0, 1, num=self.window))
        self.T, self.D = pos.shape
        assert self.D == 2, "DisappearingTrace Can only plot 2d trajectories!"
        self.sz = sz
        self.offset = 0

        trace0 = self._pad_trace(self.offset)
        self.ls = []
        for w in range(self.window):
            lsw = ax.plot(trace0[w,0],
                          trace0[w,1],
                          linestyle='-',
                          markerfacecolor=self.colors_inds[w,:],
                          marker='o',
                          markersize=max(0, sz-sz_decay*(self.window-w))
                          # markersize=sz
                          )
            self.ls.append(lsw[0])

    def _pad_trace(self, offset):
        pre_pad = self.window - offset
        post_pad = self.offset + self.window - self.T

        if pre_pad > 0:
            return np.concatenate((np.nan*np.ones((pre_pad, 2)),
                                   self.pos[:offset, :]))

        elif post_pad > 0:
            return np.concatenate((self.pos[offset:, :],
                                   np.nan*np.ones((post_pad, 2))))
        else:
            return self.pos[(offset-self.window):offset, :]

    def update(self):
        self.offset += 1
        trace = self._pad_trace(self.offset)
        for w in range(self.window):
            self.ls[w].set_data(trace[w,0], trace[w,1])

        if self.offset > self.T+self.window:
            print "WARNING: DisappearingTraceEffect has passed the length of the data!"



def initialize_moviewriter(title=None, comment=None, fps=15):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title, comment=comment)
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer

