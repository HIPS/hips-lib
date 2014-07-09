"""
Custom patches for plotting
"""
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt


class NeuronPatch(PathPatch):
    def __init__(self, xy=(0,0), size=1.0, N_points=6, **kwargs):
        """
        Make a neuron patch with diameter size and N_points around its "cell body"
        connected by concave arcs

        xy:             The center point (x,y) of the patch
        size:           The diameter of the patch
        N_points:       The number of points around the outside of the neuron
        """
        # Arrange the points around a circle
        ths = np.linspace(0, 2*np.pi, N_points+1)[:N_points]
        # Add a bit of noise
        ths += np.pi/(6*N_points) * np.random.randn(N_points)

        # Get the xy locations
        pts = np.zeros((N_points+1, 2))
        # Shrink the x axis
        pts[:-1,0] = xy[0] + size/2./1.5 * np.cos(ths)
        pts[:-1,1] = xy[1] + size/2. * np.sin(ths)
        # Close the path
        pts[-1,:] = pts[0,:]

        # Arrange control points for each segment around a circle
        ctrl_ths = np.zeros_like(ths)
        ctrl_ths[:-1] = (ths[:-1] + ths[1:])/2.0
        ctrl_ths[-1] = (ths[-1] + 2*np.pi)/2.0

        # Get the xy locations of the control points (midpoint of the Bezier curves)
        ctrl_pts = np.zeros((N_points, 2))
        ctrl_pts[:,0] = xy[0] + 0.5*size/2. * np.cos(ctrl_ths)
        ctrl_pts[:,1] = xy[1] + 0.5*size/2. * np.sin(ctrl_ths)

        # Make a list of vertices
        verts = []
        codes = []
        # Move to the starting point
        verts.append(pts[0,:])
        codes.append(Path.MOVETO)
        # Make a series of Bezier curves
        for n in range(N_points):
            # Control point
            verts.append(ctrl_pts[n,:])
            codes.append(Path.CURVE3)
            # End point
            verts.append(pts[n+1,:])
            codes.append(Path.CURVE3)

        self.verts = verts
        self.codes = codes

        # Convert the points to a curved path
        self.path = Path(verts, codes, closed=True)

        super(NeuronPatch, self).__init__(self.path, **kwargs)


def demo_neuronpatch():
    n = NeuronPatch(xy=(1,1), size=1.0, N_points=7)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # plt.plot(n.pts[:,0], n.pts[:,1])
    ax.add_patch(n)
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    plt.show()


