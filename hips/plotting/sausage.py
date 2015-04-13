import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def sausage_plot(x, y, yerr, sgax=None, **kwargs):
    T = x.size
    assert x.shape == y.shape == yerr.shape == (T,)

    # Get axis
    if sgax is None:
        sgax = plt.gca()

    # Compute envelope
    env = np.zeros((T*2,2))
    env[:,0] = np.concatenate((x, x[::-1]))
    env[:,1] = np.concatenate((y + yerr, y[::-1] - yerr[::-1]))

    # Add the patch
    sgax.add_patch(Polygon(env, **kwargs))
