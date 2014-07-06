"""
Helpers for generating colormaps
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def white_to_color_cmap(color, nsteps=256):
    # Get a red-white-black cmap
    cdict = {'red': ((0.0, 1.0, 1.0),
                       (1.0, color[0], color[0])),
                'green': ((0.0, 1.0, 1.0),
                          (1.0, color[1], color[0])),
                'blue': ((0.0, 1.0, 1.0),
                         (1.0, color[2], color[0]))}
    cmap = LinearSegmentedColormap('white_color_colormap', cdict, nsteps)
    return cmap


def combo_white_to_color_cmap(colors, nsteps=1000):
    ncolors = colors.shape[0]
    # assert colors.shape[1] == 3
    bounds = np.linspace(0,1,ncolors+1)

    # Get a red-white-black cmap
    reds = [(0.0,1.0,1.0)]
    greens = [(0.0,1.0,1.0)]
    blues = [(0.0,1.0,1.0)]
    for i,b in enumerate(bounds):
        if i == 0:
            continue
        reds.append((b, colors[i-1][0], 1.0))
        greens.append((b, colors[i-1][1], 1.0))
        blues.append((b, colors[i-1][2], 1.0))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues)}
    cmap = LinearSegmentedColormap('white_color_colormap', cdict, nsteps)
    return cmap