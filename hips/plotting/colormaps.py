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

def gradient_cmap(colors, nsteps=256, bounds=None):
    # Make a colormap that interpolates between a set of colors
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = np.linspace(0,1,ncolors)


    reds = []
    greens = []
    blues = []
    for b,c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
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

def harvard_colors():
    return [
        np.array([165,  28,  48]),    # 0.  crimson
        np.array([ 78, 132, 196]),    # 1.  blue bonnet
        np.array([ 82, 133,  76]),    # 2.  ivy
        np.array([196, 150,  26]),    # 3.  gold
        np.array([  0, 156, 163]),    # 4.  aqua
        np.array([232, 125,  30]),    # 5.  saffron
        np.array([137, 150, 160]),    # 6.  slate
        np.array([ 41,  51,  82]),    # 7.  indigo
        np.array([195, 215, 164]),    # 8.  pear
        np.array([255, 219, 109]),    # 9.  lemon
        np.array([ 43,  13,  97]),    # 10. dark purple
    ]