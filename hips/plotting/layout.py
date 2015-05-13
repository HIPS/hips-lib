import matplotlib.pyplot as plt

def create_figure(figsize=None, transparent=True,**kwargs):
    fig = plt.figure(figsize=figsize,**kwargs)
    if transparent:
        fig.patch.set_alpha(0.0)
    return fig

def create_axis_at_location(fig, left, bottom, width, height,
                            transparent=False, box=True, ticks=True):
    """
    Create axes at abolute position by scaling to figure size
    """
    w,h = fig.get_size_inches()

    # Adjust subplot so that the panel is 1.25 in high and 1 width
    ax = fig.add_axes([left/w, bottom/h, width/w, height/h])

    if transparent:
        ax.patch.set_alpha(0.0)

    if not box:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    if not ticks:
        remove_tick_marks(ax)

    return ax

def remove_plot_labels(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.yticks([])
    plt.tick_params(\
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',
        labelleft='off')

def remove_tick_marks(ax):
    # It's a real pain to remove the tick marks
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    ax.set_xticklabels([])
    ax.set_yticklabels([])

def create_legend_figure(labels, colors, size=None,
                         fontsize=9, type='line',
                         orientation='horizontal', ncol=None,
                         **kwargs):
    """
    Create a separate figure for a legend.
    :param labels:
    :param colors:
    :param orientation:
    :param type:
    :param kwargs:
    :return:
    """
    assert len(labels) == len(colors)

    dummyfig = plt.figure()
    dummyax = dummyfig.add_subplot(111)

    handles = []
    if type == 'line':
        for c in colors:
            handles.append(dummyax.plot([0,0],[1,1], color=c, **kwargs)[0])
    else:
        raise Exception('Other types of plots are not yet supported')

    # Set number of columns
    if ncol is None:
        if orientation == 'horizontal':
            ncol = len(labels)
        else:
            ncol = 1

    fig = plt.figure(figsize=size)

    plt.figlegend(handles,
                  labels,
                  loc='center',
                  fontsize=fontsize,
                  ncol=ncol)

    # Set the figure background to transparent
    fig.patch.set_alpha(0.0)


    return fig
