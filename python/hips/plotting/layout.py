import matplotlib.pyplot as plt

def create_axis_at_location(fig, left, bottom, width, height, box=True, ticks=True):
    """
    Create axes at abolute position by scaling to figure size
    """
    w,h = fig.get_size_inches()

    # Adjust subplot so that the panel is 1.25 in high and 1 width
    ax = fig.add_axes([left/w, bottom/h, width/w, height/h])

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
