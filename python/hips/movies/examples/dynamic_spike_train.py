import matplotlib.pyplot as plt
import numpy as np
from hips.movies.moviemaker import *

def dynamic_spike_train_demo():

    # Make a random spike matrix
    N = 50
    dt = 0.01
    T = 500
    rho = 0.05
    S = np.random.rand(N,T) < rho

    # Create a figure and axis for the spike train
    fig, ax = plt.subplots()
    ax.set_title('Spike train demo')

    # Create an effect to slide the spike train matrix
    effect = SlidingMatrixEffect(ax, S)

    # Create a writer to show the movie at 30 frames per sec
    fps = 30
    # Set the data time scaling. 1.0 = real time, 10 = 10x speedup, 0.1 = 10x slowdown
    tscale = 0.5
    tdata = np.arange(T)*dt
    treallife = tdata/tscale
    tframes = np.arange(treallife[-1], step=1.0/fps)
    nframes = len(tframes)
    writer = initialize_moviewriter(title='Dynamic spike train demo', fps=fps)


    with writer.saving(fig, "spike_train.mp4", 200):
        ti = 1
        for fri, tfr in enumerate(tframes):
            print "Frame %d/%d" % (fri, nframes)

            # Update the data until we exceed the frame time
            while treallife[ti] < tfr:
                effect.update()
                ti += 1

            if np.allclose(tdata[ti] % 1.0, 0):
                ax.set_title('t = %.1f' % (tdata[ti]))

            writer.grab_frame()

if __name__ == "__main__":
    dynamic_spike_train_demo()