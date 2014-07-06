import matplotlib.pyplot as plt
import numpy as np
from hips.movies.moviemaker import *

def random_walk_demo():
    writer = initialize_moviewriter(title='Random walk demo')

    fig, ax = plt.subplots()
    # remove_plot_labels(fig, ax)

    # Make a random walk trajectory
    T = 100
    D = 2
    sigma = 0.05
    trajectory = np.zeros((T,D))
    for t in np.arange(1,T):
        trajectory[t,:] = trajectory[t-1,:] + sigma*np.random.randn(1, 2)

    effect = DisappearingTraceEffect(ax, trajectory)
    ax.plot(trajectory[:,0], trajectory[:,1], 'r', alpha=0.5)


    lim = np.ceil(np.amax(np.abs(trajectory)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    with writer.saving(fig, "random_walk.mp4", 100):
        for i in range(T):
            effect.update()
            writer.grab_frame()

if __name__ == "__main__":
    random_walk_demo()