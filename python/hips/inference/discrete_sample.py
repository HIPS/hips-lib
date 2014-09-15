import numpy as np

def discrete_sample(p, N=1, values=None):
    """
    Sample N values from a discrete probability distribution vector p
    If values is given, these elements are returned
    """
    bins = np.add.accumulate(p)
    rand_inds = np.digitize(np.random.random(N), bins)
    if N==1:
        rand_inds = rand_inds[0]
    if values is not None:
        return values[rand_inds]
    else:
        return rand_inds