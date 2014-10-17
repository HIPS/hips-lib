# distutils: extra_compile_args = -O3
# cython: wraparound=False 
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
## cython: profile=True

import numpy as np
cimport numpy as np

from libc.math cimport log, exp
cpdef double log_sum_exp(double[::1] lnp):
    """
    Sample uniformly from a vector of unnormalized log probs using 
    the log-sum-exp trick
    """
    # assert np.ndim(lnp) == 1, "ERROR: logSumExpSample requires a 1-d vector"
    # lnp = np.ravel(lnp)
    # N = np.size(lnp)
    cdef int N = lnp.shape[0]
    cdef int n
    # Use logsumexp trick to calculate ln(p1 + p2 + ... + pR) from ln(pi)'s
    cdef double max_lnp = -np.Inf
    for n in range(N):
        if lnp[n] > max_lnp:
            max_lnp = lnp[n]

    # Sum up the terms, subtracting off max lnp
    cdef double sum_exp = 0
    for n in range(N):
        sum_exp += exp(lnp[n]-max_lnp)

    return log(sum_exp) + max_lnp

cpdef log_sum_exp_normalize(double[::1] lnp, double[::1] p):
    cdef int N = lnp.shape[0]
    cdef double denom = log_sum_exp(lnp)
    cdef int n

    for n in range(N):
        p[n] = exp(lnp[n] - denom)


cpdef int log_sum_exp_sample(double[::1] lnp):
    cdef int N = lnp.shape[0]
    cdef int n

    cdef double[::1] p = np.zeros(N,)
    log_sum_exp_normalize(lnp, p)
    return discrete_sample(p)

cpdef int discrete_sample(double[::1] p):
    # Randomly sample from p
    cdef int N = p.shape[0]
    cdef int choice = -1
    cdef double u = np.random.rand()
    cdef double acc = 0.0
    for n in range(N):
        acc += p[n]
        if u <= acc:
            choice = n

    return choice
