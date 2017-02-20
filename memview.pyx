# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from cython cimport boundscheck, wraparound
from cython.parallel cimport prange

import numpy as np


@boundscheck(False)
@wraparound(False)
cpdef transf_value (double[:] data, double value):

    cdef:
        long i, N
        double[:] values

    N = data.shape[0]
    values = np.zeros(N, dtype=np.float64)

    for i in prange(N, nogil=True):
        values[i] = data[i] * value
    return np.asarray(values)

cpdef transf_value_serial (double[:] data, double value):

    cdef:
        long i, N
        double[:] values

    N = data.shape[0]
    values = np.zeros(N, dtype=np.float64)

    for idx,i in enumerate(data):
        values[i] = i  * value
    return np.asarray(values)


