from libc.math cimport NAN
import numpy as np
cimport numpy as np
cimport cython

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.cpow(True)
cpdef double[:, ::1] npv(const double[::1] rates, const double[:, ::1] cashflows):
    cdef:
        long rate_len = rates.shape[0]
        long no_of_cashflows = cashflows.shape[0]
        long cashflows_len = cashflows.shape[1]
        long i, j, t
        double acc
        double[:, ::1] out

    out = np.empty(shape=(rate_len, no_of_cashflows))
    for i in range(rate_len):
        for j in range(no_of_cashflows):
            acc = 0.0
            for t in range(cashflows_len):
                acc += cashflows[j, t] / ((1.0 + rates[i]) ** t)
            out[i, j] = acc
    return out

