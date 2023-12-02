cimport cython
from cython cimport floating
from cython.parallel cimport  prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.cpow(True)
cdef floating npv_inner_loop(const floating rate, const floating[::1] cashflow) noexcept nogil:
    cdef:
        long cashflow_len = cashflow.shape[0]
        long t
        floating acc

    acc = 0.0
    for t in range(cashflow_len):
        acc += cashflow[t] / ((1.0 + rate) ** t)
    return acc


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cy_npv(
        const floating[::1] rates,
        const floating[:, ::1] cashflows,
        floating[:, ::1] out
) noexcept nogil:
    cdef:
        long rate_len = rates.shape[0]
        long no_of_cashflows = cashflows.shape[0]
        long i, j

    for i in prange(rate_len):
        for j in prange(no_of_cashflows):
            out[i, j] = npv_inner_loop(rates[i], cashflows[j])

