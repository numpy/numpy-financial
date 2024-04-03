from libc.math cimport NAN
cimport cython

cdef extern from "src/_inner_loop.h" namespace "npf":
    double nper_inner_loop(const double rate, const double pmt, const double pv, const double fv, const double when) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def nper(
    const double[::1] rates,
    const double[::1] pmts,
    const double[::1] pvs,
    const double[::1] fvs,
    const double[::1] whens,
    double[:, :, :, :, ::1] out):

    cdef:
        Py_ssize_t rate_, pmt_, pv_, fv_, when_

    for rate_ in range(rates.shape[0]):
        for pmt_ in range(pmts.shape[0]):
            for pv_ in range(pvs.shape[0]):
                for fv_ in range(fvs.shape[0]):
                    for when_ in range(whens.shape[0]):
                        out[rate_, pmt_, pv_, fv_, when_] = nper_inner_loop(
                            rates[rate_], pmts[pmt_], pvs[pv_], fvs[fv_], whens[when_]
                        )


@cython.boundscheck(False)
@cython.cdivision(True)
def npv(const double[::1] rates, const double[:, ::1] values, double[:, ::1] out):
    cdef:
        Py_ssize_t i, j, t
        double acc

    with nogil:
        for i in range(rates.shape[0]):
            for j in range(values.shape[0]):
                acc = 0.0
                for t in range(values.shape[1]):
                    if rates[i] == -1.0:
                        acc = NAN
                        break
                    else:
                        acc = acc + values[j, t] / ((1.0 + rates[i]) ** t)
                out[i, j] = acc
