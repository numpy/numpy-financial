from libc.math cimport NAN, INFINITY, log
cimport cython


cdef double nper_inner_loop(
    const double rate_,
    const double pmt_,
    const double pv_,
    const double fv_,
    const double when_
) nogil:
    if rate_ == 0.0 and pmt_ == 0.0:
        return INFINITY

    if rate_ == 0.0:
        return -(fv_ + pv_) / pmt_

    if rate_ <= -1.0:
        return NAN

    z = pmt_ * (1.0 + rate_ * when_) / rate_
    return log((-fv_ + z) / (pv_ + z)) / log(1.0 + rate_)


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
                        # We can have several ``ZeroDivisionErrors``s here
                        # At the moment we want to replicate the existing function as
                        # closely as possible however we should return financially
                        # sensible results here.
                        try:
                            res = nper_inner_loop(
                                rates[rate_], pmts[pmt_], pvs[pv_], fvs[fv_], whens[when_]
                            )
                        except ZeroDivisionError:
                            res = NAN

                        out[rate_, pmt_, pv_, fv_, when_] = res


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