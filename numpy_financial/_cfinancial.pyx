from libc.math cimport NAN


def _cnpv(const double[::1] rates, const double[:, ::1] values, double[:, ::1] out):
    cdef:
        Py_ssize_t i, j, t
        double acc

    for i in range(rates.shape[0]):
        for j in range(values.shape[0]):
            acc = 0.0
            for t in range(values.shape[1]):
                if rates[i] == -1.0:
                    acc = NAN
                    break
                else:
                    acc += values[j, t] / ((1.0 + rates[i]) ** t)
            out[i, j] = acc