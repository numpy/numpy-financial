#include <cstdint>
#include <limits>

#include "_solve.h"


double solve(
    const std::function<double(double)>& f,
    const double a,
    const double b,
    const double abs_tol,
    const uint32_t max_iter
) {
    auto lhs = a;
    auto rhs = b;
    uint32_t iters = 0;

    while (iters < max_iter) {
        const auto f_lhs = f(lhs);
        const auto f_rhs = f(rhs);
        const auto sgn_f_lhs = npf::sgn(f_lhs);
        const auto sgn_f_rhs = npf::sgn(f_rhs);

        if (sgn_f_lhs * sgn_f_rhs > 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto mid = lhs - f_lhs * (rhs - lhs) / (f_rhs - f_lhs);
        const auto f_mid = f(mid);
        const auto sgn_f_mid = npf::sgn(f_mid);

        if (std::abs(f_mid) < abs_tol) {
            return mid;
        }
        if (sgn_f_lhs * sgn_f_mid < 0) {
            rhs = mid;
        } else if (sgn_f_mid * sgn_f_rhs < 0) {
            lhs = mid;
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
        ++iters;
    }

    return std::numeric_limits<double>::quiet_NaN();
}

