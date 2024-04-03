#ifndef SOLVE_H
#define SOLVE_H

#include <cstdint>
#include <functional>

namespace npf {

inline int sgn(const double x) {
    return (x > 0) - (x < 0);
}

double solve(
    const std::function<double(double)>& f,
    const std::function<double(double)>& f_prime,
    double initial_guess,
    double abs_tol = 1e-12,
    u_int32_t max_iter = 100
);
}
#endif //SOLVE_H
