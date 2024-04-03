#include <cmath>
#include <limits>

#include "_inner_loop.h"
#include "_solve.h"


double balance(const double nper, const double rate, const double pmt, const double pv, const double fv, const double when) {
    const auto original = pv * std::pow(1.0 + rate, nper);
    const auto annuity = pmt * ( 1 + rate * when) * ( std::pow(1 + rate, nper ) - 1 ) / rate;
    return fv + original + annuity;
}


double partial_nper(
    const double nper,
    const double rate,
    const double pmt,
    const double pv,
    const double fv, // unused as constant in ``balance`` equation
    const double when
) {
    // avoid division by zero
    if (rate == 0) { return std::numeric_limits<double>::quiet_NaN(); }

    const auto rp1 = rate + 1.0;
    // cannot take log of value less that or equal to 0.
    if (rp1 <= 0.0) { return std::numeric_limits<double>::quiet_NaN(); }

    return std::pow(rp1, nper) * std::log(rp1) * (pmt * rate * when + pmt + pv * rate) / rate;
}


double nper_inner_loop(const double rate, const double pmt, const double pv, const double fv, const double when)
{
    if (rate == 0.0 && pmt == 0.0)
    {
        return std::numeric_limits<double>::infinity();
    }

    if (rate == 0.0)
    {
        return -(fv + pv) / pmt;
    }

    auto f = [rate, pmt, pv, fv, when]
             (const double guess)
             { return balance(guess, rate, pmt, pv, fv, when); };
    auto fp = [rate, pmt, pv, fv, when]
              (const double guess)
              { return partial_nper(guess, rate, pmt, pv, fv, when); };
    return npf::solve(f, fp, 0);
}
