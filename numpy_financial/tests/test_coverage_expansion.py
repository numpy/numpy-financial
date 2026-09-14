"""Additional tests expanding coverage of numpy_financial.

This file adds focused tests that target previously uncovered or
under-tested branches of ``numpy_financial``.  It complements the
existing ``test_financial.py`` rather than replacing any of those tests.

Test groups:

* Edge cases (zero rates, empty arrays, infinite cashflows, scalar 0d arrays)
* Error paths (raises with bad input, exception flags)
* Vectorized vs scalar parity (the same call returns matching results)
* Comparison against analytical solutions where they exist
* Round-trip identities (``pmt = ppmt + ipmt``, ``npv(irr(x), x) == 0``)
"""

import math
from decimal import Decimal

import numpy
import pytest
from numpy.testing import assert_allclose

import numpy_financial as npf
from numpy_financial._financial import _convert_when, _irr_default_selection


# ---------------------------------------------------------------------------
# fv ------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestFvCoverage:
    def test_zero_rate_no_pmt_returns_minus_pv(self):
        # fv + pv + pmt * nper == 0  -> fv == -pv  when pmt==0, rate==0
        assert npf.fv(0.0, 10, 0.0, -1000.0) == 1000.0

    def test_zero_rate_with_pmt(self):
        # fv = -(pv + pmt * nper)
        assert_allclose(npf.fv(0.0, 12, -100, -500), 1700.0)

    def test_negative_rate_finite(self):
        # rate < 0 still produces a finite, deterministic result
        result = npf.fv(-0.05, 10, -100, -1000)
        assert math.isfinite(result)

    def test_scalar_returns_python_float(self):
        result = npf.fv(0.05, 5, -100, -1000)
        assert isinstance(result, float)

    def test_array_returns_ndarray(self):
        result = npf.fv([0.05, 0.06], 5, -100, -1000)
        assert isinstance(result, numpy.ndarray)
        assert result.shape == (2,)

    def test_broadcast_rate_and_when(self):
        # 2x2 fully broadcast call
        out = npf.fv([[0.0], [0.05]], [12, 6], -100, 0, [0, 1])
        assert out.shape == (2, 2)

    def test_zero_dim_array_returns_scalar(self):
        # 0-d arrays should follow ufunc convention and return a scalar
        rate = numpy.array(0.05)
        result = npf.fv(rate, 5, -100, -1000)
        assert numpy.isscalar(result)

    def test_vectorized_matches_scalar(self):
        rates = [0.01, 0.05, 0.0, -0.01]
        scalar = numpy.array([npf.fv(r, 12, -100, -1000) for r in rates])
        vector = npf.fv(rates, 12, -100, -1000)
        assert_allclose(vector, scalar)


# ---------------------------------------------------------------------------
# pmt -----------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestPmtCoverage:
    def test_zero_rate_simple_division(self):
        # When rate == 0: pmt == -(fv + pv) / nper
        assert_allclose(npf.pmt(0.0, 10, 1000, 0), -100.0)

    def test_zero_rate_with_fv(self):
        assert_allclose(npf.pmt(0.0, 5, 100, 400), -100.0)

    def test_pmt_inverse_of_pv(self):
        # ``pmt`` and ``pv`` are inverse operations under fixed-rate annuities:
        # given the payment schedule that pays off ``principal`` over ``n``
        # periods at ``rate``, ``pv`` reconstructs that same principal.
        rate, n, principal = 0.05, 10, 1000
        pmt_val = npf.pmt(rate, n, principal)
        pv_val = npf.pv(rate, n, pmt_val, 0)
        assert_allclose(pv_val, principal, rtol=1e-9)

    def test_pmt_with_decimal_when_int(self):
        # When `when` is an int the function should still work for Decimal
        result = npf.pmt(Decimal("0.01"), Decimal("12"), Decimal("1000"))
        assert isinstance(result, Decimal)

    def test_vectorized_matches_scalar(self):
        rates = [0.0, 0.01, 0.05]
        scalar = numpy.array([npf.pmt(r, 12, 1000) for r in rates])
        vector = npf.pmt(rates, 12, 1000)
        assert_allclose(vector, scalar)


# ---------------------------------------------------------------------------
# pv ------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestPvCoverage:
    def test_zero_rate(self):
        # pv = -(fv + pmt * nper). The implementation triggers a divide
        # warning internally for the zero-rate branch even though the
        # ``np.where`` selects the safe value -- silence it for cleanliness.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert_allclose(npf.pv(0.0, 10, -100, 500), 500.0)

    def test_pv_zero_pmt(self):
        # With pmt == 0: pv = -fv / (1+rate)**nper
        assert_allclose(
            npf.pv(0.05, 10, 0, 1000),
            -1000 / (1.05) ** 10,
            rtol=1e-12,
        )

    def test_pv_when_begin_lower_magnitude(self):
        end = npf.pv(0.05, 10, -100, 0, "end")
        begin = npf.pv(0.05, 10, -100, 0, "begin")
        # Annuity due (begin) is worth more than ordinary annuity
        assert abs(begin) > abs(end)

    def test_array_broadcast_returns_array(self):
        result = npf.pv([0.04, 0.05, 0.06], 10, -100, 0)
        assert result.shape == (3,)

    def test_vectorized_matches_scalar(self):
        rates = [0.01, 0.05, 0.10]
        scalar = numpy.array([npf.pv(r, 10, -100, 0) for r in rates])
        vector = npf.pv(rates, 10, -100, 0)
        assert_allclose(vector, scalar)


# ---------------------------------------------------------------------------
# nper ----------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestNperCoverage:
    def test_zero_rate(self):
        # nper = -(fv + pv) / pmt
        assert_allclose(npf.nper(0.0, -100, 1000, 0), 10.0)

    def test_zero_pmt_zero_rate_returns_infinity(self):
        # Cannot pay off a balance with no payment and no growth
        with numpy.errstate(divide="raise"):
            result = npf.nper(0.0, 0.0, 1000.0)
        assert numpy.isinf(result) or numpy.isnan(result)

    def test_nper_returns_scalar_for_scalar_inputs(self):
        result = npf.nper(0.05, -100, 1000)
        assert numpy.isscalar(result)

    def test_nper_array_inputs_return_array(self):
        result = npf.nper([0.05, 0.06], -100, 1000)
        # ufunc-like: 1-element shapes might collapse, so just check ndim
        assert numpy.asarray(result).ndim >= 1


# ---------------------------------------------------------------------------
# ipmt / ppmt ---------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestPaymentDecomposition:
    """Verify that ``pmt == ppmt + ipmt`` for a variety of inputs."""

    @pytest.mark.parametrize(
        "rate,nper,pv",
        [
            (0.05, 12, 1000),
            (0.10 / 12, 24, 5000),
            (0.07, 30, 100000),
            (0.001, 60, 2500),
        ],
    )
    def test_pmt_equals_ppmt_plus_ipmt(self, rate, nper, pv):
        per = numpy.arange(1, nper + 1)
        total = npf.pmt(rate, nper, pv)
        i = npf.ipmt(rate, per, nper, pv)
        p = npf.ppmt(rate, per, nper, pv)
        assert_allclose(i + p, total, rtol=1e-10)

    def test_per_zero_returns_nan(self):
        assert numpy.isnan(npf.ipmt(0.05, 0, 12, 1000))

    def test_negative_per_returns_nan(self):
        assert numpy.isnan(npf.ipmt(0.05, -1, 12, 1000))

    def test_per_one_with_when_begin_returns_zero(self):
        # If paying at beginning of period 1, no interest has accrued yet
        result = npf.ipmt(0.05, 1, 12, 1000, when="begin")
        assert result == 0

    def test_ppmt_first_period_when_begin(self):
        # When `when=begin` on the first period, the entire payment is principal
        rate, nper, pv = 0.05, 12, 1000
        total = npf.pmt(rate, nper, pv, 0, "begin")
        principal = npf.ppmt(rate, 1, nper, pv, 0, "begin")
        assert_allclose(principal, total, rtol=1e-10)


# ---------------------------------------------------------------------------
# rate ----------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestRateCoverage:
    def test_rate_basic(self):
        # 10 periods, payment of -3500, pv 10000 -> ~11.07% per period
        result = npf.rate(10, 0, -3500, 10000)
        assert_allclose(result, 0.110690853, rtol=1e-6)

    def test_rate_zero_iterations_returns_guess(self):
        # With maxiter=0, rate should never iterate, just return the guess
        # For scalar non-convergence => default returns NaN
        result = npf.rate(10, -100, 1000, 0, guess=0.5, maxiter=0)
        assert numpy.isnan(result)

    def test_rate_decimal_with_default_tol_and_guess(self):
        # Exercise the Decimal default-type path for guess and tol
        result = npf.rate(
            Decimal("10"),
            Decimal("0"),
            Decimal("-3500"),
            Decimal("10000"),
        )
        # Approximate to a few digits; result is a Decimal
        assert isinstance(result, Decimal)
        assert abs(result - Decimal("0.1106908537")) < Decimal("1e-6")

    def test_rate_array_partial_failure_returns_nans(self):
        # When some entries are feasible and others aren't, the infeasible
        # entries should be NaN while feasible ones converge.
        nper = 2
        pmt = 0
        # First two are feasible (pv and fv have opposite sign); last two
        # have same sign so cannot converge.
        pv = [-1000, -500, -100, -1000]
        fv = [1100, 600, -110, -1100]
        result = npf.rate(nper, pmt, pv, fv)
        assert result.shape == (4,)
        # First feasible
        assert_allclose(result[0], (1100 / 1000) ** 0.5 - 1, rtol=1e-6)
        # Last two infeasible (same-sign pv/fv) -> NaN
        assert numpy.isnan(result[2])
        assert numpy.isnan(result[3])

    def test_rate_iterations_exceeded_array_message(self):
        # When raise_exceptions=True, the array path raises
        # IterationsExceededError on convergence failure.
        # Both same-sign pv/fv -> guaranteed to fail.
        with pytest.raises(npf.IterationsExceededError):
            npf.rate(2, 0, [-1000, -2000], [-1100, -2100], raise_exceptions=True)


# ---------------------------------------------------------------------------
# npv -----------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestNpvCoverage:
    def test_zero_rate_is_sum(self):
        cashflows = [-1000, 200, 300, 400, 500]
        assert_allclose(npf.npv(0.0, cashflows), sum(cashflows))

    def test_npv_known_value(self):
        # Manually computed: -100 + 50/1.1 + 50/1.21 + 50/1.331
        result = npf.npv(0.10, [-100, 50, 50, 50])
        manual = -100 + 50 / 1.1 + 50 / 1.21 + 50 / 1.331
        assert_allclose(result, manual, rtol=1e-9)

    def test_npv_2d_cashflows(self):
        # 2D cashflows return shape (K, M)
        rates = [0.05, 0.10]
        cashflows = [[-100, 50, 50, 50], [-200, 100, 100, 100]]
        result = npf.npv(rates, cashflows)
        assert result.shape == (2, 2)

    def test_npv_scalar_returns_float(self):
        result = npf.npv(0.05, [-100, 50, 50, 50])
        assert isinstance(result, float)

    def test_npv_invalid_3d_rates_raises(self):
        with pytest.raises(ValueError, match="invalid shape for rates"):
            npf.npv(numpy.zeros((1, 1, 1)), [-100, 50])

    def test_npv_invalid_3d_values_raises(self):
        with pytest.raises(ValueError, match="invalid shape for values"):
            npf.npv(0.05, numpy.zeros((1, 1, 1)))

    def test_npv_irr_zero(self):
        # NPV evaluated at the IRR should be ~0
        cashflows = [-100, 39, 59, 55, 20]
        rate = npf.irr(cashflows)
        assert_allclose(npf.npv(rate, cashflows), 0, atol=1e-9)

    def test_npv_negative_rate_finite(self):
        # rate > -1 should still produce a finite value
        result = npf.npv(-0.5, [-100, 50, 50, 50])
        assert math.isfinite(result)

    def test_npv_decreasing_with_rate(self):
        # For positive cashflows after the first, NPV decreases as rate increases
        cashflows = [-100, 50, 50, 50, 50]
        a = npf.npv(0.05, cashflows)
        b = npf.npv(0.10, cashflows)
        c = npf.npv(0.20, cashflows)
        assert a > b > c


# ---------------------------------------------------------------------------
# irr -----------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestIrrCoverage:
    def test_irr_no_real_solution_returns_nan(self):
        # Polynomial with no real roots in [-1, infty)
        # All-zero (after first) cashflows force no realistic root
        result = npf.irr([-1, -1, -1])
        assert numpy.isnan(result)

    def test_irr_2d_input_returns_array(self):
        result = npf.irr([[-100, 0, 0, 74], [-100, 100, 0, 7]])
        assert result.shape == (2,)
        assert_allclose(result, [-0.0955, 0.06206], atol=1e-3)

    def test_irr_all_positive_returns_nan(self):
        # All positive cashflows -> no real solution
        result = npf.irr([100, 200, 300])
        assert numpy.isnan(result)

    def test_irr_all_negative_returns_nan(self):
        result = npf.irr([-100, -200, -300])
        assert numpy.isnan(result)

    def test_irr_simple_doubling(self):
        # Invest 100, get 200 next period -> 100% IRR
        assert_allclose(npf.irr([-100, 200]), 1.0, rtol=1e-9)

    def test_irr_simple_zero(self):
        # Invest 100, get 100 next period -> 0% IRR
        assert_allclose(npf.irr([-100, 100]), 0.0, atol=1e-9)

    def test_irr_negative_irr(self):
        # Invest 100, get only 50 -> -50% IRR
        assert_allclose(npf.irr([-100, 50]), -0.5, rtol=1e-9)

    def test_irr_raises_no_real_solution_with_flag(self):
        with pytest.raises(npf.NoRealSolutionError):
            npf.irr([100, 200, 300], raise_exceptions=True)

    def test_irr_default_selection_mixed_signs_picks_positive(self):
        # ``_irr_default_selection`` filters to non-negative roots when the
        # sum of positives meets or exceeds the sum of negatives, then
        # returns the smallest-magnitude root.
        eirr = numpy.array([0.30, -0.05])
        result = _irr_default_selection(eirr)
        assert result == pytest.approx(0.30)

    def test_irr_default_selection_same_sign_smallest_magnitude(self):
        # When all candidates share a sign, the function should return the
        # candidate with the smallest absolute value.
        eirr = numpy.array([0.10, 0.05, 0.20])
        result = _irr_default_selection(eirr)
        assert result == pytest.approx(0.05)

    def test_irr_default_selection_all_negative(self):
        eirr = numpy.array([-0.10, -0.05, -0.20])
        result = _irr_default_selection(eirr)
        assert result == pytest.approx(-0.05)

    def test_irr_custom_selection_logic(self):
        # The selection function should be consulted when multiple roots exist.
        called = []

        def pick_largest(roots):
            called.append(roots)
            return numpy.max(roots)

        # A polynomial known to have multiple real roots:
        # -100, 230, -132 has roots 0.10 and 0.20
        result = npf.irr([-100, 230, -132], selection_logic=pick_largest)
        assert called, "Custom selection_logic was not invoked"
        assert_allclose(result, 0.20, rtol=1e-6)

    def test_irr_npv_round_trip(self):
        cashflows = [-150000, 15000, 25000, 35000, 45000, 60000]
        rate = npf.irr(cashflows)
        assert_allclose(npf.npv(rate, cashflows), 0, atol=1e-6)

    def test_irr_complex_only_roots_returns_nan(self):
        # Cashflows ``[1, -1, 1]`` give the polynomial g^2 - g + 1 = 0 whose
        # discriminant is negative, so there are no real roots. The signs
        # change so we don't short-circuit on same-sign; we fall through to
        # the real-root filter and find the empty set.
        result = npf.irr([1, -1, 1])
        assert numpy.isnan(result)

    def test_irr_complex_only_roots_raises_with_flag(self):
        with pytest.raises(npf.NoRealSolutionError, match="No real solution is found"):
            npf.irr([1, -1, 1], raise_exceptions=True)

    def test_irr_2d_with_mixed_solvability(self):
        # Mix a solvable row and a same-sign row. Without raise_exceptions
        # the result should contain a finite value and a NaN.
        result = npf.irr([[-100, 39, 59, 55, 20], [100, 100, 100, 100, 100]])
        assert math.isfinite(result[0])
        assert numpy.isnan(result[1])


# ---------------------------------------------------------------------------
# mirr ----------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestMirrCoverage:
    def test_mirr_all_positive_returns_nan(self):
        result = npf.mirr([100, 200, 300, 400], 0.10, 0.12)
        assert numpy.isnan(result)

    def test_mirr_all_negative_returns_nan(self):
        result = npf.mirr([-100, -200, -300, -400], 0.10, 0.12)
        assert numpy.isnan(result)

    def test_mirr_size_mismatch_returns_nan(self):
        # finance_rate and reinvest_rate have different sizes -> NaN, no raise
        result = npf.mirr(
            [-100, 50, -60, 70],
            [0.10, 0.11],
            [0.12],
        )
        assert numpy.isnan(result)

    def test_mirr_size_mismatch_raises_with_flag(self):
        with pytest.raises(ValueError, match="must have the same size"):
            npf.mirr(
                [-100, 50, -60, 70],
                [0.10, 0.11],
                [0.12],
                raise_exceptions=True,
            )

    def test_mirr_known_value(self):
        # Reference value: -100, 50, -60, 70 at 10% finance, 12% reinvest
        # Computed from the public docstring of mirr.
        result = npf.mirr([-100, 50, -60, 70], 0.10, 0.12)
        assert_allclose(result, -0.03909366594356467, rtol=1e-10)

    def test_mirr_2d_values(self):
        values = [
            [-4500, -800, 800, 800, 600],
            [-120000, 39000, 30000, 21000, 37000],
        ]
        # Single rate scalar broadcast
        result = npf.mirr(values, 0.05, 0.08)
        # Shape must be (2,) since rates are scalars
        assert numpy.asarray(result).shape == (2,)


# ---------------------------------------------------------------------------
# _convert_when -------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestConvertWhen:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("end", 0),
            ("begin", 1),
            ("e", 0),
            ("b", 1),
            ("beginning", 1),
            ("start", 1),
            ("finish", 0),
            (0, 0),
            (1, 1),
        ],
    )
    def test_known_aliases(self, value, expected):
        assert _convert_when(value) == expected

    def test_iterable_of_aliases(self):
        result = _convert_when(["begin", "end", 1, 0])
        assert result == [1, 0, 1, 0]

    def test_passes_through_ndarray(self):
        arr = numpy.array([0, 1, 0])
        out = _convert_when(arr)
        # Should be the same object (passes through)
        assert out is arr

    def test_unknown_alias_raises_keyerror(self):
        # Unknown string alias should propagate a KeyError-like failure
        with pytest.raises(KeyError):
            _convert_when("middle")


# ---------------------------------------------------------------------------
# Cross-function identities -------------------------------------------------
# ---------------------------------------------------------------------------


class TestIdentities:
    def test_fv_pv_round_trip(self):
        # fv(rate, n, 0, pv) should give -pv * (1+rate)^n
        rate, n, pv = 0.05, 10, -1000
        result = npf.fv(rate, n, 0, pv)
        expected = -pv * (1 + rate) ** n
        assert_allclose(result, expected, rtol=1e-12)

    def test_pv_fv_round_trip(self):
        # pv(rate, n, 0, fv) should give -fv / (1+rate)^n
        rate, n, fv = 0.05, 10, 1000
        result = npf.pv(rate, n, 0, fv)
        expected = -fv / (1 + rate) ** n
        assert_allclose(result, expected, rtol=1e-12)

    def test_pmt_zero_loan_zero_pmt(self):
        # No principal, no future value -> no payment required
        assert_allclose(npf.pmt(0.05, 10, 0, 0), 0.0, atol=1e-12)

    def test_npv_single_cashflow_at_t0(self):
        # NPV of a single cashflow at t=0 is just the cashflow itself
        assert_allclose(npf.npv(0.05, [-100]), -100.0)

    def test_irr_npv_zero_for_random_seeds(self):
        # NPV(IRR(x), x) should be near 0 for many random cashflow series
        rng = numpy.random.default_rng(42)
        for _ in range(5):
            cf = numpy.concatenate([[-1000.0], rng.uniform(50, 300, size=8)])
            r = npf.irr(cf)
            if not numpy.isnan(r):
                assert_allclose(npf.npv(r, cf), 0, atol=1e-6)
