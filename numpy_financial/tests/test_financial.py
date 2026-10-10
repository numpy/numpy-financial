import math
import warnings
from decimal import Decimal

# Don't use 'import numpy as np', to avoid accidentally testing
# the versions in numpy instead of numpy_financial.
import numpy
import pytest
from hypothesis import assume, given
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_equal,
    assert_raises,
)

import numpy_financial as npf
from numpy_financial.tests.strategies import (
    cashflow_array_like_strategy,
    cashflow_array_strategy,
    short_nicely_behaved_doubles,
    when_strategy,
)


def assert_decimal_close(actual, expected, tol=Decimal("1e-7")):
    # Check if both actual and expected are iterable (like arrays)
    if hasattr(actual, "__iter__") and hasattr(expected, "__iter__"):
        for a, e in zip(actual, expected, strict=True):
            assert abs(a - e) <= tol
    else:
        # For single value comparisons
        assert abs(actual - expected) <= tol


class TestFinancial(object):
    def test_when(self):
        # begin
        assert_equal(
            npf.rate(10, 20, -3500, 10000, 1), npf.rate(10, 20, -3500, 10000, "begin")
        )
        # end
        assert_equal(
            npf.rate(10, 20, -3500, 10000), npf.rate(10, 20, -3500, 10000, "end")
        )
        assert_equal(
            npf.rate(10, 20, -3500, 10000, 0), npf.rate(10, 20, -3500, 10000, "end")
        )

        # begin
        assert_equal(npf.pv(0.07, 20, 12000, 0, 1), npf.pv(0.07, 20, 12000, 0, "begin"))
        # end
        assert_equal(npf.pv(0.07, 20, 12000, 0), npf.pv(0.07, 20, 12000, 0, "end"))
        assert_equal(npf.pv(0.07, 20, 12000, 0, 0), npf.pv(0.07, 20, 12000, 0, "end"))

        # begin
        assert_equal(
            npf.pmt(0.08 / 12, 5 * 12, 15000.0, 0, 1),
            npf.pmt(0.08 / 12, 5 * 12, 15000.0, 0, "begin"),
        )
        # end
        assert_equal(
            npf.pmt(0.08 / 12, 5 * 12, 15000.0, 0),
            npf.pmt(0.08 / 12, 5 * 12, 15000.0, 0, "end"),
        )
        assert_equal(
            npf.pmt(0.08 / 12, 5 * 12, 15000.0, 0, 0),
            npf.pmt(0.08 / 12, 5 * 12, 15000.0, 0, "end"),
        )

        # begin
        assert_equal(
            npf.nper(0.075, -2000, 0, 100000.0, 1),
            npf.nper(0.075, -2000, 0, 100000.0, "begin"),
        )
        # end
        assert_equal(
            npf.nper(0.075, -2000, 0, 100000.0),
            npf.nper(0.075, -2000, 0, 100000.0, "end"),
        )
        assert_equal(
            npf.nper(0.075, -2000, 0, 100000.0, 0),
            npf.nper(0.075, -2000, 0, 100000.0, "end"),
        )

    def test_decimal_with_when(self):
        """
        Test that decimals are still supported if the when argument is passed
        """
        # begin
        assert_equal(
            npf.rate(
                Decimal("10"),
                Decimal("20"),
                Decimal("-3500"),
                Decimal("10000"),
                Decimal("1"),
            ),
            npf.rate(
                Decimal("10"),
                Decimal("20"),
                Decimal("-3500"),
                Decimal("10000"),
                "begin",
            ),
        )
        # end
        assert_equal(
            npf.rate(Decimal("10"), Decimal("20"), Decimal("-3500"), Decimal("10000")),
            npf.rate(
                Decimal("10"), Decimal("20"), Decimal("-3500"), Decimal("10000"), "end"
            ),
        )
        assert_equal(
            npf.rate(
                Decimal("10"),
                Decimal("20"),
                Decimal("-3500"),
                Decimal("10000"),
                Decimal("0"),
            ),
            npf.rate(
                Decimal("10"), Decimal("20"), Decimal("-3500"), Decimal("10000"), "end"
            ),
        )

        # begin
        assert_equal(
            npf.pv(
                Decimal("0.07"),
                Decimal("20"),
                Decimal("12000"),
                Decimal("0"),
                Decimal("1"),
            ),
            npf.pv(
                Decimal("0.07"), Decimal("20"), Decimal("12000"), Decimal("0"), "begin"
            ),
        )
        # end
        assert_equal(
            npf.pv(Decimal("0.07"), Decimal("20"), Decimal("12000"), Decimal("0")),
            npf.pv(
                Decimal("0.07"), Decimal("20"), Decimal("12000"), Decimal("0"), "end"
            ),
        )
        assert_equal(
            npf.pv(
                Decimal("0.07"),
                Decimal("20"),
                Decimal("12000"),
                Decimal("0"),
                Decimal("0"),
            ),
            npf.pv(
                Decimal("0.07"), Decimal("20"), Decimal("12000"), Decimal("0"), "end"
            ),
        )


class TestPV:
    def test_pv(self):
        assert_allclose(npf.pv(0.07, 20, 12000, 0), -127128.17, rtol=1e-2)

    def test_pv_decimal(self):
        assert_equal(
            npf.pv(Decimal("0.07"), Decimal("20"), Decimal("12000"), Decimal("0")),
            Decimal("-127128.1709461939327295222005"),
        )


class TestRate:
    def test_rate(self):
        assert_allclose(npf.rate(10, 0, -3500, 10000), 0.1107, rtol=1e-4)

    @pytest.mark.parametrize("number_type", [Decimal, float])
    @pytest.mark.parametrize("when", [0, 1, "end", "begin"])
    def test_rate_with_infeasible_solution(self, number_type, when):
        """
        Test when no feasible rate can be found.

        Rate will return NaN, if the Newton Raphson method cannot find a
        feasible rate within the required tolerance or number of iterations.
        This can occur if both `pmt` and `pv` have the same sign, as it is
        impossible to repay a loan by making further withdrawls.
        """
        result = npf.rate(
            number_type(12.0),
            number_type(400.0),
            number_type(10000.0),
            number_type(5000.0),
            when=when,
        )
        is_nan = Decimal.is_nan if number_type == Decimal else numpy.isnan
        assert is_nan(result)

    def test_rate_decimal(self):
        rate = npf.rate(Decimal("10"), Decimal("0"), Decimal("-3500"), Decimal("10000"))
        assert_equal(Decimal("0.1106908537142689284704528100"), rate)

    def test_gh48(self):
        """
        Test the correct result is returned with only infeasible solutions
        converted to nan.
        """
        des = [-0.39920185, -0.02305873, -0.41818459, 0.26513414, numpy.nan]
        nper = 2
        pmt = 0
        pv = [-593.06, -4725.38, -662.05, -428.78, -13.65]
        fv = [214.07, 4509.97, 224.11, 686.29, -329.67]
        actual = npf.rate(nper, pmt, pv, fv)
        assert_allclose(actual, des)

    def test_rate_maximum_iterations_exception_scalar(self):
        # Test that if the maximum number of iterations is reached,
        # then npf.rate returns IterationsExceededException
        # when raise_exceptions is set to True.
        assert_raises(
            npf.IterationsExceededError,
            npf.rate,
            Decimal(12.0),
            Decimal(400.0),
            Decimal(10000.0),
            Decimal(5000.0),
            raise_exceptions=True,
        )

    def test_rate_maximum_iterations_exception_array(self):
        # Test that if the maximum number of iterations is reached in at least
        # one rate, then npf.rate returns IterationsExceededException
        # when raise_exceptions is set to True.
        nper = 2
        pmt = 0
        pv = [-593.06, -4725.38, -662.05, -428.78, -13.65]
        fv = [214.07, 4509.97, 224.11, 686.29, -329.67]
        assert_raises(
            npf.IterationsExceededError,
            npf.rate,
            nper,
            pmt,
            pv,
            fv,
            raise_exceptions=True,
        )


class TestNpv:
    def test_npv(self):
        assert_allclose(
            npf.npv(0.05, [-15000.0, 1500.0, 2500.0, 3500.0, 4500.0, 6000.0]),
            122.89,
            rtol=1e-2,
        )

    @given(rates=short_nicely_behaved_doubles, values=cashflow_array_strategy)
    def test_fuzz(self, rates, values):
        npf.npv(rates, values)

    @pytest.mark.parametrize("rates", ([[1, 2, 3]], numpy.empty(shape=(1, 1, 1))))
    def test_invalid_rates_shape(self, rates):
        cashflows = [1, 2, 3]
        with pytest.raises(ValueError):
            npf.npv(rates, cashflows)

    @pytest.mark.parametrize("cf", ([[[1, 2, 3]]], numpy.empty(shape=(1, 1, 1))))
    def test_invalid_cashflows_shape(self, cf):
        rates = [1, 2, 3]
        with pytest.raises(ValueError):
            npf.npv(rates, cf)

    @pytest.mark.parametrize("rate", (-1, -1.0))
    def test_rate_of_negative_one_returns_nan(self, rate):
        cashflow = numpy.arange(5)
        assert numpy.isnan(npf.npv(rate, cashflow))


class TestPmt:
    def test_pmt_simple(self):
        res = npf.pmt(0.08 / 12, 5 * 12, 15000)
        tgt = -304.145914
        assert_allclose(res, tgt)

    def test_pmt_zero_rate(self):
        # Test the edge case where rate == 0.0
        res = npf.pmt(0.0, 5 * 12, 15000)
        tgt = -250.0
        assert_allclose(res, tgt)

    def test_pmt_broadcast(self):
        # Test the case where we use broadcast and
        # the arguments passed in are arrays.
        res = npf.pmt([[0.0, 0.8], [0.3, 0.8]], [12, 3], [2000, 20000])
        tgt = numpy.array([[-166.66667, -19311.258], [-626.90814, -19311.258]])
        assert_allclose(res, tgt)

    def test_pmt_decimal_simple(self):
        res = npf.pmt(Decimal("0.08") / Decimal("12"), 5 * 12, 15000)
        tgt = Decimal("-304.1459143262052370338701494")
        assert_equal(res, tgt)

    def test_pmt_decimal_zero_rate(self):
        # Test the edge case where rate == 0.0
        res = npf.pmt(Decimal("0"), Decimal("60"), Decimal("15000"))
        tgt = -250
        assert_equal(res, tgt)

    def test_pmt_decimal_broadcast(self):
        # Test the case where we use broadcast and
        # the arguments passed in are arrays.
        res = npf.pmt(
            [[Decimal("0"), Decimal("0.8")], [Decimal("0.3"), Decimal("0.8")]],
            [Decimal("12"), Decimal("3")],
            [Decimal("2000"), Decimal("20000")],
        )
        tgt = numpy.array(
            [
                [
                    Decimal("-166.6666666666666666666666667"),
                    Decimal("-19311.25827814569536423841060"),
                ],
                [
                    Decimal("-626.9081401700757748402586600"),
                    Decimal("-19311.25827814569536423841060"),
                ],
            ]
        )

        # Cannot use the `assert_allclose` because it uses isfinite under
        # the covers which does not support the Decimal type
        # See issue: https://github.com/numpy/numpy/issues/9954
        assert_equal(res[0][0], tgt[0][0])
        assert_equal(res[0][1], tgt[0][1])
        assert_equal(res[1][0], tgt[1][0])
        assert_equal(res[1][1], tgt[1][1])


class TestMirr:
    @pytest.mark.parametrize(
        "values,finance_rate,reinvest_rate,expected",
        [
            (
                [-4500, -800, 800, 800, 600, 600, 800, 800, 700, 3000],
                0.08,
                0.055,
                0.0666,
            ),
            ([-120000, 39000, 30000, 21000, 37000, 46000], 0.10, 0.12, 0.126094),
            ([100, 200, -50, 300, -200], 0.05, 0.06, 0.3428),
            ([39000, 30000, 21000, 37000, 46000], 0.10, 0.12, None),
        ],
    )
    def test_mirr(self, values, finance_rate, reinvest_rate, expected):
        result = npf.mirr(values, finance_rate, reinvest_rate)

        if expected:
            decimal_part_len = len(str(expected).split(".")[1])
            difference = 10**-decimal_part_len
            assert_allclose(result, expected, atol=difference)
        else:
            assert_(numpy.isnan(result))

    def test_mirr_broadcast(self):
        values = [
            [-4500, -800, 800, 800, 600],
            [-120000, 39000, 30000, 21000, 37000],
            [100, 200, -50, 300, -200],
        ]
        finance_rate = [0.05, 0.08, 0.10]
        reinvestment_rate = [0.08, 0.10, 0.12]
        # Found using Google sheets
        expected = numpy.array([
            [-0.1784449, -0.17328716, -0.1684366],
            [0.04627293, 0.05437856, 0.06252201],
            [0.35712458, 0.40628857, 0.44435295]
        ])
        actual = npf.mirr(values, finance_rate, reinvestment_rate)
        assert_allclose(actual, expected)

    def test_mirr_no_real_solution_exception(self):
        # Test that if there is no solution because all the cashflows
        # have the same sign, then npf.mirr returns NoRealSolutionException
        # when raise_exceptions is set to True.
        val = [39000, 30000, 21000, 37000, 46000]

        with pytest.raises(npf.NoRealSolutionError):
            npf.mirr(val, 0.10, 0.12, raise_exceptions=True)

    @given(
        values=cashflow_array_like_strategy,
        finance_rate=short_nicely_behaved_doubles,
        reinvestment_rate=short_nicely_behaved_doubles,
    )
    def test_fuzz(self, values, finance_rate, reinvestment_rate):
        assume(finance_rate.size == reinvestment_rate.size)

        # NumPy warns us of arithmetic overflow/underflow
        # this only occurs when hypothesis generates extremely large values
        # that are unlikely to ever occur in the real world.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            npf.mirr(values, finance_rate, reinvestment_rate)

    @given(
        values=cashflow_array_like_strategy,
        finance_rate=short_nicely_behaved_doubles,
        reinvestment_rate=short_nicely_behaved_doubles,
    )
    def test_mismatching_rates_raise(self, values, finance_rate, reinvestment_rate):
        assume(finance_rate.size != reinvestment_rate.size)
        with pytest.raises(ValueError):
            npf.mirr(values, finance_rate, reinvestment_rate, raise_exceptions=True)


class TestNper:
    def test_basic_values(self):
        assert_allclose(
            npf.nper([0, 0.075], -2000, 0, 100000),
            [50, 21.544944],  # Computed using Google Sheet's NPER
            rtol=1e-5,
        )

    def test_gh_18(self):
        with numpy.errstate(divide="raise"):
            assert_allclose(
                npf.nper(0.1, 0, -500, 1500),
                11.52670461,  # Computed using Google Sheet's NPER
            )

    def test_infinite_payments(self):
        with numpy.errstate(divide="raise"):
            result = npf.nper(0, -0.0, 1000)
        assert_(result == numpy.inf)

    def test_no_interest(self):
        assert_(npf.nper(0, -100, 1000) == 10)

    def test_broadcast(self):
        assert_allclose(
            npf.nper(0.075, -2000, 0, 100000.0, [0, 1]), [21.5449442, 20.76156441], 4
        )

    @given(
        rates=short_nicely_behaved_doubles,
        payments=short_nicely_behaved_doubles,
        present_values=short_nicely_behaved_doubles,
        future_values=short_nicely_behaved_doubles,
        whens=when_strategy,
    )
    def test_fuzz(self, rates, payments, present_values, future_values, whens):
        npf.nper(rates, payments, present_values, future_values, whens)


class TestPpmt:
    def test_float(self):
        assert_allclose(npf.ppmt(0.1 / 12, 1, 60, 55000), -710.25, rtol=1e-4)

    def test_decimal(self):
        result = npf.ppmt(
            Decimal("0.1") / Decimal("12"),
            Decimal("1"),
            Decimal("60"),
            Decimal("55000"),
        )
        assert_equal(
            result,
            Decimal("-710.2541257864217612489830917"),
        )

    @pytest.mark.parametrize("when", [1, "begin"])
    def test_when_is_begin(self, when):
        assert_allclose(
            npf.ppmt(0.1 / 12, 1, 60, 55000, 0, when),
            -1158.929712,  # Computed using Google Sheet's PPMT
            rtol=1e-9,
        )

    @pytest.mark.parametrize("when", [None, 0, "end"])
    def test_when_is_end(self, when):
        args = (0.1 / 12, 1, 60, 55000, 0)
        result = npf.ppmt(*args) if when is None else npf.ppmt(*args, when)
        assert_allclose(
            result,
            -710.254126,  # Computed using Google Sheet's PPMT
            rtol=1e-9,
        )

    @pytest.mark.parametrize("when", [Decimal("1"), "begin"])
    def test_when_is_begin_decimal(self, when):
        result = npf.ppmt(
            Decimal("0.08") / Decimal("12"),
            Decimal("1"),
            Decimal("60"),
            Decimal("15000."),
            Decimal("0"),
            when,
        )
        assert_decimal_close(
            result,
            Decimal("-302.131703"),  # Computed using Google Sheet's PPMT
            tol=1e-5,
        )

    @pytest.mark.parametrize("when", [None, Decimal("0"), "end"])
    def test_when_is_end_decimal(self, when):
        args = (
            Decimal("0.08") / Decimal("12"),
            Decimal("1"),
            Decimal("60"),
            Decimal("15000."),
            Decimal("0"),
        )
        result = npf.ppmt(*args) if when is None else npf.ppmt(*args, when)
        assert_decimal_close(
            result,
            Decimal("-204.145914"),  # Computed using Google Sheet's PPMT
            tol=1e-5,
        )

    @pytest.mark.parametrize(
        "args",
        [
            (0.1 / 12, 0, 60, 15000),
            (Decimal("0.012"), Decimal("0"), Decimal("60"), Decimal("15000")),
        ],
    )
    def test_invalid_per(self, args):
        # Note that math.isnan() handles Decimal NaN correctly.
        assert math.isnan(npf.ppmt(*args))

    @pytest.mark.parametrize(
        "when, desired",
        [
            (
                None,
                [-75.62318601, -76.25337923, -76.88882405, -77.52956425],
            ),
            (
                [0, 1, "end", "begin"],
                [-75.62318601, -75.62318601, -76.88882405, -76.88882405],
            ),
        ],
    )
    def test_broadcast(self, when, desired):
        args = (0.1 / 12, numpy.arange(1, 5), 24, 2000, 0)
        result = npf.ppmt(*args) if when is None else npf.ppmt(*args, when)
        assert_allclose(result, desired, rtol=1e-5)

    @pytest.mark.parametrize(
        "when, desired",
        [
            (
                None,
                [
                    Decimal("-75.62318601"),
                    Decimal("-76.25337923"),
                    Decimal("-76.88882405"),
                    Decimal("-77.52956425"),
                ],
            ),
            (
                [Decimal("0"), Decimal("1"), "end", "begin"],
                [
                    Decimal("-75.62318601"),
                    Decimal("-75.62318601"),
                    Decimal("-76.88882405"),
                    Decimal("-76.88882405"),
                ],
            ),
        ],
    )
    def test_broadcast_decimal(self, when, desired):
        args = (
            Decimal("0.1") / Decimal("12"),
            numpy.arange(1, 5),
            Decimal("24"),
            Decimal("2000"),
            Decimal("0"),
        )
        result = npf.ppmt(*args) if when is None else npf.ppmt(*args, when)
        assert_decimal_close(result, desired, tol=1e-8)


class TestIpmt:
    def test_float(self):
        assert_allclose(
            npf.ipmt(0.1 / 12, 1, 24, 2000),
            -16.666667,  # Computed using Google Sheet's IPMT
            rtol=1e-6,
        )

    def test_decimal(self):
        result = npf.ipmt(Decimal("0.1") / Decimal("12"), 1, 24, 2000)
        assert result == Decimal("-16.66666666666666666666666667")

    @pytest.mark.parametrize("when", [1, "begin"])
    def test_when_is_begin(self, when):
        assert npf.ipmt(0.1 / 12, 1, 24, 2000, 0, when) == 0

    @pytest.mark.parametrize("when", [None, 0, "end"])
    def test_when_is_end(self, when):
        if when is None:
            result = npf.ipmt(0.1 / 12, 1, 24, 2000)
        else:
            result = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, when)
        assert_allclose(result, -16.666667, rtol=1e-6)

    @pytest.mark.parametrize("when", [Decimal("1"), "begin"])
    def test_when_is_begin_decimal(self, when):
        result = npf.ipmt(
            Decimal("0.1") / Decimal("12"),
            Decimal("1"),
            Decimal("24"),
            Decimal("2000"),
            Decimal("0"),
            when,
        )
        assert result == 0

    @pytest.mark.parametrize("when", [None, Decimal("0"), "end"])
    def test_when_is_end_decimal(self, when):
        # Computed using Google Sheet's IPMT
        desired = Decimal("-16.666667")
        args = (
            Decimal("0.1") / Decimal("12"),
            Decimal("1"),
            Decimal("24"),
            Decimal("2000"),
            Decimal("0"),
        )
        result = npf.ipmt(*args) if when is None else npf.ipmt(*args, when)
        assert_decimal_close(result, desired, tol=1e-5)

    @pytest.mark.parametrize(
        "per, desired",
        [
            (0, numpy.nan),
            (1, 0),
            (2, -594.107158),
            (3, -592.971592),
        ],
    )
    def test_gh_17(self, per, desired):
        # All desired results computed using Google Sheet's IPMT
        rate = 0.001988079518355057
        result = npf.ipmt(rate, per, 360, 300000, when="begin")
        if numpy.isnan(desired):
            assert numpy.isnan(result)
        else:
            assert_allclose(result, desired, rtol=1e-6)

    def test_broadcasting(self):
        desired = [numpy.nan, -16.66666667, -16.03647345, -15.40102862, -14.76028842]
        assert_allclose(
            npf.ipmt(0.1 / 12, numpy.arange(5), 24, 2000),
            desired,
            rtol=1e-6,
        )

    def test_decimal_broadcasting(self):
        desired = [
            Decimal("-16.66666667"),
            Decimal("-16.03647345"),
            Decimal("-15.40102862"),
            Decimal("-14.76028842"),
        ]
        result = npf.ipmt(
            Decimal("0.1") / Decimal("12"),
            list(range(1, 5)),
            Decimal("24"),
            Decimal("2000"),
        )
        assert_decimal_close(result, desired, tol=1e-4)

    def test_0d_inputs(self):
        args = (0.1 / 12, 1, 24, 2000)
        # Scalar inputs should return a scalar.
        assert numpy.isscalar(npf.ipmt(*args))
        args = (numpy.array(args[0]),) + args[1:]
        # 0d array inputs should return a scalar.
        assert numpy.isscalar(npf.ipmt(*args))


class TestFv:
    def test_float(self):
        assert_allclose(
            npf.fv(0.075, 20, -2000, 0, 0),
            86609.362673042924,
            rtol=1e-10,
        )

    def test_decimal(self):
        assert_decimal_close(
            npf.fv(Decimal("0.075"), Decimal("20"), Decimal("-2000"), 0, 0),
            Decimal("86609.36267304300040536731624"),
            tol=1e-10,
        )

    @pytest.mark.parametrize("when", [1, "begin"])
    def test_when_is_begin_float(self, when):
        assert_allclose(
            npf.fv(0.075, 20, -2000, 0, when),
            93105.064874,  # Computed using Google Sheet's FV
            rtol=1e-10,
        )

    @pytest.mark.parametrize("when", [Decimal("1"), "begin"])
    def test_when_is_begin_decimal(self, when):
        result = npf.fv(
            Decimal("0.075"),
            Decimal("20"),
            Decimal("-2000"),
            Decimal("0"),
            when,
        )
        assert_decimal_close(result, Decimal("93105.064874"), tol=5)

    @pytest.mark.parametrize("when", [None, 0, "end"])
    def test_when_is_end_float(self, when):
        args = (0.075, 20, -2000, 0)
        result = npf.fv(*args) if when is None else npf.fv(*args, when)
        assert_allclose(
            result,
            86609.362673,  # Computed using Google Sheet's FV
            rtol=1e-10,
        )

    @pytest.mark.parametrize("when", [None, Decimal("0"), "end"])
    def test_when_is_end_decimal(self, when):
        args = (
            Decimal("0.075"),
            Decimal("20"),
            Decimal("-2000"),
            Decimal("0"),
        )
        result = npf.fv(*args) if when is None else npf.fv(*args, when)
        assert_decimal_close(result, Decimal("86609.362673"), tol=5)

    def test_broadcast(self):
        result = npf.fv([[0.1], [0.2]], 5, 100, 0, [0, 1])
        # All values computed using Google Sheet's FV
        desired = [[-610.510000, -671.561000], [-744.160000, -892.992000]]
        assert_allclose(result, desired, rtol=1e-10)

    def test_some_rates_zero(self):
        # Check that the logical indexing is working correctly.
        assert_allclose(
            npf.fv([0, 0.1], 5, 100, 0),
            [-500, -610.51],  # Computed using Google Sheet's FV
            rtol=1e-10,
        )


class TestIrr:
    def test_npv_irr_congruence(self):
        # IRR is defined as the rate required for the present value of
        # a series of cashflows to be zero, so we should have
        #
        # NPV(IRR(x), x) = 0.
        cashflows = numpy.array([-40000, 5000, 8000, 12000, 30000])
        assert_allclose(
            npf.npv(npf.irr(cashflows), cashflows),
            0,
            atol=1e-9,
            rtol=0,
        )

    @pytest.mark.parametrize(
        "v, desired",
        [
            ([-150000, 15000, 25000, 35000, 45000, 60000], 0.0524),
            ([-100, 0, 0, 74], -0.0955),
            ([-100, 39, 59, 55, 20], 0.28095),
            ([-100, 100, 0, -7], -0.0833),
            ([-100, 100, 0, 7], 0.06206),
            ([-5, 10.5, 1, -8, 1], 0.0886),
        ],
    )
    def test_basic_values(self, v, desired):
        assert_allclose(npf.irr(v), desired, rtol=1e-2)

    def test_trailing_zeros(self):
        assert_allclose(
            npf.irr([-5, 10.5, 1, -8, 1, 0, 0, 0]),
            0.0886,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "v",
        [
            (1, 2, 3),
            (-1, -2, -3),
        ],
    )
    def test_numpy_gh_6744(self, v):
        # Test that if there is no solution then npf.irr returns nan.
        assert numpy.isnan(npf.irr(v))

    def test_gh_15(self):
        v = [
            -3000.0,
            2.3926932267015667e-07,
            4.1672087103345505e-16,
            5.3965110036378706e-25,
            5.1962551071806174e-34,
            3.7202955645436402e-43,
            1.9804961711632469e-52,
            7.8393517651814181e-62,
            2.3072565113911438e-71,
            5.0491839233308912e-81,
            8.2159177668499263e-91,
            9.9403244366963527e-101,
            8.942410813633967e-111,
            5.9816122646481191e-121,
            2.9750309031844241e-131,
            1.1002067043497954e-141,
            3.0252876563518021e-152,
            6.1854121948207909e-163,
            9.4032980015353301e-174,
            1.0629218520017728e-184,
            8.9337141847171845e-196,
            5.5830607698467935e-207,
            2.5943122036622652e-218,
            8.9635842466507006e-230,
            2.3027710094332358e-241,
            4.3987510596745562e-253,
            6.2476630372575209e-265,
            6.598046841695288e-277,
            5.1811095266842017e-289,
            3.0250999925830644e-301,
            1.3133070599585015e-313,
        ]
        result = npf.irr(v)
        assert numpy.isfinite(result)
        # Very rough approximation taken from the issue.
        desired = -0.9999999990596069
        assert_allclose(result, desired, rtol=1e-9)

    def test_gh_39(self):
        cashflows = numpy.array(
            [
                -217500.0,
                -217500.0,
                108466.80462450592,
                101129.96439328062,
                93793.12416205535,
                86456.28393083003,
                79119.44369960476,
                71782.60346837944,
                64445.76323715414,
                57108.92300592884,
                49772.08277470355,
                42435.24254347826,
                35098.40231225296,
                27761.56208102766,
                20424.721849802358,
                13087.88161857707,
                5751.041387351768,
                -1585.7988438735192,
                -8922.639075098821,
                -16259.479306324123,
                -23596.31953754941,
                -30933.159768774713,
                -38270.0,
                -45606.8402312253,
                -52943.680462450604,
                -60280.520693675906,
                -67617.36092490121,
            ]
        )
        assert_allclose(npf.irr(cashflows), 0.12)

    def test_gh_44(self):
        # "true" value as calculated by Google sheets
        cf = [-1678.87, 771.96, 1814.05, 3520.30, 3552.95, 3584.99, 4789.91, -1]
        assert_allclose(npf.irr(cf), 1.00426, rtol=1e-4)

    def test_irr_no_real_solution_exception(self):
        # Test that if there is no solution because all the cashflows
        # have the same sign, then npf.irr returns NoRealSolutionException
        # when raise_exceptions is set to True.
        cashflows = numpy.array([40000, 5000, 8000, 12000, 30000])

        with pytest.raises(npf.NoRealSolutionError):
            npf.irr(cashflows, raise_exceptions=True)

def _cumprinc_iterative(rate, start, end, nper, pv, fv=0, when="end"):
    """Cumulative principal by summing ppmt, the definition of cumprinc.

    Used to check the closed-form version. The period bounds are coerced
    to int so that Decimal inputs can be iterated; ppmt still returns
    Decimal when the remaining arguments are Decimal.
    """
    return sum(
        npf.ppmt(rate, per, nper, pv, fv, when)
        for per in range(int(start), int(end) + 1)
    )

def _cumipmt_iterative(rate, start, end, nper, pv, fv=0, when="end"):
    """Cumulative interest by summing ipmt, the definition of cumipmt.
    """
    return sum(
        npf.ipmt(rate, per, nper, pv, fv, when)
        for per in range(int(start), int(end) + 1)
    )

class TestCumprinc:
    @pytest.mark.parametrize("when", [None, 0, "end"])
    def test_sas_reference_end(self, when):
        # SAS: CUMPRINC(0.09/12, 360, 125000, 12, 24, 0) -> $1008.23
        # https://documentation.sas.com/doc/en/edmcdc/v_031/ds2ref/p1ae3ccwcl1y4dn0zld1p5rj88yx.htm
        # SAS reports the magnitude; numpy-financial signs cash paid out
        # negative. The SAS doc prints to 2 d.p., hence rtol=1e-5.
        kwargs = dict(rate=0.09 / 12, start=12, end=24, nper=360, pv=125000)
        result = (
            npf.cumprinc(**kwargs)
            if when is None
            else npf.cumprinc(**kwargs, when=when)
        )
        assert_allclose(result, -1008.23, rtol=1e-5)

    @pytest.mark.parametrize("when", [1, "begin"])
    def test_sas_reference_begin(self, when):
        # SAS: CUMPRINC(0.09/12, 360, 125000, 12, 24, 1) -> $1000.73
        # Same source. Note 1008.23 / 1000.73 == 1 + 0.09/12, i.e. this
        # pair pins the one-period discount applied when when == 1.
        result = npf.cumprinc(
            rate=0.09 / 12, start=12, end=24, nper=360, pv=125000, when=when
        )
        assert_allclose(result, -1000.73, rtol=1e-5)

    @pytest.mark.parametrize(
        "when, desired",
        [
            (1, -1778.160172),
            ("begin", -1778.160172),
            (0, -853.996373),
            ("end", -853.996373),
        ],
    )
    def test_excel_reference_from_first_period(self, when, desired):
        # Computed using Excel's CUMPRINC:
        #   =CUMPRINC(0.09/12, 360, 125000, 1, 12, 1) -> -1778.160172
        #   =CUMPRINC(0.09/12, 360, 125000, 1, 12, 0) ->  -853.996373
        result = npf.cumprinc(
            rate=0.09 / 12, start=1, end=12, nper=360, pv=125000, when=when
        )
        assert_allclose(result, desired, rtol=1e-9)

    @pytest.mark.parametrize("when", ["end", "begin"])
    @pytest.mark.parametrize("rate", [0.09 / 12, 0.0824 / 12, 0.01, 0.1])
    @pytest.mark.parametrize(
        "start, end",
        [(1, 1), (1, 6), (1, 12), (2, 4), (5, 5), (7, 12), (12, 12)],
    )
    def test_matches_iterative(self, rate, start, end, when):
        # The closed form must agree with summing ppmt period by period.
        args = (rate, start, end, 12, 2500, 0, when)
        assert_allclose(
            npf.cumprinc(*args),
            _cumprinc_iterative(*args),
            rtol=1e-9,
        )

    @pytest.mark.parametrize("when", ["end", "begin"])
    @pytest.mark.parametrize("per", [1, 2, 6, 12])
    def test_single_period_equals_ppmt(self, per, when):
        # A one-period range is just ppmt.
        rate, nper, pv = 0.0824 / 12, 12, 2500
        assert_allclose(
            npf.cumprinc(rate, per, per, nper, pv, 0, when),
            npf.ppmt(rate, per, nper, pv, 0, when),
            rtol=1e-9,
        )

    @pytest.mark.parametrize("when", ["end", "begin"])
    def test_array_start_and_end(self, when):
        # start contains both 1 and values greater than 1 in the same call.
        rate, nper, pv = 0.09 / 12, 360, 125000
        start = numpy.array([1, 2, 12])
        end = numpy.array([12, 12, 24])
        result = npf.cumprinc(rate, start, end, nper, pv, 0, when)
        desired = [
            _cumprinc_iterative(rate, s, e, nper, pv, 0, when)
            for s, e in zip(start, end, strict=True)
        ]
        assert_allclose(result, desired, rtol=1e-9)

    @pytest.mark.parametrize("when", ["end", "begin"])
    @pytest.mark.parametrize(
        "start, end", [(1, 12), (1, 6), (7, 12), (5, 5), (2, 9)]
    )
    def test_zero_rate(self, start, end, when):
        # With no interest every payment is pure principal, so the
        # cumulative principal is pmt * count == -pv * count / nper.
        nper, pv = 12, 2500
        desired = -pv * (end - start + 1) / nper
        result = npf.cumprinc(0.0, start, end, nper, pv, 0, when)
        assert_allclose(result, desired, rtol=1e-12)


    def test_decimal(self):
        args = (Decimal("0.08") / Decimal("12"),
                Decimal("1"),
                Decimal("12"),
                Decimal("60"),
                Decimal("15000."),
                Decimal("0"),
                "end")
        result = npf.cumprinc(*args)
        assert_decimal_close(result, _cumprinc_iterative(*args), tol=Decimal("1e-5"))

    def test_scalar_input_returns_scalar(self):
        result = npf.cumprinc(
            rate=0.0824 / 12, start=1, end=12, nper=12, pv=2500
        )
        assert_(numpy.isscalar(result) or numpy.ndim(result) == 0)

    def test_broadcasting(self):
        rates = numpy.array([0.09 / 12, 0.0824 / 12, 0.01])
        result = npf.cumprinc(
            rate=rates, start=1, end=12, nper=12, pv=2500, when="end"
        )
        assert_equal(numpy.shape(result), (3,))

    @pytest.mark.parametrize(
        "start, end",
        [(0, 6), (-1, 6), (6, 5), (1, 13)],
    )
    def test_out_of_range_is_nan(self, start, end):
        # per < 1, end before start, and end beyond nper are all
        # meaningless; ipmt returns nan for per < 1, so match that.
        result = npf.cumprinc(
            rate=0.0824 / 12, start=start, end=end, nper=12, pv=2500
        )
        assert_(numpy.isnan(result))

class TestCumipmt:
    @pytest.mark.parametrize(
        "start, end, when, desired",
        [
            (12, 24, 0, -12066.88311),
            (12, 24, "end", -12066.88311),
            (1, 12, 1, -10201.33288),
            (1, 12, "begin", -10201.33288),
        ],
    )
    def test_excel_reference(self, start, end, when, desired):
        # Computed using Excel's CUMIPMT, which reports 5 d.p. here:
        #   =CUMIPMT(0.09/12, 360, 125000, 12, 24, 0) -> -12066.88311
        #   =CUMIPMT(0.09/12, 360, 125000,  1, 12, 1) -> -10201.33288
        result = npf.cumipmt(
            rate=0.09 / 12, start=start, end=end, nper=360, pv=125000, when=when
        )
        assert_allclose(result, desired, rtol=1e-8)

    @pytest.mark.parametrize("when", ["end", "begin"])
    @pytest.mark.parametrize("rate", [0.09 / 12, 0.0824 / 12, 0.01, 0.1])
    @pytest.mark.parametrize(
        "start, end",
        [(1, 1), (1, 6), (1, 12), (2, 4), (5, 5), (7, 12), (12, 12)],
    )
    def test_matches_iterative(self, rate, start, end, when):
        args = (rate, start, end, 12, 2500, 0, when)
        assert_allclose(
            npf.cumipmt(*args),
            _cumipmt_iterative(*args),
            rtol=1e-9,
            atol=1e-9,
        )

    @pytest.mark.parametrize("when", ["end", "begin"])
    @pytest.mark.parametrize("start, end", [(1, 12), (1, 6), (5, 5), (7, 12)])
    def test_zero_rate_has_no_interest(self, start, end, when):
        result = npf.cumipmt(0.0, start, end, 12, 2500, 0, when)
        assert_allclose(result, 0, atol=1e-10)

    @pytest.mark.parametrize("when", ["end", "begin"])
    @pytest.mark.parametrize("per", [1, 2, 6, 12])
    def test_single_period_equals_ipmt(self, per, when):
        # Pins the `end + 1` offset: an off-by-one returns 0, not ipmt.
        rate, nper, pv = 0.0824 / 12, 12, 2500
        assert_allclose(
            npf.cumipmt(rate, per, per, nper, pv, 0, when),
            npf.ipmt(rate, per, nper, pv, 0, when),
            rtol=1e-9,
            atol=1e-9,
        )

    @pytest.mark.parametrize("when", ["end", "begin"])
    def test_array_start_and_end(self, when):
        # Under when='begin' the first element takes the period 1
        # exemption and the other two do not, all in one call.
        rate, nper, pv = 0.09 / 12, 360, 125000
        start = numpy.array([1, 2, 12])
        end = numpy.array([12, 12, 24])
        result = npf.cumipmt(rate, start, end, nper, pv, 0, when)
        desired = [
            _cumipmt_iterative(rate, s, e, nper, pv, 0, when)
            for s, e in zip(start, end, strict=True)
        ]
        assert_allclose(result, desired, rtol=1e-9)
