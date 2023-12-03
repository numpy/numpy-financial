import math
from decimal import Decimal

# Don't use 'import numpy as np', to avoid accidentally testing
# the versions in numpy instead of numpy_financial.
import numpy
import numpy as np
import pytest
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

import numpy_financial as npf


class TestFinancial(object):
    def test_when(self):
        # begin
        assert_equal(npf.rate(10, 20, -3500, 10000, 1),
                     npf.rate(10, 20, -3500, 10000, 'begin'))
        # end
        assert_equal(npf.rate(10, 20, -3500, 10000),
                     npf.rate(10, 20, -3500, 10000, 'end'))
        assert_equal(npf.rate(10, 20, -3500, 10000, 0),
                     npf.rate(10, 20, -3500, 10000, 'end'))

        # begin
        assert_equal(npf.pv(0.07, 20, 12000, 0, 1),
                     npf.pv(0.07, 20, 12000, 0, 'begin'))
        # end
        assert_equal(npf.pv(0.07, 20, 12000, 0),
                     npf.pv(0.07, 20, 12000, 0, 'end'))
        assert_equal(npf.pv(0.07, 20, 12000, 0, 0),
                     npf.pv(0.07, 20, 12000, 0, 'end'))

        # begin
        assert_equal(npf.pmt(0.08 / 12, 5 * 12, 15000., 0, 1),
                     npf.pmt(0.08 / 12, 5 * 12, 15000., 0, 'begin'))
        # end
        assert_equal(npf.pmt(0.08 / 12, 5 * 12, 15000., 0),
                     npf.pmt(0.08 / 12, 5 * 12, 15000., 0, 'end'))
        assert_equal(npf.pmt(0.08 / 12, 5 * 12, 15000., 0, 0),
                     npf.pmt(0.08 / 12, 5 * 12, 15000., 0, 'end'))

        # begin
        assert_equal(npf.nper(0.075, -2000, 0, 100000., 1),
                     npf.nper(0.075, -2000, 0, 100000., 'begin'))
        # end
        assert_equal(npf.nper(0.075, -2000, 0, 100000.),
                     npf.nper(0.075, -2000, 0, 100000., 'end'))
        assert_equal(npf.nper(0.075, -2000, 0, 100000., 0),
                     npf.nper(0.075, -2000, 0, 100000., 'end'))

    def test_decimal_with_when(self):
        """
        Test that decimals are still supported if the when argument is passed
        """
        # begin
        assert_equal(npf.rate(Decimal('10'), Decimal('20'), Decimal('-3500'),
                              Decimal('10000'), Decimal('1')),
                     npf.rate(Decimal('10'), Decimal('20'), Decimal('-3500'),
                              Decimal('10000'), 'begin'))
        # end
        assert_equal(npf.rate(Decimal('10'), Decimal('20'), Decimal('-3500'),
                              Decimal('10000')),
                     npf.rate(Decimal('10'), Decimal('20'), Decimal('-3500'),
                              Decimal('10000'), 'end'))
        assert_equal(npf.rate(Decimal('10'), Decimal('20'), Decimal('-3500'),
                              Decimal('10000'), Decimal('0')),
                     npf.rate(Decimal('10'), Decimal('20'), Decimal('-3500'),
                              Decimal('10000'), 'end'))

        # begin
        assert_equal(npf.pv(Decimal('0.07'), Decimal('20'), Decimal('12000'),
                            Decimal('0'), Decimal('1')),
                     npf.pv(Decimal('0.07'), Decimal('20'), Decimal('12000'),
                            Decimal('0'), 'begin'))
        # end
        assert_equal(npf.pv(Decimal('0.07'), Decimal('20'), Decimal('12000'),
                            Decimal('0')),
                     npf.pv(Decimal('0.07'), Decimal('20'), Decimal('12000'),
                            Decimal('0'), 'end'))
        assert_equal(npf.pv(Decimal('0.07'), Decimal('20'), Decimal('12000'),
                            Decimal('0'), Decimal('0')),
                     npf.pv(Decimal('0.07'), Decimal('20'), Decimal('12000'),
                            Decimal('0'), 'end'))


class TestPV:
    def test_pv(self):
        assert_almost_equal(npf.pv(0.07, 20, 12000, 0), -127128.17, 2)

    def test_pv_decimal(self):
        assert_equal(npf.pv(Decimal('0.07'), Decimal('20'), Decimal('12000'),
                            Decimal('0')),
                     Decimal('-127128.1709461939327295222005'))


class TestRate:
    def test_rate(self):
        assert_almost_equal(npf.rate(10, 0, -3500, 10000), 0.1107, 4)

    @pytest.mark.parametrize('number_type', [Decimal, float])
    @pytest.mark.parametrize('when', [0, 1, 'end', 'begin'])
    def test_rate_with_infeasible_solution(self, number_type, when):
        """
        Test when no feasible rate can be found.

        Rate will return NaN, if the Newton Raphson method cannot find a
        feasible rate within the required tolerance or number of iterations.
        This can occur if both `pmt` and `pv` have the same sign, as it is
        impossible to repay a loan by making further withdrawls.
        """
        result = npf.rate(number_type(12.0),
                          number_type(400.0),
                          number_type(10000.0),
                          number_type(5000.0),
                          when=when)
        is_nan = Decimal.is_nan if number_type == Decimal else numpy.isnan
        assert is_nan(result)

    def test_rate_decimal(self):
        rate = npf.rate(Decimal('10'), Decimal('0'), Decimal('-3500'),
                        Decimal('10000'))
        assert_equal(Decimal('0.1106908537142689284704528100'), rate)

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
        assert_raises(npf.IterationsExceededError, npf.rate, Decimal(12.0),
                      Decimal(400.0), Decimal(10000.0), Decimal(5000.0),
                      raise_exceptions=True)

    def test_rate_maximum_iterations_exception_array(self):
        # Test that if the maximum number of iterations is reached in at least
        # one rate, then npf.rate returns IterationsExceededException
        # when raise_exceptions is set to True.
        nper = 2
        pmt = 0
        pv = [-593.06, -4725.38, -662.05, -428.78, -13.65]
        fv = [214.07, 4509.97, 224.11, 686.29, -329.67]
        assert_raises(npf.IterationsExceededError, npf.rate, nper,
                      pmt, pv, fv,
                      raise_exceptions=True)


class TestNpv:
    def test_npv(self):
        assert_almost_equal(
            npf.npv(0.05, [-15000.0, 1500.0, 2500.0, 3500.0, 4500.0, 6000.0]),
            122.89, 2)

    def test_npv_decimal(self):
        assert_equal(
            npf.npv(Decimal('0.05'), [-15000, 1500, 2500, 3500, 4500, 6000]),
            Decimal('122.894854950942692161628715'))

    def test_npv_broadcast(self):
        cashflows = [
            [-15000.0, 1500.0, 2500.0, 3500.0, 4500.0, 6000.0],
            [-15000.0, 1500.0, 2500.0, 3500.0, 4500.0, 6000.0],
            [-15000.0, 1500.0, 2500.0, 3500.0, 4500.0, 6000.0],
            [-15000.0, 1500.0, 2500.0, 3500.0, 4500.0, 6000.0],
        ]
        expected_npvs = [
            [122.8948549, 122.8948549, 122.8948549, 122.8948549]
        ]
        actual_npvs = npf.npv(0.05, cashflows)
        assert_allclose(actual_npvs, expected_npvs)

    @pytest.mark.parametrize("dtype", [Decimal, float])
    def test_npv_broadcast_equals_for_loop(self, dtype):
        cashflows_str = [
            ["-15000.0", "1500.0", "2500.0", "3500.0", "4500.0", "6000.0"],
            ["-25000.0", "1500.0", "2500.0", "3500.0", "4500.0", "6000.0"],
            ["-35000.0", "1500.0", "2500.0", "3500.0", "4500.0", "6000.0"],
            ["-45000.0", "1500.0", "2500.0", "3500.0", "4500.0", "6000.0"],
        ]
        rates_str = ["-0.05", "0.00", "0.05", "0.10", "0.15"]

        cashflows = numpy.array([[dtype(x) for x in cf] for cf in cashflows_str])
        rates = numpy.array([dtype(x) for x in rates_str])

        expected = numpy.empty((len(rates), len(cashflows)), dtype=dtype)
        for i, r in enumerate(rates):
            for j, cf in enumerate(cashflows):
                expected[i, j] = npf.npv(r, cf).item()

        actual = npf.npv(rates, cashflows)
        assert_equal(actual, expected)

    @pytest.mark.parametrize("rates", ([[1, 2, 3]], np.empty(shape=(1,1,1))))
    def test_invalid_rates_shape(self, rates):
        cashflows = [1, 2, 3]
        with pytest.raises(ValueError):
            npf.npv(rates, cashflows)

    @pytest.mark.parametrize("cashflows", ([[[1, 2, 3]]], np.empty(shape=(1, 1, 1))))
    def test_invalid_cashflows_shape(self, cashflows):
        rates = [1, 2, 3]
        with pytest.raises(ValueError):
            npf.npv(rates, cashflows)


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
        res = npf.pmt(Decimal('0.08') / Decimal('12'), 5 * 12, 15000)
        tgt = Decimal('-304.1459143262052370338701494')
        assert_equal(res, tgt)

    def test_pmt_decimal_zero_rate(self):
        # Test the edge case where rate == 0.0
        res = npf.pmt(Decimal('0'), Decimal('60'), Decimal('15000'))
        tgt = -250
        assert_equal(res, tgt)

    def test_pmt_decimal_broadcast(self):
        # Test the case where we use broadcast and
        # the arguments passed in are arrays.
        res = npf.pmt([[Decimal('0'), Decimal('0.8')],
                       [Decimal('0.3'), Decimal('0.8')]],
                      [Decimal('12'), Decimal('3')],
                      [Decimal('2000'), Decimal('20000')])
        tgt = numpy.array([[Decimal('-166.6666666666666666666666667'),
                            Decimal('-19311.25827814569536423841060')],
                           [Decimal('-626.9081401700757748402586600'),
                            Decimal('-19311.25827814569536423841060')]])

        # Cannot use the `assert_allclose` because it uses isfinite under
        # the covers which does not support the Decimal type
        # See issue: https://github.com/numpy/numpy/issues/9954
        assert_equal(res[0][0], tgt[0][0])
        assert_equal(res[0][1], tgt[0][1])
        assert_equal(res[1][0], tgt[1][0])
        assert_equal(res[1][1], tgt[1][1])


class TestMirr:
    @pytest.mark.parametrize("values,finance_rate,reinvest_rate,expected", [
        ([-4500, -800, 800, 800, 600, 600, 800, 800, 700, 3000], 0.08, 0.055, 0.0666),
        ([-120000, 39000, 30000, 21000, 37000, 46000], 0.10, 0.12, 0.126094),
        ([100, 200, -50, 300, -200], 0.05, 0.06, 0.3428),
        ([39000, 30000, 21000, 37000, 46000], 0.10, 0.12, None)
    ])
    def test_mirr(self, values, finance_rate, reinvest_rate, expected):
        result = npf.mirr(values, finance_rate, reinvest_rate)

        if expected:
            decimal_part_len = len(str(expected).split('.')[1])
            assert_almost_equal(result, expected, decimal_part_len)
        else:
            assert_(numpy.isnan(result))

    @pytest.mark.parametrize('number_type', [Decimal, float])
    @pytest.mark.parametrize(
        "args, expected",
        [
            ({'values': [
                '-4500', '-800', '800', '800', '600', '600', '800', '800', '700', '3000'
            ],
              'finance_rate': '0.08', 'reinvest_rate': '0.055'
              }, '0.066597175031553548874239618'
             ),
            ({'values': ['-120000', '39000', '30000', '21000', '37000', '46000'],
              'finance_rate': '0.10', 'reinvest_rate': '0.12'
              }, '0.126094130365905145828421880'
             ),
            ({'values': ['100', '200', '-50', '300', '-200'],
              'finance_rate': '0.05', 'reinvest_rate': '0.06'
              }, '0.342823387842176663647819868'
             ),
            ({'values': ['39000', '30000', '21000', '37000', '46000'],
              'finance_rate': '0.10', 'reinvest_rate': '0.12'
              }, numpy.nan
             ),
        ],
    )
    def test_mirr_decimal(self, number_type, args, expected):
        values = [number_type(v) for v in args['values']]
        result = npf.mirr(
            values,
            number_type(args['finance_rate']),
            number_type(args['reinvest_rate'])
        )

        if expected is not numpy.nan:
            assert_almost_equal(result, number_type(expected), 15)
        else:
            assert numpy.isnan(result)

    def test_mirr_no_real_solution_exception(self):
        # Test that if there is no solution because all the cashflows
        # have the same sign, then npf.mirr returns NoRealSolutionException
        # when raise_exceptions is set to True.
        val = [39000, 30000, 21000, 37000, 46000]

        with pytest.raises(npf.NoRealSolutionError):
            npf.mirr(val, 0.10, 0.12, raise_exceptions=True)


class TestNper:
    def test_basic_values(self):
        assert_allclose(
            npf.nper([0, 0.075], -2000, 0, 100000),
            [50, 21.544944],  # Computed using Google Sheet's NPER
            rtol=1e-5,
        )

    def test_gh_18(self):
        with numpy.errstate(divide='raise'):
            assert_allclose(
                npf.nper(0.1, 0, -500, 1500),
                11.52670461,  # Computed using Google Sheet's NPER
            )

    def test_infinite_payments(self):
        with numpy.errstate(divide='raise'):
            result = npf.nper(0, -0.0, 1000)
        assert_(result == numpy.inf)

    def test_no_interest(self):
        assert_(npf.nper(0, -100, 1000) == 10)

    def test_broadcast(self):
        assert_almost_equal(npf.nper(0.075, -2000, 0, 100000., [0, 1]),
                            [21.5449442, 20.76156441], 4)


class TestPpmt:
    def test_float(self):
        assert_allclose(
            npf.ppmt(0.1 / 12, 1, 60, 55000),
            -710.25,
            rtol=1e-4
        )

    def test_decimal(self):
        result = npf.ppmt(
            Decimal('0.1') / Decimal('12'),
            Decimal('1'),
            Decimal('60'),
            Decimal('55000')
        )
        assert_equal(
            result,
            Decimal('-710.2541257864217612489830917'),
        )

    @pytest.mark.parametrize('when', [1, 'begin'])
    def test_when_is_begin(self, when):
        assert_allclose(
            npf.ppmt(0.1 / 12, 1, 60, 55000, 0, when),
            -1158.929712,  # Computed using Google Sheet's PPMT
            rtol=1e-9,
        )

    @pytest.mark.parametrize('when', [None, 0, 'end'])
    def test_when_is_end(self, when):
        args = (0.1 / 12, 1, 60, 55000, 0)
        result = npf.ppmt(*args) if when is None else npf.ppmt(*args, when)
        assert_allclose(
            result,
            -710.254126,  # Computed using Google Sheet's PPMT
            rtol=1e-9,
        )

    @pytest.mark.parametrize('when', [Decimal('1'), 'begin'])
    def test_when_is_begin_decimal(self, when):
        result = npf.ppmt(
            Decimal('0.08') / Decimal('12'),
            Decimal('1'),
            Decimal('60'),
            Decimal('15000.'),
            Decimal('0'),
            when
        )
        assert_almost_equal(
            result,
            Decimal('-302.131703'),  # Computed using Google Sheet's PPMT
            decimal=5,
        )

    @pytest.mark.parametrize('when', [None, Decimal('0'), 'end'])
    def test_when_is_end_decimal(self, when):
        args = (
            Decimal('0.08') / Decimal('12'),
            Decimal('1'),
            Decimal('60'),
            Decimal('15000.'),
            Decimal('0')
        )
        result = npf.ppmt(*args) if when is None else npf.ppmt(*args, when)
        assert_almost_equal(
            result,
            Decimal('-204.145914'),  # Computed using Google Sheet's PPMT
            decimal=5,
        )

    @pytest.mark.parametrize('args', [
        (0.1 / 12, 0, 60, 15000),
        (Decimal('0.012'), Decimal('0'), Decimal('60'), Decimal('15000'))
    ])
    def test_invalid_per(self, args):
        # Note that math.isnan() handles Decimal NaN correctly.
        assert math.isnan(npf.ppmt(*args))

    @pytest.mark.parametrize('when, desired', [
        (
            None,
            [-75.62318601, -76.25337923, -76.88882405, -77.52956425],
        ), (
            [0, 1, 'end', 'begin'],
            [-75.62318601, -75.62318601, -76.88882405, -76.88882405],
        )
    ])
    def test_broadcast(self, when, desired):
        args = (0.1 / 12, numpy.arange(1, 5), 24, 2000, 0)
        result = npf.ppmt(*args) if when is None else npf.ppmt(*args, when)
        assert_allclose(result, desired, rtol=1e-5)

    @pytest.mark.parametrize('when, desired', [
        (
            None,
            [
                Decimal('-75.62318601'),
                Decimal('-76.25337923'),
                Decimal('-76.88882405'),
                Decimal('-77.52956425')
            ],
        ), (
            [Decimal('0'), Decimal('1'), 'end', 'begin'],
            [
                Decimal('-75.62318601'),
                Decimal('-75.62318601'),
                Decimal('-76.88882405'),
                Decimal('-76.88882405')
            ]
        )
    ])
    def test_broadcast_decimal(self, when, desired):
        args = (
            Decimal('0.1') / Decimal('12'),
            numpy.arange(1, 5),
            Decimal('24'),
            Decimal('2000'),
            Decimal('0')
        )
        result = npf.ppmt(*args) if when is None else npf.ppmt(*args, when)
        assert_almost_equal(result, desired, decimal=8)


class TestIpmt:
    def test_float(self):
        assert_allclose(
            npf.ipmt(0.1 / 12, 1, 24, 2000),
            -16.666667,  # Computed using Google Sheet's IPMT
            rtol=1e-6,
        )

    def test_decimal(self):
        result = npf.ipmt(Decimal('0.1') / Decimal('12'), 1, 24, 2000)
        assert result == Decimal('-16.66666666666666666666666667')

    @pytest.mark.parametrize('when', [1, 'begin'])
    def test_when_is_begin(self, when):
        assert npf.ipmt(0.1 / 12, 1, 24, 2000, 0, when) == 0

    @pytest.mark.parametrize('when', [None, 0, 'end'])
    def test_when_is_end(self, when):
        if when is None:
            result = npf.ipmt(0.1 / 12, 1, 24, 2000)
        else:
            result = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, when)
        assert_allclose(result, -16.666667, rtol=1e-6)

    @pytest.mark.parametrize('when', [Decimal('1'), 'begin'])
    def test_when_is_begin_decimal(self, when):
        result = npf.ipmt(
            Decimal('0.1') / Decimal('12'),
            Decimal('1'),
            Decimal('24'),
            Decimal('2000'),
            Decimal('0'),
            when,
        )
        assert result == 0

    @pytest.mark.parametrize('when', [None, Decimal('0'), 'end'])
    def test_when_is_end_decimal(self, when):
        # Computed using Google Sheet's IPMT
        desired = Decimal('-16.666667')
        args = (
            Decimal('0.1') / Decimal('12'),
            Decimal('1'),
            Decimal('24'),
            Decimal('2000'),
            Decimal('0')
        )
        result = npf.ipmt(*args) if when is None else npf.ipmt(*args, when)
        assert_almost_equal(result, desired, decimal=5)

    @pytest.mark.parametrize('per, desired', [
        (0, numpy.nan),
        (1, 0),
        (2, -594.107158),
        (3, -592.971592),
    ])
    def test_gh_17(self, per, desired):
        # All desired results computed using Google Sheet's IPMT
        rate = 0.001988079518355057
        result = npf.ipmt(rate, per, 360, 300000, when="begin")
        if numpy.isnan(desired):
            assert numpy.isnan(result)
        else:
            assert_allclose(result, desired, rtol=1e-6)

    def test_broadcasting(self):
        desired = [
            numpy.nan,
            -16.66666667,
            -16.03647345,
            -15.40102862,
            -14.76028842
        ]
        assert_allclose(
            npf.ipmt(0.1 / 12, numpy.arange(5), 24, 2000),
            desired,
            rtol=1e-6,
        )

    def test_decimal_broadcasting(self):
        desired = [
            Decimal('-16.66666667'),
            Decimal('-16.03647345'),
            Decimal('-15.40102862'),
            Decimal('-14.76028842')
        ]
        result = npf.ipmt(
            Decimal('0.1') / Decimal('12'),
            list(range(1, 5)),
            Decimal('24'),
            Decimal('2000')
        )
        assert_almost_equal(result, desired, decimal=4)

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
        assert_almost_equal(
            npf.fv(Decimal('0.075'), Decimal('20'), Decimal('-2000'), 0, 0),
            Decimal('86609.36267304300040536731624'),
            decimal=10,
        )

    @pytest.mark.parametrize('when', [1, 'begin'])
    def test_when_is_begin_float(self, when):
        assert_allclose(
            npf.fv(0.075, 20, -2000, 0, when),
            93105.064874,  # Computed using Google Sheet's FV
            rtol=1e-10,
        )

    @pytest.mark.parametrize('when', [Decimal('1'), 'begin'])
    def test_when_is_begin_decimal(self, when):
        result = npf.fv(
            Decimal('0.075'),
            Decimal('20'),
            Decimal('-2000'),
            Decimal('0'),
            when,
        )
        assert_almost_equal(result, Decimal('93105.064874'), decimal=5)

    @pytest.mark.parametrize('when', [None, 0, 'end'])
    def test_when_is_end_float(self, when):
        args = (0.075, 20, -2000, 0)
        result = npf.fv(*args) if when is None else npf.fv(*args, when)
        assert_allclose(
            result,
            86609.362673,  # Computed using Google Sheet's FV
            rtol=1e-10,
        )

    @pytest.mark.parametrize('when', [None, Decimal('0'), 'end'])
    def test_when_is_end_decimal(self, when):
        args = (
            Decimal('0.075'),
            Decimal('20'),
            Decimal('-2000'),
            Decimal('0'),
        )
        result = npf.fv(*args) if when is None else npf.fv(*args, when)
        assert_almost_equal(result, Decimal('86609.362673'), decimal=5)

    def test_broadcast(self):
        result = npf.fv([[0.1], [0.2]], 5, 100, 0, [0, 1])
        # All values computed using Google Sheet's FV
        desired = [[-610.510000, -671.561000],
                   [-744.160000, -892.992000]]
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
        cashflows = numpy.array([-40000.0, 5000.0, 8000.0, 12000.0, 30000.0])
        assert_allclose(
            npf.npv(npf.irr(cashflows), cashflows),
            0,
            atol=1e-10,
            rtol=0,
        )

    @pytest.mark.parametrize('v, desired', [
        ([-150000, 15000, 25000, 35000, 45000, 60000], 0.0524),
        ([-100, 0, 0, 74], -0.0955),
        ([-100, 39, 59, 55, 20], 0.28095),
        ([-100, 100, 0, -7], -0.0833),
        ([-100, 100, 0, 7], 0.06206),
        ([-5, 10.5, 1, -8, 1], 0.0886),
    ])
    def test_basic_values(self, v, desired):
        assert_almost_equal(npf.irr(v), desired, decimal=2)

    def test_trailing_zeros(self):
        assert_almost_equal(
            npf.irr([-5, 10.5, 1, -8, 1, 0, 0, 0]),
            0.0886,
            decimal=2,
        )

    @pytest.mark.parametrize('v', [
        (1, 2, 3),
        (-1, -2, -3),
    ])
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
        cashflows = numpy.array([
            -217500.0, -217500.0, 108466.80462450592, 101129.96439328062,
            93793.12416205535, 86456.28393083003, 79119.44369960476,
            71782.60346837944, 64445.76323715414, 57108.92300592884,
            49772.08277470355, 42435.24254347826, 35098.40231225296,
            27761.56208102766, 20424.721849802358, 13087.88161857707,
            5751.041387351768, -1585.7988438735192, -8922.639075098821,
            -16259.479306324123, -23596.31953754941, -30933.159768774713,
            -38270.0, -45606.8402312253, -52943.680462450604,
            -60280.520693675906, -67617.36092490121])
        assert_almost_equal(npf.irr(cashflows), 0.12)

    def test_gh_44(self):
        # "true" value as calculated by Google sheets
        cf = [-1678.87, 771.96, 1814.05, 3520.30, 3552.95, 3584.99, 4789.91, -1]
        assert_almost_equal(npf.irr(cf), 1.00426, 4)

    def test_irr_no_real_solution_exception(self):
        # Test that if there is no solution because all the cashflows
        # have the same sign, then npf.irr returns NoRealSolutionException
        # when raise_exceptions is set to True.
        cashflows = numpy.array([40000, 5000, 8000, 12000, 30000])

        with pytest.raises(npf.NoRealSolutionError):
            npf.irr(cashflows, raise_exceptions=True)

    def test_irr_maximum_iterations_exception(self):
        # Test that if the maximum number of iterations is reached,
        # then npf.irr returns IterationsExceededException
        # when raise_exceptions is set to True.
        cashflows = numpy.array([-40000, 5000, 8000, 12000, 30000])

        with pytest.raises(npf.IterationsExceededError):
            npf.irr(cashflows, maxiter=1, raise_exceptions=True)
