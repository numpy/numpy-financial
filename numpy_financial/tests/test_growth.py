from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose

import numpy_financial as npf


@pytest.mark.parametrize('rate', [-0.03, 0, 1e-14, 0.06])
@pytest.mark.parametrize('periods', [1, 4, 12, 365])
def test_rate_conversions(rate, periods):
    with localcontext() as context:
        context.prec = 60
        expected = float((1 + Decimal(str(rate)) / periods) ** periods - 1)
    actual = npf.effective_annual_rate(rate, periods)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-28)
    assert_allclose(npf.nominal_annual_rate(actual, periods), rate,
                    rtol=1e-13, atol=1e-28)
    assert np.isscalar(actual)


def test_conversion_broadcasting():
    rates = np.array([0.02, 0.04, 0.06])[:, None]
    periods = np.array([1, 4, 12, 365])
    result = npf.effective_annual_rate(rates, periods)
    assert result.shape == (3, 4)
    assert_allclose(result, (1 + rates / periods) ** periods - 1)


@pytest.mark.parametrize('function', [npf.effective_annual_rate,
                                     npf.nominal_annual_rate])
@pytest.mark.parametrize('periods', [0, -1, 1.5, np.nan, np.inf])
def test_invalid_frequency(function, periods):
    with pytest.raises(ValueError, match='positive integers'):
        function(0.05, periods)


@pytest.mark.parametrize('function,rate,periods', [
    (npf.effective_annual_rate, -12, 12),
    (npf.nominal_annual_rate, -1, 12),
    (npf.effective_annual_rate, np.nan, 12),
    (npf.nominal_annual_rate, np.inf, 12),
])
def test_invalid_rates(function, rate, periods):
    with pytest.raises(ValueError):
        function(rate, periods)


def test_changing_rates():
    assert_allclose(npf.compound_value(1000, [0.03, 0.04, 0.05]), 1124.76)
    assert_allclose(npf.compound_value(-1000, [0.03, 0.04, 0.05]), -1124.76)


def test_schedules_broadcast_axis():
    rates = np.array([[0.03, 0.04, 0.05], [-0.01, 0, 0.02]])
    principal = np.array([1000, 2000])
    expected = principal.copy().astype(float)
    for period in range(3):
        expected *= 1 + rates[:, period]
    assert_allclose(npf.compound_value(principal, rates), expected)
    assert_allclose(npf.compound_value(principal, rates.T, axis=0), expected)


def test_empty_schedule():
    assert_allclose(npf.compound_value([1000, 2000], np.empty((2, 0))), [1000, 2000])


def test_constant_rate_matches_growth():
    assert_allclose(npf.compound_value(25000, np.full(120, 0.04 / 12)),
                    25000 * (1 + 0.04 / 12) ** 120)


@pytest.mark.parametrize('rates', [[-1], [np.nan], [np.inf], 0.05])
def test_invalid_schedule(rates):
    with pytest.raises(ValueError):
        npf.compound_value(1000, rates)


def test_invalid_principal():
    with pytest.raises(ValueError):
        npf.compound_value(np.inf, [0.03])
