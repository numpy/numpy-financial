"""Annual rate conversions and compounding without intermediate cash flows."""

import numpy as np
import numpy.typing as npt

__all__ = ["effective_annual_rate", "nominal_annual_rate", "compound_value"]


def _result(value: npt.NDArray[np.float64]) -> np.float64 | npt.NDArray[np.float64]:
    return value[()] if value.ndim == 0 else value


def _annual_inputs(rate: npt.ArrayLike, periods: npt.ArrayLike):
    rate_array, periods_array = np.broadcast_arrays(
        np.asarray(rate, dtype=np.float64), np.asarray(periods, dtype=np.float64))
    if not np.all(np.isfinite(rate_array)):
        raise ValueError("rate must be finite")
    if not np.all(np.isfinite(periods_array) & (periods_array >= 1)
                  & (periods_array == np.floor(periods_array))):
        raise ValueError("periods_per_year must contain positive integers")
    return rate_array, periods_array


def effective_annual_rate(
    nominal_rate: npt.ArrayLike, periods_per_year: npt.ArrayLike,
) -> np.float64 | npt.NDArray[np.float64]:
    """Convert a nominal annual rate to an effective annual rate.

    Parameters
    ----------
    nominal_rate : array_like
        Nominal annual rate as a decimal fraction, excluding fees.
    periods_per_year : array_like
        Positive integer number of compounding periods per year.

    Returns
    -------
    out : numpy.float64 or ndarray
        Effective annual rate. Inputs follow NumPy broadcasting rules.

    Raises
    ------
    ValueError
        If inputs are nonfinite, frequencies are not positive integers, or
        the nominal rate divided by the frequency is at or below -1.

    Notes
    -----
    Computes ``(1 + nominal_rate / periods_per_year)**periods_per_year - 1``
    using log1p and expm1 for accuracy near zero. Zero and negative rates are
    supported when the periodic compounding factor is positive. Calculations
    use float64; Decimal precision is not preserved. This is a compounding
    conversion, not a fee-inclusive or regulatory APR calculation.

    Examples
    --------
    >>> round(float(effective_annual_rate(0.06, 12)), 6)
    0.061678
    """
    rate, periods = _annual_inputs(nominal_rate, periods_per_year)
    periodic = rate / periods
    if np.any(periodic <= -1):
        raise ValueError("the periodic rate must be greater than -1")
    return _result(np.asarray(np.expm1(periods * np.log1p(periodic))))


def nominal_annual_rate(
    effective_rate: npt.ArrayLike, periods_per_year: npt.ArrayLike,
) -> np.float64 | npt.NDArray[np.float64]:
    """Convert an effective annual rate to a nominal annual rate.

    Parameters
    ----------
    effective_rate : array_like
        Effective annual rate as a decimal fraction, greater than -1.
    periods_per_year : array_like
        Positive integer number of compounding periods per year.

    Returns
    -------
    out : numpy.float64 or ndarray
        Nominal annual rate. Inputs follow NumPy broadcasting rules.

    Raises
    ------
    ValueError
        If inputs are nonfinite, frequencies are not positive integers, or
        the effective rate is at or below -1.

    Notes
    -----
    Computes ``periods_per_year * ((1 + effective_rate)**(1/periods_per_year)
    - 1)`` using log1p and expm1. Zero and negative rates are supported.
    Calculations use float64; Decimal precision is not preserved. Fees are
    not included. The output is an annual nominal rate, not a periodic rate.

    Examples
    --------
    >>> round(float(nominal_annual_rate(0.06167781186449957, 12)), 6)
    0.06
    """
    rate, periods = _annual_inputs(effective_rate, periods_per_year)
    if np.any(rate <= -1):
        raise ValueError("effective_rate must be greater than -1")
    return _result(np.asarray(periods * np.expm1(np.log1p(rate) / periods)))


def compound_value(
    principal: npt.ArrayLike, rates: npt.ArrayLike, *, axis: int = -1,
) -> np.float64 | npt.NDArray[np.float64]:
    """Grow a balance through a sequence of periodic interest rates.

    Parameters
    ----------
    principal : array_like
        Initial signed balance. A positive balance produces a positive result.
    rates : array_like
        Array with at least one dimension containing finite periodic rates
        greater than -1. Each entry applies for one period; there are no
        intermediate deposits or withdrawals.
    axis : int, optional
        Period axis, default -1. Other axes represent loans or scenarios.

    Returns
    -------
    out : numpy.float64 or ndarray
        Final balance, computed as principal times the product of (1 + rates).
        Principal broadcasts against the shape remaining after removing axis.
        An empty schedule leaves principal unchanged.

    Raises
    ------
    ValueError
        If inputs are nonfinite, rates are scalar, or any rate is at or below -1.

    Notes
    -----
    Rates are per period, not automatically annualized. Unlike fv's cash-flow
    convention, this function preserves the sign of principal. Calculations
    use float64; Decimal precision is not preserved. Logarithms are used to
    accumulate growth. Results outside float64 range can overflow/underflow.

    Examples
    --------
    >>> round(float(compound_value(1000, [0.03, 0.04, 0.05])), 2)
    1124.76
    """
    balances = np.asarray(principal, dtype=np.float64)
    schedule = np.asarray(rates, dtype=np.float64)
    if schedule.ndim == 0:
        raise ValueError("rates must have a period axis; use [rate] for one period")
    if not np.all(np.isfinite(balances)):
        raise ValueError("principal must be finite")
    if not np.all(np.isfinite(schedule) & (schedule > -1)):
        raise ValueError("rates must be finite and greater than -1")
    growth = np.exp(np.sum(np.log1p(schedule), axis=axis))
    return _result(np.asarray(balances * growth))
