"""Some simple financial calculations.

patterned after spreadsheet computations.

There is some complexity in each function
so that the functions behave like ufuncs with
broadcasting and being able to be called with scalars
or arrays (or other sequences).

Functions support the :class:`decimal.Decimal` type unless
otherwise stated.
"""

from decimal import Decimal
from typing import (
    Any,
    Callable,
    Final,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
from numpy._typing import _NestedSequence  # pyright: ignore[reportPrivateImportUsage]

from . import _cfinancial

__all__ = ['fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate',
           'irr', 'npv', 'mirr',
           'NoRealSolutionError', 'IterationsExceededError']

_ArrayT = TypeVar("_ArrayT", bound=npt.NDArray[Any])
_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, covariant=True)

# accepts arrays and scalars
class _CanArray(Protocol[_ScalarT_co]):
    def __array__(self, /) -> npt.NDArray[_ScalarT_co]: ...

# accepts arrays, rejects scalars
class _CanArrayAndLen(_CanArray[_ScalarT_co], Protocol[_ScalarT_co]):
    def __len__(self, /) -> int: ...

# casts *as* float64
_AsFloat: TypeAlias = float | np.float64
_AsFloat1D: TypeAlias = _CanArrayAndLen[np.float64] | Sequence[_AsFloat]
_AsFloatND: TypeAlias = _CanArrayAndLen[np.float64] | _NestedSequence[_AsFloat]

_AsDecimal: TypeAlias = Decimal | int

# *co*ercible to float64 (assuming NEP 50 promotion rules and `same_kind` casting)
_co_float64: TypeAlias = (
    np.float64
    | np.float32
    | np.float16
    | np.integer[Any]
    | np.bool_
)
_CoFloat: TypeAlias = float | _co_float64
_CoFloat1D: TypeAlias = _CanArrayAndLen[_co_float64] | Sequence[_CoFloat]
_CoFloatND: TypeAlias = _CanArrayAndLen[_co_float64] | _NestedSequence[_CoFloat]
# concise aliases for `_CoFloat | _CoFloat1D` and `_CoFloat | _CoFloatND`
_CoFloatOr1D: TypeAlias = float | _CanArray[_co_float64] | Sequence[_CoFloat]
_CoFloatOrND: TypeAlias = float | _CanArray[_co_float64] | _NestedSequence[_CoFloat]

# coercible to (presumed to be) number-like dtypes
_co_numeric: TypeAlias = np.floating[Any] | np.integer[Any] | np.bool_ | np.object_
_CoNumeric: TypeAlias = float | Decimal | _co_numeric
_CoNumeric1D: TypeAlias = _CanArrayAndLen[_co_numeric] | Sequence[_CoNumeric]
_CoNumericND: TypeAlias = _CanArrayAndLen[_co_numeric] | _NestedSequence[_CoNumeric]
_CoNumericOrND: TypeAlias = float | Decimal | _CanArray[_co_numeric] | _NestedSequence[_CoNumeric]

_ArrayLike: TypeAlias = npt.ArrayLike | _NestedSequence[Decimal] | Decimal

_WhenOut: TypeAlias = Literal[0, 1]
_When: TypeAlias = str | int | npt.NDArray[Any] | Iterable[str | int]

#

_when_to_num: Final[Mapping[_When, _WhenOut]] = {
    'end': 0,
    "begin": 1,
    "e": 0,
    "b": 1,
    0: 0,
    1: 1,
    "beginning": 1,
    "start": 1,
    "finish": 0,
}


class NoRealSolutionError(Exception):
    """No real solution to the problem."""


class IterationsExceededError(Exception):
    """Maximum number of iterations reached."""


def _get_output_array_shape(*arrays: npt.NDArray[Any]) -> tuple[int, ...]:
    return tuple(array.shape[0] for array in arrays)


def _ufunc_like(array: np.generic | npt.NDArray[Any]) -> Any:
    try:
        # If size of array is one, return scalar
        return array.item()
    except ValueError:
        # Otherwise, return entire array
        return array.squeeze()

@overload
def _convert_when(when: _ArrayT) -> _ArrayT: ...
@overload
def _convert_when(when: str | int) -> _WhenOut: ...  # type: ignore[overload-overlap]
@overload
def _convert_when(when: Iterable[str | int]) -> list[_WhenOut]: ...
def _convert_when(when: Any) -> Any:
    # Test to see if when has already been converted to ndarray
    # This will happen if one function calls another, for example ppmt
    if isinstance(when, np.ndarray):
        return when
    try:
        return _when_to_num[when]
    except (KeyError, TypeError):
        return [_when_to_num[x] for x in when]

@overload
def fv(
    rate: _AsFloat,
    nper: _CoFloat,
    pmt: _CoFloat,
    pv: _CoFloat,
    when: _When = 'end',
) -> float: ...
@overload
def fv(
    rate: Decimal,
    nper: _AsDecimal,
    pmt: _AsDecimal,
    pv: _AsDecimal,
    when: _When = 'end',
) -> Decimal: ...
@overload
def fv(
    rate: _AsFloat1D,
    nper: _CoFloatOr1D,
    pmt: _CoFloatOr1D,
    pv: _CoFloatOr1D,
    when: _When = 'end',
) -> npt.NDArray[np.float64]: ...
@overload
def fv(
    rate: _ArrayLike,
    nper: _ArrayLike,
    pmt: _ArrayLike,
    pv: _ArrayLike,
    when: _When = 'end',
) -> Any: ...
def fv(rate, nper, pmt, pv, when: _When = 'end'):
    """Compute the future value.

    Given:
     * a present value, `pv`
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * a (fixed) payment, `pmt`, paid either
     * at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period

    Return:
       the value at the end of the `nper` periods

    Parameters
    ----------
    rate : scalar or array_like of shape(M, )
        Rate of interest as decimal (not per cent) per period
    nper : scalar or array_like of shape(M, )
        Number of compounding periods
    pmt : scalar or array_like of shape(M, )
        Payment
    pv : scalar or array_like of shape(M, )
        Present value
    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
        When payments are due ('begin' (1) or 'end' (0)).
        Defaults to {'end', 0}.

    Returns
    -------
    out : ndarray
        Future values.  If all input is scalar, returns a scalar float.  If
        any input is array_like, returns future values for each input element.
        If multiple inputs are array_like, they all must have the same shape.

    Notes
    -----
    The future value is computed by solving the equation::

     fv +
     pv*(1+rate)**nper +
     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0

    or, when ``rate == 0``::

     fv + pv + pmt * nper == 0

    References
    ----------
    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
       Open Document Format for Office Applications (OpenDocument)v1.2,
       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
       Pre-Draft 12. Organization for the Advancement of Structured Information
       Standards (OASIS). Billerica, MA, USA. [ODT Document].
       Available:
       http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
       OpenDocument-formula-20090508.odt

    Examples
    --------
    >>> import numpy as np
    >>> import numpy_financial as npf

    What is the future value after 10 years of saving $100 now, with
    an additional monthly savings of $100.  Assume the interest rate is
    5% (annually) compounded monthly?

    >>> npf.fv(0.05/12, 10*12, -100, -100)
    15692.92889433575

    By convention, the negative sign represents cash flow out (i.e. money not
    available today).  Thus, saving $100 a month at 5% annual interest leads
    to $15,692.93 available to spend in 10 years.

    If any input is array_like, returns an array of equal shape.  Let's
    compare different interest rates from the example above.

    >>> a = np.array((0.05, 0.06, 0.07))/12
    >>> npf.fv(a, 10*12, -100, -100)
    array([15692.92889434, 16569.87435405, 17509.44688102])

    """
    when = _convert_when(when)
    rate, nper, pmt, pv, when = np.broadcast_arrays(rate, nper, pmt, pv, when)

    fv_array = np.empty_like(rate)
    zero = rate == 0
    nonzero = ~zero

    fv_array[zero] = -(pv[zero] + pmt[zero] * nper[zero])

    rate_nonzero = rate[nonzero]
    temp = (1 + rate_nonzero) ** nper[nonzero]
    fv_array[nonzero] = (
            - pv[nonzero] * temp
            - pmt[nonzero] * (1 + rate_nonzero * when[nonzero]) / rate_nonzero
            * (temp - 1)
    )  # fmt: skip

    if np.ndim(fv_array) == 0:
        # Follow the ufunc convention of returning scalars for scalar
        # and 0d array inputs.
        return fv_array.item(0)
    return fv_array

@overload
def pmt(
    rate: _AsFloat,
    nper: _CoFloat,
    pv: _CoFloat,
    fv: _CoFloat = 0,
    when: _When = 'end',
) -> float: ...
@overload
def pmt(
    rate: Decimal,
    nper: _AsDecimal,
    pv: _AsDecimal,
    fv: _AsDecimal = 0,
    when: _When = 'end',
) -> Decimal: ...
@overload
def pmt(
    rate: _AsFloatND,
    nper: _CoFloatOrND,
    pv: _CoFloatOrND,
    fv: _CoFloatOrND = 0,
    when: _When = 'end',
) -> npt.NDArray[np.float64]: ...
@overload
def pmt(
    rate: _ArrayLike,
    nper: _ArrayLike,
    pv: _ArrayLike,
    fv: _ArrayLike = 0,
    when: _When = 'end',
) -> Any: ...
def pmt(rate, nper, pv, fv: Any = 0, when: _When = 'end'):
    """Compute the payment against loan principal plus interest.

    Given:
     * a present value, `pv` (e.g., an amount borrowed)
     * a future value, `fv` (e.g., 0)
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * and (optional) specification of whether payment is made
       at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period

    Return:
       the (fixed) periodic payment.

    Parameters
    ----------
    rate : array_like
        Rate of interest (per period)
    nper : array_like
        Number of compounding periods
    pv : array_like
        Present value
    fv : array_like,  optional
        Future value (default = 0)
    when : {{'begin', 1}, {'end', 0}}, {string, int}
        When payments are due ('begin' (1) or 'end' (0))

    Returns
    -------
    out : ndarray
        Payment against loan plus interest.  If all input is scalar, returns a
        scalar float.  If any input is array_like, returns payment for each
        input element. If multiple inputs are array_like, they all must have
        the same shape.

    Notes
    -----
    The payment is computed by solving the equation::

     fv +
     pv*(1 + rate)**nper +
     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0

    or, when ``rate == 0``::

      fv + pv + pmt * nper == 0

    for ``pmt``.

    Note that computing a monthly mortgage payment is only
    one use for this function.  For example, pmt returns the
    periodic deposit one must make to achieve a specified
    future balance given an initial deposit, a fixed,
    periodically compounded interest rate, and the total
    number of periods.

    References
    ----------
    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
       Open Document Format for Office Applications (OpenDocument)v1.2,
       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
       Pre-Draft 12. Organization for the Advancement of Structured Information
       Standards (OASIS). Billerica, MA, USA. [ODT Document].
       Available:
       http://www.oasis-open.org/committees/documents.php
       ?wg_abbrev=office-formulaOpenDocument-formula-20090508.odt

    Examples
    --------
    >>> import numpy_financial as npf

    What is the monthly payment needed to pay off a $200,000 loan in 15
    years at an annual interest rate of 7.5%?

    >>> npf.pmt(0.075/12, 12*15, 200000)
    np.float64(-1854.0247200054619)

    In order to pay-off (i.e., have a future-value of 0) the $200,000 obtained
    today, a monthly payment of $1,854.02 would be required.  Note that this
    example illustrates usage of `fv` having a default value of 0.

    """
    when = _convert_when(when)
    (rate, nper, pv, fv, when) = map(np.array, [rate, nper, pv, fv, when])
    temp = (1 + rate) ** nper
    mask = (rate == 0)
    masked_rate = np.where(mask, 1, rate)
    fact = np.where(mask != 0, nper,
                    (1 + masked_rate * when) * (temp - 1) / masked_rate)
    return -(fv + pv * temp) / fact

@overload
def nper(
    rate: _CoNumeric,
    pmt: _CoNumeric,
    pv: _CoNumeric,
    fv: _CoNumeric = 0,
    when: _When = 'end',
) -> float: ...
@overload
def nper(
    rate: _CoNumericND,
    pmt: _CoNumericOrND,
    pv: _CoNumericOrND,
    fv: _CoNumericOrND = 0,
    when: _When = 'end',
) -> npt.NDArray[np.float64]: ...
@overload
def nper(
    rate: _ArrayLike,
    pmt: _ArrayLike,
    pv: _ArrayLike,
    fv: _ArrayLike = 0,
    when: _When = 'end',
) -> Any: ...
def nper(rate, pmt, pv, fv: Any = 0, when: _When = 'end'):
    """Compute the number of periodic payments.

    :class:`decimal.Decimal` type is not supported.

    Parameters
    ----------
    rate : array_like
        Rate of interest (per period)
    pmt : array_like
        Payment
    pv : array_like
        Present value
    fv : array_like, optional
        Future value
    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
        When payments are due ('begin' (1) or 'end' (0))

    Notes
    -----
    The number of periods ``nper`` is computed by solving the equation::

     fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate*((1+rate)**nper-1) = 0

    but if ``rate = 0`` then::

     fv + pv + pmt*nper = 0

    Examples
    --------
    >>> import numpy as np
    >>> import numpy_financial as npf

    If you only had $150/month to pay towards the loan, how long would it take
    to pay-off a loan of $8,000 at 7% annual interest?

    >>> print(np.round(npf.nper(0.07/12, -150, 8000), 5))
    64.07335

    So, over 64 months would be required to pay off the loan.

    The same analysis could be done with several different interest rates
    and/or payments and/or total amounts to produce an entire table.

    >>> rates = [0.05, 0.06, 0.07]
    >>> payments = [100, 200, 300]
    >>> amounts = [7_000, 8_000, 9_000]
    >>> npf.nper(rates, payments, amounts).round(3)
    array([[[-30.827, -32.987, -34.94 ],
            [-20.734, -22.517, -24.158],
            [-15.847, -17.366, -18.78 ]],
    <BLANKLINE>
           [[-28.294, -30.168, -31.857],
            [-19.417, -21.002, -22.453],
            [-15.025, -16.398, -17.67 ]],
    <BLANKLINE>
           [[-26.234, -27.891, -29.381],
            [-18.303, -19.731, -21.034],
            [-14.311, -15.566, -16.722]]])
    """
    when = _convert_when(when)
    rates = np.atleast_1d(rate).astype(np.float64)
    pmts = np.atleast_1d(pmt).astype(np.float64)
    pvs = np.atleast_1d(pv).astype(np.float64)
    fvs = np.atleast_1d(fv).astype(np.float64)
    whens = np.atleast_1d(when).astype(np.float64)

    out_shape = _get_output_array_shape(rates, pmts, pvs, fvs, whens)
    out = np.empty(out_shape)
    _cfinancial.nper(rates, pmts, pvs, fvs, whens, out)
    return _ufunc_like(out)


def _value_like(arr: npt.NDArray[Any], value: Decimal | float) -> Any:
    entry = arr.item(0)
    if isinstance(entry, Decimal):
        return Decimal(value)
    return np.array(value, dtype=arr.dtype).item(0)

@overload
def ipmt(
    rate: _AsFloat,
    per: _CoFloat,
    nper: _CoFloat,
    pv: _CoFloat,
    fv: _CoFloat = 0,
    when: _When = 'end',
) -> float: ...
@overload
def ipmt(
    rate: Decimal,
    per: _AsDecimal,
    nper: _AsDecimal,
    pv: _AsDecimal,
    fv: _AsDecimal = 0,
    when: _When = 'end',
) -> Decimal: ...
@overload
def ipmt(
    rate: _AsFloat1D,
    per: _CoFloatOr1D,
    nper: _CoFloatOr1D,
    pv: _CoFloatOr1D,
    fv: _CoFloatOr1D = 0,
    when: _When = 'end',
) -> npt.NDArray[np.float64]: ...
@overload
def ipmt(
    rate: _ArrayLike,
    per: _ArrayLike,
    nper: _ArrayLike,
    pv: _ArrayLike,
    fv: _ArrayLike = 0,
    when: _When = 'end',
) -> Any: ...
def ipmt(rate, per, nper, pv, fv: Any = 0, when: _When = 'end') -> Any:
    """Compute the interest portion of a payment.

    Parameters
    ----------
    rate : scalar or array_like of shape(M, )
        Rate of interest as decimal (not per cent) per period
    per : scalar or array_like of shape(M, )
        Interest paid against the loan changes during the life or the loan.
        The `per` is the payment period to calculate the interest amount.
    nper : scalar or array_like of shape(M, )
        Number of compounding periods
    pv : scalar or array_like of shape(M, )
        Present value
    fv : scalar or array_like of shape(M, ), optional
        Future value
    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
        When payments are due ('begin' (1) or 'end' (0)).
        Defaults to {'end', 0}.

    Returns
    -------
    out : ndarray
        Interest portion of payment.  If all input is scalar, returns a scalar
        float.  If any input is array_like, returns interest payment for each
        input element. If multiple inputs are array_like, they all must have
        the same shape.

    See Also
    --------
    ppmt, pmt, pv

    Notes
    -----
    The total payment is made up of payment against principal plus interest.

    ``pmt = ppmt + ipmt``

    Examples
    --------
    >>> import numpy as np
    >>> import numpy_financial as npf

    What is the amortization schedule for a 1 year loan of $2500 at
    8.24% interest per year compounded monthly?

    >>> principal = 2500.00

    The 'per' variable represents the periods of the loan.  Remember that
    financial equations start the period count at 1!

    >>> per = np.arange(1*12) + 1
    >>> ipmt = npf.ipmt(0.0824/12, per, 1*12, principal)
    >>> ppmt = npf.ppmt(0.0824/12, per, 1*12, principal)

    Each element of the sum of the 'ipmt' and 'ppmt' arrays should equal
    'pmt'.

    >>> pmt = npf.pmt(0.0824/12, 1*12, principal)
    >>> np.allclose(ipmt + ppmt, pmt)
    True

    >>> fmt = '{0:2d} {1:8.2f} {2:8.2f} {3:8.2f}'
    >>> for payment in per:
    ...     index = payment - 1
    ...     principal = principal + ppmt[index]
    ...     print(fmt.format(payment, ppmt[index], ipmt[index], principal))
     1  -200.58   -17.17  2299.42
     2  -201.96   -15.79  2097.46
     3  -203.35   -14.40  1894.11
     4  -204.74   -13.01  1689.37
     5  -206.15   -11.60  1483.22
     6  -207.56   -10.18  1275.66
     7  -208.99    -8.76  1066.67
     8  -210.42    -7.32   856.25
     9  -211.87    -5.88   644.38
    10  -213.32    -4.42   431.05
    11  -214.79    -2.96   216.26
    12  -216.26    -1.49    -0.00

    >>> interestpd = np.sum(ipmt)
    >>> np.round(interestpd, 2)
    np.float64(-112.98)

    """
    when = _convert_when(when)
    rate, per, nper, pv, fv, when = np.broadcast_arrays(rate, per, nper,
                                                        pv, fv, when)

    total_pmt = pmt(rate, nper, pv, fv, when)
    ipmt_array = np.array(_rbl(rate, per, total_pmt, pv, when) * rate)

    # Payments start at the first period, so payments before that
    # don't make any sense.
    ipmt_array[per < 1] = _value_like(ipmt_array, np.nan)
    # If payments occur at the beginning of a period and this is the
    # first period, then no interest has accrued.
    per1_and_begin = (when == 1) & (per == 1)
    ipmt_array[per1_and_begin] = _value_like(ipmt_array, 0)
    # If paying at the beginning we need to discount by one period.
    per_gt_1_and_begin = (when == 1) & (per > 1)
    ipmt_array[per_gt_1_and_begin] = (
            ipmt_array[per_gt_1_and_begin] / (1 + rate[per_gt_1_and_begin])
    )

    if np.ndim(ipmt_array) == 0:
        # Follow the ufunc convention of returning scalars for scalar
        # and 0d array inputs.
        return ipmt_array.item(0)
    return ipmt_array


@overload
def _rbl(
    rate: _AsFloat,
    per: _CoFloat,
    pmt: _CoFloat,
    pv: _CoFloat,
    when: _When,
) -> float: ...
@overload
def _rbl(
    rate: Decimal,
    per: _AsDecimal,
    pmt: _AsDecimal,
    pv: _AsDecimal,
    when: _When,
) -> Decimal: ...
@overload
def _rbl(
    rate: _AsFloat1D,
    per: _CoFloatOr1D,
    pmt: _CoFloatOr1D,
    pv: _CoFloatOr1D,
    when: _When,
) -> npt.NDArray[np.float64]: ...
@overload
def _rbl(
    rate: _ArrayLike,
    per: _ArrayLike,
    pmt: _ArrayLike,
    pv: _ArrayLike,
    when: _When,
) -> Any: ...
def _rbl(rate, per, pmt, pv, when: _When):
    """Remaining balance on loan.

    This function is here to simply have a different name for the 'fv'
    function to not interfere with the 'fv' keyword argument within the 'ipmt'
    function.  It is the 'remaining balance on loan' which might be useful as
    it's own function, but is easily calculated with the 'fv' function.
    """
    return fv(rate, (per - 1), pmt, pv, when)


@overload
def ppmt(
    rate: _AsFloat,
    per: _CoFloat,
    nper: _CoFloat,
    pv: _CoFloat,
    fv: _CoFloat = 0,
    when: _When = 'end',
) -> float: ...
@overload
def ppmt(
    rate: Decimal,
    per: _AsDecimal,
    nper: _AsDecimal,
    pv: _AsDecimal,
    fv: _AsDecimal = 0,
    when: _When = 'end',
) -> Decimal: ...
@overload
def ppmt(
    rate: _AsFloat1D,
    per: _CoFloatOr1D,
    nper: _CoFloatOr1D,
    pv: _CoFloatOr1D,
    fv: _CoFloatOr1D = 0,
    when: _When = 'end',
) -> npt.NDArray[np.float64]: ...
@overload
def ppmt(
    rate: _ArrayLike,
    per: _ArrayLike,
    nper: _ArrayLike,
    pv: _ArrayLike,
    fv: _ArrayLike = 0,
    when: _When = 'end',
) -> Any: ...
def ppmt(rate, per, nper, pv, fv: Any = 0, when: _When = 'end'):
    """Compute the payment against loan principal.

    Parameters
    ----------
    rate : array_like
        Rate of interest (per period)
    per : array_like, int
        Amount paid against the loan changes.  The `per` is the period of
        interest.
    nper : array_like
        Number of compounding periods
    pv : array_like
        Present value
    fv : array_like, optional
        Future value
    when : {{'begin', 1}, {'end', 0}}, {string, int}
        When payments are due ('begin' (1) or 'end' (0))

    See Also
    --------
    pmt, pv, ipmt

    """
    total = pmt(rate, nper, pv, fv, when)
    return total - ipmt(rate, per, nper, pv, fv, when)


@overload
def pv(
    rate: _AsFloat,
    nper: _CoFloat,
    pmt: _CoFloat,
    fv: _CoFloat = 0,
    when: _When = 'end',
) -> np.float64: ...
@overload
def pv(
    rate: Decimal,
    nper: _AsDecimal,
    pmt: _AsDecimal,
    fv: _AsDecimal = 0,
    when: _When = 'end',
) -> Decimal: ...
@overload
def pv(
    rate: _AsFloatND,
    nper: _CoFloatOrND,
    pmt: _CoFloatOrND,
    fv: _CoFloatOrND = 0,
    when: _When = 'end',
) -> npt.NDArray[np.float64]: ...
@overload
def pv(
    rate: _ArrayLike,
    nper: _ArrayLike,
    pmt: _ArrayLike,
    fv: _ArrayLike = 0,
    when: _When = 'end',
) -> Any: ...
def pv(rate, nper, pmt, fv: Any = 0, when: _When = 'end'):
    """Compute the present value.

    Given:
     * a future value, `fv`
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * a (fixed) payment, `pmt`, paid either
     * at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period

    Return:
       the value now

    Parameters
    ----------
    rate : array_like
        Rate of interest (per period)
    nper : array_like
        Number of compounding periods
    pmt : array_like
        Payment
    fv : array_like, optional
        Future value
    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
        When payments are due ('begin' (1) or 'end' (0))

    Returns
    -------
    out : ndarray, float
        Present value of a series of payments or investments.

    Notes
    -----
    The present value is computed by solving the equation::

     fv +
     pv*(1 + rate)**nper +
     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) = 0

    or, when ``rate = 0``::

     fv + pv + pmt * nper = 0

    for `pv`, which is then returned.

    References
    ----------
    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
       Open Document Format for Office Applications (OpenDocument)v1.2,
       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
       Pre-Draft 12. Organization for the Advancement of Structured Information
       Standards (OASIS). Billerica, MA, USA. [ODT Document].
       Available:
       http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
       OpenDocument-formula-20090508.odt

    Examples
    --------
    >>> import numpy as np
    >>> import numpy_financial as npf

    What is the present value (e.g., the initial investment)
    of an investment that needs to total $15692.93
    after 10 years of saving $100 every month?  Assume the
    interest rate is 5% (annually) compounded monthly.

    >>> npf.pv(0.05/12, 10*12, -100, 15692.93)
    np.float64(-100.00067131625819)

    By convention, the negative sign represents cash flow out
    (i.e., money not available today).  Thus, to end up with
    $15,692.93 in 10 years saving $100 a month at 5% annual
    interest, one's initial deposit should also be $100.

    If any input is array_like, ``pv`` returns an array of equal shape.
    Let's compare different interest rates in the example above:

    >>> a = np.array((0.05, 0.04, 0.03))/12
    >>> npf.pv(a, 10*12, -100, 15692.93)
    array([ -100.00067132,  -649.26771385, -1273.78633713])

    So, to end up with the same $15692.93 under the same $100 per month
    "savings plan," for annual interest rates of 4% and 3%, one would
    need initial investments of $649.27 and $1273.79, respectively.

    """
    when = _convert_when(when)
    rate, nper, pmt, fv, when = map(np.asarray, [rate, nper, pmt, fv, when])
    temp = (1 + rate) ** nper
    fact = np.where(rate == 0, nper, (1 + rate * when) * (temp - 1) / rate)
    return -(fv + pmt * fact) / temp


# Computed with Sage
#  (y + (r + 1)^n*x + p*((r + 1)^n - 1)*(r*w + 1)/r)/(n*(r + 1)^(n - 1)*x -
#  p*((r + 1)^n - 1)*(r*w + 1)/r^2 + n*p*(r + 1)^(n - 1)*(r*w + 1)/r +
#  p*((r + 1)^n - 1)*w/r)


def _g_div_gp(r, n, p, x, y, w) -> Any:
    # Evaluate g(r_n)/g'(r_n), where g =
    # fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1)
    t1 = (r + 1) ** n
    t2 = (r + 1) ** (n - 1)
    g = y + t1 * x + p * (t1 - 1) * (r * w + 1) / r
    gp = (n * t2 * x
          - p * (t1 - 1) * (r * w + 1) / (r ** 2)
          + n * p * t2 * (r * w + 1) / r
          + p * (t1 - 1) * w / r)
    return g / gp


# Use Newton's iteration until the change is less than 1e-6
#  for all values or a maximum of 100 iterations is reached.
#  Newton's rule is
#  r_{n+1} = r_{n} - g(r_n)/g'(r_n)
#     where
#  g(r) is the formula
#  g'(r) is the derivative with respect to r.
def rate(
    nper,
    pmt,
    pv,
    fv,
    when: _When = 'end',
    guess: float | Decimal | None = None,
    tol: float | Decimal | None = None,
    maxiter: int = 100,
    *,
    raise_exceptions: bool = False,
) -> Any:
    """Compute the rate of interest per period.

    Parameters
    ----------
    nper : array_like
        Number of compounding periods
    pmt : array_like
        Payment
    pv : array_like
        Present value
    fv : array_like
        Future value
    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
        When payments are due ('begin' (1) or 'end' (0))
    guess : Number, optional
        Starting guess for solving the rate of interest, default 0.1
    tol : Number, optional
        Required tolerance for the solution, default 1e-6
    maxiter : int, optional
        Maximum iterations in finding the solution
    raise_exceptions: bool, optional
        Flag to raise an exception when at least one of the rates
        cannot be computed due to having reached the maximum number of
        iterations (IterationsExceededException). Set to False as default,
        thus returning NaNs for those rates.

    Notes
    -----
    The rate of interest is computed by iteratively solving the
    (non-linear) equation::

     fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) = 0

    for ``rate``.

    References
    ----------
    Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May). Open Document
    Format for Office Applications (OpenDocument)v1.2, Part 2: Recalculated
    Formula (OpenFormula) Format - Annotated Version, Pre-Draft 12.
    Organization for the Advancement of Structured Information Standards
    (OASIS). Billerica, MA, USA. [ODT Document]. Available:
    http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
    OpenDocument-formula-20090508.odt

    """
    when = _convert_when(when)
    default_type = Decimal if isinstance(pmt, Decimal) else float

    # Handle casting defaults to Decimal if/when pmt is a Decimal and
    # guess and/or tol are not given default values
    if guess is None:
        guess = default_type('0.1')

    if tol is None:
        tol = default_type('1e-6')

    nper, pmt, pv, fv, when = map(np.asarray, [nper, pmt, pv, fv, when])

    rn: Any = guess
    iterator = 0
    close: Any = False
    while (iterator < maxiter) and not np.all(close):
        rnp1 = rn - _g_div_gp(rn, nper, pmt, pv, fv, when)
        diff = abs(rnp1 - rn)
        close = diff < tol
        iterator += 1
        rn = rnp1

    if not np.all(close):
        if np.isscalar(rn):
            if raise_exceptions:
                raise IterationsExceededError('Maximum number of iterations exceeded.')
            return default_type(np.nan)
        else:
            # Return nan's in array of the same shape as rn
            # where the solution is not close to tol.
            if raise_exceptions:
                raise IterationsExceededError(f'Maximum iterations exceeded in '
                                              f'{len(close) - close.sum()} rate(s).')
            rn[~close] = np.nan
    return rn


def _irr_default_selection(eirr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ default selection logic for IRR function when there are > 1 real solutions """
     # check sign of all IRR solutions
    same_sign = np.all(eirr > 0) if eirr[0] > 0 else np.all(eirr < 0)

    # if the signs of IRR solutions are not the same, first filter potential IRR
    # by comparing the total positive and negative cash flows.
    if not same_sign:
        pos = sum(eirr[eirr > 0])
        neg = sum(eirr[eirr < 0])
        if pos >= neg:
            eirr = eirr[eirr >= 0]
        else:
            eirr = eirr[eirr < 0]

    # pick the smallest one in magnitude and return
    abs_eirr = np.abs(eirr)
    return eirr[np.argmin(abs_eirr)]


_SelectionFunc: TypeAlias = Callable[
    [npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]


@overload
def irr(
    values: Sequence[_CoFloat],
    *,
    raise_exceptions: bool = False,
    selection_logic: _SelectionFunc = ...,
) -> float: ...
@overload
def irr(
    values: Sequence[Sequence[_CoFloat]],
    *,
    raise_exceptions: bool = False,
    selection_logic: _SelectionFunc = ...,
) -> npt.NDArray[np.float64]: ...
@overload
def irr(
    values: _ArrayLike,
    *,
    raise_exceptions: bool = False,
    selection_logic: _SelectionFunc = ...,
) -> Any: ...
def irr(values, *, raise_exceptions=False, selection_logic=_irr_default_selection):
    r"""Return the Internal Rate of Return (IRR).

    This is the "average" periodically compounded rate of return
    that gives a net present value of 0.0; for a more complete explanation,
    see Notes below.

    :class:`decimal.Decimal` type is not supported.

    Parameters
    ----------
    values : array_like, shape(N,)
        Input cash flows per time period.  By convention, net "deposits"
        are negative and net "withdrawals" are positive.  Thus, for
        example, at least the first element of `values`, which represents
        the initial investment, will typically be negative.
    raise_exceptions: bool, optional
        Flag to raise an exception when the irr cannot be computed due to
        either having all cashflows of the same sign (NoRealSolutionException) or
        having reached the maximum number of iterations (IterationsExceededException).
        Set to False as default, thus returning NaNs in the two previous
        cases.
    selection_logic: function, optional
        Function for selection logic when more than 1 real solutions is found.
        User may insert their own customised function for selection
        of IRR values.The function should accept a one-dimensional array
        of numbers and return a number.


    Returns
    -------
    out : float
        Internal Rate of Return for periodic input values.

    Notes
    -----
    The IRR is perhaps best understood through an example (illustrated
    using np.irr in the Examples section below). Suppose one invests 100
    units and then makes the following withdrawals at regular (fixed)
    intervals: 39, 59, 55, 20.  Assuming the ending value is 0, one's 100
    unit investment yields 173 units; however, due to the combination of
    compounding and the periodic withdrawals, the "average" rate of return
    is neither simply 0.73/4 nor (1.73)^0.25-1.  Rather, it is the solution
    (for :math:`r`) of the equation:

    .. math:: -100 + \\frac{39}{1+r} + \\frac{59}{(1+r)^2}
     + \\frac{55}{(1+r)^3} + \\frac{20}{(1+r)^4} = 0

    In general, for `values` :math:`= [v_0, v_1, ... v_M]`,
    irr is the solution of the equation: [G]_

    .. math:: \\sum_{t=0}^M{\\frac{v_t}{(1+irr)^{t}}} = 0

    References
    ----------
    .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,
       Addison-Wesley, 2003, pg. 348.

    Examples
    --------
    >>> import numpy_financial as npf

    >>> round(npf.irr([-100, 39, 59, 55, 20]), 5)
    0.28095
    >>> round(npf.irr([-100, 0, 0, 74]), 5)
    -0.0955
    >>> round(npf.irr([-100, 100, 0, -7]), 5)
    -0.0833
    >>> round(npf.irr([-100, 100, 0, 7]), 5)
    0.06206
    >>> round(npf.irr([-5, 10.5, 1, -8, 1]), 5)
    0.0886
    >>> npf.irr([[-100, 0, 0, 74], [-100, 100, 0, 7]]).round(5)
    array([-0.0955 ,  0.06206])

    """
    values = np.atleast_2d(values)
    if values.ndim != 2:
        raise ValueError("Cashflows must be a 2D array")

    irr_results = np.empty(values.shape[0])
    for i, row in enumerate(values):
        # If all values are of the same sign, no solution exists
        # We don't perform any further calculations and exit early
        same_sign = np.all(row > 0) if row[0] > 0 else np.all(row < 0)
        if same_sign:
            if raise_exceptions:
                raise NoRealSolutionError('No real solution exists for IRR since all '
                                          'cashflows are of the same sign.')
            irr_results[i] = np.nan

    # We aim to solve eirr such that NPV is exactly zero. This can be framed as
    # simply finding the closest root of a polynomial to a given initial guess
    # as follows:
    #           V0           V1           V2           V3
    # NPV = ---------- + ---------- + ---------- + ---------- + ... = 0
    #       (1+eirr)^0   (1+eirr)^1   (1+eirr)^2   (1+eirr)^3
    #
    # by letting g = (1+eirr), we substitute to get
    #
    # NPV = V0 * 1/g^0   + V1 * 1/g^1   +  V2 * 1/x^2  +  V3 * 1/g^3  + ... = 0
    #
    # Multiplying by g^N this becomes
    #
    # V0 * g^N   + V1 * g^{N-1}   +  V2 * g^{N-2}  +  V3 * g^{N-3}  + ... = 0
    #
    # which we solve using Newton-Raphson and then reverse out the solution
    # as eirr = g - 1 (if we are close enough to a solution)
        else:
            g = np.roots(row)
            eirr = np.real(g[np.isreal(g)]) - 1

            # Realistic IRR
            eirr = eirr[eirr >= -1]

            # If no real solution
            if len(eirr) == 0:
                if raise_exceptions:
                    raise NoRealSolutionError("No real solution is found for IRR.")
                irr_results[i] = np.nan
            # If only one real solution
            elif len(eirr) == 1:
                irr_results[i] = eirr[0]
            else:
                irr_results[i] = selection_logic(eirr)

    return _ufunc_like(irr_results)


@overload
def npv(rate: _CoNumeric, values: _CoNumericND) -> float: ...
@overload
def npv(rate: _CoNumeric1D, values: _CoNumericND) -> npt.NDArray[np.float64]: ...
@overload
def npv(rate: _ArrayLike, values: _ArrayLike) -> Any: ...
def npv(rate, values):
    r"""Return the NPV (Net Present Value) of a cash flow series.

    Parameters
    ----------
    rate : scalar or array_like shape(K, )
        The discount rate.
    values : array_like, shape(M, ) or shape(M, N)
        The values of the time series of cash flows.  The (fixed) time
        interval between cash flow "events" must be the same as that for
        which `rate` is given (i.e., if `rate` is per year, then precisely
        a year is understood to elapse between each cash flow event).  By
        convention, investments or "deposits" are negative, income or
        "withdrawals" are positive; `values` must begin with the initial
        investment, thus `values[0]` will typically be negative.

    Returns
    -------
    out : float or array shape(K, M)
        The NPV of the input cash flow series `values` at the discount
        `rate`. `out` follows the ufunc convention of returning scalars
        instead of single element arrays.

    Warnings
    --------
    ``npv`` considers a series of cashflows starting in the present (t = 0).
    NPV can also be defined with a series of future cashflows, paid at the
    end, rather than the start, of each period. If future cashflows are used,
    the first cashflow `values[0]` must be zeroed and added to the net
    present value of the future cashflows. This is demonstrated in the
    examples.

    Notes
    -----
    Returns the result of: [G]_

    .. math :: \\sum_{t=0}^{M-1}{\\frac{values_t}{(1+rate)^{t}}}

    References
    ----------
    .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,
       Addison-Wesley, 2003, pg. 346.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy_financial as npf

    Consider a potential project with an initial investment of $40 000 and
    projected cashflows of $5 000, $8 000, $12 000 and $30 000 at the end of
    each period discounted at a rate of 8% per period. To find the project's
    net present value:

    >>> rate, cashflows = 0.08, [-40_000, 5_000, 8_000, 12_000, 30_000]
    >>> np.round(npf.npv(rate, cashflows), 5)
    np.float64(3065.22267)

    It may be preferable to split the projected cashflow into an initial
    investment and expected future cashflows. In this case, the value of
    the initial cashflow is zero and the initial investment is later added
    to the future cashflows net present value:

    >>> initial_cashflow = cashflows[0]
    >>> cashflows[0] = 0
    >>> np.round(npf.npv(rate, cashflows) + initial_cashflow, 5)
    np.float64(3065.22267)

    The NPV calculation may be applied to several ``rates`` and ``cashflows``
    simulatneously. This produces an array of shape ``(len(rates), len(cashflows))``.

    >>> rates = [0.00, 0.05, 0.10]
    >>> cashflows = [[-4_000, 500, 800], [-5_000, 600, 900]]
    >>> npf.npv(rates, cashflows).round(2)
    array([[-2700.  , -3500.  ],
           [-2798.19, -3612.24],
           [-2884.3 , -3710.74]])
    """
    values_inner = np.atleast_2d(values).astype(np.float64)
    rate_inner = np.atleast_1d(rate).astype(np.float64)

    if rate_inner.ndim != 1:
        msg = "invalid shape for rates. Rate must be either a scalar or 1d array"
        raise ValueError(msg)

    if values_inner.ndim != 2:
        msg = "invalid shape for values. Values must be either a 1d or 2d array"
        raise ValueError(msg)

    output_shape = _get_output_array_shape(rate_inner, values_inner)
    out = np.empty(output_shape)
    _cfinancial.npv(rate_inner, values_inner, out)
    return _ufunc_like(out)


@overload
def mirr(
    values: _CoNumericND,
    finance_rate: _CoNumeric,
    reinvest_rate: _CoNumeric,
    *,
    raise_exceptions: bool = False,
) -> float: ...
@overload
def mirr(
    values: _CoNumericND,
    finance_rate: _CoNumeric1D,
    reinvest_rate: _CoNumeric1D,
    *,
    raise_exceptions: bool = False,
) -> npt.NDArray[np.float64]: ...
@overload
def mirr(
    values: _ArrayLike,
    finance_rate: _ArrayLike,
    reinvest_rate: _ArrayLike,
    *,
    raise_exceptions: bool = False,
) -> Any: ...
def mirr(values, finance_rate, reinvest_rate, *, raise_exceptions: bool = False):
    r"""
    Return the Modified Internal Rate of Return (MIRR).

    MIRR is a financial metric that takes into account both the cost of
    the investment and the return on reinvested cash flows. It is useful
    for evaluating the profitability of an investment with multiple cash
    inflows and outflows.

    Parameters
    ----------
    values : array_like, 1D or 2D
        Cash flows, where the first value is considered a sunk cost at time zero.
        It must contain at least one positive and one negative value.
    finance_rate : scalar or 1D array
        Interest rate paid on the cash flows.
    reinvest_rate : scalar or D array
        Interest rate received on the cash flows upon reinvestment.
    raise_exceptions: bool, optional
        Flag to raise an exception when the MIRR cannot be computed due to
        having all cash flows of the same sign (NoRealSolutionException).
        Set to False as default,thus returning NaNs in the previous case.

    Returns
    -------
    out : float or 2D array
        Modified internal rate of return

    Notes
    -----
    The MIRR formula is as follows:

    .. math::

        MIRR =
        \\left( \\frac{{FV_{positive}}}{{PV_{negative}}} \\right)^{\\frac{{1}}{{n-1}}}
        * (1+r) - 1

    where:
        - \(FV_{positive}\) is the future value of positive cash flows,
        - \(PV_{negative}\) is the present value of negative cash flows,
        - \(n\) is the number of periods.
        - \(r\) is the reinvestment rate.

    Examples
    --------
    >>> import numpy_financial as npf

    Consider a project with an initial investment of -$100
    and projected cash flows of $50, -$60, and $70 at the end of each period.
    The project has a finance rate of 10% and a reinvestment rate of 12%.

    >>> npf.mirr([-100, 50, -60, 70], 0.10, 0.12)
    -0.03909366594356467

    It is also possible to supply multiple cashflows or pairs of
    finance and reinvstment rates, note that in this case the number of elements
    in each of the rates arrays must match.

    >>> values = [
    ...             [-4500, -800, 800, 800, 600],
    ...             [-120000, 39000, 30000, 21000, 37000],
    ...             [100, 200, -50, 300, -200],
    ...         ]
    >>> finance_rate = [0.05, 0.08, 0.10]
    >>> reinvestment_rate = [0.08, 0.10, 0.12]
    >>> npf.mirr(values, finance_rate, reinvestment_rate)
    array([[-0.1784449 , -0.17328716, -0.1684366 ],
           [ 0.04627293,  0.05437856,  0.06252201],
           [ 0.35712458,  0.40628857,  0.44435295]])

    Now, let's consider the scenario where all cash flows are negative.

    >>> npf.mirr([-100, -50, -60, -70], 0.10, 0.12)
    nan

    Finally, let's explore the situation where all cash flows are positive,
    and the `raise_exceptions` parameter is set to True.

    >>> npf.mirr([
    ...    100, 50, 60, 70],
    ...    0.10, 0.12,
    ...    raise_exceptions=True
    ... ) #doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    numpy_financial._financial.NoRealSolutionError:
    No real solution exists for MIRR since  all cashflows are of the same sign.
    """
    values_inner = np.atleast_2d(values).astype(np.float64)
    finance_rate_inner = np.atleast_1d(finance_rate).astype(np.float64)
    reinvest_rate_inner = np.atleast_1d(reinvest_rate).astype(np.float64)
    n = values_inner.shape[1]

    if finance_rate_inner.size != reinvest_rate_inner.size:
        if raise_exceptions:
            raise ValueError("finance_rate and reinvest_rate must have the same size")
        return np.nan

    out_shape = _get_output_array_shape(values_inner, finance_rate_inner)
    out = np.empty(out_shape)

    for i, v in enumerate(values_inner):
        for j, (rr, fr) in enumerate(
            zip(reinvest_rate_inner, finance_rate_inner, strict=True)
        ):
            pos = v > 0
            neg = v < 0

            if not (pos.any() and neg.any()):
                if raise_exceptions:
                    raise NoRealSolutionError("No real solution exists for MIRR since"
                                              " all cashflows are of the same sign.")
                out[i, j] = np.nan
            else:
                numer = np.abs(npv(rr, v * pos))
                denom = np.abs(npv(fr, v * neg))
                out[i, j] = (numer / denom) ** (1 / (n - 1)) * (1 + rr) - 1
    return _ufunc_like(out)
