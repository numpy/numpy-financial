Loans and compounding
=====================

Multiple loans and rate scenarios
---------------------------------

``pmt`` already broadcasts its inputs. Put loans on rows and rate scenarios on
columns. Rates below are nominal annual rates, compounded monthly; fees are
excluded. Payments are returned as negative cash flows for positive loans.

.. code-block:: python

    import numpy as np
    import numpy_financial as npf

    principal = np.array([250000, 15000, 20000])
    months = np.array([300, 48, 48])
    annual_rate = np.array([0.05, 0.08, 0.07])
    rate_shifts = np.array([-0.01, 0, 0.01])
    monthly_rates = (annual_rate[:, None] + rate_shifts[None, :]) / 12
    payments = npf.pmt(monthly_rates, months[:, None], principal[:, None])
    # payments.shape == (3, 3): three loans, three scenarios

For one rate per loan, pass one-dimensional arrays of the same length. Do not
add a scenario axis unless the cross-product of loans and scenarios is wanted.
For very large portfolios, process slices of loans to limit temporary memory.
Keep full precision internally and round only where a contractual schedule
requires it.

Annual rate conversions
-----------------------

``effective_annual_rate(0.06, 12)`` converts a 6% nominal annual rate compounded
monthly into its effective annual rate. ``nominal_annual_rate`` reverses the
conversion. Neither calculates fees or regulatory APR. Frequencies must be
positive integers; zero rates and negative rates with positive compounding
factors are supported.

Changing rates
--------------

``compound_value(1000, [0.03, 0.04, 0.05])`` grows an initial balance through
three periods at 3%, 4%, and 5%, producing 1124.76. Rates are per period. There
are no intermediate deposits or withdrawals. Unlike ``fv``, the initial and
final balance have the same sign.

For an array shaped ``(accounts, periods)``, the default ``axis=-1`` compounds
each account's schedule. A one-dimensional principal array can give each
account a different starting balance.

These helpers calculate in float64 and reject nonfinite inputs. They are
proposed NumPy-oriented interfaces, not a promise of exact spreadsheet error
handling or Decimal precision.
