Rate conversion and compounding
===============================

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
