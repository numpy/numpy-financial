Loan portfolios
===============

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
