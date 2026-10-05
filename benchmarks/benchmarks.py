import numpy as np

import numpy_financial as npf


class Pmt:
    """Compare scalar overhead with array throughput for loan payments."""

    param_names = ["size", "rate", "when"]
    params = [["scalar", 1, 10, 1000], [0.0, 0.05], ["end", "begin"]]

    def setup(self, size, rate, when):
        self.rate = rate
        self.nper = 120
        self.pv = 1_000_000.0
        self.fv = -100_000.0
        if size != "scalar":
            self.rate = np.full(size, rate)
            self.nper = np.full(size, self.nper)
            self.pv = np.full(size, self.pv)
        self.when = when

    def time_pmt(self, size, rate, when):
        npf.pmt(self.rate, self.nper, self.pv, self.fv, self.when)


class Npv2D:

    param_names = ["n_cashflows", "cashflow_lengths", "rates_lengths"]
    params = [
        (1, 10, 100),
        (1, 10, 100),
        (1, 10, 100),
    ]

    def __init__(self):
        self.rates = None
        self.cashflows = None

    def setup(self, n_cashflows, cashflow_lengths, rates_lengths):
        rng = np.random.default_rng(0)
        cf_shape = (n_cashflows, cashflow_lengths)
        self.cashflows = rng.standard_normal(cf_shape)
        self.rates = rng.standard_normal(rates_lengths)

    def time_for_loop(self, n_cashflows, cashflow_lengths, rates_lengths):
        for rate in self.rates:
            for cashflow in self.cashflows:
                npf.npv(rate, cashflow)

    def time_broadcast(self, n_cashflows, cashflow_lengths, rates_lengths):
        npf.npv(self.rates, self.cashflows)
