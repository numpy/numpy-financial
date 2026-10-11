import numpy as np

import numpy_financial as npf


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
