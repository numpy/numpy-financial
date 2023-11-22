import numpy as np
import numpy_financial as npf


class Npv1DCashflow:

    param_names = ["cashflow_length"]
    params = [
        (1, 10, 100, 1000),
    ]

    def __init__(self):
        self.cashflows = None

    def setup(self, cashflow_length):
        rng = np.random.default_rng(0)
        self.cashflows = rng.standard_normal(cashflow_length)

    def time_1d_cashflow(self, cashflow_length):
        npf.npv(0.08, self.cashflows)


class Npv2DCashflows:

    param_names = ["n_cashflows", "cashflow_lengths"]
    params = [
        (1, 10, 100, 1000),
        (1, 10, 100, 1000),
    ]

    def __init__(self):
        self.cashflows = None

    def setup(self, n_cashflows, cashflow_lengths):
        rng = np.random.default_rng(0)
        self.cashflows = rng.standard_normal((n_cashflows, cashflow_lengths))

    def time_2d_cashflow(self, n_cashflows, cashflow_lengths):
        npf.npv(0.08, self.cashflows)
