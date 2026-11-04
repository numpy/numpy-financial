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


class MortgageScenarios:
    """Compare per-loan calls with existing pmt broadcasting."""

    param_names = ["loans", "scenarios"]
    params = [(100, 1000, 10000), (1, 5)]

    def setup(self, loans, scenarios):
        self.principal = np.linspace(75000, 450000, loans)
        self.periods = np.resize([120, 180, 240, 300, 360], loans)
        annual_rates = np.linspace(0.03, 0.065, loans)
        shifts = np.linspace(-0.01, 0.01, scenarios) if scenarios > 1 else [0]
        self.rates = (annual_rates[:, None] + np.asarray(shifts)[None, :]) / 12

    def time_individual_calls(self, loans, scenarios):
        return np.array([
            [npf.pmt(self.rates[i, j], self.periods[i], self.principal[i])
             for j in range(scenarios)]
            for i in range(loans)
        ])

    def time_broadcast(self, loans, scenarios):
        return npf.pmt(self.rates, self.periods[:, None], self.principal[:, None])
