"""
Feasible grid of the weight-bound strategies.

Infeasible combinations (N*l > 1 or N*u < 1) are removed before the
calibration, are not logged as DSR trials and can never be selected;
combinations that force EW (N*u = 1) stay in the grid; strategies without
weight bounds are not affected by the rule.

Run with:  python -m unittest tests.test_feasible_grid
"""
import logging
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.CRITICAL)

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from src import grid                              # noqa: E402
from src.strategies.advanced import DiversifiedMeanVariance   # noqa: E402
from src.strategies.markowitz import (MeanVariance, WeightConstrainedMV,   # noqa: E402
                                      WeightLowerBoundConstrainedMV, WeightUpperBoundConstraint)
from src.engine import WalkForwardEngine          # noqa: E402


def panel(n_assets, T=140, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.004, 0.04, size=(T, n_assets))
    sch = {"window": 60, "calib_idx": np.arange(60, 110), "oos_idx": np.arange(110, T),
           "static_end": 110}
    return data, pd.bdate_range("2010-01-01", periods=T), sch


class TestRule(unittest.TestCase):
    def test_analytic_check(self):
        self.assertEqual(grid.check({"lower_bound": 0.05}, 27)[:2], (False, False))   # 1.35 > 1
        self.assertEqual(grid.check({"lower_bound": 0.02}, 27)[:2], (True, False))
        self.assertEqual(grid.check({"upper_bound": 0.10}, 7)[:2], (False, False))    # 0.70 < 1
        self.assertEqual(grid.check({"upper_bound": 0.10}, 10)[:2], (True, True))     # forces EW
        self.assertEqual(grid.check({"upper_bound": 0.20}, 10)[:2], (True, False))
        self.assertEqual(grid.check({"lower_bound": 0.05, "upper_bound": 0.10}, 14)[:2], (True, False))
        self.assertEqual(grid.check({"risk_aversion": 2.0}, 3)[:2], (True, False))

    def test_rule_agrees_with_the_strategies(self):
        """A combination is removed exactly when the strategy itself reports it infeasible."""
        mu, cov = np.full(7, 0.05), np.eye(7) * 0.04
        cases = [(WeightLowerBoundConstrainedMV, {"lower_bound": lb}, n)
                 for lb in (0.01, 0.02, 0.05) for n in (7, 10, 20, 27, 30)]
        cases += [(WeightUpperBoundConstraint, {"upper_bound": ub}, n)
                  for ub in (0.10, 0.20, 0.30) for n in (7, 8, 9, 10, 11, 14)]
        cases += [(WeightConstrainedMV, {"lower_bound": lb, "upper_bound": ub}, n)
                  for lb in (0.01, 0.05) for ub in (0.10, 0.30) for n in (7, 10, 27)]
        for cls, combo, n in cases:
            with self.subTest(cls=cls.__name__, combo=combo, n=n):
                try:
                    cls(risk_aversion=2.0, **combo).optimize_weights(np.full(n, 0.05), np.eye(n) * 0.04)
                    raised = False
                except ValueError as e:
                    raised = "Infeasible" in str(e)
                self.assertEqual(grid.is_feasible(combo, n), not raised)


class TestEngine(unittest.TestCase):
    def _run(self, cls, code, n_assets, use_filter):
        data, dates, sch = panel(n_assets)
        eng = WalkForwardEngine(cls, code, periods_per_year=12, dsr_trials="rolling")
        if use_filter:
            eng.combo_filter = grid.is_feasible
        return eng, eng.run(data, dates, rf=np.zeros(len(data)), schedule=sch)

    def test_infeasible_combinations_are_removed_and_not_counted(self):
        eng, res = self._run(WeightLowerBoundConstrainedMV, "WLBC", 27, True)
        self.assertEqual(len(eng.excluded_combos), 3)                 # l = 0.05 with 3 lambdas
        self.assertTrue(all(c["lower_bound"] == 0.05 for c in eng.excluded_combos))
        self.assertEqual(res["n_trials"], 6)
        self.assertNotEqual(res["best_params"]["lower_bound"], 0.05)
        self.assertTrue(all(t["params"]["lower_bound"] != 0.05 for t in res["trials_log"]))
        # without the rule the nine combinations compete
        _, old = self._run(WeightLowerBoundConstrainedMV, "WLBC", 27, False)
        self.assertEqual(old["n_trials"], 9)

    def test_upper_bound_grid(self):
        eng, res = self._run(WeightUpperBoundConstraint, "WUBC", 7, True)
        self.assertEqual({c["upper_bound"] for c in eng.excluded_combos}, {0.10})
        self.assertEqual(res["n_trials"], 6)
        eng, res = self._run(WeightUpperBoundConstraint, "WUBC", 10, True)     # N*u = 1 stays
        self.assertEqual(eng.excluded_combos, [])
        self.assertEqual(res["n_trials"], 9)

    def test_strategies_without_bounds_are_unaffected(self):
        for cls, code in ((MeanVariance, "MV"), (DiversifiedMeanVariance, "DMV")):
            with self.subTest(strategy=code):
                e1, a = self._run(cls, code, 9, True)
                _, b = self._run(cls, code, 9, False)
                self.assertEqual(e1.excluded_combos, [])
                np.testing.assert_array_equal(a["weights"], b["weights"])
                np.testing.assert_array_equal(a["returns"], b["returns"])
                self.assertEqual(a["best_params"], b["best_params"])
                self.assertEqual([t["sharpe"] for t in a["trials_log"]],
                                 [t["sharpe"] for t in b["trials_log"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
