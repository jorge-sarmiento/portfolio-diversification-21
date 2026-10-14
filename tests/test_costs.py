"""
Turnover against drifted weights and transaction costs.

Run with:  python -m unittest tests.test_costs
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                   # noqa: E402

from src.costs import net_returns, turnover_series   # noqa: E402


class TestTurnover(unittest.TestCase):
    def test_drift_by_hand(self):
        w = np.array([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
        r = np.array([[0.10, -0.10], [0.00, 0.00], [0.05, 0.05]])
        t = turnover_series(w, r, "drift")
        self.assertEqual(t[0], 0.0)
        # after the first period the weights drift to 0.55 and 0.45
        self.assertAlmostEqual(t[1], abs(0.5 - 0.55) + abs(0.5 - 0.45), places=14)
        # no price move in the second period: the drifted weights are the targets
        self.assertAlmostEqual(t[2], 1.0, places=14)

    def test_buy_and_hold_has_no_turnover(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.01, 0.05, size=(30, 4))
        w = np.empty((30, 4))
        w[0] = 0.25
        for k in range(1, 30):
            grown = w[k - 1] * (1.0 + r[k - 1])
            w[k] = grown / grown.sum()
        np.testing.assert_allclose(turnover_series(w, r, "drift"), 0.0, atol=1e-14)

    def test_target_convention(self):
        w = np.array([[0.5, 0.5], [0.2, 0.8]])
        np.testing.assert_allclose(turnover_series(w, None, "target"), [0.0, 0.6])

    def test_net_returns(self):
        gross, turn = np.array([0.01, 0.02]), np.array([0.0, 0.5])
        np.testing.assert_allclose(net_returns(gross, turn, cost_bps=10.0), [0.01, 0.02 - 0.0005])


if __name__ == "__main__":
    unittest.main(verbosity=2)
