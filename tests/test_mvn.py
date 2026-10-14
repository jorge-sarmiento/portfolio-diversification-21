"""
MVN (code MSV): projection onto the simplex and excess-return inputs.

Projection   Algorithm 1 of Wang and Carreira-Perpiñán (2013). On 1,000 random
             vectors -- the zero vector, vectors of the search box [0, 1]^N and
             vectors outside it -- the projection sums to one, has no negative
             component, is idempotent, satisfies the optimality conditions and
             equals the solution of the minimum-distance QP

                 min ||w - y||^2   s.t.  sum(w) = 1,  w >= 0

             solved with CLARABEL at tight tolerances (with the defaults of the
             solver its own solution is up to 2e-5 away from the projection).
Solver       the weights lie on the simplex and the search is deterministic.
Inputs       the engine estimates the moments of MVN from the returns in
             excess of the risk-free rate: mu is the mean of the total returns
             minus the mean of the rate, both annualised; the other strategies
             receive the moments of the total returns.

Run with:  python -m unittest tests.test_mvn
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from qpsolvers import solve_qp                       # noqa: E402
from sklearn.covariance import LedoitWolf            # noqa: E402

from src.engine import WalkForwardEngine, estimation_returns   # noqa: E402
from src.strategies.base import BaseStrategy         # noqa: E402
from src.strategies.markowitz import (               # noqa: E402
    MeanSquaredVariance, MeanVariance, project_simplex as P)

TIGHT = dict(tol_gap_abs=1e-13, tol_gap_rel=1e-13, tol_feas=1e-13, tol_infeas_abs=1e-13, tol_infeas_rel=1e-13)


def vectors():
    rng = np.random.default_rng(42)
    out = [np.zeros(7)]
    for k in range(999):
        n = int(rng.integers(2, 31))
        out.append(rng.uniform(0.0, 1.0, n) if k % 2 == 0 else rng.normal(0.0, 3.0, n))
    return out


class TestSimplexProjection(unittest.TestCase):
    def test_sums_to_one_and_is_non_negative(self):
        for y in vectors():
            w = P(y)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
            self.assertGreaterEqual(float(w.min()), 0.0)

    def test_idempotent(self):
        for y in vectors():
            w = P(y)
            np.testing.assert_allclose(P(w), w, rtol=0.0, atol=1e-13)

    def test_zero_vector_gives_equal_weights(self):
        np.testing.assert_allclose(P(np.zeros(7)), np.ones(7) / 7, rtol=0.0, atol=1e-15)

    def test_points_of_the_simplex_are_fixed(self):
        rng = np.random.default_rng(7)
        for n in (2, 7, 30):
            for w in (np.ones(n) / n, np.eye(n)[0], rng.dirichlet(np.ones(n))):
                np.testing.assert_allclose(P(w), w, rtol=0.0, atol=1e-15)

    def test_equals_minimum_distance_qp(self):
        worst = 0.0
        for y in vectors():
            n = len(y)
            w = solve_qp(2.0 * np.eye(n), -2.0 * y, A=np.ones((1, n)), b=np.array([1.0]),
                         lb=np.zeros(n), solver="clarabel", **TIGHT)
            self.assertIsNotNone(w)
            worst = max(worst, float(np.abs(w - P(y)).max()))
        self.assertLess(worst, 1e-8)

    def test_optimality_conditions(self):
        """KKT: w = max(y + lam, 0) for a single scalar lam, with y_i + lam <= 0 wherever w_i = 0."""
        for y in vectors():
            w = P(y)
            lam = float((w - y)[w > 0].mean())
            self.assertLess(float(np.abs((w - y)[w > 0] - lam).max()), 1e-12)
            if (w == 0).any():
                self.assertLessEqual(float((y + lam)[w == 0].max()), 1e-12)


class TestSolver(unittest.TestCase):
    def test_weights_on_the_simplex(self):
        cov = 0.02 * np.eye(6) + 0.01
        mu = np.array([0.30, 0.25, 0.60, 0.20, 0.10, 0.15])
        w, res = MeanSquaredVariance(risk_aversion=0.5).solve(mu, cov)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertGreaterEqual(float(w.min()), 0.0)
        self.assertGreater(res.nit, 0)
        # the return term dominates: the weight goes to the asset of highest mean
        self.assertEqual(int(np.argmax(w)), 2)

    def test_deterministic(self):
        mu, cov = np.array([0.05, 0.08, 0.02, 0.06]), 0.03 * np.eye(4) + 0.01
        a = MeanSquaredVariance(risk_aversion=0.5).optimize_weights(mu, cov)
        b = MeanSquaredVariance(risk_aversion=0.5).optimize_weights(mu, cov)
        np.testing.assert_array_equal(a, b)


class Recorder(BaseStrategy):
    """Records the moments the engine hands it."""
    seen = []

    def __init__(self):
        super().__init__(name="recorder")

    def optimize_weights(self, expected_returns, covariance_matrix):
        type(self).seen.append((expected_returns.copy(), covariance_matrix.copy()))
        return np.ones(len(expected_returns)) / len(expected_returns)


class ExcessRecorder(Recorder):
    excess_return_inputs = True
    seen = []


class TestExcessInputs(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.data = rng.normal(0.01, 0.05, size=(90, 4))
        self.rf = np.abs(rng.normal(0.002, 0.001, size=90))
        self.dates = pd.bdate_range("2020-01-01", periods=90)
        self.sch = {"window": 24, "calib_idx": np.arange(24, 60), "oos_idx": np.arange(60, 90), "static_end": 60}

    def test_flag(self):
        self.assertTrue(MeanSquaredVariance.excess_return_inputs)
        self.assertFalse(getattr(MeanVariance, "excess_return_inputs", False))
        np.testing.assert_array_equal(estimation_returns(self.data, self.rf, MeanVariance), self.data)
        np.testing.assert_array_equal(estimation_returns(self.data, self.rf, MeanSquaredVariance),
                                      self.data - self.rf[:, None])

    def test_moments_handed_to_the_strategies(self):
        W, clean = self.sch["window"], BaseStrategy.clean_matrix
        for cls, excess in ((Recorder, False), (ExcessRecorder, True)):
            cls.seen = []
            res = WalkForwardEngine(cls, cls.__name__, periods_per_year=12).run(
                self.data, self.dates, rf=self.rf, schedule=self.sch)
            # one call on the calibration segment, then one per OOS row
            self.assertEqual(len(cls.seen), 1 + len(self.sch["oos_idx"]))
            for (mu, cov), i in zip(cls.seen[1:], self.sch["oos_idx"]):
                window, rf_w = self.data[i - W:i], self.rf[i - W:i]
                total = window.mean(axis=0) * 12
                expected = total - rf_w.mean() * 12 if excess else total
                np.testing.assert_allclose(mu, expected, rtol=0.0, atol=1e-13)
                x = window - rf_w[:, None] if excess else window
                np.testing.assert_allclose(cov, clean(LedoitWolf().fit(x).covariance_ * 12), rtol=0.0, atol=1e-13)
            # realised returns are those of the total returns in both cases
            np.testing.assert_allclose(res["returns"], self.data[self.sch["oos_idx"]].mean(axis=1), atol=1e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
