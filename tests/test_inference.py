"""
Validation of the joint inference on simulated data.

Size   under the null (every strategy has the same Sharpe ratio as the
       benchmark) the SPA-type test must reject at 5% in about 5% of samples.
       This holds for serially independent returns (about 5-7% with 120
       observations) and at the daily sample size. With positive serial
       correlation and about 100 observations the block bootstrap
       underestimates the variance and the test is liberal: about 9% for
       AR(1) noise with coefficient 0.2; the test below bounds that case.
Power  with one clearly superior strategy, the test must reject, the
       Romano-Wolf step-down must flag that strategy and not the others, and
       the mirror test must flag a clearly inferior one.

Run with:  python -m unittest tests.test_inference
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                   # noqa: E402

from src.inference import (bootstrap_sharpes, joint_inference,   # noqa: E402
                              stationary_bootstrap_indices)

NAMES = ["EW", "A", "B", "C", "D", "E"]


def simulate(rng, T=120, K=6, extra=None, rho=0.0):
    """Correlated strategies with a common factor and AR(1) noise; equal Sharpe unless `extra`."""
    common = rng.normal(0.0, 0.03, size=(T, 1))
    eps = rng.normal(0.0, 0.02, size=(T, K))
    for t in range(1, T):
        eps[t] += rho * eps[t - 1]
    X = 0.005 + common + eps
    if extra:
        for k, shift in extra.items():
            X[:, k] += shift
    return X


class TestBootstrap(unittest.TestCase):
    def test_indices_are_blocks(self):
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(200, 400, 5, rng)
        self.assertEqual(idx.shape, (400, 200))
        self.assertTrue(((idx >= 0) & (idx < 200)).all())
        step = (np.diff(idx, axis=1) % 200) == 1
        # a block continues with probability 1 - 1/5 (a new block can also start at idx + 1)
        self.assertAlmostEqual(step.mean(), 0.8 + 0.2 / 200, delta=0.01)

    def test_resampled_sharpe_matches_direct_computation(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0.01, 0.05, size=(60, 4))
        got = bootstrap_sharpes(X, 50, 3, seed=7, chunk=20)
        rng2 = np.random.default_rng(7)
        want = []
        for n in (20, 20, 10):
            for row in stationary_bootstrap_indices(60, n, 3, rng2):
                S = X[row]
                want.append(S.mean(axis=0) / S.std(axis=0, ddof=1))
        np.testing.assert_allclose(got, np.array(want), rtol=1e-10)

    def test_same_seed_same_result(self):
        X = simulate(np.random.default_rng(2))
        a = joint_inference(X, NAMES, B=300, mean_block=3, seed=11)
        b = joint_inference(X, NAMES, B=300, mean_block=3, seed=11)
        self.assertEqual(a["spa_p_better"], b["spa_p_better"])
        np.testing.assert_array_equal(a["table"].p_better_rw.values, b["table"].p_better_rw.values)

    def test_strategy_equal_to_benchmark_is_left_out(self):
        X = simulate(np.random.default_rng(3))
        X[:, 2] = X[:, 0]
        res = joint_inference(X, NAMES, B=200, mean_block=3)
        row = res["table"].set_index("strategy").loc["B"]
        self.assertFalse(row.in_family)
        self.assertTrue(np.isnan(row.p_better_rw))
        self.assertEqual(res["n_family"], 4)
        self.assertEqual(row.excluded_reason, "identical to EW")

    def test_near_copy_of_the_benchmark_is_left_out(self):
        """
        A strategy 3 bp/year away from EW would show a large |t| and drive the
        maximum; with the tracking-error rule it gets no p-value.
        """
        rng = np.random.default_rng(4)
        X = simulate(rng)
        X[:, 2] = X[:, 0] + 0.00005 + rng.normal(0.0, 0.0001 / np.sqrt(12), size=len(X))
        free = joint_inference(X, NAMES, B=400, mean_block=3, min_te=0.0)
        row = free["table"].set_index("strategy").loc["B"]
        self.assertTrue(row.in_family)
        self.assertGreater(abs(row.t_stat), 3.0)             # the ratio blows up
        self.assertLess(row.tracking_error_ann, 0.001)
        res = joint_inference(X, NAMES, B=400, mean_block=3)
        row = res["table"].set_index("strategy").loc["B"]
        self.assertFalse(row.in_family)
        self.assertTrue(np.isnan(row.t_stat))
        self.assertEqual(row.excluded_reason, "tracking error below 10 bp/year")
        self.assertEqual(res["n_family"], 4)
        self.assertGreater(res["spa_p_better"], free["spa_p_better"])


class TestSizeAndPower(unittest.TestCase):
    def _rates(self, rho, n_sim=400):
        rng = np.random.default_rng(100)
        reject_better = reject_worse = 0
        for s in range(n_sim):
            res = joint_inference(simulate(rng, rho=rho), NAMES, B=400, mean_block=3, seed=s)
            reject_better += res["spa_p_better"] < 0.05
            reject_worse += res["spa_p_worse"] < 0.05
        return reject_better / n_sim, reject_worse / n_sim

    def test_size_with_serial_dependence_is_bounded(self):
        """AR(1) noise, coefficient 0.2, T = 120: liberal, but below 13%."""
        for rate in self._rates(rho=0.2):
            self.assertGreater(rate, 0.03)
            self.assertLess(rate, 0.13)

    def test_size_under_the_null(self):
        rng = np.random.default_rng(100)
        n_sim, reject_better, reject_worse, any_rw = 400, 0, 0, 0
        for s in range(n_sim):
            res = joint_inference(simulate(rng), NAMES, B=400, mean_block=3, seed=s)
            reject_better += res["spa_p_better"] < 0.05
            reject_worse += res["spa_p_worse"] < 0.05
            any_rw += (res["table"].p_better_rw < 0.05).any()
        # nominal 5%; the Monte Carlo error of 400 samples is about 1.1 points
        for rate in (reject_better / n_sim, reject_worse / n_sim, any_rw / n_sim):
            self.assertGreater(rate, 0.015)
            self.assertLess(rate, 0.095)

    def test_detects_a_superior_and_an_inferior_strategy(self):
        rng = np.random.default_rng(200)
        n_sim, hit_spa, hit_up, hit_down, false_up = 100, 0, 0, 0, 0
        for s in range(n_sim):
            X = simulate(rng, extra={3: 0.02, 5: -0.02}, rho=0.2)   # C clearly better, E clearly worse
            res = joint_inference(X, NAMES, B=400, mean_block=3, seed=s)
            tab = res["table"].set_index("strategy")
            hit_spa += res["spa_p_better"] < 0.05
            hit_up += tab.loc["C", "p_better_rw"] < 0.05
            hit_down += tab.loc["E", "p_worse_rw"] < 0.05
            false_up += (tab.drop("C").p_better_rw < 0.05).any()
        self.assertGreater(hit_spa / n_sim, 0.90)
        self.assertGreater(hit_up / n_sim, 0.90)
        self.assertGreater(hit_down / n_sim, 0.90)
        self.assertLess(false_up / n_sim, 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
