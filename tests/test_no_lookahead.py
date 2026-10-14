"""
No-look-ahead tests of the walk-forward engine.

Three layers, each at monthly and daily frequency:

1. Calendar   on the real panels, the estimation window of every evaluated row
              ends strictly before that row, calibration precedes the OOS
              period, and both fall inside their date ranges.
2. Engine     a spy strategy records the rows it is given; the engine must
              hand it exactly rows [t - W, t) for each evaluated row t, in the
              rolling calibration and out of sample.
3. Behaviour  changing the data at and after a row cannot change any weight
              decided before it, and changing OOS data cannot change the
              calibration.

Run with:  python -m unittest tests.test_no_lookahead
"""
import logging
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from src import config as C                          # noqa: E402
from src.data import have_snapshot                   # noqa: E402
from src.data import load_panel, schedule_indices  # noqa: E402
from src.strategies.base import BaseStrategy         # noqa: E402
from src.strategies.basic import GlobalMinimumVariance   # noqa: E402
from src.strategies.markowitz import MeanVariance    # noqa: E402
from src.engine import WalkForwardEngine          # noqa: E402

HAVE_DATA = have_snapshot()

# Synthetic designs with the proportions of the calendar of the article
DESIGNS = {
    "MONTHLY": {"T": 216, "window": 60, "calib": (60, 132), "oos": (132, 216), "ppy": 12},
    "DAILY": {"T": 4548, "window": 1260, "calib": (1278, 2788), "oos": (2788, 4548), "ppy": 252},
}


def synthetic(freq: str, n_assets: int = 4, seed: int = 0):
    d = DESIGNS[freq]
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(d["T"], n_assets))
    dates = pd.bdate_range("2008-01-01", periods=d["T"])
    sch = {"window": d["window"], "calib_idx": np.arange(*d["calib"]),
           "oos_idx": np.arange(*d["oos"]), "static_end": d["oos"][0]}
    return data, dates, sch, d["ppy"]


class Spy(BaseStrategy):
    """
    Puts all the weight on asset 0 and logs what it was given: the first and
    last row of the raw window, and the mean of column 0 implied by the
    expected-return vector (which identifies the rows the moments came from).
    """
    log = []
    mu = []
    ppy = 1

    def __init__(self, dummy: float = 0.0):
        super().__init__(name="spy")

    def set_raw_data(self, data: np.ndarray) -> None:
        # column 0 of the synthetic panel holds the row number
        Spy.log.append((int(round(data[0, 0])), int(round(data[-1, 0])), len(data)))

    def optimize_weights(self, expected_returns, covariance_matrix):
        Spy.mu.append(expected_returns[0] / Spy.ppy)
        w = np.zeros(len(expected_returns))
        w[0] = 1.0
        return w


class TestCalendar(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, "data snapshot not present")
    def test_windows_end_before_evaluated_period(self):
        for universe in C.UNIVERSES:
            for freq in C.FREQS:
                with self.subTest(universe=universe, freq=freq):
                    rets, rf = load_panel(universe, freq)
                    dates, sch = rets.index, schedule_indices(rets.index, freq)
                    w = sch["window"]
                    self.assertTrue(dates.is_monotonic_increasing and dates.is_unique)
                    self.assertEqual(w, C.ESTIMATION_WINDOW[freq])
                    for idx in (sch["calib_idx"], sch["oos_idx"]):
                        self.assertTrue((idx - w >= 0).all())
                        # last row of the window is t - 1, strictly earlier than t
                        self.assertTrue((dates[idx - 1] < dates[idx]).all())
                        self.assertTrue((np.diff(idx) == 1).all())
                    c, o = dates[sch["calib_idx"]], dates[sch["oos_idx"]]
                    self.assertGreaterEqual(c[0], pd.Timestamp(C.CALIB_START))
                    self.assertLessEqual(c[-1], pd.Timestamp(C.CALIB_END))
                    self.assertGreaterEqual(o[0], pd.Timestamp(C.OOS_START))
                    self.assertLessEqual(o[-1], pd.Timestamp(C.OOS_END))
                    self.assertLess(c[-1], o[0])
                    self.assertEqual(sch["static_end"], sch["oos_idx"][0])
                    self.assertEqual(sch["calib_idx"][-1] + 1, sch["oos_idx"][0])


class TestEngineWindows(unittest.TestCase):
    def _run(self, freq: str):
        data, dates, sch, ppy = synthetic(freq)
        data[:, 0] = np.arange(len(data))            # row number, readable by the spy
        Spy.log, Spy.mu, Spy.ppy = [], [], ppy
        eng = WalkForwardEngine(Spy, "SPY", periods_per_year=ppy,
                                force_param_grid={"dummy": [0.0, 1.0]})
        res = eng.run(data, dates, rf=np.zeros(len(data)), schedule=sch)
        return data, sch, res, list(Spy.log), np.array(Spy.mu)

    def _check(self, freq: str):
        data, sch, res, log, mu = self._run(freq)
        w, calib, oos = sch["window"], sch["calib_idx"], sch["oos_idx"]
        # static pass: 2 combinations on rows [0, static_end)
        static = log[:2]
        self.assertEqual(static, [(0, sch["static_end"] - 1, sch["static_end"])] * 2)
        # rolling calibration: 2 combinations x every calibration row
        rolling = log[2:2 + 2 * len(calib)]
        expected = [(i - w, i - 1, w) for i in calib] * 2
        self.assertEqual(rolling, expected)
        # out of sample (the call before it is the in-sample weight of the best combination)
        tail = log[-len(oos):]
        self.assertEqual(tail, [(i - w, i - 1, w) for i in oos])
        # the realised return is the one of the evaluated row itself
        np.testing.assert_array_equal(res["returns"], oos.astype(float))
        # the moments come from the same rows: mean of rows [i - w, i) of column 0
        mid = lambda idx: (2 * idx - w - 1) / 2.0
        np.testing.assert_allclose(mu[2:2 + 2 * len(calib)], np.tile(mid(calib), 2), rtol=1e-12)
        np.testing.assert_allclose(mu[-len(oos):], mid(oos), rtol=1e-12)
        np.testing.assert_allclose(mu[:2], (sch["static_end"] - 1) / 2.0, rtol=1e-12)
        # nothing the spy ever saw reaches the first OOS row before it is evaluated
        self.assertTrue(all(last < oos[0] for _, last, _ in log[:-len(oos)]))

    def test_monthly(self):
        self._check("MONTHLY")

    def test_daily(self):
        self._check("DAILY")


class TestBehaviour(unittest.TestCase):
    def _engine(self, cls, code, ppy):
        return WalkForwardEngine(cls, code, periods_per_year=ppy)

    def _check(self, freq: str, oos_rows: int):
        data, dates, sch, ppy = synthetic(freq, seed=1)
        sch = dict(sch, oos_idx=sch["oos_idx"][:oos_rows])
        if freq == "DAILY":                          # keep the daily calibration short
            sch["calib_idx"] = sch["calib_idx"][-40:]
        rf = np.full(len(data), 1e-4)
        cut = sch["oos_idx"][oos_rows // 2]          # first perturbed row
        future = data.copy()
        future[cut:] += np.random.default_rng(9).normal(0, 0.05, size=future[cut:].shape)
        all_oos = data.copy()
        all_oos[sch["oos_idx"][0]:] *= -3.0
        for cls, code in ((GlobalMinimumVariance, "GMV"), (MeanVariance, "MV")):
            with self.subTest(freq=freq, strategy=code):
                base = self._engine(cls, code, ppy).run(data, dates, rf=rf, schedule=sch)
                pert = self._engine(cls, code, ppy).run(future, dates, rf=rf, schedule=sch)
                k = oos_rows // 2
                # weights decided for rows <= cut use data before cut only
                np.testing.assert_array_equal(base["weights"][:k + 1], pert["weights"][:k + 1])
                np.testing.assert_array_equal(base["returns"][:k], pert["returns"][:k])
                self.assertFalse(np.array_equal(base["weights"][k + 1:], pert["weights"][k + 1:]))
                # calibration cannot see the OOS period
                oosp = self._engine(cls, code, ppy).run(all_oos, dates, rf=rf, schedule=sch)
                self.assertEqual(base["best_params"], oosp["best_params"])
                self.assertEqual([t["sharpe"] for t in base["trials_log"]],
                                 [t["sharpe"] for t in oosp["trials_log"]])
                np.testing.assert_array_equal(base["weights"][0], oosp["weights"][0])

    def test_monthly(self):
        self._check("MONTHLY", oos_rows=24)

    def test_daily(self):
        self._check("DAILY", oos_rows=20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
