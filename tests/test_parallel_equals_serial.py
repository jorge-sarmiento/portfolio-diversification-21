"""
Parallel execution must not change any result.

Runs the same two tasks once in this process and once in a process pool and
requires every output file to be byte-identical.

Run with:  python -m unittest tests.test_parallel_equals_serial
"""
import filecmp
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

from src.data import have_snapshot                   # noqa: E402
from src.runner import run_tasks                  # noqa: E402

HAVE_DATA = have_snapshot()
STRATEGIES = ["EW", "GMV", "MV", "WUBC", "DMV", "MD", "RP", "HRP", "SSD", "CVaR"]
UNIVERSES = ["Commodities", "Countries_Europe"]
OUTPUTS = ["oos_returns_gross.csv", "oos_returns_net.csv", "oos_turnover.csv",
           "best_params.csv", "dsr_trials.csv"] + [f"weights/{s}.csv" for s in STRATEGIES]


class TestParallelEqualsSerial(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, "data snapshot not present")
    def test_same_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            def tasks(sub):
                return [(u, "MONTHLY", Path(tmp) / sub, STRATEGIES, 12, False)
                        for u in UNIVERSES]
            run_tasks(tasks("serial"), workers=1)
            run_tasks(tasks("parallel"), workers=2)
            for u in UNIVERSES:
                for name in OUTPUTS:
                    a = Path(tmp) / "serial" / u / "MONTHLY" / name
                    b = Path(tmp) / "parallel" / u / "MONTHLY" / name
                    with self.subTest(universe=u, file=name):
                        self.assertTrue(filecmp.cmp(a, b, shallow=False))


class TestRowParallelEqualsSerial(unittest.TestCase):
    """OOS rows of MSV solved in worker processes against the serial loop."""

    @unittest.skipUnless(HAVE_DATA, "data snapshot not present")
    def test_msv_rows(self):
        from src.runner import run_task
        names = ["oos_returns_gross.csv", "oos_returns_net.csv", "oos_turnover.csv",
                 "best_params.csv", "dsr_trials.csv", "weights/MSV.csv"]
        with tempfile.TemporaryDirectory() as tmp:
            for sub, workers in (("serial", 1), ("rows", 3)):
                run_task("Commodities", "MONTHLY", Path(tmp) / sub, ["MSV"], 6, False,
                         row_workers=workers)
            for name in names:
                a = Path(tmp) / "serial" / "Commodities" / "MONTHLY" / name
                b = Path(tmp) / "rows" / "Commodities" / "MONTHLY" / name
                with self.subTest(file=name):
                    self.assertTrue(filecmp.cmp(a, b, shallow=False))

    def test_fallback_is_applied_in_order(self):
        """A failing optimiser must reuse the last valid weights, as the serial loop does."""
        import numpy as np
        import pandas as pd
        from concurrent.futures import ProcessPoolExecutor
        from src.engine import WalkForwardEngine, init_oos_worker
        from tests.helpers_flaky import Flaky
        rng = np.random.default_rng(3)
        data = rng.normal(0, 0.01, size=(80, 4))
        data[:, 0] = np.arange(80)
        dates = pd.bdate_range("2020-01-01", periods=80)
        sch = {"window": 20, "calib_idx": np.arange(20, 50), "oos_idx": np.arange(50, 80),
               "static_end": 50}
        serial = WalkForwardEngine(Flaky, "FLAKY", periods_per_year=12).run(data, dates, schedule=sch)
        with ProcessPoolExecutor(max_workers=3, initializer=init_oos_worker, initargs=(data,)) as pool:
            par = WalkForwardEngine(Flaky, "FLAKY", periods_per_year=12).run(
                data, dates, schedule=sch, oos_pool=pool)
        np.testing.assert_array_equal(serial["weights"], par["weights"])
        np.testing.assert_array_equal(serial["returns"], par["returns"])
        # the test is only meaningful if some rows did fail and some did not
        self.assertTrue((serial["weights"][0] == 0.25).all())          # first row fails -> 1/N
        self.assertGreater(len(np.unique(serial["weights"], axis=0)), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
