"""
Daily risk-free rate: the monthly RF spread over the sessions of the month.

Run with:  python -m unittest tests.test_rf
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                   # noqa: E402

from src import config as C                          # noqa: E402
from src.data import have_snapshot                   # noqa: E402
from src.data import load_panel, load_rf          # noqa: E402

HAVE_DATA = have_snapshot()


@unittest.skipUnless(HAVE_DATA, "data snapshot not present")
class TestDailyRF(unittest.TestCase):
    def setUp(self):
        self.m = load_rf("MONTHLY")
        self.d = load_rf("DAILY")
        self.window = (self.m.index >= "2008-01-01") & (self.m.index <= "2025-12-31")

    def test_monthly_sum_equals_monthly_rate(self):
        sums = self.d.groupby(self.d.index.to_period("M")).sum()
        m = self.m[self.window]
        s = sums[(sums.index >= "2008-01") & (sums.index <= "2025-12")]
        self.assertEqual(len(m), 216)
        self.assertEqual(len(s), 216)
        np.testing.assert_allclose(s.values, m.values, rtol=0, atol=1e-15)

    def test_constant_within_month_and_not_rounded(self):
        g = self.d.groupby(self.d.index.to_period("M"))
        self.assertTrue((g.nunique() == 1).all())
        d = self.d[(self.d.index >= "2008-01-01") & (self.d.index <= "2025-12-31")]
        self.assertGreater(d.nunique(), 50)              # the published column has 3 values

    def test_published_column_still_available(self):
        pub = load_rf("DAILY", daily="published")
        pub = pub[(pub.index >= "2008-01-01") & (pub.index <= "2025-12-31")]
        self.assertEqual(sorted(np.unique(np.round(pub.values * 100, 6))), [0.0, 0.01, 0.02])
        self.assertTrue(pub.index.equals(self.d[(self.d.index >= "2008-01-01")
                                                & (self.d.index <= "2025-12-31")].index))

    def test_annualised_means_agree_out_of_sample(self):
        _, rd = load_panel("Commodities", "DAILY")
        _, rm = load_panel("Commodities", "MONTHLY")
        yd = rd[rd.index >= C.OOS_START].sum() / 7
        ym = rm[rm.index >= C.OOS_START].sum() / 7
        self.assertAlmostEqual(yd, ym, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
