"""
Audit of silent fallbacks, from the saved weights only.

    python scripts/audit_fallbacks.py --run-dir results/main

For every strategy, universe and frequency: number of OOS dates whose weights
equal 1/N (tolerance 1e-10). Several strategies return 1/N when their solver
does not converge, and a strategy calibrated on an infeasible or binding
combination also ends at 1/N; the saved weights cannot tell these apart from
an optimum that happens to be 1/N, so the count is an upper bound on fallbacks.

For RP, whose internal fallback is the inverse-volatility portfolio, the
weights of each date are compared with the inverse-volatility portfolio
recomputed from the Ledoit-Wolf covariance of that date's window.

Writes <run-dir>/tables/fallback_audit.csv.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from sklearn.covariance import LedoitWolf            # noqa: E402

from src import config as C                       # noqa: E402
from src.data import load_panel                   # noqa: E402

TOL = 1e-10


def inverse_vol_matches(universe: str, freq: str, weights: pd.DataFrame) -> int:
    """Dates on which the RP weights equal the inverse-volatility portfolio of the window."""
    rets, _ = load_panel(universe, freq)
    data, w = rets.values, C.ESTIMATION_WINDOW[freq]
    pos = rets.index.get_indexer(weights.index)
    hits = 0
    for row, i in zip(weights.values, pos):
        cov = LedoitWolf().fit(data[i - w:i]).covariance_ * C.PERIODS_PER_YEAR[freq]
        # same regularisation as BaseStrategy.clean_matrix, which precedes the optimiser
        cov = (cov + cov.T) / 2 + 1e-5 * np.eye(cov.shape[0])
        iv = 1.0 / np.sqrt(np.diag(cov))
        hits += bool(np.abs(row - iv / iv.sum()).max() < TOL)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run-dir", type=Path, default=C.RESULTS_DIR / "main")
    a = ap.parse_args()
    run_dir = a.run_dir.resolve()
    rows = []
    for u in C.UNIVERSES:
        for f in C.FREQS:
            wdir = run_dir / u / f / "weights"
            if not (run_dir / u / f / "oos_returns_gross.csv").exists():
                continue
            for path in sorted(wdir.glob("*.csv")):
                s = path.stem
                w = pd.read_csv(path, index_col=0, parse_dates=True)
                n = w.shape[1]
                rec = {"universe": u, "freq": f, "strategy": s, "N": n, "oos_dates": len(w),
                       "dates_equal_1/N": int((np.abs(w.values - 1.0 / n).max(axis=1) < TOL).sum()),
                       "dates_equal_inverse_vol": np.nan}
                if s == "RP":
                    rec["dates_equal_inverse_vol"] = inverse_vol_matches(u, f, w)
                rows.append(rec)
    df = pd.DataFrame(rows)
    (run_dir / "tables").mkdir(exist_ok=True)
    df.to_csv(run_dir / "tables" / "fallback_audit.csv", index=False)
    hit = df[(df["dates_equal_1/N"] > 0) & (df.strategy != "EW")]
    print(hit.pivot_table(index="strategy", columns=["freq", "universe"], values="dates_equal_1/N",
                          aggfunc="sum", fill_value=0).to_string() if len(hit) else "no strategy at 1/N")
    rp = df[df.strategy == "RP"]
    print("RP dates equal to inverse volatility:",
          rp.set_index(["universe", "freq"])["dates_equal_inverse_vol"].to_dict())


if __name__ == "__main__":
    main()
