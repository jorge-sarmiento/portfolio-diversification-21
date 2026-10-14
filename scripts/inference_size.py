"""
Size of the joint test at the serial correlation observed in the real series.

    python scripts/inference_size.py --run-dir results/main

1. For each universe, first-order autocorrelation of the monthly return
   differences against EW (net returns), one value per strategy: median and
   maximum across strategies.
2. Simulation under the null with T = 84, as many strategies as the run has,
   and AR(1) noise with that coefficient: share of samples in which the test
   "no strategy beats EW" rejects at 5%.

Writes <run-dir>/inference/ac1_differences_MONTHLY.csv, ac1_by_universe_MONTHLY.csv and
size_calibrated.csv.
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

from src import config as C                       # noqa: E402
from src.inference import MEAN_BLOCK, joint_inference   # noqa: E402


def ac1(x: np.ndarray) -> float:
    x = x - x.mean()
    den = float(x @ x)
    return float(x[1:] @ x[:-1] / den) if den > 0 else np.nan


def simulate_null(rng, T: int, K: int, rho: float) -> np.ndarray:
    """K strategies with the same Sharpe ratio: common factor plus AR(1) noise."""
    common = rng.normal(0.0, 0.03, size=(T, 1))
    burn = 50
    e = rng.normal(0.0, 0.02, size=(T + burn, K))
    for t in range(1, T + burn):
        e[t] += rho * e[t - 1]
    return 0.005 + common + e[burn:] * np.sqrt(1 - rho ** 2)


def size(rho: float, T: int, K: int, n_sim: int, B: int, seed: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    names = ["EW"] + [f"S{k}" for k in range(1, K)]
    better = worse = 0
    for s in range(n_sim):
        res = joint_inference(simulate_null(rng, T, K, rho), names, B=B,
                              mean_block=MEAN_BLOCK["MONTHLY"], seed=s)
        better += res["spa_p_better"] < 0.05
        worse += res["spa_p_worse"] < 0.05
    return {"reject_better_%": 100 * better / n_sim, "reject_worse_%": 100 * worse / n_sim}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run-dir", type=Path, default=C.RESULTS_DIR / "main")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    run_dir = a.run_dir.resolve()
    out = run_dir / "inference"
    out.mkdir(parents=True, exist_ok=True)

    rows, K, T = [], 0, 0
    for u in C.UNIVERSES:
        path = run_dir / u / "MONTHLY" / "oos_returns_net.csv"
        if not path.exists():
            continue
        R = pd.read_csv(path, index_col=0, parse_dates=True)
        K, T = R.shape[1], len(R)
        for s in R.columns:
            if s == "EW":
                continue
            d = R[s].values - R["EW"].values
            rows.append({"universe": u, "strategy": s,
                         "ac1_difference": ac1(d) if np.abs(d).max() > 0 else np.nan,
                         "tracking_error_ann_%": float(d.std(ddof=1) * np.sqrt(12) * 100)})
    ac = pd.DataFrame(rows)
    ac.to_csv(out / "ac1_differences_MONTHLY.csv", index=False)
    g = ac.groupby("universe", sort=False).ac1_difference
    per_u = pd.DataFrame({"n_strategies": g.count(), "ac1_min": g.min(), "ac1_median": g.median(),
                          "ac1_max": g.max()}).reset_index()

    res = []
    cases = [("all universes, median", float(ac.ac1_difference.median())),
             ("all universes, maximum", float(ac.ac1_difference.max())),
             ("no serial correlation", 0.0)]
    cases += [(f"{r.universe}, median", float(r.ac1_median)) for r in per_u.itertuples()]
    cases += [(f"{r.universe}, maximum", float(r.ac1_max)) for r in per_u.itertuples()]
    for label, rho in cases:
        res.append({"case": label, "ac1": rho, "T": T, "K": K, "samples": a.sims, "B": a.boot,
                    **size(rho, T, K, a.sims, a.boot)})
        print(res[-1], flush=True)
    per_u.to_csv(out / "ac1_by_universe_MONTHLY.csv", index=False)
    pd.DataFrame(res).to_csv(out / "size_calibrated.csv", index=False)


if __name__ == "__main__":
    main()
