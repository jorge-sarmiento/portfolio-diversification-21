"""
Check of the MVN solver against an independent method.

    python scripts/check_mvn_solver.py [--dates 20] [--workers N]

For a sample of OOS dates per universe and frequency the inputs of MVN are
rebuilt as in the pipeline (annualised mean and Ledoit-Wolf covariance of the
returns of the window in excess of the risk-free rate) and two solutions of
Eq. (9),

    min  g(w) = lambda w'Sigma w - (1 - lambda)(w'mu)^2,   1'w = 1,  w >= 0,

are compared:

  solver     MeanSquaredVariance: differential evolution with the candidates
             projected onto the simplex;
  frontier   a parametric search: for a target return m, the minimum-variance
             portfolio with mu'w = m is a convex QP; g is then minimised over m
             on a grid of 200 values between min(mu) and max(mu), refined
             twice around the best point. Fixing mu'w partitions the feasible
             set, so this gives the global minimum of g up to the resolution
             of the grid.

lambda is the one selected in the run when --run-dir holds its outputs, and
--lam otherwise. Writes mvn_solver_check.csv to the run folder (or to the
current folder) and prints a summary by universe and frequency.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import warnings                                      # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from qpsolvers import solve_qp                       # noqa: E402
from sklearn.covariance import LedoitWolf            # noqa: E402

from src import config as C                          # noqa: E402
from src.data import load_panel, schedule_indices    # noqa: E402
from src.strategies.base import BaseStrategy         # noqa: E402
from src.strategies.markowitz import MeanSquaredVariance   # noqa: E402


def frontier_solution(mu: np.ndarray, cov: np.ndarray, lam: float, points: int = 200, rounds: int = 2):
    """Global minimum of g by a one-dimensional search over the target return."""
    n = len(mu)
    P, q = 2.0 * cov, np.zeros(n)
    A, lb, ub = np.vstack([np.ones(n), mu]), np.zeros(n), np.ones(n)

    def at(m):
        w = solve_qp(P, q, A=A, b=np.array([1.0, m]), lb=lb, ub=ub, solver="clarabel")
        if w is None:
            return np.inf, None
        w = np.clip(w, 0.0, None)
        w = w / w.sum()
        return lam * w @ cov @ w - (1.0 - lam) * (w @ mu) ** 2, w

    lo, hi = float(mu.min()), float(mu.max())
    best_g, best_w = np.inf, None
    for _ in range(rounds + 1):
        grid = np.linspace(lo, hi, points)
        vals = [at(m) for m in grid]
        k = int(np.argmin([v[0] for v in vals]))
        if vals[k][0] < best_g:
            best_g, best_w = vals[k]
        lo, hi = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, points - 1)])
    return float(best_g), best_w


def check(job):
    universe, freq, date, lam, excess = job
    ppy = C.PERIODS_PER_YEAR[freq]
    mu = excess.mean(axis=0) * ppy
    cov = BaseStrategy.clean_matrix(LedoitWolf().fit(excess).covariance_ * ppy)
    g = lambda w: float(lam * w @ cov @ w - (1.0 - lam) * (w @ mu) ** 2)
    t0 = time.process_time()
    w, res = MeanSquaredVariance(risk_aversion=lam).solve(mu, cov)
    cpu = time.process_time() - t0
    g_fr, w_fr = frontier_solution(mu, cov, lam)
    return {"universe": universe, "freq": freq, "date": date, "N": len(mu), "lambda": lam,
            "g_solver": g(w), "g_frontier": g_fr,
            "rel_to_frontier": (g(w) - g_fr) / max(abs(g_fr), 1e-12),
            "max_abs_weight_diff": float(np.abs(w - w_fr).max()),
            "on_simplex": bool(abs(w.sum() - 1.0) < 1e-12 and w.min() >= 0.0),
            "generations": int(res.nit), "converged": bool(res.success), "cpu_s": cpu}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run-dir", type=Path, default=C.RESULTS_DIR / "main")
    ap.add_argument("--lam", type=float, default=0.5, help="lambda when the run has no selection")
    ap.add_argument("--dates", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0, help="0 = number of physical CPU cores")
    a = ap.parse_args()
    if a.workers <= 0:
        from src.runner import physical_cores
        a.workers = physical_cores()
    run_dir = a.run_dir.resolve()
    jobs = []
    for u in C.UNIVERSES:
        for f in C.FREQS:
            rets, rf = load_panel(u, f)
            sch = schedule_indices(rets.index, f)
            params = run_dir / u / f / "best_params.csv"
            lam = a.lam
            if params.exists():
                lam = float(pd.read_csv(params).set_index("Strategy").loc["MSV", "risk_aversion"])
            excess = rets.values - rf.values[:, None]
            for k in np.unique(np.linspace(0, len(sch["oos_idx"]) - 1, a.dates).astype(int)):
                i = sch["oos_idx"][k]
                jobs.append((u, f, rets.index[i].date(), lam, excess[i - sch["window"]:i].copy()))
    print(f"{len(jobs)} dates", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        df = pd.DataFrame(list(pool.map(check, jobs, chunksize=2)))
    out = (run_dir if run_dir.exists() else Path.cwd()) / "mvn_solver_check.csv"
    df.to_csv(out, index=False)
    s = df.groupby(["freq", "universe"], sort=False).agg(
        dates=("date", "size"), N=("N", "first"), lam=("lambda", "first"), on_simplex=("on_simplex", "sum"),
        within_0_05pct=("rel_to_frontier", lambda x: int((x < 5e-4).sum())),
        largest_rel_gap=("rel_to_frontier", "max"), largest_weight_diff=("max_abs_weight_diff", "max"),
        hit_max_generations=("generations", lambda x: int((x >= 500).sum())), cpu_s=("cpu_s", "mean")).reset_index()
    pd.set_option("display.width", 220)
    print(s.round(6).to_string())
    print(f"on the simplex: {int(df.on_simplex.sum())} of {len(df)}; within 0.05% of the frontier optimum: "
          f"{int((df.rel_to_frontier < 5e-4).sum())} of {len(df)} (largest {df.rel_to_frontier.max():.2e}); wrote {out}")


if __name__ == "__main__":
    main()
