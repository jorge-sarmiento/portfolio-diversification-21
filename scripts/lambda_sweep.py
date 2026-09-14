"""
Out-of-sample Sharpe ratio of MV over a grid of lambda values: the curves of Figure 1.

    python scripts/lambda_sweep.py [--out-dir DIR]

Protocol:
  - monthly log returns from month-end prices (as _resample_to_monthly in main.py)
  - 70/30 in-sample / out-of-sample split: int(T * 0.70)
  - rolling 12-month estimation window for every universe
  - mu annualised: window.mean(axis=0) * 12
  - Sigma annualised: LedoitWolf.fit(window).covariance_ * 12
  - QP via qpsolvers/clarabel, as MeanVariance in src/strategies/markowitz.py:
      min lambda*w'*Sigma*w - mu'*w  (qpsolvers: P = 2*lambda*Sigma, q = -mu)
  - realised return: dot(r_t, w), as in the pipeline
  - OOS Sharpe: mean(r_oos)*12 / (std(r_oos, ddof=1)*sqrt(12))

Grid: 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 25.

Outputs (default results/lambda_sweep/): lambda_curve_{DS}_rolling12.csv and
lambda_curve_all_rolling12.csv. The curves of the article are stored in
archive/LAMBDA_SWEEP/, from which scripts/plot_figure1.py draws Figure 1.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from qpsolvers import solve_qp
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
OUT_DIR  = BASE / "results" / "lambda_sweep"
if "--out-dir" in sys.argv:
    OUT_DIR = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()

DATASETS = {
    "DJ30": DATA_DIR / "Dow_Jones_30.csv",
    "ETFs": DATA_DIR / "Global_ETFs.csv",
    "Tech": DATA_DIR / "Tech_Sector.csv",
}

LAMBDA_GRID = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0]


# ============================================================
# helpers
# ============================================================
def load_monthly_returns(path: Path) -> tuple:
    """
    Load daily prices CSV and resample to month-end log returns.
    Mirrors _resample_to_monthly + resample_to_monthly('M').last() in main.py.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.resample("ME").last().dropna()   # month-end prices
    prices = df.values                        # (T+1, N)
    log_ret = np.log(prices[1:] / (prices[:-1] + 1e-10))
    dates_m = df.index[1:]
    return log_ret, dates_m


def solve_mv_qp(mu, Sigma, lam):
    """
    min  lam*w'*Sigma*w - mu'*w
    qpsolvers form: min (1/2)*w'*(2*lam*Sigma)*w + (-mu)'*w
    s.t. 1'w=1, w>=0
    """
    n = len(mu)
    P  = 2.0 * lam * Sigma
    q  = -mu
    A  = np.ones((1, n))
    b  = np.array([1.0])
    lb = np.zeros(n)
    ub = np.ones(n)
    w  = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver="clarabel")
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def oos_sharpe_at_lambda(R, oos_periods, lam, window=12):
    """Rolling-window walk-forward with a fixed window; annualised OOS Sharpe ratio."""
    T, N = R.shape
    start = T - oos_periods
    realised = []

    for t in range(oos_periods):
        abs_t = start + t
        w_start = abs_t - window
        cur_win  = R[w_start : abs_t]     # rolling 12-month window

        mu    = cur_win.mean(axis=0) * 12
        Sigma = LedoitWolf().fit(cur_win).covariance_ * 12

        w = solve_mv_qp(mu, Sigma, lam)
        realised.append(float(R[abs_t] @ w))

    r = np.array(realised)
    mu_ann  = r.mean() * 12
    std_ann = r.std(ddof=1) * np.sqrt(12)
    return mu_ann / std_ann if std_ann > 0 else np.nan


# ============================================================
# main
# ============================================================
def analyze(name, path):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    R, dates = load_monthly_returns(path)
    T, N = R.shape
    is_ = int(T * 0.70)   # as in the pipeline: split_idx = int(T * 0.70)
    oos = T - is_
    print(f"  T={T}  N={N}  IS={is_}  OOS={oos}")

    rows = []
    for lam in LAMBDA_GRID:
        s = oos_sharpe_at_lambda(R, oos, lam, window=12)
        rows.append({"lambda": lam, "oos_sharpe": s})
        print(f"    lambda={lam:>6.2f}  Sharpe={s:.4f}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.insert(0, "universe", name)
    df.to_csv(OUT_DIR / f"lambda_curve_{name}_rolling12.csv", index=False)
    return df


def main():
    print("MV out-of-sample Sharpe ratio over the lambda grid")
    print(f"Protocol: rolling 12-month window, LW shrinkage, mu*12, Sigma*12")

    all_curves = []
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"[SKIP] {name}: not found at {path}")
            continue
        all_curves.append(analyze(name, path))

    if all_curves:
        combined = pd.concat(all_curves, ignore_index=True)
        combined.to_csv(OUT_DIR / "lambda_curve_all_rolling12.csv", index=False)
        print(f"\nSaved curves to: {OUT_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()
