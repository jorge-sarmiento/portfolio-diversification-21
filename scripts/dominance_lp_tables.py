"""
Linear-programming results of Block 6 (stochastic dominance), computed from the price data.

    python scripts/dominance_lp_tables.py --run-dir results/Run_<timestamp>

Outputs, written to <run-dir>/PAPER_TABLES/dominance_lp/:
    ssd_hhi_correction.csv    in-sample HHI of SSD-R with chronological and with sorted
                              cumulative sums (Table 10)
    infeasibility_rates.csv   share of rolling out-of-sample windows in which the P-Dom and
                              SSD-R programs are infeasible (Table 8), and the same share for
                              the hard return floor HSD (Section 6.3)
"""

import argparse
import io
import os
import sys
import warnings

# UTF-8 console output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))


def _latest_run():
    """Most recent results/Run_* that already contains PAPER_TABLES/."""
    rdir = os.path.join(_REPO, "results")
    if not os.path.isdir(rdir):
        return None
    runs = sorted(d for d in os.listdir(rdir)
                  if d.startswith("Run_") and os.path.isdir(os.path.join(rdir, d, "PAPER_TABLES")))
    return os.path.join(rdir, runs[-1]) if runs else None


_ap = argparse.ArgumentParser(description="Linear-programming results of Block 6 (Tables 8 and 10).")
_ap.add_argument("--run-dir", default=None,
                 help="run whose PAPER_TABLES/ receives the outputs; default: latest results/Run_*")
_ap.add_argument("--data-dir", default=os.path.join(_REPO, "data"),
                 help="price CSVs; default: <repo>/data")
_ap.add_argument("--out-dir", default=None,
                 help="default: <run-dir>/PAPER_TABLES/dominance_lp")
_args = _ap.parse_args()

RUN_DIR = os.path.abspath(_args.run_dir) if _args.run_dir else _latest_run()
if _args.out_dir:
    OUT_DIR = os.path.abspath(_args.out_dir)
elif RUN_DIR is not None and os.path.isdir(RUN_DIR):
    OUT_DIR = os.path.join(RUN_DIR, "PAPER_TABLES", "dominance_lp")
else:
    sys.exit("ERROR: no run found. Pass --run-dir results/Run_<timestamp> or --out-dir.")
DATA_DIR = os.path.abspath(_args.data_dir)

os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["Dow_Jones_30", "Global_ETFs", "Tech_Sector"]
DATASET_LABELS = {"Dow_Jones_30": "DJ30", "Global_ETFs": "ETFs", "Tech_Sector": "Tech"}


def _sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def load_prices_and_returns(dataset, monthly=False):
    """Return (log_ret: np.ndarray T×N, dates, asset_names)."""
    path = os.path.join(DATA_DIR, f"{dataset}.csv")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    if monthly:
        prices = prices.resample("ME").last()
    log_rets = np.log(prices / prices.shift(1)).dropna()
    return log_rets.values, log_rets.index, prices.columns.tolist()


def _lp_hhi(weights):
    return float(np.sum(np.asarray(weights) ** 2))


# ===========================================================================
# Table 10: SSD in-sample HHI, chronological vs sorted cumsum
# ===========================================================================
_sep("SSD in-sample HHI: chronological vs sorted cumsum")

def _ssd_lp(data, index, sorted_variant):
    """
    Run SSD LP.  Returns (weights, success_flag).
    sorted_variant=False -> chronological cumsum
    sorted_variant=True  -> cumsum after sorting by the benchmark (Eq. 22)
    """
    T, N = data.shape
    mu = np.mean(data, axis=0)   # per-period expected return (IS sample mean)
    c  = -mu                     # maximise expected return

    if sorted_variant:
        pi    = np.argsort(index)
        A_ub  = -np.cumsum(data[pi, :], axis=0)
        b_ub  = -np.cumsum(index[pi])
    else:
        A_ub  = -np.cumsum(data, axis=0)
        b_ub  = -np.cumsum(index)

    A_eq = np.ones((1, N))
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.success:
        return res.x, True
    return np.ones(N) / N, False


ssd_rows = []
for ds in DATASETS:
    for freq_label, monthly in [("MONTHLY", True), ("DAILY", False)]:
        data, dates, assets = load_prices_and_returns(ds, monthly=monthly)
        T, N = data.shape
        split = int(T * 0.70)
        is_data  = data[:split]
        is_index = is_data.mean(axis=1)   # EW benchmark over IS window

        w_orig, ok_orig = _ssd_lp(is_data, is_index, sorted_variant=False)
        w_corr, ok_corr = _ssd_lp(is_data, is_index, sorted_variant=True)

        row = {
            "Dataset":                       ds,
            "Frequency":                     freq_label,
            "IS_periods":                    split,
            "N_assets":                      N,
            "HHI_original (chrono cumsum)":  _lp_hhi(w_orig),
            "HHI_corrected (sorted cumsum)": _lp_hhi(w_corr),
            "Original_feasible":             ok_orig,
            "Corrected_feasible":            ok_corr,
        }
        ssd_rows.append(row)
        print(
            f"  {ds:15s} {freq_label}: "
            f"HHI_orig={_lp_hhi(w_orig):.4f} (feasible={ok_orig})  "
            f"HHI_corr={_lp_hhi(w_corr):.4f} (feasible={ok_corr})"
        )

ssd_df = pd.DataFrame(ssd_rows)
ssd_df.to_csv(os.path.join(OUT_DIR, "ssd_hhi_correction.csv"), index=False)
print(f"\n  -> ssd_hhi_correction.csv")


# ===========================================================================
# Table 8: infeasibility rates for HSD, P-Dom and SSD-R  (3×6 table)
#
#  Same rolling windows as the pipeline's out-of-sample phase:
#    mu_window = mean(window, axis=0) * periods_per_year   (annualized, objective)
#    benchmark  = window.mean(axis=1)                       (EW per period)
#    HSD:   A_ub = -window,           b_ub = -mean of the per-period expected returns
#    P-Dom: A_ub = -window,           b_ub = -benchmark
#    SSD-R: A_ub = -cumsum(sorted),   b_ub = -cumsum(sorted benchmark)
#  P-Dom and SSD-R give Table 8; HSD gives the result reported in Section 6.3.
# ===========================================================================
_sep("Infeasibility rates for HSD / P-Dom / SSD-R")

def _check_hsd(window, mu_period):
    """Hard floor r_t'w >= mean per-period expected return, in per-period units."""
    T, N = window.shape
    r_b   = np.mean(mu_period)
    c     = -mu_period
    A_ub  = -window
    b_ub  = np.full(T, -r_b)
    A_eq  = np.ones((1, N)); b_eq = np.array([1.0])
    res   = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0.0, 1.0)] * N, method="highs")
    return res.success

def _check_fsd(window, mu_ann, benchmark):
    T, N = window.shape
    c     = -mu_ann
    A_ub  = -window
    b_ub  = -benchmark
    A_eq  = np.ones((1, N)); b_eq = np.array([1.0])
    res   = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0.0, 1.0)] * N, method="highs")
    return res.success

def _check_ssd(window, mu_ann, benchmark):
    T, N = window.shape
    c     = -mu_ann
    pi    = np.argsort(benchmark)
    A_ub  = -np.cumsum(window[pi, :], axis=0)
    b_ub  = -np.cumsum(benchmark[pi])
    A_eq  = np.ones((1, N)); b_eq = np.array([1.0])
    res   = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0.0, 1.0)] * N, method="highs")
    return res.success


CHECKERS = {
    "HSD":   _check_hsd,
    "P-Dom": _check_fsd,
    "SSD-R": _check_ssd,
}

infeas_rates = {strat: {} for strat in CHECKERS}

for ds in DATASETS:
    for freq_label, monthly in [("MONTHLY", True), ("DAILY", False)]:
        data, dates, assets = load_prices_and_returns(ds, monthly=monthly)
        T, N = data.shape
        ppy  = 12 if monthly else 252

        split   = int(T * 0.70)
        train   = data[:split]
        test    = data[split:]
        T_test  = len(test)

        lookback = min(max(ppy, int(split * 0.2)), split)
        full_buf = np.vstack([train[-lookback:], test])   # (lookback + T_test) × N

        for strat, checker in CHECKERS.items():
            infeasible = 0
            for t in range(T_test):
                win   = full_buf[t : t + lookback]
                mu_p  = np.mean(win, axis=0)
                mu_a  = mu_p * ppy
                bench = win.mean(axis=1)
                ok = checker(win, mu_p) if strat == "HSD" else checker(win, mu_a, bench)
                if not ok:
                    infeasible += 1
            rate = infeasible / T_test
            col  = f"{DATASET_LABELS[ds]} {freq_label}"
            infeas_rates[strat][col] = rate
            print(
                f"  {strat} | {ds:15s} {freq_label}: "
                f"{infeasible}/{T_test} infeasible  ({rate:.1%})"
            )

COL_ORDER = [
    "DJ30 MONTHLY", "DJ30 DAILY",
    "ETFs MONTHLY", "ETFs DAILY",
    "Tech MONTHLY", "Tech DAILY",
]
infeas_df = pd.DataFrame(infeas_rates).T[COL_ORDER]
infeas_df.index.name = "Strategy"
infeas_df = infeas_df.map(lambda x: f"{x:.4f}")
infeas_df.to_csv(os.path.join(OUT_DIR, "infeasibility_rates.csv"))
print(f"\n  -> infeasibility_rates.csv")
print(infeas_df.to_string())

print("\n" + "="*60)
print("  All outputs written to:")
print(f"  {OUT_DIR}")
print("="*60)
