"""
Joint inference across strategies (post-processing of the saved OOS series).

Question, per universe and frequency: is any strategy better than the equally
weighted portfolio (EW), once the whole set of strategies is taken into
account? And which ones are significantly better, or worse?

Method
------
* Data: the T x K matrix of excess returns (net of costs, or gross) of the K
  strategies on the same OOS dates. Column `ref` is the benchmark (EW).
* Stationary bootstrap of Politis and Romano (1994): whole ROWS are resampled,
  in blocks of geometric length with a given mean, so the dependence across
  strategies and over time is preserved.
* Statistic of strategy k: d_k = SR_k - SR_ref (per-period Sharpe ratios),
  studentised with its bootstrap standard error, t_k = d_k / se_k. The
  bootstrap statistics are centred at the sample value and use the same
  standard error: t*_k = (d*_k - d_k) / se_k.
* (a) Reality-check / SPA-type test of H0 "no strategy beats the benchmark":
  p-value of max_k t_k under the bootstrap distribution of max_k t*_k
  (White, 2000; studentised as in Hansen, 2005).
* (b) Romano and Wolf (2005) step-down adjusted p-values for each strategy,
  one-sided (better than the benchmark).
* (c) The mirror test: which strategies are significantly worse.

Strategies excluded from the family
-----------------------------------
The studentised statistic t_k = d_k / se_k is a ratio. When a strategy is the
benchmark, or almost the benchmark, both the difference and its standard
error go to zero and the ratio is not informative: a return series that
differs from EW by a few basis points a year can show |t| of several units,
enter the maximum and drive the joint p-value, although the difference has no
economic content. Two explicit rules keep such strategies out:

  identical        the return series equals the benchmark on every date
                   (largest absolute difference below 1e-12);
  near-identical   the annualised tracking error against the benchmark,
                   std(r_k - r_ref) x sqrt(periods per year), is below
                   MIN_TRACKING_ERROR_ANN (10 basis points a year).

Excluded strategies get no p-value; the tables list them with the reason.

References
----------
Politis, D. N., and Romano, J. P. (1994). The stationary bootstrap. JASA 89.
White, H. (2000). A reality check for data snooping. Econometrica 68.
Hansen, P. R. (2005). A test for superior predictive ability. JBES 23.
Romano, J. P., and Wolf, M. (2005). Stepwise multiple testing as formalized
    data snooping. Econometrica 73.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MEAN_BLOCK = {"MONTHLY": 3, "DAILY": 21}     # 3 months / 21 sessions
MIN_TRACKING_ERROR_ANN = 0.001               # 10 bp a year against the benchmark
N_BOOT = 10_000
SEED = 42


def stationary_bootstrap_indices(T: int, B: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """(B, T) row indices of the stationary bootstrap, with wrap-around."""
    p = 1.0 / float(mean_block)
    starts = rng.integers(0, T, size=(B, T))
    new_block = rng.random((B, T)) < p
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def _sharpe_from_moments(m1: np.ndarray, m2: np.ndarray, T: int) -> np.ndarray:
    var = (m2 - m1 ** 2) * T / (T - 1)                 # ddof = 1
    return m1 / np.sqrt(np.maximum(var, 1e-300))


def bootstrap_sharpes(X: np.ndarray, B: int, mean_block: float, seed: int = SEED,
                      chunk: int = 500) -> np.ndarray:
    """(B, K) per-period Sharpe ratios of the K columns on each resample."""
    T, K = X.shape
    rng = np.random.default_rng(seed)
    X2 = X ** 2
    out = np.empty((B, K))
    for b0 in range(0, B, chunk):
        n = min(chunk, B - b0)
        idx = stationary_bootstrap_indices(T, n, mean_block, rng)
        # how many times each row enters each resample
        counts = np.bincount((idx + np.arange(n)[:, None] * T).ravel(),
                             minlength=n * T).reshape(n, T).astype(float)
        out[b0:b0 + n] = _sharpe_from_moments(counts @ X / T, counts @ X2 / T, T)
    return out


def _stepdown(t: np.ndarray, t_star: np.ndarray) -> np.ndarray:
    """Romano-Wolf step-down adjusted p-values for H_k: t_k <= 0 (one-sided)."""
    B, K = t_star.shape
    order = np.argsort(-t)                             # most significant first
    p_adj = np.empty(K)
    running = 0.0
    for j, k in enumerate(order):
        max_star = t_star[:, order[j:]].max(axis=1)    # hypotheses not yet rejected
        p = (1.0 + np.sum(max_star >= t[k])) / (B + 1.0)
        running = max(running, p)                      # enforce monotonicity
        p_adj[k] = running
    return p_adj


def joint_inference(X: np.ndarray, names: List[str], ref: str = "EW", B: int = N_BOOT,
                    mean_block: float = 3, seed: int = SEED, periods_per_year: int = 12,
                    min_te: float = MIN_TRACKING_ERROR_ANN) -> Dict:
    """
    Joint tests of every strategy against the benchmark.

    Args:
        X: (T, K) excess returns, one column per strategy, same dates.
        names: column names; `ref` must be one of them.
        periods_per_year, min_te: a strategy whose annualised tracking error
            against the benchmark is below min_te is left out of the family.

    Returns {"table": DataFrame (one row per alternative), "spa_p_better",
             "spa_p_worse", "n_family", "T", "B", "mean_block"}.
    """
    X = np.asarray(X, dtype=float)
    T, K = X.shape
    r = names.index(ref)
    alt = [k for k in range(K) if k != r]
    sr = _sharpe_from_moments(X.mean(axis=0), (X ** 2).mean(axis=0), T)
    sr_star = bootstrap_sharpes(X, B, mean_block, seed)
    d = sr[alt] - sr[r]
    d_star = sr_star[:, alt] - sr_star[:, [r]]
    se = d_star.std(axis=0, ddof=1)
    diff_ret = X[:, alt] - X[:, [r]]
    te = diff_ret.std(axis=0, ddof=1) * np.sqrt(periods_per_year)
    # equal up to floating-point noise: weights that are 1/N to the last digit
    # give returns that differ from EW by about 1e-17
    identical = np.abs(diff_ret).max(axis=0) < 1e-12
    live = (te >= min_te) & (se > 1e-12)
    reason = np.where(identical, "identical to " + ref,
                      np.where(~live, f"tracking error below {min_te * 1e4:g} bp/year", ""))
    t = np.full(len(alt), np.nan)
    t[live] = d[live] / se[live]
    t_star = (d_star[:, live] - d[live]) / se[live]

    table = pd.DataFrame({
        "strategy": [names[k] for k in alt], "sharpe": sr[alt], "sharpe_ref": sr[r],
        "diff": d, "se_boot": se, "t_stat": t,
        "p_better": np.nan, "p_better_rw": np.nan, "p_worse": np.nan, "p_worse_rw": np.nan,
        "in_family": live, "tracking_error_ann": te, "excluded_reason": reason,
    })
    spa_better = spa_worse = np.nan
    if live.any():
        tl = t[live]
        table.loc[live, "p_better"] = (1.0 + (t_star >= tl).sum(axis=0)) / (B + 1.0)
        table.loc[live, "p_worse"] = (1.0 + (-t_star >= -tl).sum(axis=0)) / (B + 1.0)
        table.loc[live, "p_better_rw"] = _stepdown(tl, t_star)
        table.loc[live, "p_worse_rw"] = _stepdown(-tl, -t_star)
        spa_better = (1.0 + np.sum(t_star.max(axis=1) >= tl.max())) / (B + 1.0)
        spa_worse = (1.0 + np.sum((-t_star).max(axis=1) >= (-tl).max())) / (B + 1.0)
    return {"table": table, "spa_p_better": float(spa_better), "spa_p_worse": float(spa_worse),
            "n_family": int(live.sum()), "T": T, "B": B, "mean_block": mean_block}


def run_inference(run_dir, B: int = N_BOOT, seed: int = SEED,
                  block_factors=(0.5, 1.0, 2.0)) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Joint inference for every finished task of a run, on net and gross excess
    returns, with the block length at half, once and twice its base value.
    Writes <run_dir>/inference/joint_tests.csv and joint_summary.csv.
    """
    from pathlib import Path
    from src import config as C
    from src.data import load_panel

    run_dir = Path(run_dir)
    out = run_dir / "inference"
    out.mkdir(parents=True, exist_ok=True)
    rows, summ = [], []
    for u in C.UNIVERSES:
        for f in C.FREQS:
            folder = run_dir / u / f
            if not (folder / "oos_returns_net.csv").exists():
                continue
            _, rf = load_panel(u, f)
            for series in ("net", "gross"):
                R = pd.read_csv(folder / f"oos_returns_{series}.csv", index_col=0, parse_dates=True)
                X = R.values - rf.reindex(R.index).values[:, None]
                for fac in block_factors:
                    block = max(1.0, MEAN_BLOCK[f] * fac)
                    res = joint_inference(X, list(R.columns), "EW", B, block, seed,
                                          periods_per_year=C.PERIODS_PER_YEAR[f])
                    tab = res["table"]
                    tab.insert(0, "block", block)
                    tab.insert(0, "series", series)
                    tab.insert(0, "freq", f)
                    tab.insert(0, "universe", u)
                    rows.append(tab)
                    fam = tab[tab.in_family]
                    summ.append({
                        "universe": u, "freq": f, "series": series, "block": block,
                        "base_block": fac == 1.0, "T": res["T"], "B": B,
                        "n_strategies": len(tab), "n_family": res["n_family"],
                        "spa_p_better": res["spa_p_better"], "spa_p_worse": res["spa_p_worse"],
                        "n_better_rw_5%": int((fam.p_better_rw < 0.05).sum()),
                        "n_worse_rw_5%": int((fam.p_worse_rw < 0.05).sum()),
                        "better_rw_5%": " ".join(fam.strategy[fam.p_better_rw < 0.05]),
                        "worse_rw_5%": " ".join(fam.strategy[fam.p_worse_rw < 0.05]),
                        "excluded_identical": " ".join(
                            tab.strategy[tab.excluded_reason.str.startswith("identical")]),
                        "excluded_near_identical": " ".join(
                            tab.strategy[tab.excluded_reason.str.startswith("tracking")]),
                    })
    if not rows:
        return None
    tests, summary = pd.concat(rows, ignore_index=True), pd.DataFrame(summ)
    tests.to_csv(out / "joint_tests.csv", index=False)
    summary.to_csv(out / "joint_summary.csv", index=False)
    return {"tests": tests, "summary": summary}
