"""
Tables of a run, built from the series written by src/runner.py.

For each universe and frequency: one row per strategy with the annualised net
return and volatility, the Sharpe ratio on net returns in excess of RF (and
its gross counterpart), the difference with the equally weighted portfolio,
annual turnover, average HHI, maximum drawdown and the Deflated Sharpe Ratio.
Cross-universe tables (strategy x universe) are written per frequency.

Conventions: costs are deducted period by period (the net series), RF is
subtracted afterwards; mean x periods and std (ddof = 1) x sqrt(periods).
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src import config as C
from src.metrics import deflated_sharpe_ratio
from src.data import load_panel


def _sharpe(excess: np.ndarray, ppy: int) -> float:
    if len(excess) < 2:
        return float("nan")
    return float(np.mean(excess) * ppy / (np.std(excess, ddof=1) * np.sqrt(ppy) + 1e-10))


def task_table(folder: Path, universe: str, freq: str) -> pd.DataFrame:
    """Metrics of every strategy found in one task folder."""
    ppy = C.PERIODS_PER_YEAR[freq]
    gross = pd.read_csv(folder / "oos_returns_gross.csv", index_col=0, parse_dates=True)
    net = pd.read_csv(folder / "oos_returns_net.csv", index_col=0, parse_dates=True)
    turn = pd.read_csv(folder / "oos_turnover.csv", index_col=0, parse_dates=True)
    trials = pd.read_csv(folder / "dsr_trials.csv") if (folder / "dsr_trials.csv").stat().st_size > 2 \
        else pd.DataFrame(columns=["Strategy", "sharpe_per_period"])
    _, rf = load_panel(universe, freq)
    rf = rf.reindex(gross.index).values
    rows = []
    for s in gross.columns:
        w = pd.read_csv(folder / "weights" / f"{s}.csv", index_col=0).values
        ex_net, ex_gross = net[s].values - rf, gross[s].values - rf
        wealth = np.cumprod(1.0 + net[s].values)
        dd = wealth / np.maximum.accumulate(wealth) - 1.0
        log = [{"sharpe": x} for x in trials.loc[trials.Strategy == s, "sharpe_per_period"]]
        sr = _sharpe(ex_net, ppy)
        dsr = (deflated_sharpe_ratio(sr, ex_net, log, ppy)
               if len(log) > 1 and len(ex_net) > 3 else float("nan"))
        rows.append({
            "Universe": universe, "Freq": freq, "Strategy": s, "OOS_obs": len(gross),
            "Return_net_ann_%": float(np.mean(net[s].values) * ppy * 100),
            "Excess_net_ann_%": float(np.mean(ex_net) * ppy * 100),
            "Vol_ann_%": float(np.std(net[s].values, ddof=1) * np.sqrt(ppy) * 100) if len(gross) > 1 else np.nan,
            "Sharpe_net": sr, "Sharpe_gross": _sharpe(ex_gross, ppy),
            "Turnover_ann": float(np.mean(turn[s].values[1:]) * ppy) if len(gross) > 1 else 0.0,
            "HHI_avg": float(np.mean(np.sum(w ** 2, axis=1))),
            "MaxDD_%": float(dd.min() * 100),
            "DSR": dsr, "N_trials": len(log),
        })
    df = pd.DataFrame(rows)
    if "EW" in set(df.Strategy):
        df["dSharpe_vs_EW"] = df.Sharpe_net - float(df.loc[df.Strategy == "EW", "Sharpe_net"].iloc[0])
    return df


def build_tables(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """Write <run_dir>/tables/ from every finished task of the run."""
    run_dir = Path(run_dir)
    out = run_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    parts: List[pd.DataFrame] = []
    for u in C.UNIVERSES:
        for f in C.FREQS:
            folder = run_dir / u / f
            if (folder / "oos_returns_net.csv").exists():
                t = task_table(folder, u, f)
                t.to_csv(out / f"metrics_{u}_{f}.csv", index=False)
                parts.append(t)
    if not parts:
        return {}
    allm = pd.concat(parts, ignore_index=True)
    allm.to_csv(out / "metrics_all.csv", index=False)
    result = {"metrics_all": allm}
    order = [s for s in C.STRATEGIES if s in set(allm.Strategy)]
    for f in C.FREQS:
        sub = allm[allm.Freq == f]
        if sub.empty:
            continue
        for col, name in (("Sharpe_net", "sharpe_net"), ("Sharpe_gross", "sharpe_gross"),
                          ("dSharpe_vs_EW", "dsharpe_vs_ew"), ("Turnover_ann", "turnover"),
                          ("HHI_avg", "hhi"), ("DSR", "dsr")):
            if col not in sub:
                continue
            piv = sub.pivot(index="Strategy", columns="Universe", values=col).reindex(order)
            piv = piv[[u for u in C.UNIVERSES if u in piv.columns]]
            piv.to_csv(out / f"{name}_{f}.csv")
            result[f"{name}_{f}"] = piv
    return result
