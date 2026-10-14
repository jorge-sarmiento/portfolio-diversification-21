"""
Cross-universe summary by ranks.

The universes differ in assets and risk, so their Sharpe ratios are not
averaged. For each frequency, every strategy is ranked inside each universe by
net Sharpe ratio (1 = best, ties share the average rank); the summary reports
the mean rank across universes and the number of universes in which the
strategy has a higher net Sharpe ratio than the equally weighted portfolio.
"""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src import config as C


def rank_summary(sharpe: pd.DataFrame, ref: str = "EW") -> pd.DataFrame:
    """Ranks of the strategies (rows) in each universe (columns) and their summary."""
    # only universes with results and strategies present in all of them are ranked
    sharpe = sharpe.dropna(axis=1, how="all").dropna(axis=0, how="any")
    # weights equal to 1/N give the Sharpe ratio of EW up to rounding: a tie
    sharpe = sharpe.round(10)
    ranks = sharpe.rank(axis=0, ascending=False, method="average")
    out = ranks.copy()
    out["mean_rank"] = ranks.mean(axis=1)
    if ref in sharpe.index:
        out["n_beats_ref"] = sharpe.gt(sharpe.loc[ref], axis=1).sum(axis=1)
        out["n_below_ref"] = sharpe.lt(sharpe.loc[ref], axis=1).sum(axis=1)
    out["n_universes"] = sharpe.shape[1]
    return out


def run_summary(run_dir) -> Optional[Dict[str, pd.DataFrame]]:
    """Write <run_dir>/summary/ranks_<FREQ>.csv from the Sharpe tables of the run."""
    run_dir = Path(run_dir)
    out = run_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for f in C.FREQS:
        path = run_dir / "tables" / f"sharpe_net_{f}.csv"
        if not path.exists():
            continue
        ranks = rank_summary(pd.read_csv(path, index_col=0))
        ranks.to_csv(out / f"ranks_{f}.csv")
        result[f"ranks_{f}"] = ranks
    return result or None
