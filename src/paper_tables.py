"""
Tables of a pipeline run from which the tables of the article are built.

Reads the artefacts that main.py writes (metrics_detailed.csv, oos_returns.csv,
oos_turnover.csv); scripts/build_paper_tables.py assembles them with the numbering
of the article.

Outputs (saved to <run_dir>/PAPER_TABLES/):
    Table_3_2_3_param_grids.csv       calibration grids                 (Table 2)
    Table_4_1_dataset_summary.csv     observations and IS/OOS split     (Table 1)
    Table_5_1_oos_<dataset>.csv       OOS performance, one per dataset  (Tables 3 and 4)
    Table_5_2_concentration.csv       HHI and effective N               (Table 4)
    Table_5_4_dsr.csv                 Deflated Sharpe Ratio             (Table 4)
    Table_5_7_cost_sensitivity.csv    net Sharpe ratio at 0/10/20 bps   (Table 5)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Block / display labels
# ----------------------------------------------------------------------------
# Block numbers are those of Section 3 of the article (Blocks 1-7).
BLOCK_MAP = {
    "EW":        (1, "Equally Weighted"),
    "GMR":       (1, "Global Max Return"),
    "GMV":       (2, "Global Min Variance"),
    "MV":        (2, "Mean-Variance"),
    "MSV":       (2, "Mean Squared Variance"),
    "SemiV":     (2, "Mean-Semivariance"),
    "CVaR":      (3, "Min-CVaR"),
    "MAD":       (3, "Mean-Abs. Deviation"),
    "WUBC_Alt":  (4, "Weight-Constr. MV"),
    "WLBC":      (4, "Weight Lower-Bound"),
    "WUBC":      (4, "Weight Upper-Bound"),
    "MD":        (5, "Max Diversification"),
    "DMV":       (5, "Diversified MV"),
    "DMV_Yager": (5, "DMV-Yager"),
    "DMV_Return":(5, "DMV-Return"),
    "DMV_Vars":  (5, "DMV-Vars"),
    "EWMV":      (5, "EWMV Mixture"),
    "SRF_LP":    (6, "Soft Return-Floor LP"),
    "FSD":       (6, "First-Order SD"),
    "SSD":       (6, "Second-Order SD"),
    "RP":        (7, "Risk Parity (ERC)"),
    "HRP":       (7, "Hierarchical RP"),
}
BLOCK_ORDER = sorted(BLOCK_MAP.keys(), key=lambda k: (BLOCK_MAP[k][0], list(BLOCK_MAP.keys()).index(k)))

# Symbol shown for each hyperparameter in the grid table.
_GRID_SYMBOLS = {
    "risk_aversion":          "λ",
    "lower_bound":            "lb",
    "upper_bound":            "ub",
    "diversification_factor": "δ",
    "mixture_factor":         "δ",
    "alpha":                  "α",
}


def _grids_doc() -> dict:
    """
    Render the calibration grids of WalkForwardEngine as table text.

    The text is generated from WalkForwardEngine._PARAM_GRIDS, which holds the
    grids searched by the engine.
    """
    from src.validator import WalkForwardEngine
    doc = {}
    for strat, grid in WalkForwardEngine._PARAM_GRIDS.items():
        parts = []
        for param, values in grid.items():
            sym = _GRID_SYMBOLS.get(param, param)
            # bounds and confidence levels are reported with two decimals
            fmt = (lambda v: f"{v:.2f}") if param in ("lower_bound", "upper_bound", "alpha") else str
            parts.append(f"{sym} ∈ {{{', '.join(fmt(v) for v in values)}}}")
        doc[strat] = "; ".join(parts)
    return doc


PARAM_GRIDS_DOC = _grids_doc()


# ============================================================================
# Helpers
# ============================================================================

_SKIP_FOLDER_NAMES = {"DAILY", "PAPER_TABLES"}

# Subfolder that holds the primary (monthly) results in the dual-frequency layout
_PRIMARY_SUBFOLDERS = {"MONTHLY"}


def _iter_metric_files(run_dir: Path, filename: str):
    """
    Yield paths to *filename* found under run_dir, searching:
      1. */MONTHLY/<filename>   - dual-frequency layout
      2. */<filename>           - single-frequency layout (skip non-dataset folders)
    Deduplicates by resolved path.
    """
    seen = set()
    # Pattern 1: MONTHLY sub-directory
    for subfolder in ("MONTHLY",):
        for p in sorted(run_dir.glob(f"*/{subfolder}/{filename}")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield p
    # Pattern 2: directly under dataset folder (single-frequency layout)
    for p in sorted(run_dir.glob(f"*/{filename}")):
        if p.parent.name in _SKIP_FOLDER_NAMES or p.parent.name in _PRIMARY_SUBFOLDERS:
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            yield p


def _load_all_metrics(run_dir: Path) -> pd.DataFrame:
    """Load all metrics_detailed.csv files from dataset sub-folders."""
    frames = []
    for p in _iter_metric_files(run_dir, "metrics_detailed.csv"):
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No metrics_detailed.csv found under {run_dir}")
    df = pd.concat(frames, ignore_index=True)
    # Normalise dataset name (strip ' - Daily' etc.)
    df["DatasetShort"] = df["Dataset"].str.split(" - ").str[0]
    return df


def _load_oos_turnover(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """Return {dataset_short_name: DataFrame(Date x strategy)} of per-period turnover."""
    out = {}
    for p in _iter_metric_files(run_dir, "oos_turnover.csv"):
        if p.parent.name in _PRIMARY_SUBFOLDERS:
            ds = p.parent.parent.name
        else:
            ds = p.parent.name
        df = pd.read_csv(p, index_col="Date", parse_dates=True)
        out[ds] = df
    return out


def _load_oos_returns(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """Return {dataset_short_name: DataFrame(Date × strategy)} for each dataset."""
    out = {}
    for p in _iter_metric_files(run_dir, "oos_returns.csv"):
        # For dual-freq layout: parent is MONTHLY -> go up one more level
        if p.parent.name in _PRIMARY_SUBFOLDERS:
            ds = p.parent.parent.name
        else:
            ds = p.parent.name
        df = pd.read_csv(p, index_col="Date", parse_dates=True)
        out[ds] = df
    return out


def _add_block(df: pd.DataFrame) -> pd.DataFrame:
    """Add Block and FullName columns from BLOCK_MAP."""
    df = df.copy()
    df["Block"] = df["Strategy"].map(lambda s: BLOCK_MAP.get(s, (99, s))[0])
    df["FullName"] = df["Strategy"].map(lambda s: BLOCK_MAP.get(s, (99, s))[1])
    return df


def _sharpe_from_returns(ret: pd.Series, factor: int = 252) -> float:
    if ret.std() < 1e-10:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(factor))


# ============================================================================
# Calibration grids (Table_3_2_3)
# ============================================================================

def table_param_grids() -> pd.DataFrame:
    """Table_3_2_3_param_grids.csv - hyperparameter grid by strategy."""
    rows = []
    for strat, grid_str in PARAM_GRIDS_DOC.items():
        block, name = BLOCK_MAP.get(strat, (0, strat))
        n_combos = 1
        for part in grid_str.split(";"):
            parts = part.strip().split("∈")
            if len(parts) < 2:
                continue
            vals = [v.strip() for v in parts[1].strip("{} ").split(",") if v.strip()]
            if vals:
                n_combos *= len(vals)
        rows.append({
            "Block":      block,
            "Strategy":   strat,
            "Full Name":  name,
            "Grid":       grid_str,
            "Combinations": n_combos,
            "Criterion":  "IS Sharpe",
        })
    # No-param strategies
    for strat in BLOCK_ORDER:
        if strat not in PARAM_GRIDS_DOC:
            block, name = BLOCK_MAP.get(strat, (0, strat))
            rows.append({
                "Block":      block,
                "Strategy":   strat,
                "Full Name":  name,
                "Grid":       "—",
                "Combinations": 0,
                "Criterion":  "—",
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["Block", "Strategy"]).reset_index(drop=True)


# ============================================================================
# Data tables (Table_4_*)
# ============================================================================

def table_dataset_summary(run_dir: Path, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Table_4_1_dataset_summary.csv - dataset summary.
    Reports the observations actually used in the run (post-resample),
    read from oos_returns.csv + metrics_detailed.csv artefacts.
    Falls back to raw daily CSV counts when artefacts are absent.
    """
    if data_dir is None:
        data_dir = run_dir.parent.parent / "data"
        if not data_dir.exists():
            data_dir = run_dir.parent / "data"

    # Try to read actual run parameters from artefacts (dual-freq: MONTHLY subfolder; single-freq: flat)
    run_info: Dict[str, dict] = {}
    for p in _iter_metric_files(run_dir, "metrics_detailed.csv"):
        ds_key = p.parent.parent.name if p.parent.name in _PRIMARY_SUBFOLDERS else p.parent.name
        try:
            df_m = pd.read_csv(p, nrows=1)
            ppy  = int(df_m["Periods_Per_Year"].iloc[0]) if "Periods_Per_Year" in df_m.columns else 252
            run_info[ds_key] = {"ppy": ppy}
        except Exception:
            pass
    for p in _iter_metric_files(run_dir, "oos_returns.csv"):
        ds_key = p.parent.parent.name if p.parent.name in _PRIMARY_SUBFOLDERS else p.parent.name
        try:
            df_oos = pd.read_csv(p, index_col=0, parse_dates=True)
            if ds_key not in run_info:
                run_info[ds_key] = {}
            run_info[ds_key]["oos_obs"]   = len(df_oos)
            run_info[ds_key]["start_oos"] = df_oos.index[0].date()
            run_info[ds_key]["end_oos"]   = df_oos.index[-1].date()
        except Exception:
            pass

    rows = []
    for p in sorted(data_dir.glob("*.csv")):
        ds = p.stem
        df_raw = pd.read_csv(p, index_col=0, parse_dates=True)
        T_raw, N = df_raw.shape

        info = run_info.get(ds, {})
        ppy  = info.get("ppy", 252)
        freq = {252: "Daily", 12: "Monthly", 4: "Quarterly"}.get(ppy, f"{ppy}/yr")

        if ppy != 252 and "oos_obs" in info:
            oos_obs = info["oos_obs"]
            # Number of monthly return periods, counted from month-end prices
            _monthly_rets = df_raw.resample("ME").last().pct_change().dropna(how="all")
            T_used   = len(_monthly_rets)
            is_obs   = T_used - oos_obs
            start    = df_raw.index[0].date()
            end      = info.get("end_oos", df_raw.index[-1].date())
        else:
            T_used = T_raw
            is_obs = int(T_raw * 0.70)
            oos_obs = T_raw - is_obs
            start = df_raw.index[0].date()
            end   = df_raw.index[-1].date()

        rows.append({
            "Dataset":      ds,
            "N Assets":     N,
            "Observations": T_used,
            "Start Date":   start,
            "End Date":     end,
            "Frequency":    freq,
            "IS Obs":       is_obs,
            "OOS Obs":      oos_obs,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Result tables (Table_5_*)
# ============================================================================

def table_oos_performance(df_all: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Table_5_1_oos_<dataset>.csv - OOS performance for one dataset.
    Returns a block-sorted table with key metrics.
    """
    cols = [
        "Strategy",
        "Annual Return (%)",
        "Annual Volatility (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Calmar Ratio",
        "Sortino Ratio",
        "Turnover (Annual)",
        "Deflated Sharpe Ratio",
    ]
    df = df_all[df_all["DatasetShort"] == dataset].copy()
    df = _add_block(df)

    available = [c for c in cols if c in df.columns]
    df = df[["Block", "FullName"] + [c for c in available if c != "Strategy"]]
    df = df.sort_values("Block", kind="stable")

    # Round numeric columns
    for c in df.select_dtypes("number").columns:
        df[c] = df[c].round(3)

    df = df.rename(columns={"FullName": "Strategy"})
    return df.reset_index(drop=True)


def table_concentration(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Table_5_2_concentration.csv - HHI and effective number of assets for all strategies × datasets.
    """
    datasets = df_all["DatasetShort"].unique()
    records = []

    for strat in BLOCK_ORDER:
        row = {"Block": BLOCK_MAP.get(strat, (0,))[0],
               "Strategy": strat,
               "Full Name": BLOCK_MAP.get(strat, (0, strat))[1]}
        for ds in sorted(datasets):
            sub = df_all[(df_all["DatasetShort"] == ds) & (df_all["Strategy"] == strat)]
            if sub.empty:
                row[f"HHI ({ds})"]   = np.nan
                row[f"EffN ({ds})"]  = np.nan
            else:
                row[f"HHI ({ds})"]   = round(float(sub["HHI (Avg)"].iloc[0]), 3)
                row[f"EffN ({ds})"]  = round(float(sub["Effective N Assets"].iloc[0]), 1)
        records.append(row)

    return pd.DataFrame(records).sort_values(["Block", "Strategy"]).reset_index(drop=True)


def table_dsr(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Table_5_4_dsr.csv - Deflated Sharpe Ratio for strategies where trials_log is available.
    NaN = no grid search (strategy has no free hyperparameters).
    """
    if "Deflated Sharpe Ratio" not in df_all.columns:
        return pd.DataFrame({"Note": ["Deflated Sharpe Ratio column not found"]})

    datasets = sorted(df_all["DatasetShort"].unique())
    records  = []
    for strat in BLOCK_ORDER:
        row = {"Block":    BLOCK_MAP.get(strat, (0,))[0],
               "Strategy": strat,
               "Full Name":BLOCK_MAP.get(strat, (0, strat))[1]}
        for ds in datasets:
            sub = df_all[(df_all["DatasetShort"] == ds) & (df_all["Strategy"] == strat)]
            val = float(sub["Deflated Sharpe Ratio"].iloc[0]) if not sub.empty else np.nan
            row[ds] = round(val, 3) if not np.isnan(val) else np.nan
        records.append(row)
    return pd.DataFrame(records).reset_index(drop=True)


def table_cost_sensitivity(run_dir: Path,
                           cost_scenarios: List[float] | None = None) -> pd.DataFrame:
    """
    Table_5_7_cost_sensitivity.csv - strategy rankings under alternative transaction cost assumptions.

    Re-computes net Sharpe Ratio for each cost scenario by applying a per-period
    deduction to the gross OOS return series:

        net_return[t] = gross_return[t] - turnover[t] * (bps / 10_000)

    where turnover[t] = sum_i |w_i[t] - w_i[t-1]| is loaded from
    oos_turnover.csv (written by run_single_backtest in main.py), the same
    convention as cost_impact in src/metrics.py.

    cost_scenarios: list of bps values; defaults to [0.0, 10.0, 20.0].
    Datasets without oos_turnover.csv are skipped with a warning.
    """
    if cost_scenarios is None:
        cost_scenarios = [0.0, 10.0, 20.0]

    df_all     = _load_all_metrics(run_dir)
    oos_ret    = _load_oos_returns(run_dir)
    oos_turn   = _load_oos_turnover(run_dir)

    records = []
    for ds_name, df_ret in oos_ret.items():
        if ds_name not in oos_turn:
            import warnings as _w
            _w.warn(
                f"table_cost_sensitivity: oos_turnover.csv not found for dataset "
                f"'{ds_name}'. Skipping (re-run the backtest to generate it).",
                stacklevel=2,
            )
            continue

        df_turn   = oos_turn[ds_name]
        ds_metrics = df_all[df_all["DatasetShort"] == ds_name]
        factor    = int(ds_metrics["Periods_Per_Year"].iloc[0]) if not ds_metrics.empty else 252

        for strat in BLOCK_ORDER:
            if strat not in df_ret.columns or strat not in df_turn.columns:
                continue
            row_base = ds_metrics[ds_metrics["Strategy"] == strat]
            turnover_ann = (float(row_base["Turnover (Annual)"].iloc[0])
                            if not row_base.empty else float("nan"))

            gross = df_ret[strat].values
            turn  = df_turn[strat].values   # per-period turnover (T,)

            # Align lengths (gross and turn must match)
            min_len = min(len(gross), len(turn))
            gross = gross[:min_len]
            turn  = turn[:min_len]

            row = {
                "Dataset":       ds_name,
                "Block":         BLOCK_MAP.get(strat, (0,))[0],
                "Strategy":      strat,
                "Full Name":     BLOCK_MAP.get(strat, (0, strat))[1],
                "Ann. Turnover": round(turnover_ann, 1),
            }
            for bps in cost_scenarios:
                cost_impact = turn * (bps / 10_000.0)
                net = gross - cost_impact
                row[f"Sharpe ({int(bps)}bps)"] = round(
                    _sharpe_from_returns(pd.Series(net), factor), 3
                )
            records.append(row)

    cols = (["Dataset", "Block", "Strategy", "Full Name", "Ann. Turnover"]
            + [f"Sharpe ({int(b)}bps)" for b in cost_scenarios])
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records).sort_values(
        ["Dataset", "Block", "Strategy"]
    ).reset_index(drop=True)


# ============================================================================
# Master function
# ============================================================================

def generate_all_tables(run_dir: Path) -> None:
    """
    Generate the tables of the run and save them to <run_dir>/PAPER_TABLES/*.csv.
    Also prints a brief summary to stdout.
    """
    out_dir = run_dir / "PAPER_TABLES"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate data directory (two levels up from run_dir, then /data)
    data_dir = run_dir.parent.parent / "data"
    if not data_dir.exists():
        data_dir = run_dir.parent / "data"

    df_all = _load_all_metrics(run_dir)
    datasets_found = sorted(df_all["DatasetShort"].unique())

    tables: Dict[str, pd.DataFrame] = {}

    # -- Calibration grids ---------------------------------------------------
    tables["3_2_3_param_grids"] = table_param_grids()

    # -- Data ----------------------------------------------------------------
    if data_dir.exists():
        try:
            tables["4_1_dataset_summary"] = table_dataset_summary(run_dir, data_dir)
        except Exception as e:
            print(f"  [!] 4_1: {e}")

    # -- Results: one OOS performance table per dataset ----------------------
    for ds in datasets_found:
        key = f"5_1_oos_{ds.replace(' ', '_')}"
        try:
            tables[key] = table_oos_performance(df_all, ds)
        except Exception as e:
            print(f"  [!] {key}: {e}")

    try:
        tables["5_2_concentration"] = table_concentration(df_all)
    except Exception as e:
        print(f"  [!] 5_2: {e}")

    try:
        tables["5_4_dsr"] = table_dsr(df_all)
    except Exception as e:
        print(f"  [!] 5_4: {e}")

    try:
        tables["5_7_cost_sensitivity"] = table_cost_sensitivity(run_dir)
    except Exception as e:
        print(f"  [!] 5_7: {e}")

    # -- Save individual CSVs ------------------------------------------------
    for name, df in tables.items():
        df.to_csv(out_dir / f"Table_{name}.csv", index=False)

    print(f"\n  Tables generated: {len(tables)}")
    for name, df in tables.items():
        print(f"    {name:<40} {df.shape[0]} rows × {df.shape[1]} cols")
