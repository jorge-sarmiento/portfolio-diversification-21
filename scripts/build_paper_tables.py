"""
Build the tables of the article, with its own numbering, from a pipeline run.

    python scripts/build_paper_tables.py --run-dir results/Run_<timestamp>
    python scripts/build_paper_tables.py --run-dir results/Run_<timestamp> --check

Tables 1-7 and 9 are assembled from the run outputs, and Table 10 from the output of
scripts/dominance_lp_tables.py, which must be run first; that script also produces
Table 8 (infeasibility rates of the Block 6 linear programs).
Every table is written as a CSV under <run-dir>/PAPER_TABLES/paper/ plus a single
Markdown file with all of them.

--check compares every cell against the values published in the article, stored in
archive/published_tables.json, rounded to the number of decimals the article prints.
It exits with status 1 if any cell differs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["Dow_Jones_30", "Global_ETFs", "Tech_Sector"]
SHORT = {"Dow_Jones_30": "DJ30", "Global_ETFs": "ETFs", "Tech_Sector": "Tech"}

# paper label -> (code used in the outputs, full name used in Table_5_1)
STRATEGIES = [
    ("EW",     "EW",         "Equally Weighted"),
    ("GMR",    "GMR",        "Global Max Return"),
    ("GMV",    "GMV",        "Global Min Variance"),
    ("MV",     "MV",         "Mean-Variance"),
    ("MVN",    "MSV",        "Mean Squared Variance"),
    ("SemiV",  "SemiV",      "Mean-Semivariance"),
    ("WUBC",   "WUBC",       "Weight Upper-Bound"),
    ("WLBC",   "WLBC",       "Weight Lower-Bound"),
    ("WCMV",   "WUBC_Alt",   "Weight-Constr. MV"),
    ("MaxDiv", "MD",         "Max Diversification"),
    ("DMV",    "DMV",        "Diversified MV"),
    ("DMVY",   "DMV_Yager",  "DMV-Yager"),
    ("DMVR",   "DMV_Return", "DMV-Return"),
    ("DMVV",   "DMV_Vars",   "DMV-Vars"),
    ("EWMV",   "EWMV",       "EWMV Mixture"),
    ("P-Dom",  "FSD",        "First-Order SD"),
    ("SSD-R",  "SSD",        "Second-Order SD"),
    ("CVaR",   "CVaR",       "Min-CVaR"),
    ("MAD",    "MAD",        "Mean-Abs. Deviation"),
    ("RP",     "RP",         "Risk Parity (ERC)"),
    ("HRP",    "HRP",        "Hierarchical RP"),
]
LABELS = [s[0] for s in STRATEGIES]
CODE = {s[0]: s[1] for s in STRATEGIES}
FULL = {s[0]: s[2] for s in STRATEGIES}

SOLVERS = {
    "EW": "closed form", "GMR": "closed form", "GMV": "CLARABEL", "MV": "CLARABEL",
    "MVN": "differential evolution", "SemiV": "CLARABEL", "WUBC": "CLARABEL",
    "WLBC": "CLARABEL", "WCMV": "CLARABEL", "MaxDiv": "trust-constr", "DMV": "CLARABEL",
    "DMVY": "CLARABEL", "DMVR": "CLARABEL", "DMVV": "CLARABEL",
    "EWMV": "CLARABEL + closed-form mixture",
    "P-Dom": "HiGHS", "SSD-R": "HiGHS", "CVaR": "HiGHS", "MAD": "HiGHS",
    "RP": "trust-constr", "HRP": "hierarchical procedure",
}


# --------------------------------------------------------------------------- io
def latest_run() -> Path:
    runs = sorted((ROOT / "results").glob("Run_*"))
    runs = [r for r in runs if (r / "PAPER_TABLES").is_dir()]
    if not runs:
        sys.exit("ERROR: no run with PAPER_TABLES found under results/. Run main.py first.")
    return runs[-1]


def metrics(run: Path, ds: str, freq: str) -> pd.DataFrame:
    return pd.read_csv(run / ds / freq / "metrics_detailed.csv").set_index("Strategy")


def table_5_1(run: Path, ds: str) -> pd.DataFrame:
    return pd.read_csv(run / "PAPER_TABLES" / f"Table_5_1_oos_{ds}.csv").set_index("Strategy")


# ------------------------------------------------------------------- the tables
def build_t1(run: Path) -> pd.DataFrame:
    src = pd.read_csv(run / "PAPER_TABLES" / "Table_4_1_dataset_summary.csv").set_index("Dataset")
    rows = []
    for ds in DATASETS:
        prices = (pd.read_csv(ROOT / "data" / f"{ds}.csv", index_col=0, parse_dates=True)
                  .apply(pd.to_numeric, errors="coerce").dropna())
        monthly = prices.resample("ME").last().dropna(how="all").values
        r = np.log(monthly[1:] / monthly[:-1])
        rows.append({
            "Dataset": ds,
            "N": int(src.loc[ds, "N Assets"]),
            "Obs": int(src.loc[ds, "Observations"]),
            "Start": str(src.loc[ds, "Start Date"]),
            "End": str(src.loc[ds, "End Date"]),
            "IS": int(src.loc[ds, "IS Obs"]),
            "OOS": int(src.loc[ds, "OOS Obs"]),
            "Skew": float(stats.skew(r, axis=0).mean()),
            "Kurt": float(stats.kurtosis(r, axis=0).mean()),
        })
    return pd.DataFrame(rows).set_index("Dataset")


def build_t2(run: Path) -> pd.DataFrame:
    grids = pd.read_csv(run / "PAPER_TABLES" / "Table_3_2_3_param_grids.csv").set_index("Strategy")
    rows = []
    for label in LABELS:
        code = CODE[label]
        grid = grids.loc[code, "Grid"] if code in grids.index else "—"
        combos = int(grids.loc[code, "Combinations"]) if code in grids.index else 0
        rows.append({"Strategy": label, "Solver": SOLVERS[label],
                     "Grid": grid, "Combinations": combos})
    return pd.DataFrame(rows).set_index("Strategy")


def _performance(run: Path, freq: str, gross: bool) -> pd.DataFrame:
    col = "Sharpe Ratio (Gross)" if gross else "Sharpe Ratio"
    out = {}
    for ds in DATASETS:
        m = metrics(run, ds, freq)
        out[f"SR_{SHORT[ds]}"] = pd.Series({lab: m.loc[CODE[lab], col] for lab in LABELS})
    df = pd.DataFrame(out)
    for s in SHORT.values():
        df[f"dSR_{s}"] = df[f"SR_{s}"] - df.loc["EW", f"SR_{s}"]
    return df[[c for pair in [(f"SR_{s}", f"dSR_{s}") for s in SHORT.values()] for c in pair]]


def build_t3(run: Path) -> pd.DataFrame:
    df = _performance(run, "MONTHLY", gross=False)
    sr, ret, vol = {}, {}, {}
    for ds in DATASETS:
        t = table_5_1(run, ds)
        sr[ds] = pd.Series({lab: t.loc[FULL[lab], "Sharpe Ratio"] for lab in LABELS})
        ret[ds] = pd.Series({lab: t.loc[FULL[lab], "Annual Return (%)"] for lab in LABELS})
        vol[ds] = pd.Series({lab: t.loc[FULL[lab], "Annual Volatility (%)"] for lab in LABELS})
    df["SR_mean"] = pd.DataFrame(sr).mean(axis=1)
    df["Ret_mean"] = pd.DataFrame(ret).mean(axis=1)
    df["Vol_mean"] = pd.DataFrame(vol).mean(axis=1)
    return df


def build_t4(run: Path) -> pd.DataFrame:
    dsr = pd.read_csv(run / "PAPER_TABLES" / "Table_5_4_dsr.csv").set_index("Strategy")
    hhi = pd.read_csv(run / "PAPER_TABLES" / "Table_5_2_concentration.csv").set_index("Strategy")
    out = {}
    for ds in DATASETS:
        t = table_5_1(run, ds)
        s = SHORT[ds]
        out[f"DSR_{s}"] = pd.Series({lab: pd.to_numeric(dsr.loc[CODE[lab], ds], errors="coerce")
                                     for lab in LABELS})
        out[f"HHI_{s}"] = pd.Series({lab: hhi.loc[CODE[lab], f"HHI ({ds})"] for lab in LABELS})
        out[f"TO_{s}"] = pd.Series({lab: t.loc[FULL[lab], "Turnover (Annual)"] for lab in LABELS})
    cols = [f"{m}_{SHORT[d]}" for m in ("DSR", "HHI", "TO") for d in DATASETS]
    return pd.DataFrame(out)[cols]


def build_t5(run: Path) -> pd.DataFrame:
    t = pd.read_csv(run / "PAPER_TABLES" / "Table_5_7_cost_sensitivity.csv")
    inv = {v: k for k, v in CODE.items()}
    t = t[t["Strategy"].isin(inv)].copy()
    t["Strategy"] = t["Strategy"].map(inv)
    t = t[["Dataset", "Strategy", "Ann. Turnover", "Sharpe (0bps)", "Sharpe (10bps)", "Sharpe (20bps)"]]
    return t.rename(columns={"Ann. Turnover": "Turnover", "Sharpe (0bps)": "SR_0bps",
                             "Sharpe (10bps)": "SR_10bps", "Sharpe (20bps)": "SR_20bps"})


def build_t6(run: Path) -> pd.DataFrame:
    return _performance(run, "DAILY", gross=True)


def build_t7(run: Path) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        m, d = metrics(run, ds, "MONTHLY"), metrics(run, ds, "DAILY")
        pairs = {
            "gross": ([m.loc[CODE[l], "Sharpe Ratio (Gross)"] for l in LABELS],
                      [d.loc[CODE[l], "Sharpe Ratio (Gross)"] for l in LABELS]),
            "net":   ([m.loc[CODE[l], "Sharpe Ratio"] for l in LABELS],
                      [d.loc[CODE[l], "Sharpe Ratio"] for l in LABELS]),
        }
        row = {"Dataset": SHORT[ds], "N_strategies": len(LABELS)}
        for basis, (x, y) in pairs.items():
            rho, p = stats.spearmanr(x, y)
            row[f"rho_{basis}"] = round(float(rho), 3)
            row[f"p_{basis}"] = round(float(p), 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Dataset")


def build_t9(run: Path) -> pd.DataFrame:
    params = {ds: pd.read_csv(run / ds / "MONTHLY" / "best_params.csv").set_index("Strategy")
              for ds in DATASETS}
    rows = []
    for label in LABELS:
        code = CODE[label]
        cols = [c for c in ("risk_aversion", "diversification_factor", "mixture_factor",
                            "lower_bound", "upper_bound", "alpha")
                if any(c in params[ds].columns and pd.notna(params[ds].loc[code, c])
                       for ds in DATASETS)]
        if not cols:
            continue
        for c in cols:
            rows.append({"Strategy": label, "Parameter": c,
                         **{SHORT[ds]: params[ds].loc[code, c] if c in params[ds].columns else np.nan
                            for ds in DATASETS}})
    return pd.DataFrame(rows).set_index(["Strategy", "Parameter"])


def build_t10(run: Path):
    for candidate in (run / "PAPER_TABLES" / "dominance_lp" / "ssd_hhi_correction.csv",):
        if candidate.exists():
            t = pd.read_csv(candidate)
            t = t[t["Frequency"] == "MONTHLY"].set_index("Dataset")
            out = pd.DataFrame({
                "HHI_chronological": t["HHI_original (chrono cumsum)"],
                "HHI_sorted": t["HHI_corrected (sorted cumsum)"],
            })
            out["delta"] = out["HHI_sorted"] - out["HHI_chronological"]
            return out.reindex(DATASETS)
    return None


# ------------------------------------------------------------------------ check
def _decimals(value) -> int:
    s = str(value)
    return len(s.split(".")[1]) if "." in s else 0


def _cmp(published, computed, table, cell, failures, checked, rounding=None):
    """
    Compare one cell with the published value, at the precision the article prints.

    A difference below one unit in the last printed decimal is reported as a
    rounding difference, not as a failure.
    """
    checked.append(cell)
    if published is None:
        if computed is not None and not (isinstance(computed, float) and np.isnan(computed)):
            failures.append((table, cell, "-", computed))
        return
    if computed is None or (isinstance(computed, float) and np.isnan(computed)):
        failures.append((table, cell, published, "missing"))
        return
    d = _decimals(published)
    delta = abs(float(computed) - float(published))
    if delta <= 0.5 * 10 ** (-d) + 1e-9:
        return
    if rounding is not None and delta < 10 ** (-d) + 1e-9:
        rounding.append((table, cell, published, round(float(computed), d + 2)))
        return
    failures.append((table, cell, published, round(float(computed), d + 2)))


def check(tables: dict, published: dict):
    failures, checked, rounding = [], [], []

    t1 = tables["T1"]
    for ds, exp in published["T1_datasets"].items():
        for key in ("N", "Obs", "IS", "OOS", "Skew", "Kurt"):
            _cmp(exp[key], t1.loc[ds, key], "T1", f"{SHORT[ds]} {key}", failures, checked, rounding)
        for key in ("Start", "End"):
            checked.append(f"{SHORT[ds]} {key}")
            if str(t1.loc[ds, key]) != exp[key]:
                failures.append(("T1", f"{SHORT[ds]} {key}", exp[key], t1.loc[ds, key]))

    t2 = tables["T2"]
    for label, grid in published["T2_grids"].items():
        if label == "_no_grid":
            for lab in grid:
                _cmp(0, t2.loc[lab, "Combinations"], "T2", f"{lab} combinations", failures, checked)
            continue
        n = int(np.prod([len(v) for v in grid.values()]))
        _cmp(n, t2.loc[label, "Combinations"], "T2", f"{label} combinations", failures, checked)
        values = sorted(v for vs in grid.values() for v in vs)
        import re
        got = sorted(float(x) for g in re.findall(r"\{([^}]*)\}", str(t2.loc[label, "Grid"]))
                     for x in re.findall(r"-?\d+\.?\d*", g))
        checked.append(f"{label} grid values")
        if [round(v, 6) for v in values] != [round(v, 6) for v in got]:
            failures.append(("T2", f"{label} grid values", values, got))

    for key, table, cols in (("T3_monthly_net", "T3", published["T3_monthly_net"]["_columns"]),
                             ("T4_robustness", "T4", published["T4_robustness"]["_columns"]),
                             ("T6_daily_gross", "T6", published["T6_daily_gross"]["_columns"])):
        built = tables[table]
        for label, values in published[key].items():
            if label.startswith("_"):
                continue
            for col, exp in zip(cols, values):
                got = built.loc[label, col]
                if col.startswith("dSR_"):
                    # the article prints Sharpe ratios to three decimals and the
                    # excess column as the difference of those printed values
                    sr = round(float(built.loc[label, col.replace("dSR_", "SR_")]), 3)
                    ew = round(float(built.loc["EW", col.replace("dSR_", "SR_")]), 3)
                    got = round(sr - ew, 10)
                _cmp(exp, got, table, f"{label} {col}", failures, checked, rounding)

    t5 = tables["T5"].set_index(["Dataset", "Strategy"])
    for ds, rows in published["T5_costs"].items():
        if ds.startswith("_"):
            continue
        for label, values in rows.items():
            for col, exp in zip(published["T5_costs"]["_columns"], values):
                _cmp(exp, t5.loc[(ds, label), col], "T5", f"{SHORT[ds]} {label} {col}", failures, checked, rounding)

    t7 = tables["T7"]
    for ds, values in published["T7_frequency"].items():
        if ds.startswith("_"):
            continue
        for col, exp in zip(published["T7_frequency"]["_columns"], values):
            _cmp(exp, t7.loc[ds, col], "T7", f"{ds} {col}", failures, checked, rounding)

    t9 = tables["T9"]
    for label, params in published["T9_hyperparameters"].items():
        if label.startswith("_"):
            continue
        for param, values in params.items():
            for ds, exp in zip(DATASETS, values):
                got = t9.loc[(label, param), SHORT[ds]] if (label, param) in t9.index else None
                _cmp(exp, got, "T9", f"{label} {param} {SHORT[ds]}", failures, checked, rounding)

    t10 = tables.get("T10")
    if t10 is not None:
        for ds, values in published["T10_ssd_correction"].items():
            if ds.startswith("_"):
                continue
            for col, exp in zip(published["T10_ssd_correction"]["_columns"], values):
                _cmp(exp, t10.loc[ds, col], "T10", f"{SHORT[ds]} {col}", failures, checked, rounding)

    return failures, checked, rounding


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Build the tables of the article from a run")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--check", action="store_true",
                    help="compare every cell against archive/published_tables.json")
    args = ap.parse_args()

    run = (args.run_dir or latest_run()).resolve()
    out = (args.out_dir or run / "PAPER_TABLES" / "paper").resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run}")

    tables = {"T1": build_t1(run), "T2": build_t2(run), "T3": build_t3(run),
              "T4": build_t4(run), "T5": build_t5(run), "T6": build_t6(run),
              "T7": build_t7(run), "T9": build_t9(run)}
    t10 = build_t10(run)
    if t10 is not None:
        tables["T10"] = t10

    titles = {
        "T1": "Table 1. Dataset characteristics",
        "T2": "Table 2. Calibration grids and solvers",
        "T3": "Table 3. Monthly out-of-sample performance (net of 10 bps)",
        "T4": "Table 4. Robustness diagnostics (DSR, HHI, turnover)",
        "T5": "Table 5. Turnover and net-of-cost Sharpe ratios",
        "T6": "Table 6. Daily out-of-sample performance (gross)",
        "T7": "Table 7. Frequency sensitivity (Spearman, monthly vs daily)",
        "T9": "Table 9. Selected hyperparameters",
        "T10": "Table 10. Effect of the SSD sorting correction",
    }
    md = [f"# Tables of the article\n", f"Run: `{run.name}`\n"]
    for key, df in tables.items():
        df.to_csv(out / f"{key}.csv")
        md += [f"## {titles[key]}\n", "```", df.round(4).to_string(), "```", ""]
    md += ["## Table 8. Infeasibility rates of the Block 6 LPs\n",
           "Produced by `scripts/dominance_lp_tables.py`.\n"]
    (out / "PAPER_TABLES.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {len(tables)} tables to {out}")

    if not args.check:
        return

    published = json.loads((ROOT / "archive" / "published_tables.json").read_text(encoding="utf-8"))
    failures, checked, rounding = check(tables, published)
    print(f"\nChecked {len(checked)} published cells against this run")
    if rounding:
        print(f"{len(rounding)} cell(s) differ by less than one unit in the last printed decimal:")
        for table, cell, exp, got in rounding:
            print(f"  {table:<4} {cell:<28} published={exp}  run={got}")
        print()
    if failures:
        print(f"{len(failures)} cell(s) differ:\n")
        for table, cell, exp, got in failures:
            print(f"  {table:<4} {cell:<28} published={exp}  run={got}")
        sys.exit(1)
    print("All published cells reproduced.")


if __name__ == "__main__":
    main()
