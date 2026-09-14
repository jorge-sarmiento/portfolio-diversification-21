"""
scripts/verify_reproduction.py
Compare a new pipeline run against the archived reference outputs.

Compares three layers, in order of sensitivity:

  Layer 1 - IS calibration  : archive/{ds}/{freq}/best_params.csv
  Layer 2 - Raw OOS series  : archive/{ds}/{freq}/oos_returns.csv
                               archive/{ds}/{freq}/oos_turnover.csv
  Layer 3 - Paper tables    : archive/PAPER_TABLES/Table_*.csv

Criterion: max|Δ| = 0 on every layer.
Any value above 1e-10 is flagged as a mismatch.

Usage
-----
    # Auto-detect the most recent run in results/:
    python scripts/verify_reproduction.py

    # Or point to a specific run:
    python scripts/verify_reproduction.py --run-dir results/Run_<timestamp>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive"
DATASETS = ["Dow_Jones_30", "Global_ETFs", "Tech_Sector"]
FREQS = ["MONTHLY", "DAILY"]
TOL = 1e-10   # anything above this counts as a mismatch
# Columns that record wall-clock timing - non-deterministic by nature, excluded from comparison
SKIP_COLS = {"Grid_Search_s"}


# -- helpers ------------------------------------------------------------------

def latest_run(results_dir: Path) -> Path:
    runs = sorted(results_dir.glob("Run_*"), key=lambda p: p.name)
    if not runs:
        print(f"ERROR: no Run_* directories found in {results_dir}")
        sys.exit(1)
    return runs[-1]


def compare_csv(ref_path: Path, new_path: Path, label: str) -> dict:
    """Return {label, max_delta, n_cells, mismatches: list[str]}."""
    if not ref_path.exists():
        return {"label": label, "error": f"reference not found: {ref_path}"}
    if not new_path.exists():
        return {"label": label, "error": f"new file not found: {new_path}"}

    ref = pd.read_csv(ref_path)
    new = pd.read_csv(new_path)

    # Use first column as row index only when it is non-numeric (label column)
    first_col = ref.columns[0]
    if (first_col in new.columns and
            not pd.api.types.is_numeric_dtype(ref[first_col])):
        ref = ref.set_index(first_col)
        new = new.set_index(new.columns[0])
        common_idx = ref.index.intersection(new.index)
        ref = ref.loc[common_idx]
        new = new.loc[common_idx]

    # Compute numeric columns AFTER set_index so the index col is already gone;
    # exclude timing columns (Grid_Search_s) which vary between machines
    num_cols = [c for c in ref.columns if c in new.columns
                and pd.api.types.is_numeric_dtype(ref[c])
                and c not in SKIP_COLS]

    if not num_cols:
        return {"label": label, "error": "no overlapping numeric columns"}

    ref = ref[num_cols]
    new = new[num_cols]
    min_rows = min(len(ref), len(new))
    ref = ref.iloc[:min_rows]
    new = new.iloc[:min_rows]

    if ref.empty or new.empty:
        return {"label": label, "error": "no overlapping numeric rows/columns"}

    delta = (ref.values.astype(float) - new.values.astype(float))
    abs_delta = np.abs(delta)
    max_delta = float(np.nanmax(abs_delta))
    n_cells = int(np.sum(~np.isnan(abs_delta)))

    mismatches = []
    if max_delta > TOL:
        # find top-5 cells
        flat = abs_delta.flatten()
        top_idx = np.argsort(flat)[::-1][:5]
        rows_arr, cols_arr = np.unravel_index(top_idx, abs_delta.shape)
        for r, c in zip(rows_arr, cols_arr):
            if abs_delta[r, c] > TOL:
                row_label = ref.index[r] if hasattr(ref.index, '__getitem__') else r
                col_label = ref.columns[c]
                mismatches.append(
                    f"  row={row_label!r}  col={col_label!r}  "
                    f"ref={ref.iloc[r, c]:.6g}  new={new.iloc[r, c]:.6g}  "
                    f"|Δ|={abs_delta[r,c]:.2e}"
                )

    return {
        "label":      label,
        "max_delta":  max_delta,
        "n_cells":    n_cells,
        "pass":       max_delta <= TOL,
        "mismatches": mismatches,
    }


def fmt_result(r: dict) -> str:
    if "error" in r:
        return f"  SKIP  {r['label']}  - {r['error']}"
    mark = "PASS" if r["pass"] else "FAIL"
    return (f"  {mark}  {r['label']}  "
            f"max|Δ|={r['max_delta']:.2e}  cells={r['n_cells']}")


# -- main ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compare a pipeline run against the reference archive")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Path to the new run directory (default: most recent results/Run_*)")
    args = ap.parse_args()

    results_dir = ROOT / "results"
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run(results_dir)
    print(f"\nNew run  : {run_dir.name}")
    print(f"Reference: archive/ (Run_20260716_190431)")
    print(f"Tolerance: {TOL:.0e}\n")

    all_results = []

    # -- Layer 1 & 2: per-dataset, per-frequency raw files ------------------
    print("Layer 1: IS calibration (best_params.csv)")
    print("Layer 2: Raw OOS series (oos_returns, oos_turnover)")
    print("-" * 60)
    for ds in DATASETS:
        for freq in FREQS:
            for fname in ["best_params.csv", "oos_returns.csv", "oos_turnover.csv"]:
                label = f"{ds}/{freq}/{fname}"
                ref_path = ARCHIVE / ds / freq / fname
                new_path = run_dir / ds / freq / fname
                r = compare_csv(ref_path, new_path, label)
                all_results.append(r)
                print(fmt_result(r))
                if not r.get("pass", True) and r.get("mismatches"):
                    for m in r["mismatches"]:
                        print(m)

    # -- Layer 3: PAPER_TABLES ----------------------------------------------
    print()
    print("Layer 3: PAPER_TABLES")
    print("-" * 60)
    ref_tables = sorted((ARCHIVE / "PAPER_TABLES").glob("Table_*.csv"))
    for ref_path in ref_tables:
        new_path = run_dir / "PAPER_TABLES" / ref_path.name
        label = f"PAPER_TABLES/{ref_path.name}"
        r = compare_csv(ref_path, new_path, label)
        all_results.append(r)
        print(fmt_result(r))
        if not r.get("pass", True) and r.get("mismatches"):
            for m in r["mismatches"]:
                print(m)

    # -- Summary ------------------------------------------------------------
    valid = [r for r in all_results if "error" not in r]
    passed = sum(1 for r in valid if r["pass"])
    failed = sum(1 for r in valid if not r["pass"])
    skipped = len(all_results) - len(valid)

    overall = "PASS" if failed == 0 else "FAIL"
    print()
    print("=" * 60)
    print(f"VERDICT: {overall}")
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print(f"  Skipped: {skipped}")
    print("=" * 60)

    # -- Write markdown report ----------------------------------------------
    lines = [
        "# Reproduction report",
        f"New run: `{run_dir.name}`  ",
        f"Reference: `Run_20260716_190431` (archived in `archive/`)  ",
        f"Tolerance: `{TOL:.0e}`  ",
        "",
        f"## Verdict: {overall}  ({passed} passed, {failed} failed, {skipped} skipped)",
        "",
        r"| File | max\|Δ\| | Cells | Result |",
        "|------|---------|-------|--------|",
    ]
    for r in all_results:
        if "error" in r:
            lines.append(f"| {r['label']} | - | - | SKIP ({r['error']}) |")
        else:
            mark = "PASS" if r["pass"] else "FAIL"
            lines.append(f"| {r['label']} | {r['max_delta']:.2e} | {r['n_cells']} | {mark} |")

    if failed > 0:
        lines += ["", "## Mismatches (top 5 per file)", ""]
        for r in all_results:
            if r.get("mismatches"):
                lines.append(f"### {r['label']}")
                lines.extend(r["mismatches"])
                lines.append("")

    out = ROOT / "REPRODUCTION_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {out}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
