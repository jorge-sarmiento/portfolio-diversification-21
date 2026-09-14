"""
Draw Figure 1 (out-of-sample Sharpe ratio of MV against lambda) from the lambda-sweep curves.

    python scripts/plot_figure1.py [--curves-dir DIR]

The curves are read from archive/LAMBDA_SWEEP/ by default, or from the output directory of
scripts/lambda_sweep.py. The dashed vertical line marks lambda = 1, the upper end of the MN21
interval under the absorbed-lambda convention.

Output: figures/lambda_curve_{DS}_rolling12.png
"""

import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description="Draw Figure 1 from the lambda-sweep curves")
ap.add_argument("--curves-dir", type=Path, default=ROOT / "archive" / "LAMBDA_SWEEP",
                help="directory with lambda_curve_{DS}_rolling12.csv; default: archive/LAMBDA_SWEEP")
args = ap.parse_args()

DATA_DIR = args.curves_dir.resolve()
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

BOUNDARY = 1.0   # upper end of the MN21 interval under the absorbed-lambda convention

DATASETS = ["DJ30", "ETFs", "Tech"]

for ds in DATASETS:
    csv_path = DATA_DIR / f"lambda_curve_{ds}_rolling12.csv"
    if not csv_path.exists():
        print(f"[SKIP] {ds}: CSV not found at {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    print(f"{ds}: {len(df)} lambda points, lambda range [{df['lambda'].min()}, {df['lambda'].max()}]")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["lambda"], df["oos_sharpe"], marker="o", linewidth=1.8)
    ax.axvline(
        BOUNDARY,
        color="red",
        linestyle="--",
        label=f"lambda = {BOUNDARY} (MN21 classical boundary)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("OOS Sharpe (annualised)")
    ax.set_title(f"MV OOS Sharpe vs lambda -- {ds} (rolling-12)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = FIG_DIR / f"lambda_curve_{ds}_rolling12.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> saved {out_path}")

print("Done.")
