"""
Post-processing of a run, from the saved OOS series to the LaTeX tables.

    python scripts/postprocess.py --run-dir results/main

Steps:
  1. tables/     metrics per universe and frequency (src/tables.py)
  2. inference/  joint inference against EW: stationary bootstrap, SPA-type
                 test and Romano-Wolf adjusted p-values (src/inference.py)
  3. summary/    ranks within universes and mean ranks (src/summary.py)
  4. tables/fallback_audit.csv   dates with weights equal to 1/N (scripts/audit_fallbacks.py)
  5. latex/      one script per table in scripts/tables/
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C                      # noqa: E402
from src.inference import N_BOOT, SEED, run_inference   # noqa: E402
from src.summary import run_summary              # noqa: E402
from src.tables import build_tables              # noqa: E402

TABLE_SCRIPTS = [
    ("tab_datasets.py", []),
    ("tab_performance.py", ["--freq", "MONTHLY"]), ("tab_performance.py", ["--freq", "DAILY"]),
    ("tab_robustness.py", ["--freq", "MONTHLY"]), ("tab_robustness.py", ["--freq", "DAILY"]),
    ("tab_costs.py", ["--freq", "MONTHLY"]), ("tab_costs.py", ["--freq", "DAILY"]),
    ("tab_hyperparams.py", ["--freq", "MONTHLY"]), ("tab_hyperparams.py", ["--freq", "DAILY"]),
    ("tab_spearman.py", []),
    ("tab_inference.py", ["--freq", "MONTHLY"]), ("tab_inference.py", ["--freq", "DAILY"]),
    ("tab_ranks.py", ["--freq", "MONTHLY"]), ("tab_ranks.py", ["--freq", "DAILY"]),
    ("tab_fallbacks.py", []),
    ("tab_compute.py", []),
    ("tab_main_ranks.py", []), ("tab_main_inference.py", []),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run-dir", type=Path, default=C.RESULTS_DIR / "main")
    ap.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap replications")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-inference", action="store_true")
    a = ap.parse_args()
    a.run_dir = a.run_dir.resolve()

    t0 = time.time()
    build_tables(a.run_dir)
    print(f"tables      {time.time() - t0:6.0f}s", flush=True)
    if not a.skip_inference:
        t0 = time.time()
        run_inference(a.run_dir, B=a.boot, seed=a.seed)
        print(f"inference   {time.time() - t0:6.0f}s  (B = {a.boot}, seed = {a.seed})", flush=True)
    run_summary(a.run_dir)
    # dates with weights equal to 1/N, read by the table of fallbacks
    subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_fallbacks.py"), "--run-dir", str(a.run_dir)],
                   capture_output=True, text=True, cwd=ROOT, check=True)

    failed = []
    for script, extra in TABLE_SCRIPTS:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "tables" / script),
                            "--run-dir", str(a.run_dir), *extra],
                           capture_output=True, text=True, cwd=ROOT / "scripts" / "tables")
        if r.returncode:
            failed.append((script, extra, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""))
        else:
            print("  " + r.stdout.strip().splitlines()[-1], flush=True)
    for script, extra, err in failed:
        print(f"  FAILED {script} {' '.join(extra)}: {err}", flush=True)
    print(f"post-processing done: {len(TABLE_SCRIPTS) - len(failed)} of {len(TABLE_SCRIPTS)} tables")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
