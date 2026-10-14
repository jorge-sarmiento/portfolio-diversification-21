"""
Command-line entry point.

    python run.py snapshot     download the data into data/snapshot/
    python run.py verify       compare the data with data/MANIFEST.json
    python run.py run          full experiment: 21 strategies, 8 universes, 2 frequencies
    python run.py run --quick  the same pipeline on a few rows, as an end-to-end check

Outputs are written to results/main/ (results/quick/ with --quick); set the
environment variable RESULTS_DIR to write them elsewhere. Tables are built
afterwards with scripts/postprocess.py.

The number of BLAS threads is fixed before NumPy is imported: one per process.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("command", choices=["snapshot", "verify", "run"])
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS threads per process (0 = library default)")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes (0 = number of physical CPU cores)")
    ap.add_argument("--quick", action="store_true", help="short run on a few rows")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints")
    return ap.parse_args()


ARGS = _parse()
os.environ["BLAS_THREADS"] = str(ARGS.threads)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import logging                                       # noqa: E402
import warnings                                      # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)        # strategy warnings are collected by the runner

import pandas as pd                                  # noqa: E402

from src import config as C                          # noqa: E402

# rows of the short run: OOS rows and rows of the rolling calibration, per frequency
QUICK_OOS = {"MONTHLY": 6, "DAILY": 16}
QUICK_CALIB = {"MONTHLY": 12, "DAILY": 16}


def cmd_snapshot() -> None:
    from src.data import build_snapshot, verify_snapshot
    build_snapshot()
    ok = verify_snapshot()
    print(f"snapshot: {sum(ok.values())} of {len(ok)} files match data/MANIFEST.json")


def cmd_verify() -> None:
    from src.data import verify_snapshot
    ok = verify_snapshot()
    for name, same in ok.items():
        print(f"  {'ok       ' if same else 'DIFFERENT'} {name}")
    print(f"{sum(ok.values())} of {len(ok)} files match data/MANIFEST.json")
    sys.exit(0 if all(ok.values()) else 1)


def cmd_run() -> None:
    """
    Two phases, so the cores are used in both:
      A. every strategy except MVN: one process per (universe, frequency) task;
      B. MVN: one task at a time, its OOS rows spread over the worker processes.
    Phase B reloads phase A from the checkpoints, so the output files of each
    task hold all the strategies.
    """
    from src.runner import physical_cores, run_task, run_tasks
    from src.tables import build_tables
    workers = ARGS.workers or physical_cores()
    resume = not ARGS.no_resume
    run_dir = C.RESULTS_DIR / ("quick" if ARGS.quick else "main")
    run_dir.mkdir(parents=True, exist_ok=True)
    oos = (lambda f: QUICK_OOS[f]) if ARGS.quick else (lambda f: None)
    calib = (lambda f: QUICK_CALIB[f]) if ARGS.quick else (lambda f: None)
    fast = [s for s in C.STRATEGIES if s not in C.ROW_PARALLEL_STRATEGIES]

    t0 = time.time()
    order = [(u, f) for f in ("DAILY", "MONTHLY") for u in C.UNIVERSES]
    a = run_tasks([(u, f, run_dir, fast, oos(f), resume, 1, calib(f)) for u, f in order], workers=workers,
                  blas_threads=ARGS.threads)
    a.to_csv(run_dir / "timing_phaseA.csv", index=False)
    t_a = time.time() - t0
    print(f"phase A done in {t_a:.0f}s: {len(fast)} strategies", flush=True)

    t0 = time.time()
    parts = []
    for f in ("MONTHLY", "DAILY"):
        for u in C.UNIVERSES:
            parts.append(run_task(u, f, run_dir, C.STRATEGIES, oos(f), True,
                                  row_workers=workers, calib_limit=calib(f)))
            print(f"  phase B done {u} {f}", flush=True)
    pd.concat(parts, ignore_index=True).to_csv(run_dir / "timing_all.csv", index=False)
    t_b = time.time() - t0
    build_tables(run_dir)
    (run_dir / "RUN_DONE").write_text(
        f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\nphaseA_s={t_a:.0f}\nphaseB_s={t_b:.0f}\nworkers={workers}\n")
    print(f"phase B done in {t_b:.0f}s. Outputs in {run_dir}", flush=True)


if __name__ == "__main__":
    {"snapshot": cmd_snapshot, "verify": cmd_verify, "run": cmd_run}[ARGS.command]()
