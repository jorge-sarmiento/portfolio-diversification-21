"""
Task runner.

A task is one (universe, frequency) pair. Tasks run in separate processes with
one BLAS thread each; inside a task the strategies run one after another and
each one is checkpointed, so an interrupted run resumes where it stopped.

Design: simple returns, risk-free rate from the Kenneth French library, fixed
calendar (60-month / 1260-session estimation window, calibration 2013-2018,
out of sample 2019-2025), turnover and costs as in src/costs.py.

Outputs of a task, in <out_dir>/<universe>/<FREQ>/:
    oos_returns_gross.csv   r_p = w'R per date and strategy
    oos_returns_net.csv     gross minus cost * turnover, period by period
    oos_turnover.csv        sum |w_t - w_t^-| per date and strategy
    weights/<STRATEGY>.csv  weights per date
    best_params.csv         selected hyperparameters, in-sample Sharpe, trials
    dsr_trials.csv          Sharpe of every candidate logged for the DSR
    timing.csv              wall-clock and CPU seconds per strategy
    grid_feasibility.csv    combinations of the grids with weight bounds: feasible,
                            forces EW, or removed as infeasible for this N
    diagnostics.csv         optimiser failures (fallback of safe_optimize), Python
                            warnings, and OOS rows whose weights equal 1/N
    checkpoints/            resume state (deleted strategies are recomputed)
"""
import itertools
import logging
import os
import pickle
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def configure_threads(blas_threads: int = 1) -> None:
    """
    Set the number of BLAS/OpenMP threads of this process and of its children.

    With one thread per process (default) the number of worker processes does
    not change any result. 0 leaves the library default; the trust-constr
    strategies (MD, RP) and MVN depend on the thread count in the last digits.
    """
    os.environ["BLAS_THREADS"] = str(blas_threads)
    for var in _THREAD_VARS:
        if blas_threads > 0:
            os.environ[var] = str(blas_threads)
        else:
            os.environ.pop(var, None)
    if blas_threads > 0:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(blas_threads)
        except ImportError:
            pass


configure_threads(int(os.environ.get("BLAS_THREADS", "1")))


def physical_cores() -> int:
    """Number of physical CPU cores; half the logical count if it cannot be read."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            return int(out)
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                      capture_output=True, text=True).stdout.strip())
        cores = set()
        phys = core = None
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if core is not None:
                        cores.add((phys, core))
                    phys = core = None
        if cores:
            return len(cores)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


class _Collector(logging.Handler):
    """Collects the warnings logged by the strategies during one strategy run."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


import numpy as np      # noqa: E402  (imported after the thread variables exist)
import pandas as pd     # noqa: E402

from src import config as C                      # noqa: E402
from src import grid                             # noqa: E402
from src.costs import net_returns, turnover_series   # noqa: E402
from src.engine import WalkForwardEngine, init_oos_worker   # noqa: E402
from src.strategies import RANDOM_SEED, STRATEGY_MAP        # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _retry(action, what: str, attempts: int = 30, wait_s: float = 2.0, must: bool = True):
    """
    Run a file operation, retrying while the file system refuses it.

    On file systems that lock files briefly (for example while they are being
    synchronised), replacing or writing a file can fail with PermissionError
    for a moment. With must=False a failure that persists is logged and
    skipped (a checkpoint that is not saved only costs recomputing after an
    interruption); with must=True it is raised after the last attempt.
    """
    for k in range(attempts):
        try:
            return action()
        except PermissionError as e:
            err = e
            time.sleep(wait_s)
    if must:
        raise err
    logging.getLogger(__name__).error(f"{what}: not saved after {attempts} attempts ({err})")
    return None


class Checkpoint:
    """Resume state of one strategy inside one task."""

    def __init__(self, folder: Path, code: str, every_s: Optional[float] = None):
        self.calib_path = folder / f"{code}.calib.pkl"
        self.oos_path = folder / f"{code}.oos.npz"
        # seconds between two saves of the OOS progress
        self.every_s = float(os.environ.get("CHECKPOINT_S", "120")) if every_s is None else every_s
        self._last = time.time()
        folder.mkdir(parents=True, exist_ok=True)

    def load_calibration(self):
        if self.calib_path.exists():
            with open(self.calib_path, "rb") as f:
                return pickle.load(f)
        return None

    def save_calibration(self, state: Dict) -> None:
        tmp = self.calib_path.with_suffix(".tmp")

        def write():
            with open(tmp, "wb") as f:
                pickle.dump(state, f)
            os.replace(tmp, self.calib_path)
        _retry(write, f"calibration checkpoint {self.calib_path.name}", must=False)

    def load_oos(self):
        if not self.oos_path.exists():
            return None
        z = np.load(self.oos_path, allow_pickle=False)
        lvw = z["last_valid_weights"]
        return {"t_done": int(z["t_done"]), "returns": z["returns"], "weights": z["weights"],
                "last_valid_weights": None if lvw.size == 0 else lvw}

    def save_oos(self, t_done, returns, weights, last_valid_weights) -> None:
        if t_done < len(returns) and time.time() - self._last < self.every_s:
            return
        self._last = time.time()
        tmp = self.oos_path.with_suffix(".tmp.npz")

        def write():
            np.savez(tmp, t_done=t_done, returns=returns, weights=weights,
                     last_valid_weights=(np.array([]) if last_valid_weights is None
                                         else last_valid_weights))
            os.replace(tmp, self.oos_path)
        # the progress saves are optional; the last one is retried for longer
        final = t_done >= len(returns)
        _retry(write, f"OOS checkpoint {self.oos_path.name}", attempts=90 if final else 15, must=False)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_inputs(universe: str, freq: str) -> Dict:
    """Return matrix, dates, risk-free rate and schedule of a task."""
    from src.data import load_panel, schedule_indices
    rets, rf = load_panel(universe, freq)
    return {"data": rets.values, "dates": rets.index, "assets": list(rets.columns),
            "rf": rf.values, "ppy": C.PERIODS_PER_YEAR[freq],
            "schedule": schedule_indices(rets.index, freq)}


# ---------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------

def run_task(universe: str, freq: str, out_dir: Path,
             strategies: List[str], oos_limit: Optional[int] = None,
             resume: bool = True, row_workers: int = 1,
             calib_limit: Optional[int] = None) -> pd.DataFrame:
    """
    Run every strategy of one (universe, frequency) pair and write its outputs.

    row_workers > 1 spreads the OOS rows of the strategies listed in
    config.ROW_PARALLEL_STRATEGIES over that many processes. The outputs
    are the same for any value.

    oos_limit and calib_limit shorten the run for smoke tests: the first
    oos_limit OOS rows and the last calib_limit rows of the rolling calibration.
    """
    configure_threads(int(os.environ.get("BLAS_THREADS", "1")))
    np.random.seed(RANDOM_SEED)

    inp = load_inputs(universe, freq)
    data, dates, rf, ppy = inp["data"], inp["dates"], inp["rf"], inp["ppy"]
    schedule = inp["schedule"]
    if oos_limit is not None:
        schedule = dict(schedule, oos_idx=schedule["oos_idx"][:oos_limit])
    if calib_limit is not None:
        schedule = dict(schedule, calib_idx=schedule["calib_idx"][-calib_limit:])

    folder = Path(out_dir) / universe / freq
    ckpt_dir = folder / "checkpoints"
    (folder / "weights").mkdir(parents=True, exist_ok=True)
    if not resume and ckpt_dir.exists():
        for p in ckpt_dir.iterdir():
            p.unlink()

    results, timing, params_rows, trial_rows, diag_rows, grid_rows = {}, [], [], [], [], []
    # Rows of strategies finished in an earlier call are kept as they were written
    old = {}
    for name in ("timing", "diagnostics"):
        path = folder / f"{name}.csv"
        old[name] = pd.read_csv(path) if resume and path.exists() and path.stat().st_size > 2 else None
    src_logger = logging.getLogger("src")
    row_pool = None
    if row_workers > 1 and any(s in C.ROW_PARALLEL_STRATEGIES for s in strategies):
        row_pool = ProcessPoolExecutor(max_workers=row_workers,
                                       initializer=init_oos_worker, initargs=(data, rf))
    for code in strategies:
        cls = STRATEGY_MAP[code]
        ckpt = Checkpoint(ckpt_dir, code)
        state = ckpt.load_oos()
        was_done = (state is not None and ckpt.calib_path.exists()
                    and int(state["t_done"]) == len(state["returns"]))
        engine = WalkForwardEngine(cls, code, periods_per_year=ppy, dsr_trials="rolling")
        # infeasible weight-bound combinations never enter the calibration
        engine.combo_filter = grid.is_feasible
        param_grid = engine._get_param_grid()
        if "lower_bound" in param_grid or "upper_bound" in param_grid:
            for values in itertools.product(*param_grid.values()):
                combo = dict(zip(param_grid, values))
                ok, forces_ew, reason = grid.check(combo, data.shape[1])
                grid_rows.append({"universe": universe, "freq": freq, "strategy": code,
                                  "N": data.shape[1], **combo, "feasible": ok,
                                  "forces_EW": forces_ew, "reason": reason})
        collector = _Collector()
        src_logger.addHandler(collector)
        cpu0 = time.process_time()
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")     # each distinct warning once
            res = engine.run(data, dates, rf=rf, schedule=schedule, checkpoint=ckpt,
                             oos_pool=row_pool if code in C.ROW_PARALLEL_STRATEGIES else None)
        wall = time.perf_counter() - t0
        # CPU seconds: this process plus the row workers. A strategy resumed
        # from a checkpoint only counts the part computed in this call.
        cpu = time.process_time() - cpu0 + res["oos_worker_cpu_s"]
        src_logger.removeHandler(collector)
        for kind, msgs in (("optimizer_failure", collector.messages),
                           ("python_warning", [f"{w.category.__name__}: {w.message}" for w in caught])):
            for msg, n in pd.Series(msgs, dtype=object).value_counts().items():
                diag_rows.append({"universe": universe, "freq": freq, "strategy": code,
                                  "kind": kind, "count": int(n), "message": str(msg)[:300]})
        n_assets = data.shape[1]
        ew_rows = int((np.abs(res["weights"] - 1.0 / n_assets).max(axis=1) < 1e-12).sum())
        if ew_rows and code != "EW":
            diag_rows.append({"universe": universe, "freq": freq, "strategy": code,
                              "kind": "oos_weights_equal_1/N", "count": ew_rows,
                              "message": f"{ew_rows} of {len(res['weights'])} OOS rows"})
        results[code] = res
        timing.append({"universe": universe, "freq": freq, "strategy": code,
                       "wall_s": wall, "calibration_s": res["grid_search_time"],
                       "n_trials": res["n_trials"], "oos_steps": len(res["returns"]),
                       "N": data.shape[1], "cpu_s": cpu,
                       "resumed_mid_run": bool(state is not None and not was_done)})
        if was_done and old["timing"] is not None and (old["timing"].strategy == code).any():
            timing[-1] = old["timing"][old["timing"].strategy == code].iloc[0].to_dict()
        if was_done and old["diagnostics"] is not None:
            diag_rows = [r for r in diag_rows if r["strategy"] != code]
            diag_rows += old["diagnostics"][old["diagnostics"].strategy == code].to_dict("records")
        params_rows.append({"Dataset": f"{universe} - {freq.capitalize()}", "Strategy": code,
                            "IS_Sharpe": res["best_is_sharpe"], "N_Trials": res["n_trials"],
                            **res["best_params"]})
        trial_rows += [{"Strategy": code, "trial": k, "sharpe_per_period": t["sharpe"],
                        "mean": t["mean"], "std": t["std"], **t["params"]}
                       for k, t in enumerate(res["trials_log"])]

    if row_pool is not None:
        row_pool.shutdown()

    # -- write the series ---------------------------------------------------
    oos_dates = next(iter(results.values()))["dates"]
    oos_rows = np.asarray(dates.get_indexer(oos_dates))
    asset_rets = data[oos_rows]
    gross, net, turn = {}, {}, {}
    for code, res in results.items():
        turn[code] = turnover_series(res["weights"], asset_rets, "drift")
        gross[code] = res["returns"]
        net[code] = net_returns(res["returns"], turn[code])
        w = pd.DataFrame(res["weights"], index=oos_dates, columns=inp["assets"])
        w.index.name = "Date"
        _retry(lambda w=w, code=code: w.to_csv(folder / "weights" / f"{code}.csv"), f"weights {code}")
    for name, d in (("oos_returns_gross", gross), ("oos_returns_net", net), ("oos_turnover", turn)):
        df = pd.DataFrame(d, index=oos_dates)
        df.index.name = "Date"
        _retry(lambda df=df, name=name: df.to_csv(folder / f"{name}.csv"), name)
    _retry(lambda: pd.DataFrame(params_rows).to_csv(folder / "best_params.csv", index=False), "best_params")
    _retry(lambda: pd.DataFrame(trial_rows).to_csv(folder / "dsr_trials.csv", index=False), "dsr_trials")
    pd.DataFrame(grid_rows).to_csv(folder / "grid_feasibility.csv", index=False)
    df_diag = pd.DataFrame(diag_rows, columns=["universe", "freq", "strategy", "kind", "count", "message"])
    if old["diagnostics"] is not None:        # rows of strategies that were not part of this call
        df_diag = pd.concat([df_diag, old["diagnostics"][~old["diagnostics"].strategy.isin(strategies)]],
                            ignore_index=True)
    _retry(lambda: df_diag.to_csv(folder / "diagnostics.csv", index=False), "diagnostics")
    df_timing = pd.DataFrame(timing)
    df_all = df_timing
    if old["timing"] is not None:             # rows of strategies that were not part of this call
        df_all = pd.concat([df_timing, old["timing"][~old["timing"].strategy.isin(strategies)]],
                           ignore_index=True)
    _retry(lambda: df_all.to_csv(folder / "timing.csv", index=False), "timing")
    return df_timing


def _worker(args: Tuple) -> pd.DataFrame:
    return run_task(*args)


def run_tasks(tasks: List[Tuple], workers: int = 1, blas_threads: int = 1) -> pd.DataFrame:
    """
    Run tasks serially (workers = 1, in this process) or in a process pool.

    Each element of `tasks` is the argument tuple of run_task. Both paths call
    the same function with the same number of BLAS threads, so their outputs
    are identical when blas_threads = 1.
    """
    configure_threads(blas_threads)
    if workers <= 1:
        return pd.concat([run_task(*t) for t in tasks], ignore_index=True)
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            out.append(fut.result())
            t = futures[fut]
            print(f"  done {t[0]} {t[1]} [{out[-1].wall_s.sum():.0f}s]", flush=True)
    return pd.concat(out, ignore_index=True)
