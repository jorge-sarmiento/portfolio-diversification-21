"""
MVN with the moments of total returns instead of excess returns.

    python scripts/diagnostics/mvn_total_moments.py [--run-dir results/main] [--workers N]
    python scripts/diagnostics/mvn_total_moments.py --solver exact

Default: MVN is run again in every task with the mean and covariance of the
total returns of the window, exactly as the main version is run with those of
the excess returns: lambda recalibrated in the static pass, differential
evolution with projection onto the simplex, same annualisation and covariance
estimator. Its out-of-sample series are compared with those of MVN in the run
(selected lambda, net Sharpe ratio, concentration, weights), and the joint
test against EW is repeated with this version in the place of MVN. The run
takes several hours and resumes from its checkpoints
(<run-dir>/diagnostics/mvn_total/).

--solver exact is a quick alternative without differential evolution: at the
lambda selected in the run, Eq. (9) is solved with both sets of moments by the
parametric search on the efficient frontier of scripts/check_mvn_solver.py.
It isolates the effect of the inputs at a common lambda and takes about half
an hour; --dates N restricts it to N evenly spaced dates per task (the series
are then not consecutive and only the weight comparisons are meaningful).

Writes mvn_total_vs_excess.csv (one row per date) and
mvn_total_vs_excess_summary.csv to <run-dir>/diagnostics/; with --solver exact
the two files end in _exact.csv and _exact_summary.csv.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

from _common import C, ROOT, np, panel, parse_args, pd, selected, sharpe

from sklearn.covariance import LedoitWolf

from src.costs import net_returns, turnover_series
from src.engine import WalkForwardEngine, init_oos_worker
from src.inference import MEAN_BLOCK, N_BOOT, SEED, joint_inference
from src.runner import Checkpoint
from src.strategies import RANDOM_SEED
from src.strategies.base import BaseStrategy
from src.strategies.markowitz import MeanSquaredVariance

sys.path.insert(0, str(ROOT / "scripts"))
from check_mvn_solver import frontier_solution      # noqa: E402


class MeanSquaredVarianceTotal(MeanSquaredVariance):
    """MVN estimated on total returns: the engine hands it the moments of the returns themselves."""
    excess_return_inputs = False


def options(ap) -> None:
    ap.add_argument("--solver", choices=["de", "exact"], default="de")
    ap.add_argument("--dates", type=int, default=0, help="with --solver exact: dates per task (0 = all)")
    ap.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap replications of the joint test")
    ap.add_argument("--universe", nargs="*", default=C.UNIVERSES, choices=C.UNIVERSES, help="default: all")


def describe(w: np.ndarray, R: np.ndarray, r: np.ndarray, ppy: int) -> dict:
    """Out-of-sample statistics of a weight matrix on the evaluated rows."""
    gross = (w * R).sum(axis=1)
    turn = turnover_series(w, R, "drift")
    net = net_returns(gross, turn)
    return {"sharpe_net": sharpe(net - r, ppy), "hhi": float((w ** 2).sum(axis=1).mean()),
            "turnover_ann": float(turn[1:].mean() * ppy) if len(turn) > 1 else float("nan"),
            "dates_one_asset_99": int((w.max(axis=1) > 0.99).sum()), "net": net}


def joint_test(folder, net_total: np.ndarray, rf: pd.Series, freq: str, boot: int) -> dict:
    """Joint test against EW on the net series of the run, with this version in the place of MVN."""
    R = pd.read_csv(folder / "oos_returns_net.csv", index_col=0, parse_dates=True)
    R["MSV"] = net_total
    X = R.values - rf.reindex(R.index).values[:, None]
    res = joint_inference(X, list(R.columns), "EW", boot, MEAN_BLOCK[freq], SEED,
                          periods_per_year=C.PERIODS_PER_YEAR[freq])
    row = res["table"].set_index("strategy").loc["MSV"]
    return {"spa_p_better": res["spa_p_better"], "spa_p_worse": res["spa_p_worse"],
            "mvn_p_better_rw": float(row.p_better_rw), "mvn_p_worse_rw": float(row.p_worse_rw)}


def run_de(a) -> None:
    """MVN with total-return moments through the engine, task by task."""
    rows, summary = [], []
    store = a.out / "mvn_total"
    for f in C.FREQS:
        for u in a.universe:
            rets, rf, sch = panel(u, f)
            data, ppy, oos = rets.values, C.PERIODS_PER_YEAR[f], sch["oos_idx"]
            folder = store / u / f
            np.random.seed(RANDOM_SEED)
            engine = WalkForwardEngine(MeanSquaredVarianceTotal, "MSV", periods_per_year=ppy, dsr_trials="rolling")
            with ProcessPoolExecutor(max_workers=a.workers, initializer=init_oos_worker,
                                     initargs=(data, rf.values)) as pool:
                res = engine.run(data, rets.index, rf=rf.values, schedule=sch,
                                 checkpoint=Checkpoint(folder / "checkpoints", "MSV"), oos_pool=pool)
            wt = res["weights"]
            pd.DataFrame(wt, index=rets.index[oos], columns=rets.columns).rename_axis("Date").to_csv(
                folder / "weights.csv")
            run = a.run_dir / u / f
            we = pd.read_csv(run / "weights" / "MSV.csv", index_col=0).values
            st, se = describe(wt, data[oos], rf.values[oos], ppy), describe(we, data[oos], rf.values[oos], ppy)
            pd.DataFrame({"net_total": st["net"]}, index=rets.index[oos]).rename_axis("Date").to_csv(
                folder / "oos_returns_net.csv")
            d = np.abs(wt - we).max(axis=1)
            task = {"universe": u, "freq": f, "N": data.shape[1], "dates": len(oos), "solver": "de",
                    "lambda_total": float(res["best_params"]["risk_aversion"]),
                    "lambda_excess": float(selected(a.run_dir, u, f, "MSV")["risk_aversion"])}
            for name, s in (("total", st), ("excess", se)):
                task.update({f"{k}_{name}": v for k, v in s.items() if k != "net"})
            task.update({"weight_diff_max": float(d.max()), "weight_diff_median": float(np.median(d)),
                         "dates_weight_diff_above_0.10": int((d > 0.10).sum())})
            task.update({f"{k}_total": v for k, v in joint_test(run, st["net"], rf, f, a.boot).items()})
            summary.append(task)
            rows += [{"universe": u, "freq": f, "date": rets.index[i].date(), "max_abs_weight_diff": d[k],
                      "hhi_total": float((wt[k] ** 2).sum()), "hhi_excess": float((we[k] ** 2).sum()),
                      "top_asset_total": rets.columns[int(wt[k].argmax())],
                      "top_asset_excess": rets.columns[int(we[k].argmax())]} for k, i in enumerate(oos)]
            print(f"  done {u} {f}", flush=True)
            pd.DataFrame(summary).to_csv(a.out / "mvn_total_vs_excess_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(a.out / "mvn_total_vs_excess.csv", index=False)
    show(pd.DataFrame(summary), ["lambda_total", "lambda_excess"])


def solve_exact(job):
    lam, ppy, window, rf_window = job
    out = []
    for returns in (window, window - rf_window[:, None]):
        mu = returns.mean(axis=0) * ppy
        cov = BaseStrategy.clean_matrix(LedoitWolf().fit(returns).covariance_ * ppy)
        out.append(frontier_solution(mu, cov, lam)[1])
    return out


def run_exact(a) -> None:
    """Both sets of moments at the lambda of the run, by the parametric search on the frontier."""
    rows, summary = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in C.FREQS:
            for u in a.universe:
                rets, rf, sch = panel(u, f)
                W, oos, ppy = sch["window"], sch["oos_idx"], C.PERIODS_PER_YEAR[f]
                if a.dates:
                    oos = oos[np.unique(np.linspace(0, len(oos) - 1, a.dates).astype(int))]
                R, r = rets.values, rf.values
                lam = float(selected(a.run_dir, u, f, "MSV")["risk_aversion"])
                res = list(pool.map(solve_exact, [(lam, ppy, R[i - W:i].copy(), r[i - W:i].copy()) for i in oos],
                                    chunksize=8))
                wt, we = np.array([x[0] for x in res]), np.array([x[1] for x in res])
                d = np.abs(wt - we).max(axis=1)
                task = {"universe": u, "freq": f, "N": R.shape[1], "dates": len(oos), "solver": "exact",
                        "lambda_total": lam, "lambda_excess": lam}
                for name, w in (("total", wt), ("excess", we)):
                    task.update({f"{k}_{name}": v for k, v in describe(w, R[oos], r[oos], ppy).items() if k != "net"})
                task.update({"weight_diff_max": float(d.max()), "weight_diff_median": float(np.median(d)),
                             "dates_weight_diff_above_0.10": int((d > 0.10).sum())})
                summary.append(task)
                rows += [{"universe": u, "freq": f, "date": rets.index[i].date(), "max_abs_weight_diff": d[k],
                          "hhi_total": float((wt[k] ** 2).sum()), "hhi_excess": float((we[k] ** 2).sum()),
                          "top_asset_total": rets.columns[int(wt[k].argmax())],
                          "top_asset_excess": rets.columns[int(we[k].argmax())]} for k, i in enumerate(oos)]
                print(f"  done {u} {f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "mvn_total_vs_excess_exact.csv", index=False)
    pd.DataFrame(summary).to_csv(a.out / "mvn_total_vs_excess_exact_summary.csv", index=False)
    show(pd.DataFrame(summary), ["lambda_excess"])


def show(s: pd.DataFrame, first) -> None:
    pd.set_option("display.width", 240)
    print(s[["freq", "universe"] + first + ["sharpe_net_total", "sharpe_net_excess", "hhi_total", "hhi_excess",
                                             "dates_one_asset_99_total", "dates_one_asset_99_excess",
                                             "weight_diff_max", "weight_diff_median"]].round(4).to_string())


def main() -> None:
    a = parse_args(__doc__.split("\n")[1], options)
    (run_exact if a.solver == "exact" else run_de)(a)


if __name__ == "__main__":
    main()
