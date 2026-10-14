"""
Return-floor family: infeasibility of the hard floor and concentration of the soft floor.

    python scripts/diagnostics/return_floor.py [--run-dir results/main]

On the estimation windows of the out-of-sample period (every monthly date; 120
evenly spaced daily dates):

  hard floor   max mu'w  s.t.  r_t'w >= mean of mu for every period t of the
               window, 1'w = 1, w >= 0. The script counts the windows in which
               the program is feasible and compares the floor with the best
               worst-period return a long-only portfolio can reach.
  soft floor   max mu'w - (1/T) sum_t xi_t  s.t.  r_t'w + xi_t >= 0, xi_t >= 0,
               1'w = 1, w >= 0. Always feasible; the script reports how
               concentrated its solution is.

Writes return_floor.csv to <run-dir>/diagnostics/.
"""
from scipy.optimize import linprog

from _common import C, np, panel, parse_args, pd, pool_map, sample


def soft_floor(window: np.ndarray) -> np.ndarray:
    """Weights of the soft return-floor program on a window of per-period returns."""
    T, n = window.shape
    c = np.concatenate([-window.mean(axis=0), np.full(T, 1.0 / T)])
    a_ub = np.hstack([-window, -np.eye(T)])
    a_eq = np.hstack([np.ones((1, n)), np.zeros((1, T))])
    res = linprog(c, A_ub=a_ub, b_ub=np.zeros(T), A_eq=a_eq, b_eq=np.array([1.0]),
                  bounds=[(0.0, 1.0)] * n + [(0.0, None)] * T, method="highs")
    if not res.success:
        raise RuntimeError(f"soft floor: solver status {res.status}")
    return res.x[:n]


def floors(job):
    universe, freq = job
    rets, _, sch = panel(universe, freq)
    data, w = rets.values, sch["window"]
    n = data.shape[1]
    idx = sample(sch["oos_idx"], freq)
    feasible, hhi, single, n_assets, maximin, floor = 0, [], 0, [], [], []
    for i in idx:
        window = data[i - w:i]
        mu = window.mean(axis=0)
        hard = linprog(-mu, A_ub=-window, b_ub=np.full(len(window), -mu.mean()),
                       A_eq=np.ones((1, n)), b_eq=np.array([1.0]), bounds=[(0.0, 1.0)] * n, method="highs")
        feasible += bool(hard.success)
        # best worst-period return any long-only portfolio can reach in the window
        c = np.zeros(n + 1)
        c[-1] = -1.0
        mm = linprog(c, A_ub=np.hstack([-window, np.ones((len(window), 1))]), b_ub=np.zeros(len(window)),
                     A_eq=np.hstack([np.ones((1, n)), np.zeros((1, 1))]), b_eq=np.array([1.0]),
                     bounds=[(0.0, 1.0)] * n + [(None, None)], method="highs")
        maximin.append(-mm.fun if mm.success else np.nan)
        floor.append(mu.mean())
        wts = soft_floor(window)
        hhi.append((wts ** 2).sum())
        single += bool(wts.max() > 0.99)
        n_assets.append(int((wts > 0.01).sum()))
    return {"universe": universe, "freq": freq, "N": n, "window": w, "dates_used": len(idx),
            "hard_feasible_dates": feasible, "hard_infeasible_%": 100.0 * (1 - feasible / len(idx)),
            "floor_mean_per_period_%": 100 * float(np.mean(floor)),
            "maximin_return_mean_per_period_%": 100 * float(np.nanmean(maximin)),
            "dates_with_maximin_below_floor": int(np.sum(np.array(maximin) < np.array(floor))),
            "soft_single_asset_dates": single, "soft_single_asset_%": 100.0 * single / len(idx),
            "soft_HHI_mean": float(np.mean(hhi)), "soft_assets_above_1%_mean": float(np.mean(n_assets))}


def main() -> None:
    a = parse_args(__doc__.split("\n")[1])
    jobs = [(u, f) for f in C.FREQS for u in C.UNIVERSES]
    df = pd.DataFrame(pool_map(floors, jobs, a.workers))
    df.to_csv(a.out / "return_floor.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
