"""
DMVY: share of calibration dates on which each grid combination returns 1/N (Table 10).

    python scripts/diagnostics/dmvy_collapse.py [--run-dir results/main]

For every combination of the DMVY grid, universe and frequency, DMVY is solved
on each estimation window of the calibration period and the dates whose
weights equal 1/N are counted. The combination selected in the run is flagged.

Writes dmvy_collapse_by_grid.csv, dmvy_share_equal_1N_<FREQ>.csv and
tab_dmvy_collapse.tex to <run-dir>/diagnostics/.
"""
import itertools

from _common import C, TOL, moments, np, panel, parse_args, pd, pool_map, selected, table_helpers, write_tex

from src.engine import WalkForwardEngine
from src.strategies.advanced import DMVYager


def collapse(job):
    universe, freq, run_dir = job
    rets, _, sch = panel(universe, freq)
    data, w, ppy = rets.values, sch["window"], C.PERIODS_PER_YEAR[freq]
    n = data.shape[1]
    grid = WalkForwardEngine._PARAM_GRIDS["DMV_Yager"]
    mom = [moments(data[i - w:i], ppy) for i in sch["calib_idx"]]
    sel = selected(run_dir, universe, freq, "DMV_Yager")
    rows = []
    for values in itertools.product(*grid.values()):
        combo = dict(zip(grid, values))
        ew, dist = 0, []
        for mu, cov in mom:
            d = np.abs(DMVYager(**combo).safe_optimize(mu, cov) - 1.0 / n)
            ew += bool(d.max() < TOL)
            dist.append(d.sum())
        rows.append({"universe": universe, "freq": freq, **combo, "calibration_dates": len(mom),
                     "dates_equal_1/N": ew, "share_equal_1/N_%": 100.0 * ew / len(mom),
                     "mean_L1_distance_to_1/N": float(np.mean(dist)),
                     "selected": bool(all(np.isclose(float(sel[k]), v) for k, v in combo.items()))})
    return rows


def main() -> None:
    a = parse_args(__doc__.split("\n")[1])
    jobs = [(u, f, a.run_dir) for f in C.FREQS for u in C.UNIVERSES]
    df = pd.DataFrame([r for part in pool_map(collapse, jobs, a.workers) for r in part])
    df.to_csv(a.out / "dmvy_collapse_by_grid.csv", index=False)

    t = table_helpers()
    rows = []
    for f in C.FREQS:
        p = df[df.freq == f].pivot_table(index=["risk_aversion", "diversification_factor"], columns="universe",
                                         values="share_equal_1/N_%")[C.UNIVERSES]
        p.to_csv(a.out / f"dmvy_share_equal_1N_{f}.csv")
        chosen = df[(df.freq == f) & df.selected].set_index("universe")
        rows.append(t.panel(f"{f.capitalize()} frequency", len(C.UNIVERSES) + 2))
        for (lam, delta), r in p.iterrows():
            cells = []
            for u in C.UNIVERSES:
                mark = (np.isclose(chosen.loc[u, "risk_aversion"], lam)
                        and np.isclose(chosen.loc[u, "diversification_factor"], delta))
                cells.append(t.num(float(r[u]), 1) + ("$^{*}$" if mark else ""))
            rows.append([f"{lam:g}", f"{delta:g}"] + cells)
        print(f, "\n", p.round(1).to_string())
    write_tex(a.out / "tab_dmvy_collapse.tex",
              "DMVY: percentage of calibration dates on which the weights equal $1/N$, by grid combination "
              "and universe.",
              "tab:dmvy_collapse", [r"$\lambda$", r"$\delta$"] + [t.UNIVERSE_LABEL[u] for u in C.UNIVERSES], rows,
              colspec="rr" + "r" * len(C.UNIVERSES),
              note="Each combination of the grid is solved on every estimation window of the calibration "
                   "period (2013--2018). An asterisk marks the combination selected in the run.")


if __name__ == "__main__":
    main()
