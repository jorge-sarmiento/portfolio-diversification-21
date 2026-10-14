"""
SSD: second-order dominance constraints on chronological against sorted cumulative sums (Table 11).

    python scripts/diagnostics/ssd_sorting.py [--run-dir results/main]

Second-order stochastic dominance over a benchmark compares cumulative sums of
the returns ordered by the benchmark, not in chronological order. The script
solves both linear programs, with the equally weighted portfolio as benchmark,
on the estimation windows of the out-of-sample period (every monthly date; 120
evenly spaced daily dates) and on the static segment before it, and reports
the concentration of the two solutions and the dates on which they differ.

Writes ssd_sorting.csv and tab_ssd_sorting.tex to <run-dir>/diagnostics/.
"""
from scipy.optimize import linprog

from _common import C, np, panel, parse_args, pd, pool_map, sample, table_helpers, write_tex


def ssd_lp(data: np.ndarray, index: np.ndarray, sorted_variant: bool):
    """Maximum-mean portfolio whose cumulative returns dominate those of the index."""
    n = data.shape[1]
    if sorted_variant:
        order = np.argsort(index)
        a_ub, b_ub = -np.cumsum(data[order, :], axis=0), -np.cumsum(index[order])
    else:
        a_ub, b_ub = -np.cumsum(data, axis=0), -np.cumsum(index)
    res = linprog(-data.mean(axis=0), A_ub=a_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
                  bounds=[(0.0, 1.0)] * n, method="highs")
    return (res.x, True) if res.success else (np.ones(n) / n, False)


def compare(job):
    universe, freq = job
    rets, _, sch = panel(universe, freq)
    data, w = rets.values, sch["window"]
    seg = data[:sch["static_end"]]
    wc, okc = ssd_lp(seg, seg.mean(axis=1), False)
    ws, oks = ssd_lp(seg, seg.mean(axis=1), True)
    row = {"universe": universe, "freq": freq, "N": data.shape[1], "segment_rows": len(seg),
           "HHI_chronological_static": float((wc ** 2).sum()), "HHI_sorted_static": float((ws ** 2).sum()),
           "feasible_chronological_static": okc, "feasible_sorted_static": oks}
    idx = sample(sch["oos_idx"], freq)
    hc, hs, fc, fs, dif = [], [], 0, 0, 0
    for i in idx:
        window = data[i - w:i]
        x, okx = ssd_lp(window, window.mean(axis=1), False)
        y, oky = ssd_lp(window, window.mean(axis=1), True)
        hc.append((x ** 2).sum())
        hs.append((y ** 2).sum())
        fc += okx
        fs += oky
        dif += bool(np.abs(x - y).max() > 1e-8)
    row.update({"oos_dates_used": len(idx), "HHI_chronological_oos_mean": float(np.mean(hc)),
                "HHI_sorted_oos_mean": float(np.mean(hs)), "feasible_chronological_oos": fc,
                "feasible_sorted_oos": fs, "dates_where_solutions_differ": dif})
    row["delta_HHI_static"] = row["HHI_sorted_static"] - row["HHI_chronological_static"]
    row["delta_HHI_oos"] = row["HHI_sorted_oos_mean"] - row["HHI_chronological_oos_mean"]
    return row


def main() -> None:
    a = parse_args(__doc__.split("\n")[1])
    jobs = [(u, f) for f in C.FREQS for u in C.UNIVERSES]
    df = pd.DataFrame(pool_map(compare, jobs, a.workers))
    df.to_csv(a.out / "ssd_sorting.csv", index=False)

    t = table_helpers()
    rows = []
    for f in C.FREQS:
        rows.append(t.panel(f.capitalize(), 8))
        for r in df[df.freq == f].itertuples():
            rows.append([t.UNIVERSE_LABEL[r.universe], t.num(r.HHI_chronological_static), t.num(r.HHI_sorted_static),
                         t.num(r.delta_HHI_static), t.num(r.HHI_chronological_oos_mean),
                         t.num(r.HHI_sorted_oos_mean), t.num(r.delta_HHI_oos),
                         f"{r.dates_where_solutions_differ}/{r.oos_dates_used}"])
    write_tex(a.out / "tab_ssd_sorting.tex",
              "Effect of the SSD sorting correction on portfolio concentration (HHI): chronological cumulative "
              "sums against the pre-sorted formulation of Eq.~(22).",
              "tab:ssd_sorting",
              ["Universe", "Chron.", "Sorted", r"$\Delta$", "Chron.", "Sorted", r"$\Delta$", "Different"], rows,
              colspec="lrrrrrrr",
              top=r" & \multicolumn{3}{c}{Static segment} & \multicolumn{3}{c}{Out-of-sample mean} & \\"
                  r" \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
              note="The chronological columns correspond to cumulative sums in chronological order; the sorted "
                   "columns use Eq.~(22). Static segment: one program on all the data prior to the out-of-sample "
                   "period. Out-of-sample mean: average HHI over the rolling windows (the 84 monthly dates; 120 "
                   "equally spaced dates at daily frequency). Different: dates on which the two formulations give "
                   "different portfolios. Both formulations are feasible in every case.")
    print(df[["universe", "freq", "HHI_chronological_oos_mean", "HHI_sorted_oos_mean", "delta_HHI_oos",
              "dates_where_solutions_differ", "oos_dates_used"]].round(4).to_string())


if __name__ == "__main__":
    main()
