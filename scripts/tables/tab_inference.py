"""Table: joint inference against EW -- SPA-type p-values and Romano-Wolf adjusted p-values (--freq)."""
from _common import PAPER_NAME, UNIVERSE_LABEL, num, pd, panel, parse_args, strategies, universes, write_table


def main() -> None:
    a = parse_args(__doc__, freq=True)
    inf = a.run_dir / "inference"
    summ = pd.read_csv(inf / "joint_summary.csv")
    tests = pd.read_csv(inf / "joint_tests.csv")
    summ, tests = summ[summ.freq == a.freq], tests[tests.freq == a.freq]
    cols = universes(summ.universe.unique())
    n = len(cols) + 1
    base = summ[summ.base_block]
    blocks = sorted(summ.block.unique())

    def line(label, series, col, block=None, nd=3):
        d = (summ[summ.block == block] if block is not None else base)
        d = d[d.series == series].set_index("universe")
        return [label] + [num(float(d.loc[u, col]), nd) if nd else str(int(d.loc[u, col])) for u in cols]

    rows = [panel("Panel A: joint tests (net returns unless stated)", n),
            line(r"$p$: no strategy beats EW", "net", "spa_p_better"),
            line(r"$p$: no strategy is worse than EW", "net", "spa_p_worse"),
            line(r"$p$: no strategy beats EW (gross)", "gross", "spa_p_better")]
    for b in blocks:
        rows.append(line(f"\\quad block = {b:g}", "net", "spa_p_better", block=b))
    rows += [line("Strategies better than EW (5\\%)", "net", "n_better_rw_5%", nd=0),
             line("Strategies worse than EW (5\\%)", "net", "n_worse_rw_5%", nd=0),
             line("Strategies in the family", "net", "n_family", nd=0)]

    main_tests = tests[(tests.series == "net") & tests.block.isin(base.block.unique())]
    for title, col in ((r"Panel B: Romano--Wolf adjusted $p$-values, better than EW", "p_better_rw"),
                       (r"Panel C: Romano--Wolf adjusted $p$-values, worse than EW", "p_worse_rw")):
        piv = main_tests.pivot(index="strategy", columns="universe", values=col)
        rows += [r"\midrule", panel(title, n)]
        rows += [[PAPER_NAME[s]] + [num(piv.loc[s, u]) for u in cols] for s in strategies(piv.index)]
    f = a.freq.lower()
    B, T = int(base.B.iloc[0]), int(base["T"].iloc[0])
    excluded = []
    net = base[base.series == "net"].set_index("universe")
    for col, why in (("excluded_identical", "return series identical to EW"),
                     ("excluded_near_identical",
                      "annualised tracking error against EW below 10 basis points")):
        items = [f"{', '.join(PAPER_NAME[s] for s in str(net.loc[u, col]).split())} ({UNIVERSE_LABEL[u]})"
                 for u in cols if isinstance(net.loc[u, col], str) and net.loc[u, col].strip()]
        if items:
            excluded.append(why + ": " + "; ".join(items))
    excl_note = (" Left out of the family -- " + ". ".join(excluded) + "."
                 if excluded else " No strategy was left out of the family.")
    write_table(a.out / f"tab_inference_{f}.tex",
                f"Joint inference against the equally weighted portfolio ({a.freq.lower()} frequency).",
                f"tab:inference_{f}", ["", ] + [UNIVERSE_LABEL[u] for u in cols], rows,
                note=f"Stationary bootstrap of the $T \\times 21$ matrix of excess returns "
                     f"($T = {T}$, $B = {B}$, mean block {base.block.iloc[0]:g}); the statistic is the "
                     "difference in Sharpe ratio with EW, studentised with its bootstrap standard "
                     "error. Panel A reports the $p$-value of the maximum statistic and its "
                     "sensitivity to the block length. A strategy that is EW, or almost EW, is left "
                     "out of the family because its studentised statistic is a ratio of two "
                     "quantities close to zero; it is shown as ``--''." + excl_note)


if __name__ == "__main__":
    main()
