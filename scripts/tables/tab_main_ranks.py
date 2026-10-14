"""Compact table for the main text: mean rank and universes above EW, monthly and daily side by side."""
from _common import PAPER_NAME, num, pd, parse_args, strategies, write_table


def main() -> None:
    a = parse_args(__doc__)
    s = a.run_dir / "summary"
    m = pd.read_csv(s / "ranks_MONTHLY.csv", index_col=0)
    d = pd.read_csv(s / "ranks_DAILY.csv", index_col=0)
    order = sorted(strategies(m.index), key=lambda x: m.loc[x, "mean_rank"])

    def cells(t, x):
        if x not in t.index:
            return ["--", "--"]
        n = int(t.loc[x, "n_universes"])
        return [num(float(t.loc[x, "mean_rank"]), 2), "--" if x == "EW" else f"{int(t.loc[x, 'n_beats_ref'])}/{n}"]

    rows = [[PAPER_NAME[x]] + cells(m, x) + cells(d, x) for x in order]
    write_table(a.out / "tab_main_ranks.tex",
                "Rank summary across the eight universes: mean rank by net Sharpe ratio and number of "
                "universes in which the strategy has a higher net Sharpe ratio than EW.",
                "tab:main_ranks", ["Strategy", "Mean rank", "Above EW", "Mean rank", "Above EW"], rows,
                colspec="lrrrr",
                top=r" & \multicolumn{2}{c}{Monthly} & \multicolumn{2}{c}{Daily} \\"
                    r" \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
                note="Strategies are ranked within each universe (1 = highest net Sharpe ratio; ties share "
                     "the average rank) and ordered here by monthly mean rank. Sharpe ratios are not "
                     "averaged across universes.")


if __name__ == "__main__":
    main()
