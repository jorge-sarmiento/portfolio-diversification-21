"""Table: rank of each strategy within each universe and mean rank (--freq)."""
from _common import PAPER_NAME, UNIVERSE_LABEL, num, pd, parse_args, strategies, universes, write_table


def main() -> None:
    a = parse_args(__doc__, freq=True)
    ranks = pd.read_csv(a.run_dir / "summary" / f"ranks_{a.freq}.csv", index_col=0)
    cols = universes(ranks.columns)
    order = sorted(strategies(ranks.index), key=lambda x: ranks.loc[x, "mean_rank"])
    rows = [[PAPER_NAME[x]] + [f"{ranks.loc[x, u]:g}" for u in cols]
            + [num(float(ranks.loc[x, "mean_rank"]), 2),
               "--" if x == "EW" else f"{int(ranks.loc[x, 'n_beats_ref'])}/{len(cols)}"]
            for x in order]
    f = a.freq.lower()
    write_table(a.out / f"tab_ranks_{f}.tex",
                f"Rank of each strategy within each universe by net Sharpe ratio ({f} frequency), "
                "ordered by mean rank.",
                f"tab:ranks_{f}",
                ["Strategy"] + [UNIVERSE_LABEL[u] for u in cols] + ["Mean rank", "Beats EW"], rows,
                note="Rank 1 is the highest net Sharpe ratio in the universe; ties share the average "
                     "rank. Sharpe ratios are not averaged across universes.")


if __name__ == "__main__":
    main()
