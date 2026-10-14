"""Table: Deflated Sharpe Ratio, average HHI and annual turnover by universe (--freq)."""
from _common import PAPER_NAME, UNIVERSE_LABEL, num, pd, panel, parse_args, strategies, universes, write_table


def main() -> None:
    a = parse_args(__doc__, freq=True)
    t = a.run_dir / "tables"
    parts = [("Panel A: Deflated Sharpe Ratio", "dsr", 3), ("Panel B: average HHI", "hhi", 3),
             ("Panel C: annualised turnover", "turnover", 2)]
    rows, cols = [], None
    for i, (title, name, nd) in enumerate(parts):
        df = pd.read_csv(t / f"{name}_{a.freq}.csv", index_col=0)
        cols = universes(df.columns)
        if i:
            rows.append(r"\midrule")
        rows.append(panel(title, len(cols) + 1))
        rows += [[PAPER_NAME[s]] + [num(df.loc[s, u], nd) for u in cols] for s in strategies(df.index)]
    f = a.freq.lower()
    write_table(a.out / f"tab_robustness_{f}.tex",
                f"{a.freq.capitalize()} robustness diagnostics: Deflated Sharpe Ratio (DSR), average "
                "portfolio concentration (HHI) and annualised turnover.",
                f"tab:robustness_{f}", ["Strategy"] + [UNIVERSE_LABEL[u] for u in cols], rows,
                note="DSR is reported for strategies with calibrated hyperparameters; its trials are the "
                     "Sharpe ratios of the candidates in the rolling calibration (static calibration "
                     "for MVN). Turnover is measured against the weights drifted by prices before "
                     "rebalancing. ``--'' denotes a strategy without hyperparameters.")


if __name__ == "__main__":
    main()
