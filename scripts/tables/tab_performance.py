"""Table: out-of-sample net Sharpe ratio by universe, with the difference to EW (--freq MONTHLY or DAILY)."""
from _common import PAPER_NAME, UNIVERSE_LABEL, num, pd, panel, parse_args, strategies, universes, write_table


def main() -> None:
    a = parse_args(__doc__, freq=True)
    t = a.run_dir / "tables"
    sr = pd.read_csv(t / f"sharpe_net_{a.freq}.csv", index_col=0)
    ds = pd.read_csv(t / f"dsharpe_vs_ew_{a.freq}.csv", index_col=0)
    cols = universes(sr.columns)
    n = len(cols) + 1
    rows = [panel("Panel A: Sharpe ratio (net of 10 bps, in excess of the risk-free rate)", n)]
    rows += [[PAPER_NAME[s]] + [num(sr.loc[s, u]) for u in cols] for s in strategies(sr.index)]
    rows += [r"\midrule", panel(r"Panel B: difference with the equally weighted portfolio, $\Delta$SR", n)]
    rows += [[PAPER_NAME[s]] + [num(ds.loc[s, u], 2, plus=True) for u in cols] for s in strategies(ds.index)]
    f = a.freq.lower()
    write_table(a.out / f"tab_performance_{f}.tex",
                f"{a.freq.capitalize()} out-of-sample performance by asset universe.",
                f"tab:{f}", ["Strategy"] + [UNIVERSE_LABEL[u] for u in cols], rows,
                note="Sharpe ratios are computed on returns net of a 10 bps proportional transaction "
                     "cost and in excess of the risk-free rate, annualised. Sharpe ratios are not "
                     "averaged across universes; see the rank summary.")


if __name__ == "__main__":
    main()
