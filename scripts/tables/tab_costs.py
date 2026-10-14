"""Table: net Sharpe ratio under transaction costs of 0, 10 and 20 basis points (--freq)."""
from _common import C, PAPER_NAME, UNIVERSE_LABEL, np, num, pd, panel, parse_args, strategies, write_table

from src.data import load_panel

COSTS_BPS = (0, 10, 20)


def main() -> None:
    a = parse_args(__doc__, freq=True)
    ppy = C.PERIODS_PER_YEAR[a.freq]
    tables = {c: {} for c in COSTS_BPS}
    for u in C.UNIVERSES:
        folder = a.run_dir / u / a.freq
        if not (folder / "oos_returns_gross.csv").exists():
            continue
        gross = pd.read_csv(folder / "oos_returns_gross.csv", index_col=0, parse_dates=True)
        turn = pd.read_csv(folder / "oos_turnover.csv", index_col=0, parse_dates=True)
        rf = load_panel(u, a.freq)[1].reindex(gross.index).values[:, None]
        for c in COSTS_BPS:
            ex = gross.values - turn.values * (c / 10000.0) - rf       # costs first, then RF
            sr = ex.mean(axis=0) * ppy / (ex.std(axis=0, ddof=1) * np.sqrt(ppy) + 1e-10)
            tables[c][u] = pd.Series(sr, index=gross.columns)
    cols = list(tables[0])
    out_csv = pd.concat({c: pd.DataFrame(tables[c]) for c in COSTS_BPS}, names=["cost_bps", "Strategy"])
    (a.run_dir / "tables").mkdir(exist_ok=True)
    out_csv.to_csv(a.run_dir / "tables" / f"cost_sensitivity_{a.freq}.csv")
    rows = []
    for i, c in enumerate(COSTS_BPS):
        df = pd.DataFrame(tables[c])
        if i:
            rows.append(r"\midrule")
        rows.append(panel(f"Panel {'ABC'[i]}: {c} bps", len(cols) + 1))
        rows += [[PAPER_NAME[s]] + [num(df.loc[s, u]) for u in cols] for s in strategies(df.index)]
    f = a.freq.lower()
    write_table(a.out / f"tab_costs_{f}.tex",
                f"{a.freq.capitalize()} Sharpe ratios under proportional transaction costs of 0, 10 and "
                "20 basis points.",
                f"tab:costs_{f}", ["Strategy"] + [UNIVERSE_LABEL[u] for u in cols], rows,
                note="Costs are deducted period by period from the gross return before subtracting "
                     "the risk-free rate; the 10 bps panel reproduces the headline Sharpe ratios.")


if __name__ == "__main__":
    main()
