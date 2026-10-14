"""
MV: out-of-sample Sharpe ratio against the risk-aversion parameter (Figure 1).

    python scripts/diagnostics/lambda_sweep.py [--run-dir results/main]

For lambda between 0.1 and 50, MV is run out of sample at monthly frequency in
the eight universes with lambda held fixed, and its net Sharpe ratio in excess
of the risk-free rate is compared with the lambda selected in the run and with
the equally weighted portfolio.

Writes lambda_sweep_MONTHLY.csv, lambda_sweep_summary.csv and
fig_lambda_sweep.pdf / .png to <run-dir>/diagnostics/.
"""
from _common import C, moments, np, panel, parse_args, pd, pool_map, selected, sharpe

from src.costs import net_returns, turnover_series
from src.strategies.markowitz import MeanVariance

LAMBDAS = [0.1, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0, 35.0, 50.0]
TITLE = {
    "DJ30": "Dow Jones 30 (N = 27)", "Tech": "Tech (N = 11)", "Global_ETFs": "Global ETFs (N = 14)",
    "Countries_Europe": "Countries: Europe (N = 8)",
    "Countries_APAC_Americas": "Countries: APAC and Americas (N = 9)",
    "Commodities": "Commodities (N = 7)", "French_10": "French 10 industries",
    "French_30": "French 30 industries",
}


def sweep(universe: str):
    rets, rf, sch = panel(universe, "MONTHLY")
    data, w, idx = rets.values, sch["window"], sch["oos_idx"]
    mom = [moments(data[i - w:i], 12) for i in idx]
    rows = []
    for lam in LAMBDAS:
        strat = MeanVariance(risk_aversion=lam)
        W = np.array([strat.safe_optimize(mu, cov) for mu, cov in mom])
        gross = (W * data[idx]).sum(axis=1)
        to = turnover_series(W, data[idx], "drift")
        r = rf.values[idx]
        rows.append({"universe": universe, "lambda": lam, "sharpe_net": sharpe(net_returns(gross, to) - r, 12),
                     "sharpe_gross": sharpe(gross - r, 12), "turnover_ann": float(to[1:].mean() * 12),
                     "hhi": float((W ** 2).sum(axis=1).mean())})
    return rows


def main() -> None:
    a = parse_args(__doc__.split("\n")[1])
    df = pd.DataFrame([r for part in pool_map(sweep, C.UNIVERSES, min(a.workers, 8)) for r in part])
    table = pd.read_csv(a.run_dir / "tables" / "sharpe_net_MONTHLY.csv", index_col=0)
    sel = pd.DataFrame([{
        "universe": u, "lambda_selected": float(selected(a.run_dir, u, "MONTHLY", "MV")["risk_aversion"]),
        "sharpe_net_run_MV": float(table.loc["MV", u]), "sharpe_net_run_EW": float(table.loc["EW", u]),
        "sharpe_net_run_GMV": float(table.loc["GMV", u])} for u in C.UNIVERSES])
    best = df.loc[df.groupby("universe").sharpe_net.idxmax(), ["universe", "lambda", "sharpe_net"]]
    best.columns = ["universe", "lambda_best_oos", "sharpe_net_best_oos"]
    summ = sel.merge(best, on="universe")
    for lam, name in ((0.1, "sharpe_net_at_0.1"), (1.0, "sharpe_net_at_1"), (50.0, "sharpe_net_at_50")):
        summ[name] = summ.universe.map(df[df["lambda"] == lam].set_index("universe").sharpe_net)
    df.to_csv(a.out / "lambda_sweep_MONTHLY.csv", index=False)
    summ.to_csv(a.out / "lambda_sweep_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True)
    for ax, u in zip(axes.ravel(), C.UNIVERSES):
        d, s = df[df.universe == u], summ[summ.universe == u].iloc[0]
        ax.plot(d["lambda"], d.sharpe_net, marker="o", ms=3.5, lw=1.5, color="#1f4e79", label="MV, net")
        ax.axhline(s.sharpe_net_run_EW, color="#7f7f7f", lw=1.0, ls=":", label="EW")
        ax.axvline(1.0, color="#c0392b", lw=1.0, ls="--", label=r"$\lambda = 1$")
        ax.axvline(s.lambda_selected, color="#2e7d32", lw=1.0, ls="-.", label=r"selected $\lambda$")
        ax.set_xscale("log")
        ax.set_title(TITLE[u], fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel(r"$\lambda$ (log scale)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("OOS Sharpe ratio (net, excess)", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(a.out / "fig_lambda_sweep.png", dpi=200)
    fig.savefig(a.out / "fig_lambda_sweep.pdf")
    plt.close(fig)
    print(summ.round(3).to_string())
    print(f"wrote {a.out / 'fig_lambda_sweep.pdf'}")


if __name__ == "__main__":
    main()
