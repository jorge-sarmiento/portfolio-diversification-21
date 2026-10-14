"""Table: Spearman rank correlation between monthly and daily Sharpe-ratio rankings, by universe."""
from _common import UNIVERSE_LABEL, num, pd, parse_args, universes, write_table

from scipy import stats


def main() -> None:
    a = parse_args(__doc__)
    t = a.run_dir / "tables"
    rows, recs = [], []
    pairs = {k: (pd.read_csv(t / f"sharpe_{k}_MONTHLY.csv", index_col=0),
                 pd.read_csv(t / f"sharpe_{k}_DAILY.csv", index_col=0)) for k in ("gross", "net")}
    for u in universes(pairs["net"][0].columns):
        row = [UNIVERSE_LABEL[u]]
        rec = {"universe": u}
        for k in ("gross", "net"):
            m, d = pairs[k]
            if u not in d.columns:
                row += ["--", "--"]
                continue
            both = pd.concat([m[u], d[u]], axis=1, keys=["m", "d"]).dropna()
            rho, p = stats.spearmanr(both.m, both.d)
            row += [num(float(rho)), num(float(p))]
            rec.update({f"rho_{k}": rho, f"p_{k}": p, "n_strategies": len(both)})
        rows.append(row)
        recs.append(rec)
    pd.DataFrame(recs).to_csv(t / "spearman_monthly_daily.csv", index=False)
    write_table(a.out / "tab_spearman.tex",
                "Spearman rank correlation between monthly and daily Sharpe-ratio rankings across "
                "the twenty-one strategies.",
                "tab:spearman", ["Universe", r"$\rho$", "$p$-value", r"$\rho$", "$p$-value"], rows,
                top=r" & \multicolumn{2}{c}{Gross--gross} & \multicolumn{2}{c}{Net--net (10 bps)} \\"
                    r" \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
                note="Sharpe ratios in excess of the risk-free rate at both frequencies.")


if __name__ == "__main__":
    main()
