"""
SemiV and MVN: dates with a single asset and coincidence with GMR.

    python scripts/diagnostics/concentration.py [--run-dir results/main]

Read-only on the weights saved by the run. For SemiV and MVN, by universe and
frequency: number of out-of-sample dates on which one asset holds more than
99% of the portfolio, number of those dates on which it is the asset held by
GMR, and average HHI. For the single-asset dates the script also reports the
sign of the mean of that asset in the estimation window (mean of the returns
in excess of the risk-free rate for MVN, which is estimated on them): Eq. (9)
rewards the squared portfolio mean, so MVN can hold an asset of negative mean.

Writes concentration_semiv_mvn.csv to <run-dir>/diagnostics/.
"""
from _common import C, np, panel, parse_args, pd, selected

CODES = {"SemiV": "SemiV", "MSV": "MVN"}


def main() -> None:
    a = parse_args(__doc__.split("\n")[1])
    rows = []
    for f in C.FREQS:
        for u in C.UNIVERSES:
            folder = a.run_dir / u / f
            rets, rf, sch = panel(u, f)
            W, oos, R = sch["window"], sch["oos_idx"], rets.values
            gmr = pd.read_csv(folder / "weights" / "GMR.csv", index_col=0).values
            for code, name in CODES.items():
                w = pd.read_csv(folder / "weights" / f"{code}.csv", index_col=0).values
                one = w.max(axis=1) > 0.99
                pick = w.argmax(axis=1)
                same = one & (pick == gmr.argmax(axis=1))
                negative = largest = 0
                for k in np.flatnonzero(one):
                    i = oos[k]
                    mu = R[i - W:i].mean(axis=0) - (rf.values[i - W:i].mean() if code == "MSV" else 0.0)
                    negative += bool(mu[pick[k]] < 0)
                    largest += bool(pick[k] == mu.argmax())
                rows.append({"strategy": name, "universe": u, "freq": f, "N": w.shape[1],
                             "lambda_selected": float(selected(a.run_dir, u, f, code)["risk_aversion"]),
                             "oos_dates": len(w), "dates_one_asset_above_99%": int(one.sum()),
                             "share_one_asset_%": 100.0 * one.mean(), "dates_same_asset_as_GMR": int(same.sum()),
                             "share_same_as_GMR_%": 100.0 * same.mean(),
                             "dates_weights_equal_GMR": int((np.abs(w - gmr).max(axis=1) < 1e-6).sum()),
                             "HHI_mean": float((w ** 2).sum(axis=1).mean()),
                             "mean_L1_distance_to_GMR": float(np.abs(w - gmr).sum(axis=1).mean()),
                             "single_asset_mean_negative": negative, "single_asset_is_largest_mean": largest})
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "concentration_semiv_mvn.csv", index=False)
    pd.set_option("display.width", 240)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
