"""Table: fallbacks by strategy and universe -- optimiser failures and OOS dates with 1/N weights."""
from _common import C, PAPER_NAME, UNIVERSE_LABEL, pd, panel, parse_args, strategies, write_table


def main() -> None:
    a = parse_args(__doc__)
    parts, oos, present = [], {}, {}
    for u in C.UNIVERSES:
        for f in C.FREQS:
            folder = a.run_dir / u / f
            if (folder / "diagnostics.csv").exists():
                d = pd.read_csv(folder / "diagnostics.csv")
                parts.append(d)
                oos[(u, f)] = len(pd.read_csv(folder / "oos_returns_gross.csv", usecols=[0]))
                present[(u, f)] = set(pd.read_csv(folder / "oos_returns_gross.csv", nrows=0).columns[1:])
    diag = pd.concat(parts, ignore_index=True)
    diag = diag[diag.kind != "python_warning"]
    (a.run_dir / "tables").mkdir(exist_ok=True)
    counts = (diag.groupby(["kind", "strategy", "universe", "freq"])["count"].sum()
              .reset_index().sort_values(["kind", "strategy", "universe", "freq"]))
    counts.to_csv(a.run_dir / "tables" / "fallbacks.csv", index=False)
    cols = [u for u in C.UNIVERSES if (u, "MONTHLY") in oos or (u, "DAILY") in oos]

    def block(kind, title, as_share):
        sub = diag[diag.kind == kind]
        rows = [panel(title, len(cols) + 1)]
        for s in strategies(sub.strategy.unique()):
            cells = []
            for u in cols:
                vals = []
                for f in C.FREQS:
                    if s not in present.get((u, f), set()):
                        vals.append("n.r.")                    # strategy not run in this task
                        continue
                    c = sub[(sub.strategy == s) & (sub.universe == u) & (sub.freq == f)]["count"].sum()
                    if as_share:
                        vals.append(f"{int(c)}/{oos.get((u, f), 0)}" if c else "0")
                    else:
                        vals.append(str(int(c)))
                cells.append("--" if all(v == "0" for v in vals) else " | ".join(vals))
            if all(c == "--" for c in cells):
                continue
            rows.append([PAPER_NAME[s]] + cells)
        return rows

    audit_path = a.run_dir / "tables" / "fallback_audit.csv"
    if audit_path.exists():
        # counts recomputed from the saved weights (tolerance 1e-10), including RP
        audit = pd.read_csv(audit_path)
        audit = audit[audit.strategy != "EW"]
        aud = audit.rename(columns={"dates_equal_1/N": "count"}).assign(kind="oos_weights_equal_1/N")
        diag = pd.concat([diag[diag.kind != "oos_weights_equal_1/N"],
                          aud[aud["count"] > 0][["universe", "freq", "strategy", "kind", "count"]]],
                         ignore_index=True)
    rows = block("oos_weights_equal_1/N",
                 "Panel A: out-of-sample dates with weights equal to $1/N$ (monthly | daily)", True)
    if audit_path.exists():
        rp = audit[audit.strategy == "RP"].set_index(["universe", "freq"])["dates_equal_inverse_vol"]
        rows += [r"\midrule",
                 panel("Panel B: RP dates equal to the inverse-volatility portfolio (monthly | daily)",
                       len(cols) + 1),
                 ["RP"] + [" | ".join(f"{int(rp.get((u, f), 0))}/{oos.get((u, f), 0)}" for f in C.FREQS)
                           for u in cols]]
    panel_c = block("optimizer_failure",
                    "Panel C: distinct optimiser failures resolved by the fallback (monthly | daily)", False)
    none_failed = len(panel_c) == 1
    if not none_failed:
        rows += [r"\midrule"] + panel_c
    write_table(a.out / "tab_fallbacks.tex",
                "Fallbacks by strategy and universe.",
                "tab:fallbacks", ["Strategy"] + [UNIVERSE_LABEL[u] for u in cols], rows,
                colspec="l" + "c" * len(cols),
                note="Panel A counts the out-of-sample dates on which the weights of a strategy equal "
                     "the equally weighted portfolio (tolerance $10^{-10}$), out of the total number "
                     "of dates; it is an upper bound on fallbacks, since an optimum can also be "
                     "$1/N$. Panel B compares RP, whose internal fallback is the inverse-volatility "
                     "portfolio, with that portfolio recomputed on each date. "
                     + ("No optimiser failure was reported in the calibration or the evaluation: "
                        "the grid combinations that are infeasible for the number of assets "
                        "($N l > 1$ or $N u < 1$) are removed before the calibration. " if none_failed else
                        "Panel C counts the distinct failures reported by the optimiser during "
                        "calibration and evaluation, each resolved by reusing the previous weights "
                        "(or $1/N$). ")
                     + "Strategies without entries are omitted; ``n.r.'' marks a strategy that had "
                     "not been run in that universe and frequency.")


if __name__ == "__main__":
    main()
