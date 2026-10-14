"""Compact table for the main text: joint tests against EW at both frequencies (Panel A of the inference tables)."""
from _common import PAPER_NAME, UNIVERSE_LABEL, num, pd, parse_args, universes, write_table


def main() -> None:
    a = parse_args(__doc__)
    summ = pd.read_csv(a.run_dir / "inference" / "joint_summary.csv", keep_default_na=False)
    base = summ[(summ.base_block.astype(str) == "True") & (summ.series == "net")]
    cols = universes(base.universe.unique())

    def names(text):
        return ", ".join(PAPER_NAME[s] for s in str(text).split()) if str(text).strip() else "--"

    rows = []
    for u in cols:
        row = [UNIVERSE_LABEL[u]]
        for f in ("MONTHLY", "DAILY"):
            r = base[(base.universe == u) & (base.freq == f)].iloc[0]
            row += [num(float(r.spa_p_better)), names(r["better_rw_5%"]),
                    num(float(r.spa_p_worse)), names(r["worse_rw_5%"])]
        rows.append(row)
    B = int(float(base.B.iloc[0]))
    blocks = {f: float(base[base.freq == f].block.iloc[0]) for f in ("MONTHLY", "DAILY")}
    T = {f: int(float(base[base.freq == f]["T"].iloc[0])) for f in ("MONTHLY", "DAILY")}
    write_table(a.out / "tab_main_inference.tex",
                "Joint inference against the equally weighted portfolio, by universe and frequency.",
                "tab:main_inference",
                ["Universe", r"$p_{\mathrm{better}}$", "Better", r"$p_{\mathrm{worse}}$", "Worse",
                 r"$p_{\mathrm{better}}$", "Better", r"$p_{\mathrm{worse}}$", "Worse"],
                rows, colspec="lrlrlrlrl",
                top=r" & \multicolumn{4}{c}{Monthly} & \multicolumn{4}{c}{Daily} \\"
                    r" \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
                note=r"$p_{\mathrm{better}}$ ($p_{\mathrm{worse}}$): $p$-value of the hypothesis that no "
                     "strategy has a higher (lower) Sharpe ratio than EW, from the maximum of the "
                     "studentised Sharpe differences under a stationary bootstrap of the "
                     f"$T \\times 21$ matrix of net excess returns ($B = {B}$; $T = {T['MONTHLY']}$ "
                     f"months with mean block {blocks['MONTHLY']:g}, $T = {T['DAILY']}$ sessions with mean "
                     f"block {blocks['DAILY']:g}). ``Better'' and ``Worse'' list the strategies whose "
                     "Romano--Wolf adjusted $p$-value is below 0.05. Strategies whose returns coincide "
                     "with EW, or track it within 10 basis points a year, are left out of the family; "
                     "the complete tables are in the appendix.")


if __name__ == "__main__":
    main()
