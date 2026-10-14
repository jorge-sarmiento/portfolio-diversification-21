"""Table: computing cost measured in the run -- CPU hours by strategy group and frequency, share of MVN."""
import math

from _common import num, pd, parse_args, write_table


def main() -> None:
    a = parse_args(__doc__)
    timing = pd.concat([pd.read_csv(p) for p in sorted(a.run_dir.glob("*/*/timing.csv"))], ignore_index=True)
    timing["group"] = timing.strategy.where(timing.strategy == "MSV", "other")
    h = timing.groupby(["freq", "group"]).cpu_s.sum().div(3600).unstack(fill_value=0.0)
    h = h.reindex(["MONTHLY", "DAILY"]).fillna(0.0)
    for col in ("MSV", "other"):
        if col not in h:
            h[col] = 0.0
    tot_msv, tot_other = h["MSV"].sum(), h["other"].sum()
    total = tot_msv + tot_other
    done = {}
    if (a.run_dir / "RUN_DONE").exists():
        done = dict(l.split("=", 1) for l in (a.run_dir / "RUN_DONE").read_text().split() if "=" in l)
    wall_a, wall_b = float(done.get("phaseA_s", "nan")) / 3600, float(done.get("phaseB_s", "nan")) / 3600
    summary = pd.DataFrame([{
        "cpu_hours_total": total, "cpu_hours_MSV": tot_msv, "share_MSV_%": 100 * tot_msv / total,
        "cpu_hours_other_20": tot_other, "wall_hours_other_20": wall_a, "wall_hours_MSV": wall_b,
        "tasks": timing.groupby(["universe", "freq"]).ngroups,
    }])
    (a.run_dir / "tables").mkdir(exist_ok=True)
    summary.to_csv(a.run_dir / "tables" / "compute.csv", index=False)
    timing.groupby(["freq", "strategy"]).cpu_s.sum().div(3600).unstack(0).to_csv(
        a.run_dir / "tables" / "compute_by_strategy.csv")
    rows = [[f.capitalize(), num(h.loc[f, "other"], 2), num(h.loc[f, "MSV"], 2),
             num(h.loc[f, "other"] + h.loc[f, "MSV"], 2)] for f in h.index]
    rows += [r"\midrule", ["Total", num(tot_other, 2), num(tot_msv, 2), num(total, 2)]]
    wall = ""
    if not (math.isnan(wall_a) or math.isnan(wall_b)):
        wall = (f"Wall-clock time: {wall_a:.1f} h for the twenty strategies and {wall_b:.1f} h for MVN, "
                "with one linear-algebra thread per process. ")
    write_table(a.out / "tab_compute.tex",
                "Computing cost of the experiment, in CPU hours.",
                "tab:compute", ["Frequency", "Twenty strategies", "MVN", "Total"], rows,
                note=f"MVN accounts for {100 * tot_msv / total:.0f}\\% of the CPU time. " + wall
                     + "CPU time is measured in each process, including the worker processes that "
                     "solve the MVN rebalancing dates.")


if __name__ == "__main__":
    main()
