"""Table: characteristics of the asset universes (N, observations, calibration, OOS, skewness, kurtosis)."""
from _common import C, UNIVERSE_LABEL, num, parse_args, write_table

from scipy import stats

from src.data import load_panel, schedule_indices


def main() -> None:
    a = parse_args(__doc__)
    rows = []
    for u in C.UNIVERSES:
        Rm, _ = load_panel(u, "MONTHLY")
        Rd, _ = load_panel(u, "DAILY")
        sm, sd = schedule_indices(Rm.index, "MONTHLY"), schedule_indices(Rd.index, "DAILY")
        rows.append([UNIVERSE_LABEL[u], str(Rm.shape[1]),
                     f"{Rm.index[0]:%Y-%m}", f"{Rm.index[-1]:%Y-%m}",
                     str(len(Rm)), str(len(sm["calib_idx"])), str(len(sm["oos_idx"])),
                     str(int((Rd.index >= "2008-01-01").sum())), str(len(sd["calib_idx"])),
                     str(len(sd["oos_idx"])),
                     num(float(stats.skew(Rm.values, axis=0).mean())),
                     num(float(stats.kurtosis(Rm.values, axis=0).mean()))])
    write_table(
        a.out / "tab_datasets.tex",
        "Characteristics of the asset universes. Monthly and daily simple returns over a common "
        "calendar; Skew and Kurt are cross-sectional averages of asset-level skewness and excess "
        "kurtosis of monthly returns.",
        "tab:datasets",
        ["Universe", "$N$", "Start", "End", "Obs", "Calib.", "OOS", "Obs", "Calib.", "OOS", "Skew", "Kurt"],
        rows, colspec="lrllrrrrrrrr",
        top=r" & & & & \multicolumn{3}{c}{Monthly} & \multicolumn{3}{c}{Daily} & & \\"
            r" \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        note=f"Estimation window: {C.ESTIMATION_WINDOW['MONTHLY']} months "
             f"({C.ESTIMATION_WINDOW['DAILY']} sessions at daily frequency). "
             "Calibration: 2013-01 to 2018-12. Out-of-sample: 2019-01 to 2025-12. "
             "Observations are counted from 2008-01. The daily panels also hold the 19 sessions of "
             "December 2007 (from 2007-12-04): the last one completes the 1260-session estimation "
             "window of the first calibration date (2013-01-02), and the nineteen belong to the "
             "static calibration segment of MVN (all sessions before 2019). They enter no "
             "out-of-sample statistic.")


if __name__ == "__main__":
    main()
