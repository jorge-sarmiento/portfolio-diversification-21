"""Table: hyperparameters selected by the calibration, marking those on the boundary of the grid (--freq)."""
from _common import (C, PAPER_NAME, PARAM_LABEL, UNIVERSE_LABEL, np, panel, pd, parse_args, strategies,
                     write_table)

from src.engine import WalkForwardEngine

GRIDS = WalkForwardEngine._PARAM_GRIDS
DAGGER = r"$^{\dagger}$"


def cell(code: str, row: pd.Series) -> str:
    """Selected values of one strategy in one universe; a dagger marks grid boundaries."""
    parts = []
    for p, grid in GRIDS[code].items():
        v = row.get(p, np.nan)
        if pd.isna(v):
            return "--"
        txt = f"{v:g}"
        if np.isclose(v, min(grid)) or np.isclose(v, max(grid)):
            txt += DAGGER
        parts.append(txt)
    return ", ".join(parts)


def bound_values(g: pd.DataFrame, mask: pd.Series) -> str:
    """Bound values responsible for a set of combinations, e.g. 'l=0.05, u=0.1'."""
    out = []
    for col, sym, test in (("lower_bound", "l", lambda n, v: n * v >= 1 - 1e-9),
                           ("upper_bound", "u", lambda n, v: n * v <= 1 + 1e-9)):
        if col in g and mask.any():
            vals = sorted({v for n, v in zip(g.loc[mask, "N"], g.loc[mask, col])
                           if pd.notna(v) and test(n, v)})
            out += [f"${sym}$={v:g}" for v in vals]
    return ", ".join(out) if out else "--"


def effective_grid_rows(run_dir, freq, cols):
    """Panel with the effective grid of the weight-bound strategies in each universe."""
    grids = {}
    for u in cols:
        path = run_dir / u / freq / "grid_feasibility.csv"
        if path.exists() and path.stat().st_size > 2:
            grids[u] = pd.read_csv(path)
    if not grids:
        return [], None
    rows, flat = [], []
    codes = [s for s in strategies(GRIDS) if any(s in set(g.strategy) for g in grids.values())]
    for s in codes:
        lines = {"feasible": [], "excluded": [], "forces EW": []}
        for u in cols:
            g = grids.get(u)
            g = g[g.strategy == s] if g is not None else None
            if g is None or g.empty:
                for k in lines:
                    lines[k].append("--")
                continue
            feas, ew = g.feasible.astype(bool), g.forces_EW.astype(bool)
            lines["feasible"].append(f"{int(feas.sum())}/{len(g)}")
            lines["excluded"].append(bound_values(g, ~feas))
            lines["forces EW"].append(bound_values(g, ew))
            flat.append({"strategy": s, "universe": u, "N": int(g.N.iloc[0]), "combinations": len(g),
                         "feasible": int(feas.sum()), "excluded_infeasible": int((~feas).sum()),
                         "forces_EW": int(ew.sum())})
        for k, cells in lines.items():
            rows.append([PAPER_NAME[s] if k == "feasible" else "", k] + cells)
    return rows, pd.DataFrame(flat)


def main() -> None:
    a = parse_args(__doc__, freq=True)
    best, cols = {}, []
    for u in C.UNIVERSES:
        path = a.run_dir / u / a.freq / "best_params.csv"
        if path.exists():
            best[u] = pd.read_csv(path).set_index("Strategy")
            cols.append(u)
    codes = [s for s in strategies(GRIDS) if any(s in best[u].index for u in cols)]
    rows, flat = [], []
    for s in codes:
        params = ", ".join(f"${PARAM_LABEL[p]}$" for p in GRIDS[s])
        rows.append([PAPER_NAME[s], params] + [cell(s, best[u].loc[s]) if s in best[u].index else "--"
                                               for u in cols])
        for u in cols:
            if s in best[u].index:
                for p, grid in GRIDS[s].items():
                    v = best[u].loc[s].get(p, np.nan)
                    flat.append({"strategy": s, "universe": u, "parameter": p, "selected": v,
                                 "grid_min": min(grid), "grid_max": max(grid),
                                 "on_boundary": bool(np.isclose(v, min(grid)) or np.isclose(v, max(grid)))})
    flat = pd.DataFrame(flat)
    (a.run_dir / "tables").mkdir(exist_ok=True)
    flat.to_csv(a.run_dir / "tables" / f"hyperparameters_{a.freq}.csv", index=False)
    share = 100.0 * flat.on_boundary.mean() if len(flat) else float("nan")
    grid_rows, grid_flat = effective_grid_rows(a.run_dir, a.freq, cols)
    grid_note = ""
    if grid_rows:
        rows = ([panel("Panel A: selected hyperparameters", len(cols) + 2)] + rows + [r"\midrule",
                panel("Panel B: effective grid of the weight-bound strategies", len(cols) + 2)] + grid_rows)
        grid_flat.to_csv(a.run_dir / "tables" / f"effective_grid_{a.freq}.csv", index=False)
        grid_note = (" Panel B: number of feasible combinations out of the grid; bound values removed "
                     "before the calibration because they are infeasible for the number of assets "
                     "($N l > 1$ or $N u < 1$), which are not counted as DSR trials; and feasible "
                     "bound values that force the equally weighted portfolio ($N u = 1$).")
    f = a.freq.lower()
    write_table(a.out / f"tab_hyperparams_{f}.tex",
                f"Hyperparameters selected by the calibration ({a.freq.lower()} frequency). Values are "
                "selected once per strategy and universe on 2013--2018 and held fixed out of sample.",
                f"tab:hyperparams_{f}", ["Strategy", "Parameter(s)"] + [UNIVERSE_LABEL[u] for u in cols],
                rows, colspec="ll" + "r" * len(cols),
                note=DAGGER + " selected value on the boundary of its calibration grid "
                     f"({share:.0f}\\% of the {len(flat)} selected values). "
                     "MVN uses the static calibration." + grid_note)


if __name__ == "__main__":
    main()
