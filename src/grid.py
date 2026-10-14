"""
Feasibility of the weight-bound combinations of the calibration grids.

The grids of WLBC, WUBC and WCMV (code WUBC_Alt) hold lower bounds l and upper
bounds u that are fixed numbers, while the number of assets N changes with the
universe. With the budget constraint sum(w) = 1:

    N * l > 1   no portfolio satisfies w_i >= l          -> infeasible
    N * u < 1   no portfolio satisfies w_i <= u          -> infeasible
    N * u = 1   (or N * l = 1) the only portfolio is 1/N -> feasible, forces EW

Infeasible combinations are removed from the grid before the calibration and
are not counted as trials of the Deflated Sharpe Ratio. Combinations that
force EW are feasible and stay in the grid; they are flagged so the tables
can report them.
"""
from typing import Dict, Tuple

TOL = 1e-9


def check(combo: Dict[str, float], n_assets: int) -> Tuple[bool, bool, str]:
    """(feasible, forces_ew, reason) of one hyperparameter combination."""
    lower, upper = combo.get("lower_bound"), combo.get("upper_bound")
    if lower is not None and n_assets * lower > 1 + TOL:
        return False, False, f"N*l = {n_assets * lower:.2f} > 1"
    if upper is not None and n_assets * upper < 1 - TOL:
        return False, False, f"N*u = {n_assets * upper:.2f} < 1"
    if upper is not None and abs(n_assets * upper - 1) <= TOL:
        return True, True, "N*u = 1"
    if lower is not None and abs(n_assets * lower - 1) <= TOL:
        return True, True, "N*l = 1"
    return True, False, ""


def is_feasible(combo: Dict[str, float], n_assets: int) -> bool:
    return check(combo, n_assets)[0]
