"""
Soft Return-Floor LP (SRF_LP).

Formulation (b=0, eta=1, frozen ex ante):

    max_{w, xi}  mu'w - (1/T) * sum_t xi_t

    s.t.  r_t'w + xi_t >= 0   for t = 1,...,T
          xi_t >= 0
          1'w  = 1
          w    >= 0

Setting xi_t = max(0, -r_t'w) absorbs any shortfall below zero. The feasible
region is non-empty for every return matrix R (e.g. w = 1/N, xi_t = max(0,
-r_t'(1/N))), so solver failure is always a numerical issue and raises
explicitly instead of falling back silently.

Units: mu_raw and R are per-period values taken directly from the estimation
window, as in the MAD and CVaR implementations, so the objective and the
constraints share the same units (see Section 6.3 of the article).

Mapping to scipy.optimize.linprog (minimisation):
    Variables: x = [w (N), xi (T)]
    c     = [-mu_raw;       (1/T) * ones(T)]
    A_ub  = [-R, -I_T]                     (T rows, N+T cols)
    b_ub  = zeros(T)                        (r_t'w + xi_t >= 0)
    A_eq  = [ones(N)', zeros(T)']           (1'w = 1)
    b_eq  = [1]
    bounds = [(0,1)]*N + [(0,None)]*T

Per-window diagnostics (list of dicts, one entry per optimize_weights() call):
    xi_active   : number of t with xi_t > 1e-6
    cardinality : number of assets with w_i > 1e-6
    hhi         : Herfindahl-Hirschman Index (sum_i w_i^2)

References:
    Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
    Value-at-Risk. Journal of Risk, 2(3), 21-41.

    Roy, A. D. (1952). Safety first and the holding of assets.
    Econometrica, 20(3), 431-449.

    Fishburn, P. C. (1977). Mean-risk analysis with risk associated with
    below-target returns. American Economic Review, 67(2), 116-126.
"""

import numpy as np
from scipy.optimize import linprog as _linprog
from src.strategies.base import BaseStrategy

_TOL = 1e-6


class SoftReturnFloor(BaseStrategy):
    """
    Soft Return-Floor LP portfolio: b=0, eta=1 (class-level constants; not tuned).

    Requires raw return data via set_raw_data() before each optimize_weights() call.
    Raises RuntimeError on LP solver failure: the problem is always feasible, so
    a solver failure indicates a numerical issue that must not be silently absorbed.
    """

    B: float   = 0.0   # floor level (ex ante frozen)
    ETA: float = 1.0   # shortfall penalty weight (ex ante frozen)

    def __init__(self):
        super().__init__(name="Soft Return-Floor LP (b=0, eta=1)")
        self._raw_data: np.ndarray | None = None
        self._window_diagnostics: list = []

    def set_raw_data(self, data: np.ndarray) -> None:
        self._raw_data = data

    def get_window_diagnostics(self) -> list:
        """Return per-window diagnostics accumulated since construction."""
        return list(self._window_diagnostics)

    def optimize_weights(self,
                         expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Solve the Soft Return-Floor LP and return portfolio weights.

        expected_returns and covariance_matrix are accepted to satisfy the
        BaseStrategy interface but are NOT used in the LP objective; per-period
        means from _raw_data are used instead to keep units consistent with r_t.
        """
        if self._raw_data is None or self._raw_data.shape[0] < 2:
            raise RuntimeError(
                "SRF_LP.optimize_weights: set_raw_data() must be called with "
                "at least 2 periods before optimisation."
            )

        R   = self._raw_data         # (T, N)  per-period returns
        T, N = R.shape
        mu_raw = R.mean(axis=0)      # (N,)    per-period mean

        # Objective: min  -mu'w + (1/T)*sum(xi)
        c = np.concatenate([-mu_raw, np.full(T, 1.0 / T)])

        # T inequality constraints: r_t'w + xi_t >= 0  ->  -r_t'w - xi_t <= 0
        A_ub = np.zeros((T, N + T))
        A_ub[:, :N] = -R
        np.fill_diagonal(A_ub[:, N:], -1.0)
        b_ub = np.zeros(T)

        # Equality: 1'w = 1
        A_eq = np.zeros((1, N + T))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w in [0,1]; xi in [0, inf)
        bounds = [(0.0, 1.0)] * N + [(0.0, None)] * T

        result = _linprog(c, A_ub=A_ub, b_ub=b_ub,
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(
                f"SRF_LP: HiGHS failed (status={result.status}, "
                f"message='{result.message}'). T={T}, N={N}, "
                f"R.min={R.min():.6f}, R.max={R.max():.6f}."
            )

        w   = result.x[:N]
        xi  = result.x[N:]

        self._window_diagnostics.append({
            "xi_active":   int(np.sum(xi > _TOL)),
            "cardinality": int(np.sum(w  > _TOL)),
            "hhi":         float(np.sum(w ** 2)),
        })

        return w
