"""
Mean-variance family: Blocks 2 and 4 of the paper.

    MV    (Eq. 6)   mean-variance with absorbed lambda
    SemiV (Eq. 8)   mean-semivariance on the semi-covariance matrix of Eq. (7)
    MVN   (Eq. 9)   mean squared variance-normalised (non-convex)
    WLBC  (Eq. 12)  MV with a lower bound on every weight
    WUBC  (Eq. 13)  MV with an upper bound on every weight
    WCMV  (Eq. 14)  MV with both bounds

References:
    Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
    Estrada, J. (2008). Mean-Semivariance Optimization. J. Applied Finance, 18(1), 57-72.
    Jagannathan, R., & Ma, T. (2003). Risk Reduction in Large Portfolios. J. Finance, 58(4).
"""
import numpy as np
from qpsolvers import solve_qp
from src.strategies.base import BaseStrategy


class MeanVariance(BaseStrategy):
    """
    Mean-Variance portfolio (MV), Eq. (6) of the paper.

        max_w  μ'w - λ w'Σw     s.t.  Σw_i = 1,  0 ≤ w_i ≤ 1

    λ is the effective risk weight: the solver receives P = 2λΣ, so the
    quadratic term it evaluates is (1/2) w'(2λΣ)w = λ w'Σw.

    References:
        Markowitz, H. (1952). Portfolio Selection. J. Finance, 7(1), 77-91.
    """

    def __init__(self, risk_aversion: float = 1.0):
        super().__init__(name=f"Mean Variance (λ={risk_aversion})")
        self.lamb = risk_aversion
        if risk_aversion < 0:
            raise ValueError("Risk aversion must be non-negative")

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)

        # qpsolvers form min (1/2) w'Pw + q'w, with P = 2 lambda Sigma and q = -mu
        P = 2.0 * self.lamb * covariance_matrix
        q = -expected_returns
        A = np.ones((1, N))
        b = np.array([1.0])
        lb = np.zeros(N)
        ub = np.ones(N)

        w = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='clarabel')
        if w is None:
            raise ValueError("MV optimization failed: CLARABEL returned None")
        return w


class MeanSquaredVariance(MeanVariance):
    """
    Mean Squared Variance-Normalised portfolio (MVN), Eq. (9) of the paper,
    after Fernández-Navarro et al. (2021).

    Objective:
        min_w  lambda*w'Sigma*w - (1-lambda)*(w'mu)^2
        s.t.   sum(w) = 1,  w >= 0

    Matrix form:
        min_w  (1/2)*w'Hw,   H = 2*(lambda*Sigma - (1-lambda)*mu*mu')

    H is generally indefinite for 0 < lambda < 1, so the problem is non-convex.
    Fernández-Navarro et al. (2021) give a mixed-integer linear programming
    formulation; here the problem is solved with differential evolution, as
    described in Section 3.2 of the article.

    Solver (MVN*, see README): SciPy differential evolution over the box
    [0, 1]^N; the budget constraint is imposed by normalising the solution
    after the search.

    References:
        Fernández-Navarro, F., Martínez-Nieto, L., Carbonero-Ruz, M., &
        Montero-Romero, T. (2021). Mean Squared Variance Portfolio: A
        Mixed-Integer Linear Programming Formulation.
        Mathematics, 9(3), 223. https://doi.org/10.3390/math9030223
    """

    def __init__(self, risk_aversion: float = 0.5, seed: int = 42):
        super().__init__(risk_aversion)
        self.name = f"Mean Squared Variance (lambda={risk_aversion})"
        self.seed = seed
        if not (0 <= risk_aversion <= 1):
            raise ValueError("Normalized risk aversion must be in [0, 1]")

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)
        lam = self.lamb

        # H is generally indefinite for 0 < lambda < 1: the problem is non-convex
        H = 2.0 * (lam * covariance_matrix
                   - (1.0 - lam) * np.outer(expected_returns, expected_returns))

        # Differential evolution over the box [0, 1]^N; the budget constraint
        # is imposed by normalising the returned vector (see MVN* in README).
        from scipy.optimize import differential_evolution
        result = differential_evolution(
            lambda w: 0.5 * w @ H @ w,
            bounds=[(0.0, 1.0)] * N,
            seed=self.seed, maxiter=500, popsize=20, polish=True,
            tol=1e-8,
        )
        w = np.maximum(result.x, 0.0)
        total = np.sum(w)
        return w / total if total > 1e-10 else np.ones(N) / N


class WeightConstrainedMV(MeanVariance):
    """
    Weight-Constrained Mean-Variance (WCMV), Eq. (14) of the paper.

        max_w  μ'w - λ w'Σw     s.t.  Σw_i = 1,  l ≤ w_i ≤ u

    References:
        Jagannathan, R., & Ma, T. (2003). Risk Reduction in Large Portfolios:
        Why Imposing the Wrong Constraints Helps. J. Finance, 58(4), 1651-1683.
    """

    def __init__(self, risk_aversion: float = 1.0,
                 lower_bound: float = 0.0, upper_bound: float = 1.0):
        super().__init__(risk_aversion)
        self.name = f"Constrained MV (λ={risk_aversion}, [{lower_bound:.2f}, {upper_bound:.2f}])"
        self.lb = lower_bound
        self.ub = upper_bound
        if not (0 <= lower_bound <= upper_bound <= 1):
            raise ValueError("Invalid bounds: must satisfy 0 ≤ lb ≤ ub ≤ 1")

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)

        # Feasibility checks
        if N * self.lb > 1 + 1e-6:
            raise ValueError(f"Infeasible: {N} assets with lb={self.lb}")
        if N * self.ub < 1 - 1e-6:
            raise ValueError(f"Infeasible: {N} assets with ub={self.ub}")

        P = 2.0 * self.lamb * covariance_matrix
        q = -expected_returns
        A = np.ones((1, N))
        b = np.array([1.0])
        lb = np.full(N, self.lb)
        ub = np.full(N, self.ub)

        w = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='clarabel')
        if w is None:
            raise ValueError("Constrained MV failed: CLARABEL returned None")
        return w


class WeightLowerBoundConstrainedMV(MeanVariance):
    """
    Weight Lower-Bound Constraint (WLBC), Eq. (12) of the paper.

        max_w  μ'w - λ w'Σw     s.t.  Σw_i = 1,  l ≤ w_i ≤ 1

    The problem is feasible only if N * l ≤ 1.

    References:
        Jagannathan, R., & Ma, T. (2003). J. Finance, 58(4), 1651-1683.
    """

    def __init__(self, risk_aversion: float = 1.0, lower_bound: float = 0.01):
        super().__init__(risk_aversion)
        self.name = f"WLBC (λ={risk_aversion}, lb={lower_bound})"
        self.lower_bound = lower_bound
        if not (0 <= lower_bound <= 1):
            raise ValueError("Lower bound must be in [0, 1]")

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)
        if N * self.lower_bound > 1 + 1e-6:
            raise ValueError(f"Infeasible: {N} assets with lb={self.lower_bound}")

        P = 2.0 * self.lamb * covariance_matrix
        q = -expected_returns
        A = np.ones((1, N))
        b = np.array([1.0])
        lb = np.full(N, self.lower_bound)
        ub = np.ones(N)

        w = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='clarabel')
        return w if w is not None else np.ones(N) / N


class MeanSemivariance(BaseStrategy):
    """
    Mean-Semivariance portfolio (SemiV), Eqs. (7) and (8) of the paper.

    Semi-covariance matrix, with threshold τ (τ = 0 in the article):
        Σ⁻ᵢⱼ = (1/T) Σₜ min(rᵢₜ - τ, 0) · min(rⱼₜ - τ, 0)

    Optimisation:
        max_w   (1-λ) · μ'w - λ · w'Σ⁻w
        s.t.    Σwᵢ = 1,  wᵢ ≥ 0

    Σ⁻ = D'D / T with D = min(R - τ, 0), so it is positive semi-definite and
    the problem is convex. If no return falls below τ, Σ⁻ = 0 and only the
    return term remains.

    Requires raw return data via set_raw_data() for Σ⁻ construction.

    References:
        Estrada, J. (2008). Mean-Semivariance Optimization: A Heuristic Approach.
        Journal of Applied Finance, 18(1), 57-72.

        Markowitz, H. (1959). Portfolio Selection: Efficient Diversification
        of Investments. John Wiley & Sons. Chapter 9.
    """

    def __init__(self, risk_aversion: float = 0.5, threshold: float = 0.0):
        super().__init__(name=f"Mean-Semivariance (λ={risk_aversion}, τ={threshold})")
        self.lambda_param = risk_aversion
        self.threshold = threshold
        self._raw_data = None

        if not (0 <= risk_aversion <= 1):
            raise ValueError("Lambda must be in [0, 1]")

    def set_raw_data(self, data: np.ndarray):
        """Store raw returns for semi-covariance computation."""
        self._raw_data = data

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)

        # Semi-covariance matrix Σ⁻, Eq. (7)
        if self._raw_data is not None:
            data = self._raw_data
            T = data.shape[0]
            # Filter: only downside deviations below threshold
            downside = np.minimum(data - self.threshold, 0.0)  # (T, N)
            semi_cov = (downside.T @ downside) / T              # (N, N) - always PSD
        else:
            # Fallback: use full covariance (approximation)
            semi_cov = covariance_matrix

        lam = self.lambda_param

        # min lam*(w'*Sigma_minus*w) - (1-lam)*(w'*mu)
        # Standard QP: (1/2)*w'*(2*lam*Sigma_minus)*w + (-(1-lam)*mu)'*w
        # Sigma_minus is always PSD (product D'D), so the QP is convex.
        P = 2.0 * lam * semi_cov
        q = -(1.0 - lam) * expected_returns
        A = np.ones((1, N))
        b = np.array([1.0])
        lb = np.zeros(N)
        ub = np.ones(N)

        w = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='clarabel')
        return w if w is not None else np.ones(N) / N


class WeightUpperBoundConstraint(BaseStrategy):
    """
    Weight Upper-Bound Constraint (WUBC), Eq. (13) of the paper.

        max_w  μ'w - λ w'Σw     s.t.  Σw_i = 1,  0 ≤ w_i ≤ u

    The problem is feasible only if N * u ≥ 1.

    References:
        Jagannathan, R., & Ma, T. (2003). J. Finance, 58(4), 1651-1683.
    """

    def __init__(self, risk_aversion: float = 1.0, upper_bound: float = 0.20):
        super().__init__(name=f"MV Upper-Bound (λ={risk_aversion}, ub={upper_bound})")
        self.risk_aversion = risk_aversion
        self.upper_bound = upper_bound
        if not (0 < upper_bound <= 1.0):
            raise ValueError("Upper bound must be in (0, 1]")

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)
        if self.upper_bound < 1.0/N:
            raise ValueError(f"Infeasible: {self.upper_bound} < 1/{N}")

        P = 2.0 * self.risk_aversion * covariance_matrix
        q = -expected_returns
        A = np.ones((1, N))
        b = np.array([1.0])
        lb = np.zeros(N)
        ub = np.full(N, self.upper_bound)

        w = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='clarabel')
        return w if w is not None else np.ones(N) / N
