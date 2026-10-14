"""
Mean-variance family: Blocks 2 and 4 of the paper.

    MV    (Eq. 6)   mean-variance with absorbed lambda
    SemiV (Eq. 8)   mean-semivariance on the semi-covariance matrix of Eq. (7)
    MVN   (Eq. 9)   mean squared variance-normalised (non-convex), on excess returns
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


def project_simplex(y: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the simplex {w >= 0, sum(w) = 1}.

    Algorithm 1 of Wang and Carreira-Perpiñán (2013):

        sort y into u_1 >= u_2 >= ... >= u_N
        rho = max{ j : u_j + (1 - sum_{i<=j} u_i) / j > 0 }
        lam = (1 - sum_{i<=rho} u_i) / rho
        w_i = max(y_i + lam, 0)
    """
    y = np.asarray(y, dtype=float)
    u = np.sort(y)[::-1]
    shift = (1.0 - np.cumsum(u)) / np.arange(1, len(y) + 1)
    rho = np.flatnonzero(u + shift > 0)[-1]
    return np.maximum(y + shift[rho], 0.0)


class MeanSquaredVariance(MeanVariance):
    """
    Mean Squared Variance-Normalised portfolio (MVN), Eq. (9) of the paper.

        min_w  λ w'Σw - (1 - λ)(w'μ)²     s.t.  Σw_i = 1,  w_i ≥ 0

    μ and Σ are the annualised mean and covariance of the returns in excess
    of the risk-free rate (the engine supplies them to strategies that set
    excess_return_inputs). In matrix form the objective is (1/2) w'Hw with
    H = 2(λΣ - (1 - λ)μμ'), which is generally indefinite for 0 < λ < 1, so
    the problem is non-convex.

    Solver: SciPy differential evolution over the box [0, 1]^N. Each candidate
    y is mapped to the simplex by Euclidean projection, the objective is
    evaluated at P(y), and the weights returned are P(y*).

    References:
        Carbonero-Ruz, M., Fernández-Navarro, F., Durán-Rosal, A. M., &
        Pérez-Rodríguez, J. (2025). A hybrid optimization and data-driven
        approach to understand the role of the risk-aversion profile parameter
        in portfolio optimization problems with shorting constraints.
        Operations Research Perspectives, 15, 100353.
        https://doi.org/10.1016/j.orp.2025.100353
        Fernández-Navarro, F., Martínez-Nieto, L., Carbonero-Ruz, M., &
        Montero-Romero, T. (2021). Mean Squared Variance Portfolio: A
        Mixed-Integer Linear Programming Formulation. Mathematics, 9(3), 223.
        Wang, W., & Carreira-Perpiñán, M. Á. (2013). Projection onto the
        probability simplex: An efficient algorithm with a simple proof, and
        an application. arXiv:1309.1541.
    """

    excess_return_inputs = True

    def __init__(self, risk_aversion: float = 0.5, seed: int = 42):
        super().__init__(risk_aversion)
        self.name = f"Mean Squared Variance (lambda={risk_aversion})"
        self.seed = seed
        if not (0 <= risk_aversion <= 1):
            raise ValueError("Normalised risk aversion must be in [0, 1]")

    def solve(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray):
        """Weights P(y*) and the result object of the differential evolution."""
        N = len(expected_returns)
        lam = self.lamb
        H = 2.0 * (lam * covariance_matrix
                   - (1.0 - lam) * np.outer(expected_returns, expected_returns))

        def objective(y: np.ndarray) -> float:
            w = project_simplex(y)
            return 0.5 * w @ H @ w

        from scipy.optimize import differential_evolution
        result = differential_evolution(
            objective,
            bounds=[(0.0, 1.0)] * N,
            seed=self.seed, maxiter=500, popsize=20, polish=True,
            tol=1e-8,
        )
        return project_simplex(result.x), result

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        return self.solve(expected_returns, covariance_matrix)[0]


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
