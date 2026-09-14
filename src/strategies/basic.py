"""
Benchmark and single-moment strategies (Blocks 1, 2 and 5 of the paper).

    EW     (Eq. 3)   equally weighted, 1/N
    GMV    (Eq. 5)   global minimum variance
    GMR    (Eq. 4)   full allocation to the highest expected return
    MaxDiv (Eq. 15)  maximum diversification ratio
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from qpsolvers import solve_qp
from src.strategies.base import BaseStrategy

class EquallyWeighted(BaseStrategy):
    """
    Equally weighted portfolio (EW), Eq. (3) of the paper.

    Allocates the same fraction of capital to every asset, w_i = 1/N. It carries
    no estimation error in mu or Sigma, which makes it a demanding benchmark.

    References:
        DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive
        Diversification: How Inefficient is the 1/N Portfolio Strategy?
        Review of Financial Studies.
    """
    def __init__(self):
        super().__init__(name="Equally Weighted")

    def optimize_weights(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)
        return np.ones(N) / N


class GlobalMinimumVariance(BaseStrategy):
    """
    Global Minimum Variance portfolio (GMV), Eq. (5) of the paper.

    The left-most point of the efficient frontier, and the only efficient
    allocation that does not depend on expected returns, whose estimation error
    is far larger than that of the covariance matrix.

        min_w  w' Sigma w     s.t.  sum(w) = 1,  0 <= w_i <= 1

    References:
        Jagannathan, R., & Ma, T. (2003). Risk Reduction in Large Portfolios:
        Why Imposing the Wrong Constraints Helps. Journal of Finance.
    """
    def __init__(self):
        super().__init__(name="Global Minimum Variance")

    def optimize_weights(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        N = covariance_matrix.shape[0]

        # qpsolvers form min (1/2) w'Pw + q'w, with P = 2 Sigma and q = 0
        P = 2.0 * covariance_matrix
        q = np.zeros(N)
        A = np.ones((1, N))
        b = np.array([1.0])
        lb = np.zeros(N)
        ub = np.ones(N)

        w = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='clarabel')
        return w if w is not None else np.ones(N) / N


class GlobalMaximumReturn(BaseStrategy):
    """
    Global Maximum Return portfolio (GMR), Eq. (4) of the paper.

    Allocates the entire budget to the asset with the highest expected return.
    It is a maximally concentrated reference point, not a recommended strategy.
    """
    def __init__(self):
        super().__init__(name="Global Maximum Return")

    def optimize_weights(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)
        weights = np.zeros(N)
        idx_max = np.argmax(expected_returns)
        weights[idx_max] = 1.0
        return weights


class MaxDiversification(BaseStrategy):
    """
    Maximum Diversification portfolio (MaxDiv), Eq. (15) of the paper.

    Maximises the diversification ratio, the weighted average asset volatility
    divided by portfolio volatility, which rewards uncorrelated assets:

        max_w  (w' sigma) / sqrt(w' Sigma w)

    where sigma is the vector of individual volatilities.

    References:
        Choueifaty, Y., & Coignard, Y. (2008). Toward Maximum Diversification.
        Journal of Portfolio Management.
    """
    def __init__(self):
        super().__init__(name="Max Diversification")

    def optimize_weights(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        N = covariance_matrix.shape[0]
        volatilities = np.sqrt(np.diag(covariance_matrix))

        # Negative diversification ratio, -DR(w)
        def objective(w):
            sigma_p = np.sqrt(w @ covariance_matrix @ w)
            return -(w @ volatilities) / (sigma_p + 1e-10)

        # Gradient of -DR: -sigma / sigma_p + (w'sigma) Sigma w / sigma_p^3
        def gradient(w):
            sigma_p2 = w @ covariance_matrix @ w          # portfolio variance
            sigma_p = np.sqrt(sigma_p2) + 1e-10
            wvol = w @ volatilities
            return -volatilities / sigma_p + wvol * (covariance_matrix @ w) / (sigma_p * sigma_p2)

        # trust-constr takes LinearConstraint objects, not the SLSQP dict format
        eq_constraint = LinearConstraint(np.ones((1, N)), lb=1.0, ub=1.0)
        bounds_obj = Bounds(lb=0.0, ub=1.0)

        result = minimize(objective, np.ones(N) / N, method='trust-constr',
                          jac=gradient, bounds=bounds_obj,
                          constraints=eq_constraint,
                          options={'gtol': 1e-9, 'maxiter': 1000, 'verbose': 0})
        return result.x if result.success else np.ones(N) / N
