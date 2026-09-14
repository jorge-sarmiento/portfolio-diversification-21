"""
Diversification-penalised mean-variance strategies (Block 5 of the paper).

These strategies extend classical mean-variance with penalties on the distance
to the equally weighted allocation, or by mixing the optimised weights with it:

    DMV   (Eq. 16)  isotropic Tikhonov penalty, delta * w'w
    DMVY  (Eq. 17)  L1 (Yager entropy, z = 1) penalty on w - 1/N
    DMVR  (Eq. 18)  return-weighted quadratic penalty, delta * diag(mu^-2)
    DMVV  (Eq. 19)  variance-weighted quadratic penalty, delta * diag(sigma^-4)
    EWMV  (Eq. 20)  ex-post mixture of the MV solution with 1/N

References:
    Schmidt, A. B. (2019). Managing portfolio diversity within the mean variance
    theory. Annals of Operations Research, 282(1-2), 315-329.
    DeMiguel, V., Garlappi, L., Nogales, F. J., & Uppal, R. (2009). A generalized
    approach to portfolio optimization. Management Science, 55(5), 798-812.
    Yu, J.-R., Lee, W.-Y., & Chiou, W.-J. P. (2014). Diversified portfolios with
    different entropy measures. Applied Mathematics and Computation, 241, 47-63.
"""
import numpy as np
from qpsolvers import solve_qp
from src.strategies.base import BaseStrategy


class DiversifiedMeanVariance(BaseStrategy):
    """
    Diversified Mean-Variance (DMV), Eq. (16) of the paper.

    Mathematical Formulation:
        min_w  (1/2) * w^T * H * w + f^T * w

        where:
            H = 2*(λ*Σ + δ*I)
            f = -μ

        Subject to:
            1^T w = 1
            w_i ≥ 0

    The identity term is a Tikhonov penalty: it regularises an ill-conditioned
    covariance matrix and, as delta grows, pulls the solution toward equal weights.

    Hyperparameters:
        - risk_aversion (λ): Variance penalty [0, ∞)
        - diversification_factor (δ): Identity regularization [0, ∞)
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 diversification_factor: float = 1.0):
        """
        Initialize DMV strategy.

        Args:
            risk_aversion: λ - variance penalty coefficient
            diversification_factor: δ - identity matrix penalty
        """
        super().__init__(
            name=f"Diversified MV (λ={risk_aversion}, δ={diversification_factor})"
        )
        self.risk_aversion = risk_aversion
        self.diversification_factor = diversification_factor

    def optimize_weights(self,
                         expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Solve the DMV quadratic program.

        Quadratic Program:
            min  (1/2)*w^T*H*w + f^T*w
            s.t. 1^T*w = 1, w ≥ 0
        """
        N = len(expected_returns)

        H = 2 * (self.risk_aversion * covariance_matrix +
                 self.diversification_factor * np.eye(N))

        f = -expected_returns

        # qpsolvers: min (1/2)*w'*P*w + q'*w  ->  P=H, q=f
        A    = np.ones((1, N))
        b_eq = np.array([1.0])
        lb   = np.zeros(N)
        ub   = np.ones(N)

        w = solve_qp(H, f, A=A, b=b_eq, lb=lb, ub=ub, solver='clarabel')
        return w if w is not None else np.ones(N) / N


class DMVYager(BaseStrategy):
    """
    DMV with Yager entropy penalty, z = 1 (DMVY), Eq. (17) of the paper.

    Formulation:
        min_{w,d}  λ·w'Σw − μ'w + (δ/2)·Σdᵢ

        s.t.  dᵢ ≥  wᵢ − 1/N   ∀i     (->  wᵢ − dᵢ ≤  1/N)
              dᵢ ≥  1/N − wᵢ   ∀i     (-> −wᵢ − dᵢ ≤ −1/N)
              Σwᵢ = 1
              0 ≤ wᵢ ≤ 1,  dᵢ ≥ 0

    Variables: x = [w(N), d(N)], with dᵢ = |wᵢ − 1/N| at the optimum.

    With z = 2 the penalty Σ(wᵢ − 1/N)² equals ‖w‖² − 1/N on the simplex and
    therefore coincides with the Tikhonov penalty of DMV up to a constant
    (Appendix H of the article); the L1 penalty is not of that form.

    Solver: CLARABEL via qpsolvers.

    References:
        Yager, R. R. (1995). Measures of entropy and fuzziness related to
        aggregation operators. Information Sciences, 82(3-4), 147-166.
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 diversification_factor: float = 1.0):
        super().__init__(
            name=f"DMV-Yager (λ={risk_aversion}, δ={diversification_factor})"
        )
        self.risk_aversion = risk_aversion
        self.diversification_factor = diversification_factor

    def optimize_weights(self,
                         expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)
        lam = self.risk_aversion
        delta = self.diversification_factor
        inv_n = 1.0 / N

        # --- QP matrices over x = [w(N), d(N)] ---
        # Objective: (1/2)*x'*P*x + q'*x
        #   P[0:N,0:N] = 2λΣ  (so (1/2)*2λ*w'Σw = λw'Σw)
        P = np.zeros((2*N, 2*N))
        P[:N, :N] = 2.0 * lam * covariance_matrix
        q = np.concatenate([-expected_returns, np.full(N, delta / 2.0)])

        # Inequality G*x <= h  (2N rows)
        #   row i:    wᵢ − dᵢ ≤  1/N
        #   row N+i: −wᵢ − dᵢ ≤ −1/N
        G = np.zeros((2*N, 2*N))
        G[:N,  :N]  =  np.eye(N)   # wᵢ
        G[:N,  N:]  = -np.eye(N)   # −dᵢ
        G[N:,  :N]  = -np.eye(N)   # −wᵢ
        G[N:,  N:]  = -np.eye(N)   # −dᵢ
        h = np.concatenate([np.full(N, inv_n), np.full(N, -inv_n)])

        # Equality: Σwᵢ = 1
        A_eq = np.zeros((1, 2*N))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w ∈ [0,1], d ∈ [0, ∞)
        lb = np.zeros(2*N)
        ub = np.concatenate([np.ones(N), np.full(N, np.inf)])

        x = solve_qp(P, q, G=G, h=h, A=A_eq, b=b_eq, lb=lb, ub=ub,
                     solver='clarabel')
        if x is not None:
            return x[:N]
        return np.ones(N) / N

class DMVReturn(BaseStrategy):
    """
    DMV with return-weighted penalty (DMVR), Eq. (18) of the paper.

        min_w  λ w'Σw - μ'w + δ Σᵢ cᵢ (wᵢ - 1/N)²,   cᵢ = med(μ_safe²) / μ_safe,ᵢ²

    Expanding the square and dropping the constant gives the quadratic program
        H = 2 (λΣ + δ diag(c)),   f = -μ - (2δ/N) c
    """

    def __init__(self, risk_aversion: float = 1.0,
                 diversification_factor: float = 1.0):
        super().__init__(
            name=f"DMV-Return (λ={risk_aversion}, δ={diversification_factor})"
        )
        self.risk_aversion = risk_aversion
        self.diversification_factor = diversification_factor

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)

        # Guard against division by zero
        mu_safe = np.where(np.abs(expected_returns) < 1e-8, 1e-8, expected_returns)

        inv_mu_sq_raw = 1.0 / (mu_safe ** 2)

        # Median scaling keeps the penalty weights c_i of order one
        scale = np.median(mu_safe ** 2)
        inv_mu_sq = inv_mu_sq_raw * scale

        H = 2 * (self.risk_aversion * covariance_matrix +
                 self.diversification_factor * np.diag(inv_mu_sq))

        f = -expected_returns - (self.diversification_factor * 2 * (1.0/N) * inv_mu_sq)

        A    = np.ones((1, N))
        b_eq = np.array([1.0])
        lb   = np.zeros(N)
        ub   = np.ones(N)

        w = solve_qp(H, f, A=A, b=b_eq, lb=lb, ub=ub, solver='clarabel')
        return w if w is not None else np.ones(N) / N


class DMVVars(BaseStrategy):
    """
    DMV with variance-weighted penalty (DMVV), Eq. (19) of the paper.

        min_w  λ w'Σw - μ'w + δ Σᵢ cᵢ (wᵢ - 1/N)²,   cᵢ = med(σ⁴) / σᵢ⁴

    with σ² = diag(Σ). Expanding the square and dropping the constant gives
        H = 2 (λΣ + δ diag(c)),   f = -μ - (2δ/N) c
    """

    def __init__(self, risk_aversion: float = 1.0,
                 diversification_factor: float = 1.0):
        super().__init__(
            name=f"DMV-Vars (λ={risk_aversion}, δ={diversification_factor})"
        )
        self.risk_aversion = risk_aversion
        self.diversification_factor = diversification_factor

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)

        variances = np.diag(covariance_matrix)
        vars_safe = np.where(variances < 1e-8, 1e-8, variances)

        # Inverse squared variances, median-scaled so that the penalty
        # weights c_i are of order one
        inv_vars_sq_raw = 1.0 / (vars_safe ** 2)
        scale = np.median(variances ** 2)
        inv_vars_sq = inv_vars_sq_raw * scale

        H = 2 * (self.risk_aversion * covariance_matrix +
                 self.diversification_factor * np.diag(inv_vars_sq))

        f = -expected_returns - (self.diversification_factor * 2 * (1.0/N) * inv_vars_sq)

        A    = np.ones((1, N))
        b_eq = np.array([1.0])
        lb   = np.zeros(N)
        ub   = np.ones(N)

        w = solve_qp(H, f, A=A, b=b_eq, lb=lb, ub=ub, solver='clarabel')
        return w if w is not None else np.ones(N) / N

class EWMV_Mixture(BaseStrategy):
    """
    Equally Weighted - Mean Variance Mixture (EWMV), Eq. (20) of the paper.

    The mixture is applied ex post: the MV problem is solved first and its
    solution is then shrunk toward the equally weighted portfolio.

    Mathematical Formulation:
        Step 1 (Solve MV):
            w_MV = argmin_w { λ·w^T·Σ·w - w^T·μ }

        Step 2 (Mixture):
            w_final = δ·w_MV + (1-δ)·w_EW

        where w_EW = (1/N)·1

    delta = 1 recovers the MV solution and delta = 0 the equally weighted
    portfolio.

    References:
        DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive
        diversification. Review of Financial Studies, 22(5), 1915-1953.

        Jiang, C., Du, J., & An, Y. (2019). Combining the minimum-variance and
        equally-weighted portfolios: Can portfolio performance be improved?
        Economic Modelling, 80, 260-274.
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 mixture_factor: float = 0.5):
        """
        Initialize EWMV mixture strategy.

        Args:
            risk_aversion: λ - variance penalty for MV step
            mixture_factor: δ - weight on MV (1-δ on 1/N)
        """
        super().__init__(
            name=f"EWMV Mixture (λ={risk_aversion}, δ={mixture_factor})"
        )
        self.risk_aversion = risk_aversion
        self.mixture_factor = mixture_factor

        if not (0 <= mixture_factor <= 1):
            raise ValueError("Mixture factor must be in [0, 1]")

    def optimize_weights(self,
                         expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Solve the MV problem, then mix its solution with 1/N.
        """
        N = len(expected_returns)

        # Step 1: Solve pure MV via CLARABEL
        H = 2.0 * self.risk_aversion * covariance_matrix
        f = -expected_returns
        A = np.ones((1, N))
        b_eq = np.array([1.0])
        lb = np.zeros(N)
        ub = np.ones(N)

        w_mv_sol = solve_qp(H, f, A=A, b=b_eq, lb=lb, ub=ub, solver='clarabel')
        w_mv = w_mv_sol if w_mv_sol is not None else np.ones(N) / N

        # Step 2: Mix with 1/N
        w_ew = np.ones(N) / N
        w_final = self.mixture_factor * w_mv + (1 - self.mixture_factor) * w_ew

        # Ensure normalization (numerical stability)
        w_final = w_final / np.sum(w_final)

        return w_final
