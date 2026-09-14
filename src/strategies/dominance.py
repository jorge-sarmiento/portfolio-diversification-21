"""
Stochastic dominance and alternative risk measures.

    P-Dom (Eq. 21) and SSD-R (Eq. 22)   Block 6 of the paper
    MAD (Eq. 10) and CVaR (Eq. 11)      Block 3 of the paper

References:
    Hadar, J. & Russell, W. (1969). Rules for Ordering Uncertain Prospects.
    American Economic Review, 59(1), 25-34.

    Post, T. (2003). Empirical Tests for Stochastic Dominance Efficiency.
    Journal of Finance, 58(5), 1905-1931.

    Dentcheva, D. & Ruszczyński, A. (2003). Optimization with Stochastic
    Dominance Constraints. SIAM Journal on Optimization, 14(2), 548-566.

    Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
    Value-at-Risk. Journal of Risk, 2(3), 21-41.

    Konno, H. & Yamazaki, H. (1991). Mean-Absolute Deviation Portfolio
    Optimization Model and Its Applications to Tokyo Stock Market.
    Management Science, 37(5), 519-531.
"""
import numpy as np
from scipy.optimize import linprog as _linprog
from src.strategies.base import BaseStrategy


class FirstOrderSD(BaseStrategy):
    """
    Pathwise Dominance (P-Dom), Eq. (21) of the paper.

    The constraints impose state-by-state (almost-sure) dominance over the
    benchmark:
        R_t' w >= index_t   for t = 1,...,T
    This is sufficient, but not necessary, for first-order stochastic
    dominance, which only requires F_w(x) <= F_bench(x) for all x.

    Formulation (conservative LP approach of Post, 2003):
        max w^T μ
        s.t.  -data @ w <= -index  (pointwise dominance, all t)
              Σw_i = 1,  w_i >= 0

    Mapping to scipy linprog:
        c    = -mu
        A_ub = -data    (T x N)
        b_ub = -index   (T,)
        A_eq = ones(1,N),  b_eq = [1]
        lb = 0,  ub = 1

    References:
        Hadar, J. & Russell, W. (1969). Rules for Ordering Uncertain Prospects.
        American Economic Review, 59(1), 25-34.
    """

    def __init__(self):
        super().__init__(name="First-Order SD")
        self._data = None
        self._index = None

    def set_benchmark_data(self, data: np.ndarray, index: np.ndarray):
        self._data = data
        self._index = index

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        from scipy.optimize import linprog

        if self._data is None or self._index is None:
            raise ValueError("FSD requires data. Call set_benchmark_data() first.")

        N = len(expected_returns)
        c = -expected_returns
        A_ub = -self._data
        b_ub = -self._index
        A_eq = np.ones((1, N))
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0) for _ in range(N)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        return result.x if result.success else np.ones(N) / N


class SecondOrderSD(BaseStrategy):
    """
    SSD necessary-condition relaxation (SSD-R), Eq. (22) of the paper.

    Benchmark returns are sorted in ascending order (pi = argsort) and the same
    permutation is applied to the asset returns before the cumulative sums are
    taken. Because pi orders the benchmark only, the left-hand side is at least
    the sum of the t smallest portfolio returns, so every portfolio that
    dominates the benchmark in the second-order sense is feasible, and the
    feasible set may contain others.

    Formulation:
        max  mu' w
        s.t. cumsum(R[pi,:]) w >= cumsum(index[pi])   for t = 1,...,T
             Σw_i = 1,  0 <= w_i <= 1

    where pi = argsort(index) sorts benchmark returns ascending.
    The permutation is fixed by the benchmark alone, so constraints remain
    linear in w.

    Mapping to scipy linprog:
        c    = -mu
        A_ub = -cumsum(data[pi, :], axis=0)   (T x N)
        b_ub = -cumsum(index[pi])              (T,)
        A_eq = ones(1,N),  b_eq = [1]
        lb = 0,  ub = 1

    References:
        Hadar, J. & Russell, W. (1969). Rules for Ordering Uncertain Prospects.
        American Economic Review, 59(1), 25-34.

        Dentcheva, D. & Ruszczyński, A. (2003). Optimization with Stochastic
        Dominance Constraints. SIAM Journal on Optimization, 14(2), 548-566.
    """

    def __init__(self):
        super().__init__(name="Second-Order SD")
        self._data = None
        self._index = None

    def set_benchmark_data(self, data: np.ndarray, index: np.ndarray):
        self._data = data
        self._index = index

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        from scipy.optimize import linprog

        if self._data is None or self._index is None:
            raise ValueError("SSD requires data. Call set_benchmark_data() first.")

        N = len(expected_returns)
        c = -expected_returns

        # Sort benchmark ascending -> pi determines CDF ordering.
        # Apply same permutation to asset rows to keep constraints linear in w.
        pi = np.argsort(self._index)
        A_ub = -np.cumsum(self._data[pi, :], axis=0)   # (T, N)
        b_ub = -np.cumsum(self._index[pi])              # (T,)

        A_eq = np.ones((1, N))
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0) for _ in range(N)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        return result.x if result.success else np.ones(N) / N


class MinCVaR(BaseStrategy):
    """
    Minimum Conditional Value-at-Risk (CVaR) portfolio.

    Eq. (11) of the paper, written as the linear program of Rockafellar and
    Uryasev (2000):

        min_{w, zeta}  zeta + 1/(T(1-alpha)) * sum_t u_t

        s.t.  u_t >= -R_t'w - zeta,   t = 1,...,T   (loss beyond VaR)
              u_t >= 0,                t = 1,...,T
              1'w = 1
              w >= 0

    Variables: x = [w (N), zeta (1), u (T)]  ->  N+1+T total.

    zeta is the VaR (endogenous decision variable).
    alpha is the confidence level (e.g. 0.95 = CVaR at 5% tail).

    Requires raw return data via set_raw_data(); falls back to 1/N if absent.

    References:
        Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
        Value-at-Risk. Journal of Risk, 2(3), 21-41.

        Rockafellar, R.T. & Uryasev, S. (2002). Conditional Value-at-Risk for
        General Loss Distributions. Journal of Banking & Finance, 26(7),
        1443-1471.
    """

    def __init__(self, alpha: float = 0.95):
        super().__init__(name=f"Min-CVaR (alpha={alpha})")
        self.alpha = alpha
        self._raw_data = None

    def set_raw_data(self, data: np.ndarray):
        self._raw_data = data

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)
        R = self._raw_data
        if R is None or R.shape[0] < 2:
            return np.ones(N) / N

        T = R.shape[0]
        alpha = self.alpha
        coeff_u = 1.0 / (T * (1.0 - alpha))

        # Objective: min zeta + coeff_u * sum(u)
        c = np.concatenate([np.zeros(N), [1.0], np.full(T, coeff_u)])

        # Inequality: -R[t]'w - zeta - u[t] <= 0  (T rows)
        A_ub = np.zeros((T, N + 1 + T))
        A_ub[:, :N] = -R
        A_ub[:, N] = -1.0
        np.fill_diagonal(A_ub[:, N + 1:], -1.0)
        b_ub = np.zeros(T)

        # Equality: sum(w) = 1
        A_eq = np.zeros((1, N + 1 + T))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w in [0,1]; zeta unbounded; u in [0, inf)
        bounds = [(0.0, 1.0)] * N + [(None, None)] + [(0.0, None)] * T

        result = _linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
        if result.success:
            return result.x[:N]
        return np.ones(N) / N


class MAD(BaseStrategy):
    """
    Mean-Absolute Deviation (MAD) portfolio, Konno & Yamazaki (1991).

    Eq. (10) of the paper: the Konno and Yamazaki (1991) model in scalarised
    form, with the minimum-return constraint replaced by the risk weight lambda:

        max_{w}  mu'w - lambda * (1/T) * sum_t d_t

        s.t.  d_t >=  (R_t - mu)'w,   t = 1,...,T
              d_t >= -(R_t - mu)'w,   t = 1,...,T
              d_t >= 0,               t = 1,...,T
              1'w = 1
              w >= 0

    For linprog (minimization): min  -mu'w + lambda * (1/T) * sum_t d_t

    Variables: x = [w (N), d (T)]  ->  N+T total.
    mu = (1/T) sum_t R_t  (sample mean from the estimation window).

    As lambda -> 0 the problem reduces to max mu'w, the GMR portfolio.

    Requires raw return data via set_raw_data(); falls back to 1/N if absent.

    References:
        Konno, H. & Yamazaki, H. (1991). Mean-Absolute Deviation Portfolio
        Optimization Model and Its Applications to Tokyo Stock Market.
        Management Science, 37(5), 519-531.
    """

    def __init__(self, risk_aversion: float = 2.0):
        super().__init__(name=f"MAD (lambda={risk_aversion})")
        self.risk_aversion = risk_aversion
        self._raw_data = None

    def set_raw_data(self, data: np.ndarray):
        self._raw_data = data

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = len(expected_returns)
        R = self._raw_data
        if R is None or R.shape[0] < 2:
            return np.ones(N) / N

        T = R.shape[0]
        lam = self.risk_aversion
        mu_raw = R.mean(axis=0)          # per-period mean  (N,)
        dev = R - mu_raw                 # (R_t - mu)       (T, N)

        # Objective: min -mu'w + (lambda/T)*sum(d)
        c = np.concatenate([-mu_raw, np.full(T, lam / T)])

        # 2T inequality constraints
        A_ub = np.zeros((2 * T, N + T))
        # Constraints (R_t - mu)'w - d_t <= 0
        A_ub[:T, :N] = dev
        np.fill_diagonal(A_ub[:T, N:], -1.0)
        # Constraints -(R_t - mu)'w - d_t <= 0
        A_ub[T:, :N] = -dev
        np.fill_diagonal(A_ub[T:, N:], -1.0)
        b_ub = np.zeros(2 * T)

        # Equality: sum(w) = 1
        A_eq = np.zeros((1, N + T))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w in [0,1]; d in [0, inf)
        bounds = [(0.0, 1.0)] * N + [(0.0, None)] * T

        result = _linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
        if result.success:
            return result.x[:N]
        return np.ones(N) / N
