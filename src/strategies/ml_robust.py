"""
Risk-budgeting and hierarchical allocation strategies (Block 7 of the paper).

    RP  (Eq. 23)  equal risk contribution, solved with trust-constr
    HRP           hierarchical risk parity (algorithmic; no closed-form objective)

References:
    López de Prado, M. (2016). Building Diversified Portfolios that Outperform
    Out of Sample. Journal of Portfolio Management, 42(4), 59-69.

    Maillard, S., Roncalli, T., & Teiletche, J. (2010). The properties of
    equally weighted risk contribution portfolios. Journal of Portfolio Management.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from src.strategies.base import BaseStrategy


class RiskParity(BaseStrategy):
    """
    Risk Parity (Equal Risk Contribution), Eq. (23) of the paper.

    Objective:
        min_w  Σ_i Σ_j (RC_i - RC_j)²,   RC_i = w_i (Σw)_i / sqrt(w'Σw)

    evaluated as Σ_i Σ_j (w_i (Σw)_i - w_j (Σw)_j)² / (w'Σw). The optimiser is
    warm-started at the inverse-volatility allocation.

    References:
        Maillard, S., Roncalli, T., & Teiletche, J. (2010).
        The properties of equally weighted risk contribution portfolios.
        Journal of Portfolio Management, 36(4), 60-70.

        Qian, E. (2005). Risk Parity Portfolios. PanAgora Asset Management.
    """

    def __init__(self):
        super().__init__(name="Risk Parity (ERC)")

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        N = covariance_matrix.shape[0]

        # Initial guess: Inverse volatility
        volatilities = np.sqrt(np.diag(covariance_matrix))
        initial_guess = (1.0 / volatilities) / np.sum(1.0 / volatilities)

        def objective(w):
            """Sum of squared pairwise differences of the risk contributions."""
            sigma_w = covariance_matrix @ w
            rc = w * sigma_w  # Risk contributions
            port_risk = w @ sigma_w

            # Vectorised pairwise differences
            diff = rc[:, None] - rc[None, :]  # (N, N) matrix
            obj = np.sum(diff ** 2) / (port_risk + 1e-10)

            return obj

        # trust-constr takes LinearConstraint objects, not the SLSQP dict format
        eq_constraint = LinearConstraint(np.ones((1, N)), lb=1.0, ub=1.0)
        bounds_obj = Bounds(lb=0.0, ub=1.0)

        result = minimize(objective, initial_guess, method='trust-constr',
                         bounds=bounds_obj, constraints=eq_constraint,
                         options={'maxiter': 300, 'gtol': 1e-9, 'verbose': 0})

        if not result.success:
            return (1.0 / volatilities) / np.sum(1.0 / volatilities)
        return result.x


class HierarchicalRiskParity(BaseStrategy):
    """
    Hierarchical Risk Parity (HRP), López de Prado (2016).

    Algorithm:
        1. Tree clustering on the distance d_ij = sqrt(0.5 (1 - ρ_ij)), single
           linkage.
        2. Quasi-diagonalisation: Σ is reordered so that similar assets are
           adjacent.
        3. Recursive bisection: the weight of each cluster is split between its
           two halves in inverse proportion to their variance.

    No matrix inversion is required and the strategy has no hyperparameters.

    References:
        López de Prado, M. (2016). Building Diversified Portfolios that
        Outperform Out of Sample. JPM, 42(4), 59-69.
    """

    def __init__(self):
        super().__init__(name="Hierarchical Risk Parity (HRP)")

    def optimize_weights(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        std_devs = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)

        # Clip correlation to [-1, 1] before sqrt to avoid NaN from floating-point
        # values slightly outside [-1, 1] (e.g. 1.0000000002 due to rounding).
        correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)
        distance = np.sqrt(0.5 * (1 - correlation_matrix))
        np.fill_diagonal(distance, 0)

        distance_condensed = squareform(distance, checks=False)
        linkage_matrix = linkage(distance_condensed, method='single')

        sort_ix = self._get_quasi_diag(linkage_matrix)
        cov_sorted = covariance_matrix[np.ix_(sort_ix, sort_ix)]
        weights_sorted = self._recursive_bisection(cov_sorted)

        weights = np.zeros(len(expected_returns))
        weights[sort_ix] = weights_sorted
        return weights

    def _get_quasi_diag(self, linkage_matrix: np.ndarray) -> list:
        """
        López de Prado (2016), quasi-diagonalisation.

        The linkage tree is traversed in preorder with an explicit stack: for
        each internal node the right child is pushed first and the left child
        second, so the left subtree is expanded first (LIFO).
        """
        N = int(linkage_matrix[-1, 3])   # total number of leaves
        # Stack initialised with the two children of the root
        root_left  = int(linkage_matrix[-1, 0])
        root_right = int(linkage_matrix[-1, 1])
        stack = [root_right, root_left]  # LIFO: the left child is processed first
        result = []

        while stack:
            node = stack.pop()
            if node < N:
                # Leaf node: append to the result
                result.append(node)
            else:
                # Internal node: push its children (right child first)
                idx = node - N
                stack.append(int(linkage_matrix[idx, 1]))  # right child
                stack.append(int(linkage_matrix[idx, 0]))  # left child

        return result

    def _recursive_bisection(self, cov: np.ndarray) -> np.ndarray:
        """
        López de Prado (2016), recursive bisection.
        """
        N = cov.shape[0]
        weights = np.ones(N)
        clusters = [list(range(N))]

        while clusters:
            # Split every cluster into two halves
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    new_clusters.extend([cluster[:mid], cluster[mid:]])
            clusters = new_clusters

            # Split the weight between each pair of adjacent clusters
            for i in range(0, len(clusters) - 1, 2):
                left_cluster  = clusters[i]
                right_cluster = clusters[i + 1]

                var_left  = self._cluster_variance(cov, left_cluster)
                var_right = self._cluster_variance(cov, right_cluster)

                alpha = 1.0 - var_left / (var_left + var_right + 1e-10)
                weights[left_cluster]  *= alpha
                weights[right_cluster] *= (1.0 - alpha)

        return weights

    def _cluster_variance(self, cov: np.ndarray, cluster: list) -> float:
        """Cluster variance using Inverse Variance Portfolio."""
        cov_cluster = cov[np.ix_(cluster, cluster)]
        inv_diag = 1.0 / (np.diag(cov_cluster) + 1e-10)
        w_ivp = inv_diag / np.sum(inv_diag)
        return w_ivp.T @ cov_cluster @ w_ivp
