"""
Abstract base class shared by every allocation strategy.

Provides the optimisation wrapper used by the walk-forward engine: covariance
regularisation, the strategy-specific optimiser, constraint enforcement, and a
fallback policy for windows in which the optimiser fails.
"""
from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Failures already reported: each distinct message is logged once per run
_REPORTED_FAILURES = set()


class BaseStrategy(ABC):
    """
    Abstract base class for asset allocation strategies.

    Attributes:
        name (str): Strategy identifier used in logs and outputs.
        last_valid_weights (Optional[np.ndarray]): Cached weights for fallback.
    """

    def __init__(self, name: str):
        self.name = name
        self.last_valid_weights: Optional[np.ndarray] = None

    @abstractmethod
    def optimize_weights(self,
                         expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the optimal portfolio weights.

        Args:
            expected_returns: Vector mu of shape (N,).
            covariance_matrix: Matrix Sigma of shape (N, N).

        Returns:
            np.ndarray: Weight vector normalised so that sum(w) = 1.
        """
        pass

    def safe_optimize(self,
                      expected_returns: np.ndarray,
                      covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Fail-safe optimisation wrapper.

        In a walk-forward backtest a window may produce a singular covariance
        matrix, an infeasible parameter combination, or a convergence failure.
        This method catches the failure and applies a recovery policy:

            1. Optimiser fails -> reuse the weights of the previous window.
            2. No previous window -> use 1/N (equally weighted).

        Each distinct failure is logged once per run.
        """
        N = covariance_matrix.shape[0]

        try:
            # 1. Regularise the covariance matrix
            clean_cov = self.clean_matrix(covariance_matrix)

            # 2. Strategy-specific optimisation
            weights = self.optimize_weights(expected_returns, clean_cov)

            # 3. Enforce the long-only and budget constraints
            weights = self.verify_constraints(weights)

            # 4. Update the fallback cache
            self.last_valid_weights = weights
            return weights

        except Exception as e:
            key = (self.name, str(e))
            if key not in _REPORTED_FAILURES:
                _REPORTED_FAILURES.add(key)
                logger.warning(f"Optimization failed [{self.name}]: {str(e).rstrip('.')}. "
                               f"Using the previous weights (1/N if there are none).")

            if self.last_valid_weights is not None and len(self.last_valid_weights) == N:
                return self.last_valid_weights
            else:
                return np.ones(N) / N

    def verify_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        Enforce the long-only and full-investment constraints.

            1. w_i >= 0      (no short selling)
            2. sum(w_i) = 1  (full investment)
        """
        # Clip to remove floating-point residuals such as -1e-16
        w_clipped = np.maximum(weights, 0.0)

        # L1 normalisation
        total_weight = np.sum(w_clipped)
        if total_weight <= 1e-8:
            # Degenerate case (zero vector): fall back to 1/N
            return np.ones(len(weights)) / len(weights)

        return w_clipped / total_weight

    @staticmethod
    def clean_matrix(matrix: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Tikhonov regularisation ensuring a symmetric positive-definite matrix.

            Sigma_clean = (Sigma + Sigma^T) / 2 + epsilon * I

        Symmetrisation removes floating-point asymmetries; the ridge term shifts
        every eigenvalue by +epsilon, which guarantees positive definiteness
        without an eigendecomposition.
        """
        # 1. Symmetrise
        matrix = (matrix + matrix.T) / 2

        # 2. Ridge term
        return matrix + epsilon * np.eye(matrix.shape[0])
