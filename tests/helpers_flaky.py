"""Test double: a strategy whose optimiser fails on some windows."""
import numpy as np

from src.strategies.base import BaseStrategy


class Flaky(BaseStrategy):
    """Fails when the last row number of its window is a multiple of 3 (and on the first OOS row)."""

    def __init__(self):
        super().__init__(name="flaky")
        self._last_row = None

    def set_raw_data(self, data: np.ndarray) -> None:
        self._last_row = int(round(data[-1, 0]))      # column 0 holds the row number

    def optimize_weights(self, expected_returns, covariance_matrix):
        if self._last_row % 3 == 0 or self._last_row == 49:
            raise RuntimeError("no solution")
        w = np.arange(1, len(expected_returns) + 1, dtype=float) + self._last_row
        return w / w.sum()
