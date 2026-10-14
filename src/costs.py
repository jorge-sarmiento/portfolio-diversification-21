"""
Turnover and transaction costs.

Turnover is measured against the weights drifted by the returns of the period,
and proportional costs are deducted period by period.
"""
from typing import Optional

import numpy as np

from src import config as C


def turnover_series(weights: np.ndarray,
                    asset_returns: Optional[np.ndarray] = None,
                    convention: str = "drift") -> np.ndarray:
    """
    Per-period turnover, sum_i |w_{i,t} - w_{i,t}^-|, with zero in the first period.

    "target": w_t^- is the previous target vector w_{t-1}.
    "drift":  w_t^- is w_{t-1} after one period of price moves,
              w_{t-1} * (1 + R_{t-1}) / (1 + w_{t-1}' R_{t-1}),
              which requires the simple asset returns of the evaluated rows.
    """
    T = len(weights)
    out = np.zeros(T)
    if T < 2:
        return out
    if convention == "target":
        prev = weights[:-1]
    elif convention == "drift":
        grown = weights[:-1] * (1.0 + asset_returns[:-1])
        prev = grown / grown.sum(axis=1, keepdims=True)
    else:
        raise ValueError(convention)
    out[1:] = np.abs(weights[1:] - prev).sum(axis=1)
    return out


def net_returns(gross: np.ndarray, turnover: np.ndarray,
                cost_bps: float = C.TRANSACTION_COST_BPS) -> np.ndarray:
    """Costs are deducted period by period; RF is subtracted afterwards."""
    return gross - turnover * (cost_bps / 10000.0)
