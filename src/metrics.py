"""
Deflated Sharpe Ratio.

References:
    Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
    Journal of Portfolio Management, 40(5), 94-107.
"""
from typing import Dict, List

import numpy as np
import scipy.stats as stats


def deflated_sharpe_ratio(observed_sharpe: float,
                          returns: np.ndarray,
                          trials_log: List[Dict],
                          periods_per_year: int = 252) -> float:
    """
    Deflated Sharpe Ratio (DSR).

    Adjusts the Sharpe ratio for the selection bias introduced by testing
    several hyperparameter combinations (Bailey and López de Prado, 2014).
    """
    n = len(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)

    # Extract Sharpe values from trials
    trial_sharpes = [t['sharpe'] for t in trials_log if 'sharpe' in t]

    if len(trial_sharpes) < 2:
        return 0.5  # No trials to compare

    # Variance of the trial Sharpe ratios (ddof=1), Bailey and López de Prado (2014)
    var_sharpes = np.var(trial_sharpes, ddof=1)

    # Number of trials
    N_trials = len(trial_sharpes)

    # De-annualise the observed Sharpe ratio
    sr0 = observed_sharpe / np.sqrt(periods_per_year)

    # Expected maximum Sharpe ratio under the null (Bailey & López de Prado, 2014)
    euler_gamma = 0.5772156649  # Euler-Mascheroni constant

    term1 = (1 - euler_gamma) * stats.norm.ppf(1 - 1 / N_trials)
    term2 = euler_gamma * stats.norm.ppf(1 - 1 / (N_trials * np.e))

    expected_max_sr = np.sqrt(var_sharpes) * (term1 + term2)

    # DSR calculation
    numerator = (sr0 - expected_max_sr) * np.sqrt(n - 1)
    denominator = np.sqrt(1 - skew * sr0 + ((kurt + 2) / 4) * sr0**2)

    dsr_value = stats.norm.cdf(numerator / (denominator + 1e-10))

    return dsr_value
