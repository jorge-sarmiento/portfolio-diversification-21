"""
Portfolio performance metrics computed for every strategy.

Metrics, by category:
- Performance: Returns, Sharpe, Sortino, Calmar, Omega
- Risk: Drawdown, CVaR, Tail Ratio, Ulcer Index
- Efficiency: Turnover, HHI, Win Rate, Profit Factor
- Statistical: DSR, PSR, Bootstrap CI, Jarque-Bera
- Relative: Alpha, Beta, Information Ratio, Tracking Error

References:
    Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.
    Journal of Portfolio Management, 40(5), 94-107.

    López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.

    Lo, A. W. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal.
"""
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression


def calculate_metrics(
    returns: np.ndarray,
    weights: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    trials_log: Optional[List[Dict]] = None,
    risk_free_rate: float = 0.0,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Performance metrics of a portfolio return series.

    Args:
        returns: Strategy returns (T,)
        weights: Historical weights (T, N)
        benchmark_returns: Benchmark returns for relative metrics
        trials_log: Grid search history for DSR
        risk_free_rate: Annual risk-free rate
        transaction_cost_bps: Trading cost in basis points
        periods_per_year: 252 for daily, 12 for monthly

    Returns:
        Dictionary of metrics
    """
    metrics = {}
    T = len(returns)

    if T < 10:
        raise ValueError(f"Insufficient data: T={T} < 10")

    # Validate dimensions
    if len(weights) != T:
        if len(weights) == T - 1:
            # Weight history one row shorter than the returns: repeat the first row
            weights = np.vstack([weights[0], weights])
        else:
            raise ValueError(f"Weight dimension mismatch: {len(weights)} vs {T}")

    FACTOR = periods_per_year
    rf_daily = risk_free_rate / FACTOR

    # =========================================================================
    # 1. TRANSACTION COSTS & NET RETURNS
    # =========================================================================

    if len(weights) > 1:
        # Turnover = sum of absolute weight changes
        weight_changes = np.abs(np.diff(weights, axis=0))
        turnover_series = np.sum(weight_changes, axis=1)
        avg_turnover = np.mean(turnover_series)

        # The first period has no turnover: prepend a zero cost
        cost_impact = np.concatenate([[0.0], turnover_series * (transaction_cost_bps / 10000.0)])
    else:
        avg_turnover = 0.0
        cost_impact = np.zeros(T)

    net_returns = returns - cost_impact

    metrics['Turnover (Annual)'] = avg_turnover * FACTOR
    metrics['Transaction Cost (bps)'] = transaction_cost_bps
    metrics['Total Costs (%)'] = np.sum(cost_impact) * 100

    # =========================================================================
    # 2. BASIC PERFORMANCE METRICS
    # =========================================================================

    mean_ret_ann = np.mean(net_returns) * FACTOR
    volatility_ann = np.std(net_returns, ddof=1) * np.sqrt(FACTOR)

    metrics['Annual Return (%)'] = mean_ret_ann * 100
    metrics['Annual Volatility (%)'] = volatility_ann * 100

    # Sharpe Ratio (net and gross)
    sharpe = (mean_ret_ann - risk_free_rate) / (volatility_ann + 1e-10)
    metrics['Sharpe Ratio'] = sharpe

    # Gross Sharpe (before costs)
    mean_gross = np.mean(returns) * FACTOR
    vol_gross = np.std(returns, ddof=1) * np.sqrt(FACTOR)
    metrics['Sharpe Ratio (Gross)'] = (mean_gross - risk_free_rate) / (vol_gross + 1e-10)

    # =========================================================================
    # 3. RISK METRICS
    # =========================================================================

    # Max Drawdown
    cum_returns = np.cumprod(1 + net_returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / (peak + 1e-10)
    max_dd = np.min(drawdown)

    metrics['Max Drawdown (%)'] = max_dd * 100

    # Calmar Ratio
    metrics['Calmar Ratio'] = mean_ret_ann / (abs(max_dd) + 1e-10)

    # Sortino Ratio (downside deviation)
    target = 0.0
    downside = net_returns[net_returns < target]
    if len(downside) > 1:
        downside_std = np.std(downside, ddof=1) * np.sqrt(FACTOR)
        sortino = (mean_ret_ann - risk_free_rate) / (downside_std + 1e-10)
    else:
        sortino = np.nan
    metrics['Sortino Ratio'] = sortino

    # Omega Ratio
    gains = net_returns[net_returns > 0].sum()
    losses = np.abs(net_returns[net_returns < 0].sum())
    metrics['Omega Ratio'] = gains / (losses + 1e-10)

    # VaR (95%)
    metrics['VaR 95% (%)'] = np.percentile(net_returns, 5) * 100

    # CVaR (95%) - Conditional Value at Risk
    var_95 = np.percentile(net_returns, 5)
    cvar_returns = net_returns[net_returns <= var_95]
    metrics['CVaR 95% (%)'] = np.mean(cvar_returns) * 100 if len(cvar_returns) > 0 else var_95 * 100

    # Ulcer Index (root mean squared drawdown)
    ulcer = np.sqrt(np.mean(drawdown**2))
    metrics['Ulcer Index'] = ulcer

    # Pain Ratio: annual return over the Ulcer Index
    metrics['Pain Ratio'] = mean_ret_ann / (ulcer + 1e-10)

    # =========================================================================
    # 4. TAIL RISK METRICS
    # =========================================================================

    # Tail Ratio
    perc_95 = np.percentile(net_returns, 95)
    perc_5 = np.percentile(net_returns, 5)
    metrics['Tail Ratio'] = abs(perc_95 / (perc_5 + 1e-10))

    # Skewness and Kurtosis
    skew = stats.skew(net_returns)
    kurt = stats.kurtosis(net_returns)  # Excess kurtosis
    metrics['Skewness'] = skew
    metrics['Excess Kurtosis'] = kurt

    # Jarque-Bera test for normality
    n = T
    jb_stat = (n / 6) * (skew**2 + (kurt**2 / 4))
    jb_pvalue = 1 - stats.chi2.cdf(jb_stat, 2)
    metrics['Jarque-Bera Stat'] = jb_stat
    metrics['Jarque-Bera p-value'] = jb_pvalue
    metrics['Normal Distribution'] = 'Yes' if jb_pvalue > 0.05 else 'No'

    # =========================================================================
    # 5. EFFICIENCY METRICS
    # =========================================================================

    # Win Rate
    metrics['Win Rate (%)'] = (np.sum(net_returns > 0) / T) * 100

    # Profit Factor
    total_gains = np.sum(net_returns[net_returns > 0])
    total_losses = abs(np.sum(net_returns[net_returns < 0]))
    metrics['Profit Factor'] = total_gains / (total_losses + 1e-10)

    # Average Win / Average Loss
    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1e-10
    metrics['Avg Win / Avg Loss'] = avg_win / avg_loss

    # Recovery Factor
    total_return = cum_returns[-1] - 1.0
    metrics['Recovery Factor'] = total_return / (abs(max_dd) + 1e-10)

    # =========================================================================
    # 6. DRAWDOWN ANALYSIS
    # =========================================================================

    # Identify all drawdown periods
    dd_periods = _identify_drawdown_periods(cum_returns, peak)

    if len(dd_periods) > 0:
        metrics['Avg Drawdown (%)'] = np.mean([dd['depth'] for dd in dd_periods]) * 100
        metrics['Max DD Duration (days)'] = max([dd['duration'] for dd in dd_periods])
        metrics['Avg DD Duration (days)'] = np.mean([dd['duration'] for dd in dd_periods])
        metrics['Number of Drawdowns'] = len(dd_periods)
    else:
        metrics['Avg Drawdown (%)'] = 0.0
        metrics['Max DD Duration (days)'] = 0
        metrics['Avg DD Duration (days)'] = 0
        metrics['Number of Drawdowns'] = 0

    # =========================================================================
    # 7. DIVERSIFICATION (STRUCTURAL)
    # =========================================================================

    # Herfindahl-Hirschman Index (concentration)
    hhi_series = np.sum(weights**2, axis=1)
    metrics['HHI (Avg)'] = np.mean(hhi_series)
    metrics['HHI (Max)'] = np.max(hhi_series)
    metrics['HHI (Min)'] = np.min(hhi_series)

    # Effective Number of Assets (1/HHI)
    metrics['Effective N Assets'] = 1.0 / metrics['HHI (Avg)']

    # Weight Stability (correlation of consecutive weights)
    if len(weights) > 1:
        correlations = []
        for t in range(1, len(weights)):
            if np.std(weights[t-1]) > 1e-10 and np.std(weights[t]) > 1e-10:
                corr = np.corrcoef(weights[t-1], weights[t])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        metrics['Weight Stability'] = np.mean(correlations) if correlations else 0.0
    else:
        metrics['Weight Stability'] = 1.0

    # =========================================================================
    # 8. STATISTICAL ROBUSTNESS
    # =========================================================================

    # Bootstrap Confidence Interval for Sharpe Ratio
    boot_sharpes = _bootstrap_sharpe(net_returns, n_bootstrap=1000,
                                     periods_per_year=FACTOR,
                                     risk_free_rate=risk_free_rate)
    metrics['Sharpe CI 95% Lower'] = np.percentile(boot_sharpes, 2.5)
    metrics['Sharpe CI 95% Upper'] = np.percentile(boot_sharpes, 97.5)
    metrics['Sharpe CI Width'] = metrics['Sharpe CI 95% Upper'] - metrics['Sharpe CI 95% Lower']

    # Probabilistic Sharpe Ratio (PSR)
    psr = _probabilistic_sharpe_ratio(
        net_returns,
        benchmark_sharpe=0.0,
        periods_per_year=FACTOR
    )
    metrics['Probabilistic Sharpe Ratio'] = psr

    # Deflated Sharpe Ratio (DSR)
    if trials_log and len(trials_log) > 1:
        dsr = _deflated_sharpe_ratio(
            observed_sharpe=sharpe,
            returns=net_returns,
            trials_log=trials_log,
            periods_per_year=FACTOR
        )
        metrics['Deflated Sharpe Ratio'] = dsr
    else:
        metrics['Deflated Sharpe Ratio'] = np.nan

    # Minimum Track Record Length
    if sharpe > 0:
        mtrl = _minimum_track_record_length(
            net_returns,
            observed_sharpe=sharpe,
            benchmark_sharpe=0.0,
            periods_per_year=FACTOR
        )
        metrics['Min Track Record (years)'] = mtrl / FACTOR
    else:
        metrics['Min Track Record (years)'] = np.inf

    # Sharpe Ratio adjusted for autocorrelation (Lo, 2002)
    sr_adjusted = _autocorrelation_adjusted_sharpe(net_returns, sharpe)
    metrics['Sharpe Ratio (AC-Adjusted)'] = sr_adjusted

    # =========================================================================
    # 9. RELATIVE PERFORMANCE (vs Benchmark)
    # =========================================================================

    if benchmark_returns is not None and len(benchmark_returns) == T:
        # Align lengths
        port_ret = net_returns
        bench_ret = benchmark_returns

        # Excess returns
        excess_returns = port_ret - bench_ret

        # Tracking Error
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(FACTOR)
        metrics['Tracking Error (%)'] = tracking_error * 100

        # Information Ratio
        metrics['Information Ratio'] = (np.mean(excess_returns) * FACTOR) / (tracking_error + 1e-10)

        # Alpha and Beta (regression)
        alpha, beta = _compute_alpha_beta(port_ret, bench_ret, risk_free_rate, FACTOR)
        metrics['Alpha (Annualized %)'] = alpha * 100
        metrics['Beta'] = beta

        # Correlation with benchmark
        metrics['Correlation vs Benchmark'] = np.corrcoef(port_ret, bench_ret)[0, 1]

    else:
        metrics['Tracking Error (%)'] = np.nan
        metrics['Information Ratio'] = np.nan
        metrics['Alpha (Annualized %)'] = np.nan
        metrics['Beta'] = np.nan
        metrics['Correlation vs Benchmark'] = np.nan

    # =========================================================================
    # 10. STABILITY INDEX (Martínez-Nieto et al., 2021)
    # =========================================================================
    # SI = R^2 of the linear regression of log(wealth) on time: it measures
    # how linear the cumulative growth is. SI = 1 is perfectly steady growth,
    # SI close to 0 is erratic growth.
    log_wealth = np.log(cum_returns + 1e-10)
    time_idx = np.arange(len(log_wealth)).reshape(-1, 1)
    ss_res = np.sum((log_wealth - (np.polyval(np.polyfit(np.arange(len(log_wealth)),
                                                          log_wealth, 1),
                                               np.arange(len(log_wealth)))))**2)
    ss_tot = np.sum((log_wealth - np.mean(log_wealth))**2)
    metrics['Stability Index'] = 1.0 - ss_res / (ss_tot + 1e-10)

    return metrics


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _bootstrap_sharpe(returns: np.ndarray,
                      n_bootstrap: int = 1000,
                      periods_per_year: int = 252,
                      risk_free_rate: float = 0.0,
                      seed: int = 42) -> np.ndarray:
    """Bootstrap confidence interval for Sharpe Ratio."""
    n = len(returns)
    rng = np.random.default_rng(seed)   # local RNG - no global state side-effect
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    samples = returns[idx]              # (n_bootstrap, n)
    mean_ret = samples.mean(axis=1) * periods_per_year
    std_ret  = samples.std(axis=1, ddof=1) * np.sqrt(periods_per_year)
    return (mean_ret - risk_free_rate) / (std_ret + 1e-10)


def _probabilistic_sharpe_ratio(returns: np.ndarray,
                                benchmark_sharpe: float = 0.0,
                                periods_per_year: int = 252) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).

    Probability that the true Sharpe ratio exceeds a benchmark value (zero by
    default), allowing for skewness and excess kurtosis.
    """
    n = len(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)

    # Observed Sharpe (non-annualized for formula)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sr_observed = mean_ret / (std_ret + 1e-10)

    # Adjustment for skewness and excess kurtosis
    denominator = np.sqrt(1 - skew * sr_observed + ((kurt + 2) / 4) * sr_observed**2)

    # Test statistic
    numerator = (sr_observed - benchmark_sharpe) * np.sqrt(n - 1)

    z_score = numerator / (denominator + 1e-10)

    # CDF of standard normal
    psr = stats.norm.cdf(z_score)

    return psr


def _deflated_sharpe_ratio(observed_sharpe: float,
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

    # De-annualize observed Sharpe for formula
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


def _minimum_track_record_length(returns: np.ndarray,
                                 observed_sharpe: float,
                                 benchmark_sharpe: float = 0.0,
                                 periods_per_year: int = 252,
                                 confidence_level: float = 0.95) -> float:
    """
    Minimum Track Record Length (MinTRL).

    Number of observations needed to conclude, at the given confidence level,
    that the Sharpe ratio exceeds the benchmark Sharpe ratio.
    """
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)

    # Z-score for confidence level
    z_alpha = stats.norm.ppf(confidence_level)

    # De-annualize
    sr_obs = observed_sharpe / np.sqrt(periods_per_year)
    sr_bench = benchmark_sharpe / np.sqrt(periods_per_year)

    # Adjustment for non-normality
    adjustment = 1 - skew * sr_obs + ((kurt - 1) / 4) * sr_obs**2

    # MinTRL formula
    min_trl = (z_alpha / (sr_obs - sr_bench + 1e-10))**2 * adjustment

    return max(1, min_trl)


def _autocorrelation_adjusted_sharpe(returns: np.ndarray,
                                     observed_sharpe: float,
                                     max_lag: int = 10) -> float:
    """
    Sharpe Ratio adjusted for serial correlation (Lo, 2002).

    If returns are autocorrelated, the standard Sharpe formula
    overstates the true risk-adjusted performance.
    """
    n = len(returns)

    # Compute autocorrelations
    acf_sum = 0
    for k in range(1, min(max_lag + 1, n)):
        rho_k = np.corrcoef(returns[:-k], returns[k:])[0, 1]
        if not np.isnan(rho_k):
            acf_sum += (n - k) / n * rho_k

    # Adjustment factor
    adjustment = np.sqrt(1 + 2 * acf_sum)

    # Adjusted Sharpe
    sr_adjusted = observed_sharpe / adjustment

    return sr_adjusted


def _compute_alpha_beta(portfolio_returns: np.ndarray,
                       benchmark_returns: np.ndarray,
                       risk_free_rate: float,
                       periods_per_year: int) -> Tuple[float, float]:
    """Compute Alpha and Beta via linear regression."""
    # Excess returns
    excess_port = portfolio_returns - (risk_free_rate / periods_per_year)
    excess_bench = benchmark_returns - (risk_free_rate / periods_per_year)

    # Regression
    X = excess_bench.reshape(-1, 1)
    y = excess_port

    model = LinearRegression()
    model.fit(X, y)

    alpha_daily = model.intercept_
    beta = model.coef_[0]

    # Annualize alpha
    alpha_annual = alpha_daily * periods_per_year

    return alpha_annual, beta


def _identify_drawdown_periods(cumulative: np.ndarray,
                               running_max: np.ndarray) -> List[Dict]:
    """Identify all distinct drawdown periods."""
    drawdowns = []
    in_dd = False
    dd_start = 0

    for t in range(len(cumulative)):
        if cumulative[t] < running_max[t]:
            if not in_dd:
                # Start of new drawdown
                in_dd = True
                dd_start = t
        else:
            if in_dd:
                # End of drawdown
                in_dd = False
                dd_end = t - 1

                # Compute depth
                dd_cumulative = cumulative[dd_start:dd_end+1]
                dd_max = running_max[dd_start:dd_end+1]
                depth = np.min((dd_cumulative - dd_max) / (dd_max + 1e-10))

                drawdowns.append({
                    'start': dd_start,
                    'end': dd_end,
                    'depth': depth,
                    'duration': dd_end - dd_start + 1
                })

    # Handle ongoing drawdown at end
    if in_dd:
        dd_end = len(cumulative) - 1
        dd_cumulative = cumulative[dd_start:dd_end+1]
        dd_max = running_max[dd_start:dd_end+1]
        depth = np.min((dd_cumulative - dd_max) / (dd_max + 1e-10))

        drawdowns.append({
            'start': dd_start,
            'end': dd_end,
            'depth': depth,
            'duration': dd_end - dd_start + 1
        })

    return drawdowns
