"""
Walk-Forward validation engine with nested hyperparameter optimization.

Protocol:
- Chronological train/test split (no look-ahead bias)
- Grid search on in-sample data
- Rolling window out-of-sample testing
- Complete trial logging for DSR calculation

References:
    Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies.
    Wiley. (Chapter 11: Walk-Forward Analysis)

    Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
    Journal of Portfolio Management, 40(5), 94-107.
"""
import time
import numpy as np
import pandas as pd
import itertools
import logging
from typing import List, Dict, Any, Tuple, Type, Optional
from sklearn.covariance import LedoitWolf
from src.strategies.base import BaseStrategy

# Logging for experiment traceability
logger = logging.getLogger(__name__)


class WalkForwardEngine:
    """
    Nested walk-forward validation engine.

    Protocol:
        1. In-sample phase (first 70% of the sample):
           - grid search over the hyperparameters, scored by the Sharpe ratio of
             a rolling mini-backtest with Ledoit-Wolf covariance (static
             calibration for MVN)
           - every combination is logged for the Deflated Sharpe Ratio

        2. Out-of-sample phase (last 30%):
           - the selected hyperparameters are held fixed
           - weights are re-optimised at each step on a rolling window
           - the weights applied at t use data up to t-1 only

    Hyperparameters are selected once; weights are re-optimised at every
    rebalancing date.

    Attributes:
        strategy_class: Class to instantiate (not instance)
        strategy_name: Human-readable name
        trials_log: Complete history of IS attempts (for DSR)
        best_params: Winning hyperparameters from grid search
        periods_per_year: Data frequency (252=daily, 12=monthly)
    """

    def __init__(self, strategy_class: Type[BaseStrategy], strategy_name: str,
                 periods_per_year: int = 12,
                 calibration_estimator: str = "lw_rolling",
                 force_param_grid: Optional[Dict[str, List]] = None):
        """
        Initialize validation engine.

        Args:
            strategy_class: Uninstantiated strategy class
            strategy_name: Descriptive name for logging
            periods_per_year: Data frequency (252=daily, 12=monthly)
            calibration_estimator: IS covariance protocol for grid search.
                "sample_static" - sample covariance on the full IS period;
                    calibration and evaluation then use different estimators.
                "lw_rolling" (default) - rolling LW mini-backtest within IS using
                    the same lookback_window as OOS, eliminating the estimator
                    asymmetry between calibration and evaluation.
            force_param_grid: If not None, overrides _PARAM_GRIDS for this instance.
        """
        self.strategy_class = strategy_class
        self.strategy_name = strategy_name
        self.calibration_estimator = calibration_estimator
        self._force_param_grid: Optional[Dict[str, List]] = force_param_grid
        self.trials_log: List[Dict[str, Any]] = []
        self.best_params: Dict[str, Any] = {}
        self.best_params_static: Dict[str, Any] = {}     # always sample_static result
        self.best_is_sharpe: float = float('nan')
        self.best_is_sharpe_static: float = float('nan')
        self._grid_search_time: float = 0.0
        self.periods_per_year = periods_per_year
        self.is_returns: np.ndarray = np.array([])
        self.is_weights: np.ndarray = np.array([])
        self._oos_strategy = None   # strategy instance after OOS run (for diagnostics)

    def run(self,
            data: np.ndarray,
            dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Execute complete walk-forward validation.

        Args:
            data: Return matrix (T × N)
            dates: Datetime index

        Returns:
            Dictionary with:
                - returns: OOS return series
                - weights: OOS weight matrix
                - dates: OOS dates
                - best_params: Optimal hyperparameters
                - trials_log: All IS attempts
        """
        T, N = data.shape

        # Report short samples
        if T < 100:
            logger.info(f"Sample size: T={T} periods.")

        # 70/30 split for IS/OOS
        split_idx = int(T * 0.70)

        if split_idx < 50 or (T - split_idx) < 20:
            logger.info(f"Split: IS={split_idx}, OOS={T-split_idx} periods.")

        # In-Sample data (training)
        train_data = data[:split_idx]

        # Lookback window - shared by grid search (lw_rolling mode) and OOS rolling.
        # Formula: max(1 year, 20% of IS), capped at IS length.
        lookback_window = min(
            max(self.periods_per_year, int(len(train_data) * 0.2)),
            len(train_data)
        )

        logger.info(
            f"[{self.strategy_name}] Starting validation: "
            f"IS={split_idx}, OOS={T-split_idx}, lookback={lookback_window}"
        )

        # --- PHASE 1: HYPERPARAMETER OPTIMIZATION ---
        param_grid = self._get_param_grid()

        if param_grid:
            logger.info(f"Grid search: {self._count_combinations(param_grid)} combinations "
                        f"[calibration_estimator={self.calibration_estimator}]")
            start_time = time.time()
            self.best_params = self._grid_search(train_data, param_grid, lookback_window)
            self._grid_search_time = time.time() - start_time
            logger.info(
                f"Best params: {self.best_params} "
                f"(search took {self._grid_search_time:.2f}s)"
            )
        else:
            self.best_params = {}
            self.best_params_static = {}
            logger.info("No hyperparameters to optimize (using defaults)")
            # Compute IS returns for no-param strategies using the full train set
            try:
                mu_is = np.mean(train_data, axis=0) * self.periods_per_year
                S_is  = np.cov(train_data, rowvar=False) * self.periods_per_year
                _strat_is = self.strategy_class()
                if hasattr(_strat_is, 'set_raw_data'):
                    _strat_is.set_raw_data(train_data)
                if hasattr(_strat_is, 'set_benchmark_data'):
                    _strat_is.set_benchmark_data(train_data, train_data.mean(axis=1))
                self.is_weights = _strat_is.safe_optimize(mu_is, S_is)
                self.is_returns = train_data @ self.is_weights
            except Exception as _e:
                logger.debug(f"IS returns (no-param) failed: {_e}")

        # --- PHASE 2: OUT-OF-SAMPLE TESTING ---
        test_data = data[split_idx:]
        test_dates = dates[split_idx:]

        history_buffer = train_data[-lookback_window:]

        oos_returns, oos_weights = self._run_rolling_test(
            history_buffer=history_buffer,
            test_data=test_data,
            params=self.best_params
        )

        return {
            'returns': oos_returns,
            'weights': oos_weights,
            'dates': test_dates,
            'best_params': self.best_params,
            'best_params_static': self.best_params_static,
            'best_is_sharpe': self.best_is_sharpe,
            'best_is_sharpe_static': self.best_is_sharpe_static,
            'calibration_estimator': self.calibration_estimator,
            'trials_log': self.trials_log,
            'grid_search_time': self._grid_search_time,
            'n_trials': len(self.trials_log),
            'is_returns': self.is_returns,
            'is_weights': self.is_weights,
            'oos_strategy': self._oos_strategy,
        }

    # -----------------------------------------------------------------------
    # Per-strategy-code grid definitions.
    # Keys must match the strategy codes used in STRATEGY_MAP (main.py).
    # Strategies absent from this dict have no tunable hyperparameters
    # (EW, GMV, GMR, MD, FSD, SSD, SRF_LP, RP, HRP).
    # -----------------------------------------------------------------------
    _PARAM_GRIDS: Dict[str, Dict[str, List[float]]] = {
        # --- Block 2: Mean-Variance and Extensions ---
        # λ > 0; the grid includes values below and above one (Section 6.2).
        'MV':         {'risk_aversion': [0.5, 2.0, 10.0]},
        # MSV = MVN (Eq. 9): convex-combination form, lambda in [0, 1].
        'MSV':        {'risk_aversion': [0.1, 0.3, 0.5, 0.7, 0.9]},
        # SemiV (Eq. 8): convex-combination form, lambda in [0, 1].
        'SemiV':      {'risk_aversion': [0.1, 0.3, 0.5, 0.7, 0.9]},

        # --- Block 4: Weight Constraints ---
        'WLBC':       {'risk_aversion': [0.5, 2.0, 10.0],
                       'lower_bound':   [0.01, 0.02, 0.05]},
        'WUBC':       {'risk_aversion': [0.5, 2.0, 10.0],
                       'upper_bound':   [0.10, 0.20, 0.30]},
        # WUBC_Alt = WCMV: simultaneous lb and ub constraints
        'WUBC_Alt':   {'risk_aversion': [0.5, 2.0, 10.0],
                       'lower_bound':   [0.01, 0.02, 0.05],
                       'upper_bound':   [0.10, 0.20, 0.30]},

        # --- Block 5: Diversification Incentives ---
        'DMV':        {'risk_aversion':          [0.5, 2.0, 10.0],
                       'diversification_factor': [0.1, 0.5, 2.0]},
        'DMV_Yager':  {'risk_aversion':          [0.5, 2.0, 10.0],
                       'diversification_factor': [0.1, 0.5, 2.0]},
        'DMV_Return': {'risk_aversion':          [0.5, 2.0, 10.0],
                       'diversification_factor': [0.1, 0.5, 2.0]},
        'DMV_Vars':   {'risk_aversion':          [0.5, 2.0, 10.0],
                       'diversification_factor': [0.1, 0.5, 2.0]},
        'EWMV':       {'risk_aversion':  [0.5, 2.0, 10.0],
                       'mixture_factor': [0.1, 0.3, 0.5, 0.7, 0.9]},

        # --- Block 3: Alternative Risk Measures ---
        # CVaR: alpha is the confidence level (0.90/0.95/0.99 standard choices)
        'CVaR':  {'alpha': [0.90, 0.95, 0.99]},
        # MAD: risk_aversion is the lambda of Eq. (10)
        'MAD':   {'risk_aversion': [0.5, 2.0, 10.0]},
    }

    def _get_param_grid(self) -> Dict[str, List[float]]:
        """
        Return the hyperparameter grid for this strategy.

        Grids are looked up by strategy code in _PARAM_GRIDS, independently of
        the display name each strategy class gives itself.
        Strategies not in _PARAM_GRIDS have no free hyperparameters and
        return an empty dict (no grid search, so no Deflated Sharpe Ratio).

        If force_param_grid was supplied at construction time,
        it overrides the class-level definition for this instance only.
        """
        if self._force_param_grid is not None:
            return dict(self._force_param_grid)
        return dict(self._PARAM_GRIDS.get(self.strategy_name, {}))

    def _count_combinations(self, grid: Dict[str, List]) -> int:
        """Count total hyperparameter combinations."""
        if not grid:
            return 0
        return int(np.prod([len(v) for v in grid.values()]))

    # Strategies solved by differential evolution are calibrated with the static
    # in-sample protocol: the rolling mini-backtest would require one global
    # optimisation per in-sample step and combination (Section 4.2).
    _LW_ROLLING_SKIP: set = {'MSV'}

    def _grid_search(self,
                     train_data: np.ndarray,
                     param_grid: Dict[str, List],
                     lookback_window: int = 12) -> Dict[str, Any]:
        """
        Exhaustive grid search maximising IS Sharpe Ratio.

        Two calibration modes (self.calibration_estimator):

        "sample_static":
            Estimates mu and Sigma once on the full IS period using sample
            covariance (ddof=1). Produces static IS weights and Sharpe, and
            leaves an estimator asymmetry (sample IS vs Ledoit-Wolf OOS).

        "lw_rolling" (default):
            Mirrors the OOS rolling protocol within the IS segment: at each
            IS step t in {0,...,T_is-lookback_window-1}, fits LW shrinkage on
            train_data[t:t+lookback_window], optimises weights, records the
            realized return at train_data[t+lookback_window]. IS Sharpe is
            computed from these rolling IS returns. Eliminates the estimator
            asymmetry between calibration and evaluation.
            Excluded for MSV, whose evolutionary calibration is prohibitive under
            the rolling mini-backtest (see _LW_ROLLING_SKIP and Section 4.2).

        In either mode, the sample_static results are also stored in
        self.best_params_static and self.best_is_sharpe_static.

        Args:
            train_data:      In-sample return matrix (T_is x N).
            param_grid:      Hyperparameter search space.
            lookback_window: Rolling window size (must equal the OOS value
                             computed in run()).
        Returns:
            Best hyperparameter combination dict.
        """
        param_names  = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        # ------------------------------------------------------------------ #
        # STEP 1 - sample_static pass (always runs; provides trials_log      #
        #          for DSR and the secondary comparison result)               #
        # ------------------------------------------------------------------ #
        mu_static = np.mean(train_data, axis=0) * self.periods_per_year
        S_static  = np.cov(train_data, rowvar=False) * self.periods_per_year

        best_sharpe_static = -np.inf
        best_combo_static  = {}

        for combo_values in combinations:
            combo_dict = dict(zip(param_names, combo_values))
            try:
                strategy = self.strategy_class(**combo_dict)
                if hasattr(strategy, 'set_raw_data'):
                    strategy.set_raw_data(train_data)
                weights = strategy.safe_optimize(mu_static, S_static)
                port_ret = train_data @ weights
                mean_ret = np.mean(port_ret)
                std_ret  = np.std(port_ret, ddof=1)
                sharpe   = mean_ret / (std_ret + 1e-10)
                self.trials_log.append({
                    'sharpe': sharpe, 'mean': mean_ret, 'std': std_ret,
                    'params': combo_dict, 'strategy': self.strategy_name,
                })
                if sharpe > best_sharpe_static:
                    best_sharpe_static = sharpe
                    best_combo_static  = combo_dict.copy()
            except Exception as e:
                logger.debug(f"Static combo {combo_dict} failed: {e}")

        if not best_combo_static and combinations:
            best_combo_static = dict(zip(param_names, combinations[0]))
            logger.warning(f"All static combos failed; using default: {best_combo_static}")

        self.best_params_static    = best_combo_static
        self.best_is_sharpe_static = (
            best_sharpe_static * np.sqrt(self.periods_per_year)
            if best_sharpe_static > -np.inf else float('nan')
        )

        # ------------------------------------------------------------------ #
        # STEP 2 - decide active calibration path                            #
        # ------------------------------------------------------------------ #
        use_rolling = (
            self.calibration_estimator == "lw_rolling"
            and self.strategy_name not in self._LW_ROLLING_SKIP
        )
        if self.calibration_estimator == "lw_rolling" and not use_rolling:
            logger.info(
                f"[{self.strategy_name}] calibrated with the static in-sample protocol "
                f"(non-convex optimiser)."
            )

        n_roll = len(train_data) - lookback_window
        if use_rolling and n_roll < 5:
            logger.warning(
                f"[{self.strategy_name}] Only {n_roll} rolling IS steps "
                f"(lookback={lookback_window}, T_is={len(train_data)}); "
                f"falling back to sample_static."
            )
            use_rolling = False

        if not use_rolling:
            # Use sample_static results
            if best_sharpe_static > -np.inf:
                self.best_is_sharpe = self.best_is_sharpe_static
            self._set_is_weights_static(best_combo_static, mu_static, S_static, train_data)
            return best_combo_static

        # ------------------------------------------------------------------ #
        # STEP 3 - lw_rolling: rolling IS mini-backtest                      #
        # ------------------------------------------------------------------ #
        best_sharpe_rolling = -np.inf
        best_combo_rolling  = {}

        for combo_values in combinations:
            combo_dict   = dict(zip(param_names, combo_values))
            roll_returns = np.full(n_roll, np.nan)

            for t in range(n_roll):
                window = train_data[t : t + lookback_window]
                try:
                    lw    = LedoitWolf()
                    cov_w = lw.fit(window).covariance_ * self.periods_per_year
                    mu_w  = np.mean(window, axis=0) * self.periods_per_year
                    strat = self.strategy_class(**combo_dict)
                    if hasattr(strat, 'set_raw_data'):
                        strat.set_raw_data(window)
                    if hasattr(strat, 'set_benchmark_data'):
                        strat.set_benchmark_data(window, window.mean(axis=1))
                    w = strat.safe_optimize(mu_w, cov_w)
                    roll_returns[t] = train_data[t + lookback_window] @ w
                except Exception as _e:
                    logger.debug(f"lw_rolling IS t={t} {combo_dict}: {_e}")

            valid = roll_returns[~np.isnan(roll_returns)]
            if len(valid) < 3:
                continue
            sharpe = np.mean(valid) / (np.std(valid, ddof=1) + 1e-10)
            if sharpe > best_sharpe_rolling:
                best_sharpe_rolling = sharpe
                best_combo_rolling  = combo_dict.copy()

        if not best_combo_rolling and combinations:
            best_combo_rolling = dict(zip(param_names, combinations[0]))
            logger.warning(f"All rolling combos failed; using default: {best_combo_rolling}")

        if best_sharpe_rolling > -np.inf:
            self.best_is_sharpe = best_sharpe_rolling * np.sqrt(self.periods_per_year)

        # IS weights: optimise with LW on the last lookback_window IS periods
        # (mirrors the first OOS step's estimation window)
        if best_combo_rolling:
            try:
                last_w  = train_data[-lookback_window:]
                lw      = LedoitWolf()
                cov_is  = lw.fit(last_w).covariance_ * self.periods_per_year
                mu_is   = np.mean(last_w, axis=0) * self.periods_per_year
                bs = self.strategy_class(**best_combo_rolling)
                if hasattr(bs, 'set_raw_data'):
                    bs.set_raw_data(last_w)
                if hasattr(bs, 'set_benchmark_data'):
                    bs.set_benchmark_data(last_w, last_w.mean(axis=1))
                self.is_weights = bs.safe_optimize(mu_is, cov_is)
                self.is_returns = train_data @ self.is_weights
            except Exception as e:
                logger.debug(f"lw_rolling IS weights failed: {e}")

        return best_combo_rolling

    def _set_is_weights_static(self,
                               best_combo: Dict[str, Any],
                               mu: np.ndarray,
                               S: np.ndarray,
                               train_data: np.ndarray) -> None:
        """Compute IS weights and returns for the sample_static path."""
        if not best_combo:
            return
        try:
            bs = self.strategy_class(**best_combo)
            if hasattr(bs, 'set_raw_data'):
                bs.set_raw_data(train_data)
            self.is_weights = bs.safe_optimize(mu, S)
            self.is_returns = train_data @ self.is_weights
        except Exception as e:
            logger.debug(f"Could not compute IS weights (static): {e}")

    def _run_rolling_test(self,
                          history_buffer: np.ndarray,
                          test_data: np.ndarray,
                          params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rolling window out-of-sample backtest.

        Simulation Protocol:
            At each time t:
            1. Use data from [t-window, t-1]
            2. Estimate μ, Σ
            3. Optimize weights w_t
            4. Apply w_t to returns at time t
            5. Record realized return and weights

        This prevents look-ahead bias (cannot use future data).

        Args:
            history_buffer: Initial lookback data
            test_data: Future data (OOS)
            params: Fixed hyperparameters from grid search

        Returns:
            (returns, weights) arrays for OOS period
        """
        T_test, N = test_data.shape
        window_size = len(history_buffer)

        # Concatenate for easy windowing
        full_data = np.vstack([history_buffer, test_data])

        # Pre-allocate outputs
        oos_returns = np.zeros(T_test)
        oos_weights = np.zeros((T_test, N))

        # Instantiate strategy with best params
        strategy = self.strategy_class(**params)

        # Progress bar - use tqdm if available, otherwise silent
        try:
            from tqdm import tqdm
            _iter = tqdm(
                range(T_test),
                desc=f"  OOS [{self.strategy_name}]",
                unit="step",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                leave=False,   # erase bar when done; main.py prints the summary line
            )
        except ImportError:
            _iter = range(T_test)

        for t in _iter:
            # Estimation window: rows t, ..., t + window_size - 1 of full_data
            window_start = t
            window_end = t + window_size
            current_window = full_data[window_start:window_end]

            # Handle strategies that need raw data (e.g. MeanSemivariance)
            if hasattr(strategy, 'set_raw_data'):
                strategy.set_raw_data(current_window)

            # Strategies that need benchmark data (P-Dom and SSD-R, codes FSD and SSD);
            # the benchmark is the equally weighted portfolio
            if hasattr(strategy, 'set_benchmark_data'):
                benchmark_returns = current_window.mean(axis=1)
                strategy.set_benchmark_data(current_window, benchmark_returns)

            # OOS estimation: Ledoit-Wolf shrinkage on the rolling window.
            # IS grid search uses either sample_static (full IS, sample cov) or
            # lw_rolling (rolling LW, same protocol as here), controlled by
            # calibration_estimator.
            mu_window = np.mean(current_window, axis=0) * self.periods_per_year
            try:
                lw = LedoitWolf()
                cov_window = lw.fit(current_window).covariance_ * self.periods_per_year
            except Exception:
                cov_window = np.cov(current_window, rowvar=False) * self.periods_per_year

            # Optimize - safe_optimize handles all failures internally
            w_optimal = strategy.safe_optimize(mu_window, cov_window)

            # Realized return at time t
            r_realized = test_data[t] @ w_optimal

            oos_returns[t] = r_realized
            oos_weights[t] = w_optimal

        self._oos_strategy = strategy   # expose for per-step diagnostics (e.g. SRF_LP)
        return oos_returns, oos_weights
