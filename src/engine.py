"""
Walk-forward validation engine with nested hyperparameter optimisation.

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


# ---------------------------------------------------------------------------
# Out-of-sample rows solved in worker processes
# ---------------------------------------------------------------------------
# Each OOS row is an independent problem: its inputs are the rows of the
# estimation window and the fixed hyperparameters. The only link between
# consecutive rows is the fallback of BaseStrategy.safe_optimize (reuse the
# last valid weights when the optimiser fails), which the parent process
# applies in order. The numbers are therefore the same as in the serial loop.

_POOL_DATA: Optional[np.ndarray] = None
_POOL_RF: Optional[np.ndarray] = None


def init_oos_worker(data: np.ndarray, rf: Optional[np.ndarray] = None) -> None:
    """Pool initializer: keep the return matrix and the risk-free rate in the worker process."""
    global _POOL_DATA, _POOL_RF
    _POOL_DATA, _POOL_RF = data, rf


def estimation_returns(data: np.ndarray, rf: Optional[np.ndarray],
                       strategy_class: Type[BaseStrategy]) -> np.ndarray:
    """
    Returns from which the inputs of a strategy are estimated: in excess of the
    risk-free rate for the strategies that set excess_return_inputs, the
    returns themselves for the rest.
    """
    if rf is not None and getattr(strategy_class, "excess_return_inputs", False):
        return data - np.asarray(rf, dtype=float)[:, None]
    return data


def solve_oos_row(args: Tuple) -> Tuple[Optional[np.ndarray], float]:
    """
    Weights of one OOS row (None when the optimiser failed) and the CPU
    seconds the worker spent on it.

    Same steps as one iteration of the serial loop, on a fresh strategy
    instance: a fresh instance has no cached weights, so it reports a failure
    instead of applying the fallback itself.
    """
    cpu0 = time.process_time()
    strategy_class, params, i, window_size, periods_per_year = args
    current_window = estimation_returns(_POOL_DATA[i - window_size : i],
                                        None if _POOL_RF is None else _POOL_RF[i - window_size : i],
                                        strategy_class)
    strategy = strategy_class(**params)
    if hasattr(strategy, 'set_raw_data'):
        strategy.set_raw_data(current_window)
    if hasattr(strategy, 'set_benchmark_data'):
        strategy.set_benchmark_data(current_window, current_window.mean(axis=1))
    mu_window = np.mean(current_window, axis=0) * periods_per_year
    try:
        cov_window = LedoitWolf().fit(current_window).covariance_ * periods_per_year
    except Exception:
        cov_window = np.cov(current_window, rowvar=False) * periods_per_year
    w = strategy.safe_optimize(mu_window, cov_window)
    return (w if strategy.last_valid_weights is not None else None,
            time.process_time() - cpu0)


class WalkForwardEngine:
    """
    Nested walk-forward validation engine.

    Protocol:
        1. Calibration phase:
           - grid search over the hyperparameters, scored by the Sharpe ratio of
             a rolling mini-backtest with Ledoit-Wolf covariance (static
             calibration for MVN)
           - every combination is logged for the Deflated Sharpe Ratio

        2. Out-of-sample phase:
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
                 force_param_grid: Optional[Dict[str, List]] = None,
                 dsr_trials: str = "static"):
        """
        Initialise the validation engine.

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
            dsr_trials: Source of the trials logged for the Deflated Sharpe Ratio.
                "static" - Sharpe ratios of the candidates in the static
                    in-sample pass.
                "rolling" - Sharpe ratios of the candidates in the rolling
                    mini-backtest that selects the hyperparameter; strategies
                    calibrated with the static pass only (MSV) keep its trials.
        """
        if dsr_trials not in ("static", "rolling"):
            raise ValueError(dsr_trials)
        self.dsr_trials = dsr_trials
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
        self.oos_worker_cpu_s: float = 0.0   # CPU seconds spent by row workers (timing only)
        # Optional rule (combination, n_assets) -> bool. Combinations for which it
        # returns False are removed from the grid before any calibration pass and
        # are not logged as trials. None keeps the full grid.
        self.combo_filter = None
        self.excluded_combos: List[Dict[str, Any]] = []

    def run(self,
            data: np.ndarray,
            dates: pd.DatetimeIndex,
            rf: Optional[np.ndarray] = None,
            schedule: Optional[Dict[str, Any]] = None,
            checkpoint: Optional[Any] = None,
            oos_pool: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute complete walk-forward validation.

        Args:
            data: Return matrix (T × N)
            dates: Datetime index
            rf: Per-period risk-free rate (T,), in the units of `data`. When
                given, every Sharpe ratio used for selection and logged for the
                DSR is computed on returns in excess of it. None means rf = 0.
            schedule: Explicit design, as returned by data.schedule_indices:
                {"window", "calib_idx", "oos_idx", "static_end"}. Every
                evaluated row t is estimated on rows [t - window, t). None
                uses a 70/30 split of the sample.
            checkpoint: Optional object with load_calibration / save_calibration /
                load_oos / save_oos, used to resume an interrupted run.
            oos_pool: Optional process pool created with init_oos_worker(data).
                The OOS rows are then solved in its workers; the results are
                identical to the serial loop.

        Returns:
            Dictionary with:
                - returns: OOS return series
                - weights: OOS weight matrix
                - dates: OOS dates
                - best_params: Optimal hyperparameters
                - trials_log: All IS attempts
        """
        T, N = data.shape
        self._rf = None if rf is None else np.asarray(rf, dtype=float)
        # returns used to estimate the inputs (excess returns for MVN); the
        # realised portfolio returns are always computed from `data`
        self._est = estimation_returns(data, self._rf, self.strategy_class)

        if schedule is None:
            # 70/30 split for IS/OOS
            split_idx = int(T * 0.70)

            if T < 100:
                logger.info(f"Sample size: T={T} periods.")
            if split_idx < 50 or (T - split_idx) < 20:
                logger.info(f"Split: IS={split_idx}, OOS={T-split_idx} periods.")

            # Lookback window - shared by grid search (lw_rolling mode) and OOS rolling.
            # Formula: max(1 year, 20% of IS), capped at IS length.
            lookback_window = min(
                max(self.periods_per_year, int(split_idx * 0.2)),
                split_idx
            )
            calib_idx = np.arange(lookback_window, split_idx)
            oos_idx = np.arange(split_idx, T)
        else:
            split_idx = int(schedule["static_end"])
            lookback_window = int(schedule["window"])
            calib_idx = np.asarray(schedule["calib_idx"], dtype=int)
            oos_idx = np.asarray(schedule["oos_idx"], dtype=int)
            if min(calib_idx.min(), oos_idx.min()) < lookback_window:
                raise ValueError("an evaluated row has fewer prior rows than the window")

        # Static in-sample segment: every row before the first OOS row
        train_data = data[:split_idx]

        logger.info(
            f"[{self.strategy_name}] Starting validation: "
            f"IS={split_idx}, OOS={len(oos_idx)}, lookback={lookback_window}"
        )

        # --- PHASE 1: HYPERPARAMETER OPTIMISATION ---
        param_grid = self._get_param_grid()
        saved = checkpoint.load_calibration() if checkpoint is not None else None

        if saved is not None:
            # Resumed run: the calibration was already done
            self.best_params = saved["best_params"]
            self.best_params_static = saved["best_params_static"]
            self.best_is_sharpe = saved["best_is_sharpe"]
            self.best_is_sharpe_static = saved["best_is_sharpe_static"]
            self.trials_log = saved["trials_log"]
            self._grid_search_time = saved["grid_search_time"]
        elif param_grid:
            logger.info(f"Grid search: {self._count_combinations(param_grid)} combinations "
                        f"[calibration_estimator={self.calibration_estimator}]")
            start_time = time.time()
            self.best_params = self._grid_search(train_data, param_grid, lookback_window,
                                                 data=data, calib_idx=calib_idx)
            self._grid_search_time = time.time() - start_time
            logger.info(
                f"Best params: {self.best_params} "
                f"(search took {self._grid_search_time:.2f}s)"
            )
        else:
            self.best_params = {}
            self.best_params_static = {}
            logger.info("No hyperparameters to optimise (using defaults)")
            # Compute IS returns for no-param strategies using the full train set
            try:
                train_est = self._est[:split_idx]
                mu_is = np.mean(train_est, axis=0) * self.periods_per_year
                S_is  = np.cov(train_est, rowvar=False) * self.periods_per_year
                _strat_is = self.strategy_class()
                if hasattr(_strat_is, 'set_raw_data'):
                    _strat_is.set_raw_data(train_est)
                if hasattr(_strat_is, 'set_benchmark_data'):
                    _strat_is.set_benchmark_data(train_data, train_data.mean(axis=1))
                self.is_weights = _strat_is.safe_optimize(mu_is, S_is)
                self.is_returns = train_data @ self.is_weights
            except Exception as _e:
                logger.debug(f"IS returns (no-param) failed: {_e}")

        if checkpoint is not None and saved is None:
            checkpoint.save_calibration({
                "best_params": self.best_params,
                "best_params_static": self.best_params_static,
                "best_is_sharpe": self.best_is_sharpe,
                "best_is_sharpe_static": self.best_is_sharpe_static,
                "trials_log": self.trials_log,
                "grid_search_time": self._grid_search_time,
            })

        # --- PHASE 2: OUT-OF-SAMPLE TESTING ---
        test_dates = dates[oos_idx]

        oos_returns, oos_weights = self._run_rolling_test(
            data=data,
            oos_idx=oos_idx,
            window_size=lookback_window,
            params=self.best_params,
            checkpoint=checkpoint,
            oos_pool=oos_pool,
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
            'oos_worker_cpu_s': self.oos_worker_cpu_s,
        }

    # -----------------------------------------------------------------------
    # Per-strategy-code grid definitions.
    # Keys must match the strategy codes of src/strategies/__init__.py.
    # Strategies absent from this dict have no tunable hyperparameters
    # (EW, GMV, GMR, MD, FSD, SSD, RP, HRP).
    # -----------------------------------------------------------------------
    _PARAM_GRIDS: Dict[str, Dict[str, List[float]]] = {
        # --- Block 2: Mean-Variance and Extensions ---
        # λ > 0; the grid includes values below and above one.
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
    # optimisation per in-sample step and combination.
    _LW_ROLLING_SKIP: set = {'MSV'}

    def _grid_search(self,
                     train_data: np.ndarray,
                     param_grid: Dict[str, List],
                     lookback_window: int = 12,
                     data: Optional[np.ndarray] = None,
                     calib_idx: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
            the rolling mini-backtest (see _LW_ROLLING_SKIP).

        In either mode, the sample_static results are also stored in
        self.best_params_static and self.best_is_sharpe_static.

        Args:
            train_data:      In-sample return matrix (T_is x N).
            param_grid:      Hyperparameter search space.
            lookback_window: Rolling window size (must equal the OOS value
                             computed in run()).
            data:            Full return matrix; row t of the rolling
                             mini-backtest is estimated on data[t - window:t].
            calib_idx:       Rows evaluated by the rolling mini-backtest.
                             Default: every row of train_data after the first
                             window.
        Returns:
            Best hyperparameter combination dict.

        Sharpe ratios are computed on returns in excess of the risk-free rate
        passed to run() (rf = 0 when none was given).
        """
        if data is None:
            data = train_data
        if calib_idx is None:
            calib_idx = np.arange(lookback_window, len(train_data))
        rf = getattr(self, "_rf", None)
        est = getattr(self, "_est", None)
        if est is None or est.shape != data.shape:
            est = data
        train_est = est[:len(train_data)]
        param_names  = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        if self.combo_filter is not None:
            n_assets = train_data.shape[1]
            keep = [self.combo_filter(dict(zip(param_names, c)), n_assets) for c in combinations]
            self.excluded_combos = [dict(zip(param_names, c))
                                    for c, k in zip(combinations, keep) if not k]
            combinations = [c for c, k in zip(combinations, keep) if k]

        # ------------------------------------------------------------------ #
        # STEP 1 - sample_static pass (always runs; provides trials_log      #
        #          for DSR and the secondary comparison result)               #
        # ------------------------------------------------------------------ #
        mu_static = np.mean(train_est, axis=0) * self.periods_per_year
        S_static  = np.cov(train_est, rowvar=False) * self.periods_per_year

        best_sharpe_static = -np.inf
        best_combo_static  = {}

        for combo_values in combinations:
            combo_dict = dict(zip(param_names, combo_values))
            try:
                strategy = self.strategy_class(**combo_dict)
                if hasattr(strategy, 'set_raw_data'):
                    strategy.set_raw_data(train_est)
                weights = strategy.safe_optimize(mu_static, S_static)
                port_ret = train_data @ weights
                if rf is not None:
                    port_ret = port_ret - rf[:len(train_data)]
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

        n_roll = len(calib_idx)
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
        rolling_trials: List[Dict[str, Any]] = []

        # Moments of each estimation window. They do not depend on the
        # hyperparameters, so they are computed once and reused by every
        # combination; a window whose estimation fails is stored as None.
        moments: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        for i in calib_idx:
            window = est[i - lookback_window : i]
            try:
                lw    = LedoitWolf()
                cov_w = lw.fit(window).covariance_ * self.periods_per_year
                mu_w  = np.mean(window, axis=0) * self.periods_per_year
                moments.append((mu_w, cov_w))
            except Exception as _e:
                logger.debug(f"lw_rolling IS row={i}: {_e}")
                moments.append(None)

        for combo_values in combinations:
            combo_dict   = dict(zip(param_names, combo_values))
            roll_returns = np.full(n_roll, np.nan)

            for t, i in enumerate(calib_idx):
                if moments[t] is None:
                    continue
                window = est[i - lookback_window : i]
                try:
                    mu_w, cov_w = moments[t]
                    strat = self.strategy_class(**combo_dict)
                    if hasattr(strat, 'set_raw_data'):
                        strat.set_raw_data(window)
                    if hasattr(strat, 'set_benchmark_data'):
                        strat.set_benchmark_data(window, window.mean(axis=1))
                    w = strat.safe_optimize(mu_w, cov_w)
                    roll_returns[t] = data[i] @ w
                    if rf is not None:
                        roll_returns[t] -= rf[i]
                except Exception as _e:
                    logger.debug(f"lw_rolling IS t={t} {combo_dict}: {_e}")

            valid = roll_returns[~np.isnan(roll_returns)]
            if len(valid) < 3:
                continue
            sharpe = np.mean(valid) / (np.std(valid, ddof=1) + 1e-10)
            rolling_trials.append({
                'sharpe': sharpe, 'mean': np.mean(valid), 'std': np.std(valid, ddof=1),
                'params': combo_dict, 'strategy': self.strategy_name,
            })
            if sharpe > best_sharpe_rolling:
                best_sharpe_rolling = sharpe
                best_combo_rolling  = combo_dict.copy()

        # Trials for the DSR: the candidates of the search that made the selection
        if self.dsr_trials == "rolling" and rolling_trials:
            self.trials_log = rolling_trials

        if not best_combo_rolling and combinations:
            best_combo_rolling = dict(zip(param_names, combinations[0]))
            logger.warning(f"All rolling combos failed; using default: {best_combo_rolling}")

        if best_sharpe_rolling > -np.inf:
            self.best_is_sharpe = best_sharpe_rolling * np.sqrt(self.periods_per_year)

        # IS weights: optimise with LW on the last lookback_window IS periods
        # (mirrors the first OOS step's estimation window)
        if best_combo_rolling:
            try:
                last_w  = train_est[-lookback_window:]
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
                          data: np.ndarray,
                          oos_idx: np.ndarray,
                          window_size: int,
                          params: Dict[str, Any],
                          checkpoint: Optional[Any] = None,
                          oos_pool: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rolling window out-of-sample backtest.

        Simulation Protocol:
            For each evaluated row t:
            1. Use data from [t-window, t-1]
            2. Estimate μ, Σ
            3. Optimise weights w_t
            4. Apply w_t to returns at time t
            5. Record realized return and weights

        This prevents look-ahead bias (cannot use future data).

        Args:
            data: Full return matrix
            oos_idx: Rows evaluated out of sample
            window_size: Length of the estimation window
            params: Fixed hyperparameters from grid search
            checkpoint: Optional resume object (see run())
            oos_pool: Optional process pool (see run())

        Returns:
            (returns, weights) arrays for OOS period
        """
        T_test, N = len(oos_idx), data.shape[1]

        # Pre-allocate outputs
        oos_returns = np.zeros(T_test)
        oos_weights = np.zeros((T_test, N))

        # Instantiate strategy with best params
        strategy = self.strategy_class(**params)

        # Resume: rows already evaluated and the fallback cache of the strategy
        t_start = 0
        if checkpoint is not None:
            state = checkpoint.load_oos()
            if state is not None and len(state["returns"]) == T_test:
                t_start = int(state["t_done"])
                oos_returns[:t_start] = state["returns"][:t_start]
                oos_weights[:t_start] = state["weights"][:t_start]
                strategy.last_valid_weights = state["last_valid_weights"]

        if oos_pool is not None:
            # Rows solved in worker processes, collected in order
            jobs = [(self.strategy_class, params, int(oos_idx[t]), window_size,
                     self.periods_per_year) for t in range(t_start, T_test)]
            for t, (w_optimal, cpu_s) in zip(range(t_start, T_test),
                                             oos_pool.map(solve_oos_row, jobs)):
                self.oos_worker_cpu_s += cpu_s
                if w_optimal is None:
                    # Fallback of safe_optimize: previous valid weights, else 1/N
                    lvw = strategy.last_valid_weights
                    w_optimal = lvw if lvw is not None and len(lvw) == N else np.ones(N) / N
                else:
                    strategy.last_valid_weights = w_optimal
                oos_returns[t] = data[oos_idx[t]] @ w_optimal
                oos_weights[t] = w_optimal
                if checkpoint is not None:
                    checkpoint.save_oos(t + 1, oos_returns, oos_weights,
                                        strategy.last_valid_weights)
            self._oos_strategy = strategy
            return oos_returns, oos_weights

        for t in range(T_test):
            if t < t_start:
                continue
            # Estimation window: the window_size rows that precede row i
            i = oos_idx[t]
            current_window = self._est[i - window_size : i]

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
            r_realized = data[i] @ w_optimal

            oos_returns[t] = r_realized
            oos_weights[t] = w_optimal

            if checkpoint is not None:
                checkpoint.save_oos(t + 1, oos_returns, oos_weights,
                                    strategy.last_valid_weights)

        self._oos_strategy = strategy   # exposed for diagnostics
        return oos_returns, oos_weights
