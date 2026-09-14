"""
Main entry point of the portfolio comparison pipeline.

Runs the walk-forward backtest for the selected strategies and datasets, at
monthly frequency and, in dual-frequency mode, also daily, and writes the
outputs from which the tables of the article are built under results/Run_<ts>/.

Usage:
    python main.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import HybridDataLoader
from src.validator import WalkForwardEngine
from src.metrics import calculate_metrics

# ============================================================================
# STRATEGY IMPORTS
# ============================================================================

# Benchmarks and single-moment strategies (4)
from src.strategies.basic import (
    EquallyWeighted, GlobalMinimumVariance,
    GlobalMaximumReturn, MaxDiversification
)

# Markowitz Family (6)
from src.strategies.markowitz import (
    MeanVariance,                      # MV
    MeanSquaredVariance,               # MSV   (MVN in the paper)
    WeightConstrainedMV,               # WUBC_Alt (WCMV in the paper)
    WeightLowerBoundConstrainedMV,     # WLBC
    MeanSemivariance,                  # SemiV
    WeightUpperBoundConstraint         # WUBC
)

# Stochastic dominance and alternative risk measures (5)
from src.strategies.dominance import (
    SecondOrderSD,
    FirstOrderSD,
    MinCVaR,
    MAD
)
from src.strategies.srf_lp import SoftReturnFloor

# Diversification-penalised mean-variance (5)
from src.strategies.advanced import (
    DiversifiedMeanVariance,
    DMVYager,
    DMVReturn,
    DMVVars,
    EWMV_Mixture
)

# Risk budgeting and hierarchical allocation (2)
from src.strategies.ml_robust import (
    RiskParity,
    HierarchicalRiskParity
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FOLDER = Path("data")
TRANSACTION_COST_BPS = 10.0  # basis points, deducted period by period
RANDOM_SEED = 42             # Global seed for reproducibility
np.random.seed(RANDOM_SEED)

# ============================================================================
# STRATEGY REGISTRY
# The 21 strategies of the article, plus SRF_LP, which the article reports as a
# negative result and excludes from the comparison tables.
# ============================================================================

STRATEGY_MAP = {
    # === BASIC BENCHMARKS (4) ===
    'EW': EquallyWeighted,
    'GMV': GlobalMinimumVariance,
    'GMR': GlobalMaximumReturn,
    'MD': MaxDiversification,

    # === MARKOWITZ FAMILY (6) ===
    'MV': MeanVariance,
    'MSV': MeanSquaredVariance,
    'SemiV': MeanSemivariance,
    'WUBC': lambda: WeightUpperBoundConstraint(upper_bound=0.20),
    'WLBC': lambda: WeightLowerBoundConstrainedMV(lower_bound=0.02),
    'WUBC_Alt': lambda: WeightConstrainedMV(lower_bound=0.02, upper_bound=0.20),

    # === STOCHASTIC DOMINANCE (3) ===
    'FSD':    FirstOrderSD,
    'SSD':    SecondOrderSD,
    'SRF_LP': SoftReturnFloor,

    # === DIVERSIFICATION PENALTIES (5) ===
    'DMV': DiversifiedMeanVariance,
    'DMV_Yager': DMVYager,
    'DMV_Return': DMVReturn,
    'DMV_Vars': DMVVars,
    'EWMV': EWMV_Mixture,

    # === RISK BUDGETING (2) ===
    'RP': RiskParity,
    'HRP': HierarchicalRiskParity,

    # === TAIL RISK / ALTERNATIVE RISK (2) ===
    'CVaR': MinCVaR,
    'MAD': MAD,
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def scan_datasets(folder: Path) -> List[Path]:
    """
    Discover all CSV files in data folder.
    """
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"\nCreated folder '{folder}'. Please add CSV files and restart.")
        print(f"  Expected format: CSV with columns as assets, rows as time periods")
        sys.exit(0)

    files = sorted(list(folder.glob("*.csv")))

    if not files:
        print(f"\nNo CSV files found in '{folder}'")
        print(f"  Add at least one dataset to proceed")
        sys.exit(0)

    return files


def select_datasets(all_files: List[Path], mode: str = 'interactive') -> List[Path]:
    """
    Interactive dataset selection.

    Args:
        all_files: Available CSV files
        mode: 'interactive' or 'all'
    """
    if mode == 'all':
        return all_files

    print(f"\n{'='*70}")
    print(f"  AVAILABLE DATASETS")
    print(f"{'='*70}")

    for i, f in enumerate(all_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  [{i+1}] {f.name:<35} ({size_mb:>6.2f} MB)")

    print(f"  [{len(all_files)+1}] ALL DATASETS")
    print(f"{'='*70}")

    while True:
        try:
            sel = input("\nSelect dataset numbers (e.g., 1,3,5) or 'ALL': ").strip()

            if sel.upper() == 'ALL' or sel == str(len(all_files)+1):
                return all_files

            indices = [int(x.strip()) - 1 for x in sel.split(',')]
            selected = [all_files[i] for i in indices if 0 <= i < len(all_files)]

            if selected:
                return selected

            print("  Invalid selection, please try again")

        except (ValueError, IndexError):
            print("  Invalid input format")


def select_strategies(mode: str = 'interactive') -> List[str]:
    """
    Interactive strategy selection with 22 strategies.

    Args:
        mode: 'interactive', 'all', or 'quick'
    """
    if mode == 'all':
        return list(STRATEGY_MAP.keys())

    if mode == 'quick':
        # Quick test: Core strategies only
        return ['EW', 'GMV', 'MV', 'DMV', 'HRP']

    print(f"\n{'='*70}")
    print(f"  AVAILABLE STRATEGIES (22 total)")
    print(f"{'='*70}")
    print(f"\n  Benchmarks (4):        {', '.join(['EW', 'GMV', 'GMR', 'MD'])}")
    print(f"  Markowitz Family (6):  {', '.join(['MV', 'MSV', 'SemiV', 'WUBC', 'WLBC', 'WUBC_Alt'])}")
    print(f"  Diversification (5):   {', '.join(['DMV', 'DMV_Yager', 'DMV_Return', 'DMV_Vars', 'EWMV'])}")
    print(f"  Risk budgeting (2):    {', '.join(['RP', 'HRP'])}")
    print(f"  Dominance (3):         {', '.join(['FSD', 'SSD', 'SRF_LP'])}")
    print(f"  Alternative risk (2):  {', '.join(['CVaR', 'MAD'])}")
    print(f"\n  Shortcuts:")
    print(f"    'ALL'    - Run all 22 strategies")
    print(f"    'QUICK' - Run core strategies only (EW, GMV, MV, DMV, HRP)")
    print(f"{'='*70}")

    while True:
        strat_input = input("\nEnter strategy codes (comma-separated): ").strip()

        if strat_input.upper() == 'ALL':
            return list(STRATEGY_MAP.keys())

        if strat_input.upper() == 'QUICK':
            return ['EW', 'GMV', 'MV', 'DMV', 'HRP']

        # match codes case-insensitively, but keep the canonical spelling
        canonical = {code.upper(): code for code in STRATEGY_MAP}
        codes = [s.strip().upper() for s in strat_input.split(',')]
        valid_codes = [canonical[c] for c in codes if c in canonical]

        if valid_codes:
            return valid_codes

        print("  No valid strategies selected")


def _resample_to_monthly(data_matrix: np.ndarray,
                         dates: pd.DatetimeIndex,
                         loader: "HybridDataLoader"):
    """
    Resample a daily return matrix to monthly returns.
    Returns (data_monthly, dates_monthly, 12).
    """
    prices = np.exp(np.cumsum(data_matrix, axis=0))
    prices = np.vstack([np.ones(data_matrix.shape[1]), prices])
    df_prices = pd.DataFrame(
        prices,
        index=pd.to_datetime(
            [dates[0] - pd.tseries.offsets.BusinessDay(1)] + list(dates)
        )
    )
    df_monthly = loader.resample_to_monthly(df_prices)
    prices_m = df_monthly.values
    data_m = np.log(prices_m[1:] / (prices_m[:-1] + 1e-10))
    dates_m = df_monthly.index[1:]
    return data_m, dates_m


def run_monthly_backtest(datasets: List[Path],
                         strategies: List[str],
                         base_dir: Path):
    """
    Option 1 - Monthly backtest (primary analysis).
    Resamples daily CSVs to monthly returns and runs walk-forward validation.
    """
    print(f"\n{'='*70}")
    print(f"  MONTHLY BACKTEST  (primary analysis)")
    print(f"{'='*70}")
    print(f"  Datasets:   {len(datasets)}")
    print(f"  Strategies: {len(strategies)}")
    print(f"  Frequency:  Monthly (resampled from daily)")
    print(f"{'='*70}\n")

    loader = HybridDataLoader(engine='pandas')

    for dataset_path in datasets:
        ds_name = dataset_path.stem
        print(f"\n>>> Processing: {ds_name}")

        try:
            ((data_daily, ppy_orig), loaded_dates), _ = loader.load_and_preprocess(dataset_path)
            dates_daily = loaded_dates if loaded_dates is not None else pd.date_range(
                end=datetime.today(), periods=len(data_daily), freq='B'
            )
        except Exception as e:
            print(f"    Failed to load: {e}")
            continue

        if ppy_orig == 12:
            data_m, dates_m = data_daily, dates_daily
            print(f"    Data already monthly, using as-is")
        else:
            data_m, dates_m = _resample_to_monthly(data_daily, dates_daily, loader)
            print(f"    Resampled {len(data_daily)} daily -> {len(data_m)} monthly periods")

        out_dir = base_dir / ds_name
        run_single_backtest(data_m, dates_m, 12, strategies, out_dir,
                            f"{ds_name} - Monthly")

    # Tables built from the monthly results
    try:
        from src.paper_tables import generate_all_tables
        generate_all_tables(base_dir)
        print(f"  Paper tables saved to {base_dir / 'PAPER_TABLES'}")
    except Exception as e:
        print(f"  Paper tables failed: {e}")

    print(f"\n{'='*70}")
    print(f"  MONTHLY BACKTEST COMPLETE")
    print(f"  Output directory: {base_dir}")
    print(f"{'='*70}\n")


def run_dual_frequency_analysis(datasets: List[Path],
                                strategies: List[str],
                                base_dir: Path):
    """
    Option 2 - Dual-frequency analysis.
    Run 1: Monthly (primary).  Run 2: Daily (robustness).
    """
    print(f"\n{'='*70}")
    print(f"  DUAL-FREQUENCY ANALYSIS  (Monthly primary + Daily robustness)")
    print(f"{'='*70}")
    print(f"  Datasets:   {len(datasets)}")
    print(f"  Strategies: {len(strategies)}")
    print(f"  Mode:       Monthly (Run 1) + Daily (Run 2)")
    print(f"{'='*70}\n")

    loader = HybridDataLoader(engine='pandas')

    for dataset_path in datasets:
        ds_name = dataset_path.stem
        print(f"\n>>> Processing: {ds_name}")
        print(f"    {'-'*60}")

        try:
            ((data_daily, ppy_orig), loaded_dates), _ = loader.load_and_preprocess(dataset_path)
            dates_daily = loaded_dates if loaded_dates is not None else pd.date_range(
                end=datetime.today(), periods=len(data_daily), freq='B'
            )
            print(f"    Daily data:  {data_daily.shape[0]} periods x {data_daily.shape[1]} assets")
            print(f"    Timeline:    {dates_daily[0].date()} to {dates_daily[-1].date()}")
        except Exception as e:
            print(f"    Failed to load dataset: {e}")
            continue

        # ====================================================================
        # RUN 1: MONTHLY (primary)
        # ====================================================================
        print(f"\n    RUN 1: Monthly (primary analysis)")
        print(f"    {'-'*60}")

        if ppy_orig == 12:
            data_m, dates_m = data_daily, dates_daily
            print(f"    Data already monthly, using as-is")
        else:
            data_m, dates_m = _resample_to_monthly(data_daily, dates_daily, loader)
            print(f"    Resampled {len(data_daily)} daily -> {len(data_m)} monthly periods")

        monthly_dir = base_dir / ds_name / "MONTHLY"
        run_single_backtest(
            data_m, dates_m, 12,
            strategies, monthly_dir, f"{ds_name} - Monthly"
        )

        # ====================================================================
        # RUN 2: DAILY (robustness)
        # ====================================================================
        if ppy_orig != 12:
            print(f"\n    RUN 2: Daily (robustness check)")
            print(f"    {'-'*60}")

            daily_dir = base_dir / ds_name / "DAILY"
            run_single_backtest(
                data_daily, dates_daily, ppy_orig,
                strategies, daily_dir, f"{ds_name} - Daily"
            )
        else:
            print(f"\n    Data is already monthly; daily robustness check skipped")

    # Tables use the MONTHLY subfolder as primary source
    try:
        from src.paper_tables import generate_all_tables
        generate_all_tables(base_dir)
        print(f"  Paper tables saved to {base_dir / 'PAPER_TABLES'}")
    except Exception as e:
        print(f"  Paper tables failed: {e}")

    print(f"\n{'='*70}")
    print(f"  DUAL-FREQUENCY ANALYSIS COMPLETE")
    print(f"  Output directory: {base_dir}")
    print(f"{'='*70}\n")


def run_single_backtest(data_matrix: np.ndarray,
                        dates: pd.DatetimeIndex,
                        periods_per_year: int,
                        strategies: List[str],
                        output_dir: Path,
                        label: str) -> pd.DataFrame:
    """
    Run backtest on single frequency.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark returns (EW)
    ew_returns = data_matrix @ (np.ones(data_matrix.shape[1]) / data_matrix.shape[1])

    # Storage
    strategy_returns  = {}
    strategy_weights  = {}
    results_list      = []
    best_params_rows  = []
    srf_lp_diag_rows  = []   # per-window SRF_LP diagnostics
    oos_dates         = None

    # Run each strategy
    import time as _time
    for strat_code in strategies:
        try:
            print(f"    Running {strat_code:<15}", end="", flush=True)

            StrategyClass = STRATEGY_MAP.get(strat_code)
            if not StrategyClass:
                print(f" Not found")
                continue

            strategy = StrategyClass()
            t_start = _time.time()
            engine = WalkForwardEngine(
                type(strategy),
                strat_code,
                periods_per_year=periods_per_year
            )
            results = engine.run(data_matrix, dates)
            elapsed = _time.time() - t_start

            metrics = calculate_metrics(
                returns=results['returns'],
                weights=results['weights'],
                benchmark_returns=ew_returns[len(data_matrix) - len(results['returns']):],
                trials_log=results.get('trials_log'),
                transaction_cost_bps=TRANSACTION_COST_BPS,
                periods_per_year=periods_per_year
            )

            metrics['Strategy'] = strat_code
            metrics['Dataset']  = label
            metrics['Periods_Per_Year'] = periods_per_year

            results_list.append(metrics)
            strategy_returns[strat_code] = results['returns']
            strategy_weights[strat_code] = results['weights']

            # Track OOS dates (same for all strategies)
            if oos_dates is None:
                oos_dates = results['dates']

            # Best-params row for paper tables
            param_row = {
                'Dataset':       label,
                'Strategy':      strat_code,
                'IS_Sharpe':     results.get('best_is_sharpe', float('nan')),
                'N_Trials':      results.get('n_trials', 0),
                'Grid_Search_s': results.get('grid_search_time', 0.0),
            }
            param_row.update(results.get('best_params', {}))
            best_params_rows.append(param_row)

            # Collect SRF_LP per-window OOS diagnostics
            if strat_code == 'SRF_LP':
                oos_strat = results.get('oos_strategy')
                if oos_strat is not None and hasattr(oos_strat, 'get_window_diagnostics'):
                    for step_idx, d in enumerate(oos_strat.get_window_diagnostics()):
                        srf_lp_diag_rows.append({
                            'Dataset': label, 'oos_step': step_idx,
                            **d,
                        })

            sr_val = metrics.get('Sharpe Ratio (Ann.)', metrics.get('Sharpe Ratio', float('nan')))
            print(f" SR={sr_val:.2f}  [{elapsed:.1f}s]")

        except Exception as e:
            print(f" Error: {str(e)[:60]}")

    # -- Save the CSVs from which the tables are built ---------------------
    df_metrics = pd.DataFrame(results_list)
    df_metrics.to_csv(output_dir / "metrics_detailed.csv", index=False)

    if best_params_rows:
        pd.DataFrame(best_params_rows).to_csv(output_dir / "best_params.csv", index=False)

    if strategy_returns and oos_dates is not None:
        try:
            oos_len = min(len(v) for v in strategy_returns.values())
            df_oos = pd.DataFrame(
                {s: r[:oos_len] for s, r in strategy_returns.items()},
                index=oos_dates[:oos_len]
            )
            df_oos.index.name = "Date"
            df_oos.to_csv(output_dir / "oos_returns.csv")
        except Exception as _e:
            print(f"    oos_returns.csv save failed: {_e}")

    # oos_turnover.csv - per-period sum(|Δw|) for each strategy.
    # First period has zero turnover (no prior weights to compare against).
    # Used by table_cost_sensitivity to compute exact per-period cost impact.
    if strategy_weights and oos_dates is not None:
        try:
            oos_len = min(len(v) for v in strategy_weights.values())
            turn_dict: Dict[str, np.ndarray] = {}
            for s, W in strategy_weights.items():
                W_clip = W[:oos_len]
                if W_clip.shape[0] > 1:
                    delta = np.abs(np.diff(W_clip, axis=0)).sum(axis=1)   # (T-1,)
                    turn_dict[s] = np.concatenate([[0.0], delta])          # (T,)
                else:
                    turn_dict[s] = np.zeros(oos_len)
            df_turn = pd.DataFrame(turn_dict, index=oos_dates[:oos_len])
            df_turn.index.name = "Date"
            df_turn.to_csv(output_dir / "oos_turnover.csv")
        except Exception as _e:
            print(f"    oos_turnover.csv save failed: {_e}")

    if srf_lp_diag_rows:
        try:
            srf_diag_dir = Path(output_dir) / "diagnostics" / "srf_lp"
            srf_diag_dir.mkdir(parents=True, exist_ok=True)
            # Use a filename derived from the output_dir to avoid collisions
            safe_label = label.replace(" ", "_").replace("/", "-")
            pd.DataFrame(srf_lp_diag_rows).to_csv(
                srf_diag_dir / f"window_diagnostics_{safe_label}.csv", index=False
            )
        except Exception as _e:
            print(f"    SRF_LP diagnostics save failed: {_e}")

    print(f"      Saved {len(results_list)} results")

    return df_metrics


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main orchestrator."""
    print(f"\n{'='*70}")
    print(f"  PORTFOLIO OPTIMIZATION FRAMEWORK")
    print(f"  21 strategies of the article + SRF_LP (reported as excluded)")
    print(f"{'='*70}")

    # Scan available datasets
    available_files = scan_datasets(DATA_FOLDER)
    print(f"\n  Found {len(available_files)} dataset(s)")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results = Path(f"results/Run_{timestamp}")

    # Mode selection
    print(f"\n{'='*70}")
    print(f"  EXECUTION MODE")
    print(f"{'='*70}")
    print(f"  1. Monthly Backtest       (primary analysis, comparable with the paper)")
    print(f"  2. Dual-Frequency         (Monthly primary + Daily robustness)")
    print(f"{'='*70}")

    mode = input("\nSelect mode (1 or 2): ").strip()

    selected_datasets  = select_datasets(available_files, mode='interactive')
    selected_strategies = select_strategies(mode='interactive')

    print(f"\n{'='*70}")
    print(f"  CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Datasets:   {', '.join([f.stem for f in selected_datasets])}")
    print(f"  Strategies: {', '.join(selected_strategies)}")
    if mode == '2':
        print(f"  Mode:       Dual-Frequency (Monthly + Daily)")
        print(f"  Est. time:  ~{len(selected_datasets) * len(selected_strategies) * 20} min")
    else:
        print(f"  Mode:       Monthly Backtest")
        print(f"  Est. time:  ~{len(selected_datasets) * len(selected_strategies) * 10} min")
    print(f"{'='*70}")

    confirm = input("\nProceed? (Y/n): ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("\n  Cancelled by user")
        return

    if mode == '2':
        run_dual_frequency_analysis(selected_datasets, selected_strategies, base_results)
    else:
        run_monthly_backtest(selected_datasets, selected_strategies, base_results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n  Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
