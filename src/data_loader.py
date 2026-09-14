"""
Data loading utilities.

Reads a price panel, resamples it when required, and returns log returns
together with the inferred sampling frequency and the date index.
Uses log returns for portfolio calculations (time-additive property).

References:
    Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). The Econometrics of
    Financial Markets. Princeton University Press.
"""
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, Optional
from functools import wraps

PathLike = Union[str, Path]


def measure_performance(func):
    """
    Benchmark decorator for execution time and memory usage.

    Tracks peak memory via tracemalloc and wall-clock time via perf_counter.

    Returns:
        Tuple of (function_result, metrics_dict)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Ensure clean slate for memory tracking
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()

        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            tracemalloc.stop()
            raise RuntimeError(f"Error in {func.__name__}: {str(e)}") from e

        elapsed_time = time.perf_counter() - start_time

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = {
            'execution_time_sec': elapsed_time,
            'peak_memory_mb': peak_mem / (1024 * 1024),
            'current_memory_mb': current_mem / (1024 * 1024),
            'method': func.__name__
        }

        return result, metrics

    return wrapper


class HybridDataLoader:
    """
    Loads a price panel and returns log returns, sampling frequency and dates.

    Attributes:
        engine: backend used to read the CSV; only 'pandas' is supported
    """

    def __init__(self, engine: str = 'pandas'):
        """
        Initialize data loader with specified engine.

        Args:
            engine: Computational backend selection
        """
        self.engine = engine
        self._validate_engine()

    def _validate_engine(self) -> None:
        """Ensure selected engine is available."""
        if self.engine != 'pandas':
            raise ValueError(f"Unsupported engine '{self.engine}'; only 'pandas' is available")

    @measure_performance
    def load_and_preprocess(self, file_path: PathLike) -> Tuple[Tuple[np.ndarray, int], Optional[pd.DatetimeIndex]]:
        """
        Load CSV file and compute log returns matrix.

        Frequency detection:
            - if the CSV has a date column, the frequency is inferred from it
            - otherwise monthly data are assumed

        Log returns r_t = ln(P_t / P_{t-1}) are used because they are additive
        over time: ln(P_t / P_0) = sum_i r_i.

        Args:
            file_path: Path to CSV (columns=assets, rows=time)

        Returns:
            Tuple of ((returns_matrix, periods_per_year), dates)
            - returns_matrix: Clean returns matrix (T × N)
            - periods_per_year: Detected frequency (252=daily, 12=monthly, etc.)
            - dates: DatetimeIndex or None (can be discarded)

        Raises:
            FileNotFoundError: If CSV doesn't exist
            ValueError: If data contains all NaN after cleaning
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        try:
            log_returns, dates = self._process_pandas(path)

            # Automatic frequency detection
            if dates is not None and len(dates) > 1:
                periods_per_year = self.infer_frequency_from_dates(dates)
            else:
                # Fallback: assume monthly data
                periods_per_year = 12
                warnings.warn(
                    f"No date column detected in {path.name}. "
                    f"Assuming MONTHLY data (12 periods/year). "
                    f"Add a date column for automatic detection."
                )

            return (log_returns, periods_per_year), dates

        except Exception as e:
            import traceback
            print(f"\n  ERROR loading {path.name}: {str(e)}")
            print(traceback.format_exc())
            raise

    def _process_pandas(self, path: Path) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Read the CSV with pandas and compute log returns.
        """
        df = pd.read_csv(path)

        # Detect the date column
        date_col = None
        dates = None

        # 1) look for a date column by name
        date_keywords = ['date', 'time', 'fecha', 'periodo', 'day', 'timestamp', 'dia']
        for col in df.columns:
            if any(kw in str(col).lower() for kw in date_keywords):
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')
                    # accept it if at least half of the values parse as dates
                    if dates.notna().sum() / len(dates) > 0.5:
                        date_col = col
                        print(f"  Detected date column by name: '{col}'")
                        df = df.drop(columns=[col])
                        break
                    else:
                        dates = None
                except Exception:
                    continue

        # 2) try the first column
        if dates is None and df.shape[1] > 1:
            try:
                test_dates = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                # accept it if at least half of the values parse as dates
                if test_dates.notna().sum() / len(test_dates) > 0.5:
                    dates = test_dates
                    first_col_name = df.columns[0]
                    print(f"  Detected date column by position: '{first_col_name}' (first column)")
                    df = df.iloc[:, 1:]
            except Exception:
                pass

        # 3) try every column
        if dates is None:
            for i, col in enumerate(df.columns):
                try:
                    test_dates = pd.to_datetime(df[col], errors='coerce')
                    valid_ratio = test_dates.notna().sum() / len(test_dates)
                    if valid_ratio > 0.5:  # at least half parse as dates
                        dates = test_dates
                        print(f"  Detected date column by content: '{col}' (column {i+1}, {valid_ratio*100:.1f}% valid dates)")
                        df = df.drop(columns=[col])
                        break
                except Exception:
                    continue

        # No date column found: warn and fall back to a sequential index
        if dates is None:
            print(f"  WARNING: No date column detected. Will use sequential index.")
            if len(df) > 0:
                print(f"      CSV columns: {list(df.columns[:5])}...")

        # Robust numeric coercion (handles mixed types)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Remove rows with any NaN
        df.dropna(inplace=True)

        if df.empty:
            raise ValueError(f"No valid data remaining after cleaning in {path}")

        prices = df.values
        epsilon = 1e-10

        # Log returns: r_t = ln(P_t / P_{t-1})
        log_returns = np.log(prices[1:] / (prices[:-1] + epsilon))

        # Adjust dates to match returns
        if dates is not None and len(dates) > 1:
            # Need to align dates with data after dropna
            # If we dropped rows in df, we must drop corresponding dates
            if len(dates) > len(df):
                 dates = dates[df.index]

            dates = dates[1:]  # Skip first date

            # Ensure dates is a DatetimeIndex
            if not isinstance(dates, pd.DatetimeIndex):
                dates = pd.DatetimeIndex(dates)

        return log_returns, dates

    def infer_frequency_from_dates(self, dates: pd.DatetimeIndex) -> int:
        """
        Infer data frequency from datetime index.

        The median gap between consecutive dates is used, so weekends and
        holidays do not distort the estimate.

        Detects:
            - Daily: ~1 day gaps -> 252 periods/year
            - Monthly: ~20-30 day gaps -> 12 periods/year
            - Quarterly: ~90 day gaps -> 4 periods/year
            - Annual: ~365 day gaps -> 1 period/year

        Args:
            dates: Pandas DatetimeIndex or Series

        Returns:
            periods_per_year (int)
        """
        if len(dates) < 2:
            warnings.warn("Insufficient dates to infer frequency, assuming monthly")
            return 12

        # Ensure dates is a DatetimeIndex
        if not isinstance(dates, pd.DatetimeIndex):
            try:
                dates = pd.to_datetime(dates)
                if isinstance(dates, pd.Series):
                    dates = pd.DatetimeIndex(dates)
            except Exception as e:
                print(f"  WARNING: Could not convert dates to DatetimeIndex: {e}")
                return 252  # Default to daily

        # Remove duplicates and sort
        try:
            # Convert to numpy datetime64, unique, then back to DatetimeIndex
            dates_array = dates.values  # numpy array of datetime64
            dates_sorted = np.sort(dates_array)
            # Remove duplicates by checking consecutive differences
            mask = np.concatenate([[True], dates_sorted[1:] != dates_sorted[:-1]])
            dates_unique = dates_sorted[mask]
            dates = pd.DatetimeIndex(dates_unique)
        except Exception as e:
            print(f"  WARNING: Error processing dates: {e}, using original dates")
            # On failure, keep the original dates

        # Calculate gaps
        diffs = dates[1:] - dates[:-1]

        # Convert gaps to days robustly
        try:
            # Method 1: If it's a TimedeltaIndex, use .days
            if hasattr(diffs, 'days'):
                gap_days = diffs.days
            # Method 2: If it's a Series, convert each element
            elif isinstance(diffs, pd.Series):
                gap_days = np.array([d.days if hasattr(d, 'days') else float(d) / 86400e9 for d in diffs])
            # Method 3: Convert to total seconds and divide
            else:
                gap_days = np.array([d.total_seconds() / 86400 for d in diffs])
        except Exception as e:
            print(f"  WARNING: Error converting timedeltas to days: {e}")
            print(f"      Type of diffs: {type(diffs)}")
            print(f"      Assuming DAILY (252 periods/year)")
            return 252

        # Convert to numpy array if not already
        if not isinstance(gap_days, np.ndarray):
            gap_days = np.array(gap_days)

        # Remove invalid values (NaN, negative, zero)
        gap_days = gap_days[~np.isnan(gap_days)]
        gap_days = gap_days[gap_days > 0]

        # Safety check
        if len(gap_days) == 0:
            print(f"  WARNING: No valid date gaps, assuming DAILY (252 periods/year)")
            return 252

        # Calculate median
        median_gap = np.median(gap_days)

        # Safety check for NaN
        if np.isnan(median_gap):
            print(f"  WARNING: median_gap is NaN, assuming DAILY (252 periods/year)")
            return 252

        # Decision thresholds with some tolerance
        if median_gap <= 3:
            freq = 252  # Daily (allowing for weekends)
            freq_name = "DAILY"
        elif median_gap <= 45:
            freq = 12   # Monthly
            freq_name = "MONTHLY"
        elif median_gap <= 120:
            freq = 4    # Quarterly
            freq_name = "QUARTERLY"
        else:
            freq = 1    # Annual
            freq_name = "ANNUAL"

        print(f"  Frequency detected: {freq_name} (median gap: {median_gap:.1f} days, n={len(dates)} obs)")

        # More than 200 observations cannot be annual data: treat them as daily
        if freq == 1 and len(dates) > 200:
            print(f"  {len(dates)} observations detected as annual; treating them as daily")
            freq = 252

        return freq

    def resample_to_monthly(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Convert daily prices to monthly (end-of-month).

        Args:
            daily_prices: DataFrame with DatetimeIndex

        Returns:
            Monthly prices (end-of-month observations)
        """
        # Ensure datetime index exists
        if not isinstance(daily_prices.index, pd.DatetimeIndex):
            warnings.warn("No datetime index found, creating synthetic dates")
            daily_prices.index = pd.date_range(
                end=pd.Timestamp.today(),
                periods=len(daily_prices),
                freq='B'  # Business days
            )

        # Resample to month-end
        monthly_prices = daily_prices.resample('M').last()

        # Remove any resulting NaNs
        monthly_prices = monthly_prices.dropna()

        return monthly_prices
