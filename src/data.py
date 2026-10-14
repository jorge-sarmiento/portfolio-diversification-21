"""
Data layer.

build_snapshot()   downloads Yahoo adjusted closes and the Kenneth French files
                   into data/snapshot/.
verify_snapshot()  compares the files with the SHA-256 manifest of the data used
                   in the article (data/MANIFEST.json).
load_panel()       returns simple returns (T x N) and the per-period risk-free
                   rate, both in decimals and on the same dates.

Returns are SIMPLE returns throughout. French files come in percent and code
missing values as -99.99; both are handled here.
"""
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src import config as C


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_snapshot(data_dir: Path = C.DATA_DIR, force: bool = False) -> None:
    """Download every input into data_dir; files already present are kept unless force is set."""
    import requests
    import yfinance as yf

    data_dir.mkdir(parents=True, exist_ok=True)
    end_exclusive = (pd.Timestamp(C.DOWNLOAD_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for name, tickers in C.YAHOO_UNIVERSES.items():
        out = data_dir / f"prices_{name}.csv"
        if out.exists() and not force:
            continue
        raw = yf.download(tickers, start=C.DOWNLOAD_START, end=end_exclusive,
                          auto_adjust=True, progress=False)
        prices = raw["Close"][tickers].dropna(how="all")
        prices.index.name = "Date"
        prices.to_csv(out)
        print(f"  {name}: {prices.shape[0]} rows x {prices.shape[1]} tickers")

    files = [f for u in C.FRENCH_UNIVERSES.values() for f in u.values()] + list(C.FRENCH_RF.values())
    for fname in files:
        out = data_dir / fname
        if out.exists() and not force:
            continue
        r = requests.get(C.FRENCH_BASE_URL + fname, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"  {fname}: {len(r.content)} bytes")


def verify_snapshot(data_dir: Path = C.DATA_DIR, manifest: Path = C.MANIFEST) -> Dict[str, bool]:
    """True for every file of the manifest that is present with the recorded hash."""
    files = json.loads(manifest.read_text())["files"]
    return {name: (data_dir / name).exists() and sha256(data_dir / name) == meta["sha256"]
            for name, meta in files.items()}


def have_snapshot(data_dir: Path = C.DATA_DIR, manifest: Path = C.MANIFEST) -> bool:
    """Whether every data file listed in the manifest is present."""
    if not manifest.exists():
        return False
    return all((data_dir / name).exists() for name in json.loads(manifest.read_text())["files"])


# ---------------------------------------------------------------------------
# Kenneth French files
# ---------------------------------------------------------------------------

def read_french(path: Path, daily: bool) -> pd.DataFrame:
    """
    First data block of a Kenneth French CSV, as decimals with NaN for -99.99.

    For the industry files the first block holds the value-weighted returns;
    for the factor files it holds Mkt-RF, SMB, HML and RF.
    """
    with zipfile.ZipFile(path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    pat = re.compile(r"^\s*\d{8}\s*," if daily else r"^\s*\d{6}\s*,")
    i0 = next(i for i, ln in enumerate(lines) if pat.match(ln))
    i1 = next((i for i in range(i0, len(lines)) if not pat.match(lines[i])), len(lines))
    header = [h.strip() for h in lines[i0 - 1].split(",")]
    block = next((lines[j].strip() for j in range(i0 - 2, max(i0 - 6, -1), -1)
                  if lines[j].strip()), "")
    if "Industry" in path.name and "Value Weighted" not in block:
        raise ValueError(f"{path.name}: first block is not value-weighted ({block!r})")

    df = pd.read_csv(io.StringIO("\n".join(lines[i0:i1])), header=None)
    df.columns = ["Date"] + header[1:]
    dates = pd.to_datetime(df["Date"].astype(str).str.strip(),
                           format="%Y%m%d" if daily else "%Y%m")
    if not daily:
        dates = dates + pd.offsets.MonthEnd(0)
    df = df.drop(columns="Date").apply(pd.to_numeric, errors="coerce")
    df.index = pd.DatetimeIndex(dates, name="Date")
    df = df.mask(df <= C.FRENCH_MISSING)          # -99.99 (and -999) -> NaN
    return df / 100.0                              # percent -> decimal


def load_rf(freq: str, data_dir: Path = C.DATA_DIR, daily: str = "monthly_over_sessions") -> pd.Series:
    """
    Per-period risk-free rate of the French library, in decimals.

    Monthly: the RF column of the monthly factor file.

    Daily: the monthly RF divided by the number of sessions of that month,
    without rounding, so the daily rates of a month add up to its monthly rate.
    The RF column of the daily file is not used: it is published with two
    decimals of a percent (only 0.00, 0.01 and 0.02 in 2008-2025), which
    overstates the mean rate of 2019-2025 by 0.19 points a year. Pass
    daily="published" to read that column as it is.
    """
    monthly = read_french(data_dir / C.FRENCH_RF["monthly"], False)["RF"]
    if freq != "DAILY":
        return monthly
    published = read_french(data_dir / C.FRENCH_RF["daily"], True)["RF"]
    if daily == "published":
        return published
    if daily != "monthly_over_sessions":
        raise ValueError(daily)
    month_end = published.index + pd.offsets.MonthEnd(0)
    sessions = pd.Series(1, index=month_end).groupby(level=0).transform("sum").values
    rf = pd.Series(monthly.reindex(month_end).values / sessions, index=published.index, name="RF")
    if rf.isna().any():
        raise ValueError("daily sessions without a monthly RF value")
    return rf


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _yahoo_returns(name: str, freq: str, data_dir: Path) -> pd.DataFrame:
    prices = pd.read_csv(data_dir / f"prices_{name}.csv", index_col="Date", parse_dates=True)
    prices = prices[C.YAHOO_UNIVERSES[name]]
    if freq == "MONTHLY":
        prices = prices.resample("ME").last()
    return prices.pct_change().iloc[1:]            # simple returns


def load_panel(name: str, freq: str, data_dir: Path = C.DATA_DIR
               ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Simple returns of a universe and the risk-free rate on the same dates.

    Monthly panels start in 2008-01 (first full month after the download
    start); daily panels start on the second session of the download window.
    Rows are restricted to dates present in both the universe and the RF file.
    """
    if name in C.YAHOO_UNIVERSES:
        rets = _yahoo_returns(name, freq, data_dir)
    else:
        daily = freq == "DAILY"
        rets = read_french(data_dir / C.FRENCH_UNIVERSES[name]["daily" if daily else "monthly"], daily)

    first = pd.Timestamp("2008-01-01") if freq == "MONTHLY" else pd.Timestamp(C.DOWNLOAD_START)
    rets = rets[(rets.index >= first) & (rets.index <= pd.Timestamp(C.DOWNLOAD_END))]
    if freq == "DAILY" and name not in C.YAHOO_UNIVERSES:
        # same first session as the Yahoo panels (their first return is the
        # second session of the download window)
        rets = rets.iloc[1:]

    rf = load_rf(freq, data_dir)
    common = rets.index.intersection(rf.index)
    return rets.loc[common], rf.loc[common]


def schedule_indices(dates: pd.DatetimeIndex, freq: str) -> Dict:
    """
    Row positions of the design on a given date index.

    Every evaluated row t uses the ESTIMATION_WINDOW rows immediately before
    it, rows [t - W, t), so the estimation window always ends one period
    before the period it evaluates.
    """
    w = C.ESTIMATION_WINDOW[freq]
    pos = np.arange(len(dates))
    calib = pos[(dates >= C.CALIB_START) & (dates <= C.CALIB_END)]
    oos = pos[(dates >= C.OOS_START) & (dates <= C.OOS_END)]
    if calib[0] < w:
        raise ValueError(f"{freq}: first calibration row {calib[0]} has fewer than {w} prior rows")
    return {"window": w, "calib_idx": calib, "oos_idx": oos, "static_end": int(oos[0])}
