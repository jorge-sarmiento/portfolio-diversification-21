"""
Re-download the price panels from Yahoo Finance.

The snapshot used in the article is not distributed with the repository (see
data/README.md); this script is for running the pipeline on more recent prices.
It will NOT reproduce the published numbers: Yahoo revises adjusted closes
retroactively whenever a corporate action (dividend, split, spin-off) is applied
to historical records.

SHA-256 of the snapshot used in the article (hex, case-insensitive):
  Dow_Jones_30.csv : A2ABFF69E358E09EC0FA3AC0B549C1396FEBE344DF491F9F2BF6874BC0109100
  Global_ETFs.csv  : 2E86DE65D5891CEEA9CA221D44E71B77C076E08183D28D72B692943A69FD0E68
  Tech_Sector.csv  : 203AF671E5858056F7D782FD5D93E138DB8F2F78A7DFE4798D81B257FFE863A5

Requirements: yfinance (pip install yfinance==1.2.0)

Usage:
    python download_data.py [--out-dir data]
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

# -- Dataset definitions (exact tickers and date ranges from the paper) --------

DATASETS = {
    "Dow_Jones_30": {
        "tickers": [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
            "DIS", "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM",
            "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV",
            "UNH", "V", "VZ", "WMT",
        ],
        "start": "2019-03-01",
        "end":   "2023-12-31",
    },
    "Global_ETFs": {
        "tickers": [
            "DBC", "EEM", "EWJ", "GLD", "HYG", "IEF", "IWM", "LQD",
            "QQQ", "SLV", "SPY", "TLT", "VGK", "VNQ",
        ],
        "start": "2015-01-01",
        "end":   "2023-12-31",
    },
    "Tech_Sector": {
        "tickers": [
            "AAPL", "ADBE", "AMD", "AMZN", "CRM", "CSCO", "GOOGL",
            "INTC", "MSFT", "NFLX", "NVDA", "PYPL",
        ],
        "start": "2015-07-01",
        "end":   "2023-12-31",
    },
}

ARCHIVED_SHA256 = {
    "Dow_Jones_30.csv": "a2abff69e358e09ec0fa3ac0b549c1396febe344df491f9f2bf6874bc0109100",
    "Global_ETFs.csv":  "2e86de65d5891ceea9ca221d44e71b77c076e08183d28d72b692943a69fd0e68",
    "Tech_Sector.csv":  "203af671e5858056f7d782fd5d93e138db8f2f78a7dfe4798d81b257ffe863a5",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(name: str, cfg: dict, out_dir: Path) -> None:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance is required: pip install yfinance==1.2.0")

    print(f"Downloading {name} ({len(cfg['tickers'])} tickers, {cfg['start']} to {cfg['end']})...")
    raw = yf.download(
        cfg["tickers"],
        start=cfg["start"],
        end=cfg["end"],
        auto_adjust=True,
        progress=False,
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices[cfg["tickers"]]          # enforce column order
    prices = prices.dropna(how="all")

    out_path = out_dir / f"{name}.csv"
    prices.to_csv(out_path)
    print(f"  Saved {len(prices)} rows -> {out_path}")

    h = sha256(out_path)
    expected = ARCHIVED_SHA256.get(f"{name}.csv", "")
    if h.lower() == expected:
        print(f"  SHA-256 MATCHES archived snapshot: exact reproduction is possible.")
    else:
        print(f"  SHA-256 DIFFERS from archived snapshot (expected {expected[:16]}...).")
        print(f"  Numerical results will differ from the paper. This is expected when")
        print(f"  downloading after Yahoo Finance has revised its historical prices.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download price data for the paper pipeline")
    ap.add_argument("--out-dir", default="data", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("WARNING: see module docstring re retroactive Yahoo adjustments")
    print("=" * 60)
    for name, cfg in DATASETS.items():
        download_dataset(name, cfg, args.out_dir)
    print("\nDone. Place CSV files in data/ before running main.py.")


if __name__ == "__main__":
    main()
