"""Experimental design: universes, dates, strategy set and paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "snapshot"       # downloaded files (not tracked)
MANIFEST = ROOT / "data" / "MANIFEST.json"  # SHA-256 of the files used in the article
# Run outputs; set the environment variable RESULTS_DIR to write them elsewhere.
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

# -- Download window --------------------------------------------------------
DOWNLOAD_START = "2007-12-01"
DOWNLOAD_END = "2025-12-31"               # inclusive

# -- Universes --------------------------------------------------------------
YAHOO_UNIVERSES = {
    "DJ30": [                                     # Dow Jones 30 without DOW, V and WBA
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
        "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "VZ", "WMT",
    ],
    "Tech": [                                     # without PYPL
        "AAPL", "ADBE", "AMD", "AMZN", "CRM", "CSCO", "GOOGL", "INTC",
        "MSFT", "NFLX", "NVDA",
    ],
    "Global_ETFs": [
        "DBC", "EEM", "EWJ", "GLD", "HYG", "IEF", "IWM", "LQD", "QQQ",
        "SLV", "SPY", "TLT", "VGK", "VNQ",
    ],
    "Countries_Europe": ["EWG", "EWU", "EWQ", "EWI", "EWP", "EWL", "EWN", "EWD"],
    "Countries_APAC_Americas": [
        "EWA", "EWC", "EWJ", "EWH", "EWS", "EWW", "EWZ", "EWY", "EWT",
    ],
    "Commodities": ["GLD", "SLV", "DBC", "USO", "UNG", "DBA", "DBB"],
}

# Kenneth French library: value-weighted industry portfolios and the RF series
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_UNIVERSES = {
    "French_10": {"monthly": "10_Industry_Portfolios_CSV.zip",
                  "daily": "10_Industry_Portfolios_daily_CSV.zip"},
    "French_30": {"monthly": "30_Industry_Portfolios_CSV.zip",
                  "daily": "30_Industry_Portfolios_daily_CSV.zip"},
}
FRENCH_RF = {"monthly": "F-F_Research_Data_Factors_CSV.zip",
             "daily": "F-F_Research_Data_Factors_daily_CSV.zip"}
FRENCH_MISSING = -99.99                   # missing-value code of the library

UNIVERSES = list(YAHOO_UNIVERSES) + list(FRENCH_UNIVERSES)
FREQS = ["MONTHLY", "DAILY"]
PERIODS_PER_YEAR = {"MONTHLY": 12, "DAILY": 252}

# -- Dates common to every universe ----------------------------------------
ESTIMATION_WINDOW = {"MONTHLY": 60, "DAILY": 1260}
CALIB_START, CALIB_END = "2013-01-01", "2018-12-31"
OOS_START, OOS_END = "2019-01-01", "2025-12-31"

# -- Strategies, by code (see src/strategies/__init__.py) -------------------
STRATEGIES = [
    "EW", "GMV", "GMR", "MD",
    "MV", "MSV", "SemiV", "WUBC", "WLBC", "WUBC_Alt",
    "FSD", "SSD",
    "DMV", "DMV_Yager", "DMV_Return", "DMV_Vars", "EWMV",
    "RP", "HRP",
    "CVaR", "MAD",
]

TRANSACTION_COST_BPS = 10.0

# Strategies whose OOS rows are spread over worker processes inside a task.
# MVN (differential evolution) takes seconds per row; the rest take
# milliseconds. Results do not depend on this setting.
ROW_PARALLEL_STRATEGIES = ["MSV"]
