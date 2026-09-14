# Price data

The price panels are not redistributed here: they come from Yahoo Finance, whose terms do
not allow redistribution. Place the three CSV files in this folder before running the
pipeline.

| File | Assets | Daily obs. | SHA-256 |
|------|--------|-----------|---------|
| `Dow_Jones_30.csv` | 29 | 1205 | `A2ABFF69E358E09EC0FA3AC0B549C1396FEBE344DF491F9F2BF6874BC0109100` |
| `Global_ETFs.csv` | 14 | 2264 | `2E86DE65D5891CEEA9CA221D44E71B77C076E08183D28D72B692943A69FD0E68` |
| `Tech_Sector.csv` | 12 | 2138 | `203AF671E5858056F7D782FD5D93E138DB8F2F78A7DFE4798D81B257FFE863A5` |

Format: one CSV per universe, a `Date` column plus one column of adjusted closing prices
per ticker, daily frequency.

## Option A: the archived snapshot (required to reproduce the published numbers)

The snapshot used for the results in the article is available on reasonable request from
the corresponding author, Jorge Sarmiento-Laya (jrsarmientolaya@al.uloyola.es). Verify the
files against the hashes above, for example:

```bash
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" data/Dow_Jones_30.csv
```

## Option B: re-download (exploratory only)

```bash
pip install yfinance
python download_data.py
```

This downloads the same tickers and date ranges, but Yahoo Finance revises adjusted closing
prices retroactively whenever a corporate action is applied, so the result will differ from
the snapshot and will not reproduce the published tables.
