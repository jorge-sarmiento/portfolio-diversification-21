# Portfolio Diversification Revisited

Replication code for:

> Sarmiento-Laya, J., & Fernández-Navarro, F. *Portfolio Diversification Revisited: An
> Out-of-Sample Comparison of Twenty-One Strategies Across Asset Universes and Rebalancing
> Frequencies.* Manuscript submitted to Mathematics, 2026.

Twenty-one portfolio optimization strategies are evaluated out of sample on three asset
universes (Dow Jones 30, Global ETFs, Technology equities) at monthly and daily rebalancing
frequencies, under a single walk-forward protocol: Ledoit-Wolf shrinkage on a rolling
estimation window, in-sample grid search, transaction costs of 10 basis points, and the
Deflated Sharpe Ratio as the multiple-testing correction.

Running `main.py`, followed by the scripts listed below, reproduces every table and figure
of the article.

---

## Quick start

```bash
conda env create -f environment.yml     # exact solver stack used in the article
conda activate portfolio-21
# place the three price CSV files in data/ first (see data/README.md)
python main.py                          # menu: mode 2, ALL datasets, ALL strategies
```

Reproducing all three universes at both frequencies takes about six hours on a desktop
machine. Results are written to `results/Run_<timestamp>/`.

To check the installation first, answer the menu with `1`, `1`, `EW,GMV,MV` and `Y`: this
runs three strategies on the Dow Jones 30 monthly panel in about a minute.

Once the full run has finished:

```bash
python scripts/dominance_lp_tables.py --run-dir results/Run_<timestamp>          # Tables 8 and 10
python scripts/build_paper_tables.py  --run-dir results/Run_<timestamp> --check  # Tables 1-7, 9, 10
python scripts/verify_reproduction.py --run-dir results/Run_<timestamp>
```

`build_paper_tables.py` writes the tables in the numbering of the article to
`results/Run_<timestamp>/PAPER_TABLES/paper/` and, with `--check`, compares every cell with
the value printed in the article. `verify_reproduction.py` compares the run with the
reference outputs in `archive/`.

Figure 1 is computed from the price data alone:

```bash
python scripts/lambda_sweep.py          # out-of-sample Sharpe ratio of MV over the λ grid
python scripts/plot_figure1.py          # draws Figure 1 from archive/LAMBDA_SWEEP/
```

`plot_figure1.py --curves-dir results/lambda_sweep` draws it from the curves just computed.

---

## Strategies

`Code` is the identifier used in the output files; `Paper` is the label used in the article.

| Block | Paper | Code | Objective | Solver |
|-------|-------|------|-----------|--------|
| 1 Benchmarks | EW | `EW` | Eq. (3), 1/N | closed form |
| 1 Benchmarks | GMR | `GMR` | Eq. (4), argmax of μ | closed form |
| 2 Mean–variance and extensions | GMV | `GMV` | Eq. (5) | CLARABEL |
| 2 Mean–variance and extensions | MV | `MV` | Eq. (6) | CLARABEL |
| 2 Mean–variance and extensions | SemiV | `SemiV` | Eqs. (7)–(8), semicovariance | CLARABEL |
| 2 Mean–variance and extensions | MVN\* | `MSV` | Eq. (9), non-convex | differential evolution |
| 3 Alternative risk measures | CVaR | `CVaR` | Eq. (11) | HiGHS |
| 3 Alternative risk measures | MAD | `MAD` | Eq. (10) | HiGHS |
| 4 Weight constraints | WLBC | `WLBC` | Eq. (12) | CLARABEL |
| 4 Weight constraints | WUBC | `WUBC` | Eq. (13) | CLARABEL |
| 4 Weight constraints | WCMV | `WUBC_Alt` | Eq. (14) | CLARABEL |
| 5 Diversification incentives | MaxDiv | `MD` | Eq. (15) | trust-constr |
| 5 Diversification incentives | DMV | `DMV` | Eq. (16) | CLARABEL |
| 5 Diversification incentives | DMVY | `DMV_Yager` | Eq. (17), L1 penalty | CLARABEL |
| 5 Diversification incentives | DMVR | `DMV_Return` | Eq. (18) | CLARABEL |
| 5 Diversification incentives | DMVV | `DMV_Vars` | Eq. (19) | CLARABEL |
| 5 Diversification incentives | EWMV | `EWMV` | Eq. (20), ex-post mixture | CLARABEL |
| 6 Stochastic dominance | P-Dom | `FSD` | Eq. (21) | HiGHS |
| 6 Stochastic dominance | SSD-R | `SSD` | Eq. (22), sorted cumsum | HiGHS |
| 7 Risk-based allocation | RP | `RP` | Eq. (23), equal risk contribution | trust-constr |
| 7 Risk-based allocation | HRP | `HRP` | hierarchical clustering | hierarchical procedure |

A twenty-second strategy, the soft return-floor LP (`SRF_LP`), is computed by the pipeline
but excluded from the results of the article, where it is reported as a negative result
(Section 6.3). Its per-window diagnostics are written to
`<dataset>/{MONTHLY,DAILY}/diagnostics/srf_lp/` inside the run directory.

\* **MVN.** Eq. (9) is solved with SciPy's differential evolution over the box [0, 1]^N; the
budget constraint is imposed by normalising the solution after the search.

---

## Data

The price panels are **not** redistributed here: they come from Yahoo Finance, whose terms do
not allow redistribution. `data/README.md` lists the three files, their SHA-256 hashes and the
format expected by the pipeline.

- **Snapshot used in the article** (required to reproduce the published numbers): available
  on reasonable request from the corresponding author, Jorge Sarmiento-Laya (jrsarmientolaya@al.uloyola.es).
- **Re-download** (exploratory only): `python download_data.py` fetches the same tickers and
  date ranges, but Yahoo revises adjusted closing prices retroactively, so the results will
  differ from the published ones.

Sample period: 2019-03-20 to 2023-12-29 (DJ30, 29 assets) and 2015-01-02 to 2023-12-29
(ETFs, 14 assets; Tech, 12 assets). WBA was dropped from the Dow Jones 30 universe because of
incomplete price data.

---

## Environment

| | Exact reproduction | Approximate |
|---|---|---|
| Install | `conda env create -f environment.yml` | `pip install -r requirements.txt` |
| BLAS | MKL | OpenBLAS (PyPI wheels) |
| Agreement with `archive/` | max\|Δ\| = 0 | differences of order 1e-6 in returns |

Python 3.13, CLARABEL 0.11.1 for convex quadratic programs, HiGHS 1.8.0 (bundled in SciPy
1.16.3) for linear programs, and SciPy `differential_evolution` and `trust-constr` for the
non-convex problems. The stochastic components (MVN and the RP warm start) use seed 42.

---

## What each artefact of the paper comes from

| Artefact | Produced by | File |
|----------|-------------|------|
| **Tables 1–7, 9, 10, in the numbering of the article** | `scripts/build_paper_tables.py` | `PAPER_TABLES/paper/` |
| Table 1, dataset characteristics | `main.py` | `PAPER_TABLES/Table_4_1_dataset_summary.csv` |
| Table 1, skewness and kurtosis | `scripts/build_paper_tables.py` | `PAPER_TABLES/paper/T1.csv` |
| Table 2, calibration grids | `main.py` | `PAPER_TABLES/Table_3_2_3_param_grids.csv` |
| Table 3, monthly performance | `main.py` | `PAPER_TABLES/Table_5_1_oos_<dataset>.csv` |
| Table 4, robustness (DSR, HHI, turnover) | `main.py` | `Table_5_4_dsr.csv`, `Table_5_2_concentration.csv`, `Table_5_1_oos_<dataset>.csv` |
| Table 5, cost sensitivity | `main.py` | `PAPER_TABLES/Table_5_7_cost_sensitivity.csv` |
| Table 6, daily performance | `main.py` | `<dataset>/DAILY/metrics_detailed.csv` |
| Table 7, frequency sensitivity | `scripts/build_paper_tables.py` | `PAPER_TABLES/paper/T7.csv` |
| Table 8, LP infeasibility rates | `scripts/dominance_lp_tables.py` | `PAPER_TABLES/dominance_lp/infeasibility_rates.csv` |
| Table 9, selected hyperparameters | `main.py` | `<dataset>/MONTHLY/best_params.csv` |
| Table 10, SSD sorting correction | `scripts/dominance_lp_tables.py` | `PAPER_TABLES/dominance_lp/ssd_hhi_correction.csv` |
| Figure 1, MV Sharpe vs λ | `scripts/lambda_sweep.py`, `scripts/plot_figure1.py` | `figures/lambda_curve_*.png` |
| Section 6.3, return-floor results | `main.py`, `scripts/dominance_lp_tables.py` | `diagnostics/srf_lp/`, `infeasibility_rates.csv` (HSD) |

---

## Output structure

```
results/Run_<timestamp>/
  {Dow_Jones_30,Global_ETFs,Tech_Sector}/
    {MONTHLY,DAILY}/
      best_params.csv          in-sample selected hyperparameters
      oos_returns.csv          out-of-sample return series per strategy
      oos_turnover.csv         out-of-sample turnover per strategy
      metrics_detailed.csv     performance metrics per strategy
      diagnostics/srf_lp/      per-window diagnostics of SRF_LP
  PAPER_TABLES/                tables built from the run
    dominance_lp/              Tables 8 and 10
    paper/                     tables in the numbering of the article
```

`archive/` holds the outputs of the reference run (`Run_20260716_190431`) used in the
article, the λ-sweep curves behind Figure 1 (`LAMBDA_SWEEP/`), and the values printed in the
article (`published_tables.json`).

---

## Protocol

- 70/30 in-sample / out-of-sample split, walk-forward with one rebalancing step per period.
- Estimation window: 12 months (DJ30), 14 months (ETFs, Tech), 252–316 trading days at daily
  frequency.
- μ and Σ are annualised (×12 monthly, ×252 daily); Σ is shrunk with Ledoit-Wolf towards the
  scaled identity target, in calibration and in evaluation alike.
- Hyperparameters are selected once, by in-sample Sharpe ratio over a rolling mini-backtest,
  and held fixed throughout the out-of-sample phase. MVN is calibrated statically over the
  full in-sample segment, as stated in the article.
- Portfolio returns are the linear combination of log returns; costs of 10 bps are deducted
  period by period from turnover.

---

## License

MIT for the code, see [LICENSE](LICENSE). The price data is not distributed with this
repository; see `data/README.md`.
