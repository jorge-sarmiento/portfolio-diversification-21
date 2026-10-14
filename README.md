# Portfolio Diversification Revisited

Replication code for:

> Sarmiento-Laya, J. R., & Fernández-Navarro, F. *Portfolio Diversification Revisited: An
> Out-of-Sample Comparison of Twenty-One Strategies Across Asset Universes and Rebalancing
> Frequencies.* Mathematics, 2026.

## What the code replicates

Twenty-one portfolio strategies are evaluated out of sample on eight asset universes, at
monthly and daily rebalancing frequency, under one walk-forward protocol:

- simple returns; Ledoit-Wolf covariance on a rolling estimation window of 60 months
  (1,260 sessions at daily frequency);
- hyperparameters selected once on 2013-01 to 2018-12 by the Sharpe ratio of a rolling
  mini-backtest, then held fixed; MVN, solved by differential evolution, is calibrated
  instead by its in-sample Sharpe ratio on all the data before 2019;
- out-of-sample period 2019-01 to 2025-12 (84 months, 1,760 sessions);
- transaction costs of 10 basis points per unit of turnover, measured against the weights
  drifted by prices, deducted period by period;
- Sharpe ratios on net returns in excess of the risk-free rate;
- joint inference against the equally weighted portfolio (stationary bootstrap, SPA-type
  test, Romano-Wolf adjusted p-values), rank summaries across universes and the Deflated
  Sharpe Ratio.

| Universe | Assets | Source |
|---|---|---|
| Dow Jones 30 | 27 stocks (without DOW, V and WBA) | Yahoo Finance |
| Tech Sector | 11 stocks (without PYPL) | Yahoo Finance |
| Global ETFs | 14 multi-asset ETFs | Yahoo Finance |
| Countries: Europe | 8 country ETFs | Yahoo Finance |
| Countries: Asia-Pacific and Americas | 9 country ETFs | Yahoo Finance |
| Commodities | 7 commodity ETFs | Yahoo Finance |
| 10 Industries | 10 value-weighted industry portfolios | Kenneth French data library |
| 30 Industries | 30 value-weighted industry portfolios | Kenneth French data library |

All series are in US dollars, from January 2008 to December 2025; an asset is included only
if it has a complete history over that period. The tickers are listed in `src/config.py`.

### Strategies

| Code in the output files | Name in the article |
|---|---|
| EW, GMV, GMR | EW, GMV, GMR |
| MD | MaxDiv |
| MV, SemiV | MV, SemiV |
| MSV | MVN |
| WLBC, WUBC | WLBC, WUBC |
| WUBC_Alt | WCMV |
| DMV, EWMV | DMV, EWMV |
| DMV_Yager, DMV_Return, DMV_Vars | DMVY, DMVR, DMVV |
| FSD, SSD | P-Dom, SSD-R |
| CVaR, MAD | CVaR, MAD |
| RP, HRP | RP, HRP |

MVN minimises `λ w'Σw − (1 − λ)(w'μ)²` subject to `Σw = 1`, `w ≥ 0`, where `μ` and `Σ` are the
annualised mean and covariance of the returns in excess of the risk-free rate. The problem
is non-convex and is solved by differential evolution; each candidate is mapped to the
simplex by Euclidean projection (`src/strategies/markowitz.py`).

## Revised version

This version of the code accompanies the revised manuscript. The changes below were made in
response to the peer review of the first submission. The code and reference outputs of the
original submission remain available at commit `f6b967a`.

| What changed | Why |
|---|---|
| Eight asset universes instead of three, all observed from January 2008 to December 2025: country ETFs for Europe and for Asia-Pacific and the Americas, commodity ETFs, and the Kenneth French 10 and 30 industry portfolios were added | The reviewers asked for more asset classes (international equities, sector portfolios, commodities) and a longer sample; the industry portfolios are free of survivorship bias by construction |
| Estimation window of 60 months (1,260 sessions) instead of 12, and the same calendar in every universe: calibration 2013–2018, out of sample 2019–2025 | The original samples gave between 18 and 33 monthly out-of-sample observations, set by the listing date of a few assets, and 12 months is too short to estimate the mean vector; every universe now has 84 monthly and 1,760 daily out-of-sample observations |
| Simple returns instead of log returns | The weighted sum of log returns only approximates the portfolio return; with simple returns `w'r` is exact |
| Sharpe ratios on returns in excess of the one-month US Treasury bill rate, in the evaluation, in the selection of hyperparameters and in the Deflated Sharpe Ratio | A zero risk-free rate lowers each Sharpe ratio by `rf/σ`, which depends on the volatility of the strategy, so it can change the ranking |
| Turnover measured against the weights drifted by prices, with costs deducted period by period | With simple returns the weights drift between rebalancing dates, and the trades needed are those that restore the target weights |
| Joint test of the twenty strategies against the equally weighted portfolio (stationary bootstrap, SPA-type test, Romano-Wolf adjusted p-values) | The Deflated Sharpe Ratio only corrects the selection within each strategy; the reviewers asked for a formal joint test across strategies |
| Comparison across universes by ranks instead of the average Sharpe ratio | An average of Sharpe ratios across universes does not correspond to any portfolio an investor could hold |
| Weight-bound combinations that admit no portfolio for a given number of assets are removed from the calibration grids | A combination with `N·l > 1` or `N·u < 1` has no feasible portfolio and must not compete in the calibration or count as a trial of the Deflated Sharpe Ratio |
| MVN: each candidate of the differential evolution is projected onto the simplex, and the mean and covariance are those of excess returns | The original code searched the box [0, 1]^N and normalised the solution afterwards; projecting every candidate keeps the search on the feasible set of Eq. (9), and excess-return inputs follow Carbonero-Ruz et al. (2025) |
| The soft return-floor LP, which the original code ran as a twenty-second strategy, is now computed only in `scripts/diagnostics/return_floor.py`, together with the hard floor | The article reports the return-floor family as a negative result and does not include it among the twenty-one strategies compared |
| Scripts for the diagnostics of the article (Figure 1, Tables 10 and 11 and others), a test suite, and a data manifest with SHA-256 hashes | So that every result of the article can be reproduced and checked |

## Repository structure

```
run.py                    command-line entry point: data, full run
src/
  config.py               universes, dates, strategy set, paths
  data.py                 download, manifest check, returns and risk-free rate
  engine.py               walk-forward engine: calibration and out-of-sample loop
  runner.py               tasks, process parallelism, checkpoints
  costs.py                turnover against drifted weights and transaction costs
  grid.py                 feasibility of the weight-bound combinations of the grids
  metrics.py              Deflated Sharpe Ratio
  tables.py               metric tables of a run
  inference.py            joint inference against EW
  summary.py              rank summaries across universes
  strategies/             the 21 strategies
scripts/
  postprocess.py          tables, inference, ranks and LaTeX tables of a run
  tables/                 one script per LaTeX table
  audit_fallbacks.py      dates on which a strategy holds 1/N
  inference_size.py       size of the joint test at the observed serial correlation
  check_mvn_solver.py     MVN solver against a parametric search on the efficient frontier
  diagnostics/            diagnostics reported in the article (Figure 1, Tables 10 and 11, and others)
tests/                    unit and integration tests
data/MANIFEST.json        SHA-256 of the data files used in the article
```

## Data

- Yahoo Finance: daily closes adjusted for dividends and splits (`yfinance`,
  `auto_adjust=True`), 2007-12-01 to 2025-12-31. Monthly returns use the last close of
  each month.
- Kenneth French data library: value-weighted 10 and 30 industry portfolios, monthly and
  daily, and the risk-free rate. The daily risk-free rate is the monthly rate divided by
  the number of sessions of the month.

`python run.py snapshot` downloads every file into `data/snapshot/`. The price files are
not redistributed in this repository. `data/MANIFEST.json` records the SHA-256 of the
files used in the article and `python run.py verify` compares a download against it.
Adjusted prices downloaded at a later date can differ from the frozen files, because
Yahoo Finance restates them after each dividend; results then differ accordingly. The data
snapshot used for the reported results is available from the corresponding author upon
reasonable request.

## Installation

```bash
conda env create -f environment.yml
conda activate portfolio-21
```

or, with pip:

```bash
pip install -r requirements.txt
```

`environment.yml` pins the solver stack used for the article (Python 3.13, NumPy and SciPy
linked to MKL). `requirements.txt` lists the same package versions for `pip`; those wheels
link OpenBLAS, and results agree to about 1e-6.

## Reproducing the results

```bash
python run.py snapshot                  # download the data
python run.py verify                    # compare it with data/MANIFEST.json
python run.py run                       # full experiment, outputs in results/main/
python scripts/postprocess.py           # tables, inference, ranks, LaTeX tables
python scripts/inference_size.py        # size of the joint test
python scripts/check_mvn_solver.py      # MVN solver against the efficient-frontier search
```

`python run.py run --quick` runs the same pipeline on a few rows (about ten minutes) and
`python -m unittest discover -s tests -t .` runs the tests. A run that is interrupted
resumes from its checkpoints when the command is repeated. Set the environment variable
`RESULTS_DIR` to write the outputs outside the repository folder.

For each universe and frequency, `results/main/<universe>/<FREQUENCY>/` holds the gross and
net out-of-sample return series, turnover, weights, selected hyperparameters and
diagnostics of every strategy. `scripts/postprocess.py` writes:

| Output in `results/main/latex/` | Content | Script in `scripts/tables/` |
|---|---|---|
| `tab_datasets.tex` | Universes, sample sizes and periods | `tab_datasets.py` |
| `tab_performance_{monthly,daily}.tex` | Net Sharpe ratio by strategy and universe | `tab_performance.py` |
| `tab_robustness_{monthly,daily}.tex` | Turnover, concentration, Deflated Sharpe Ratio | `tab_robustness.py` |
| `tab_costs_{monthly,daily}.tex` | Sensitivity to transaction costs | `tab_costs.py` |
| `tab_hyperparams_{monthly,daily}.tex` | Selected hyperparameters and effective grids | `tab_hyperparams.py` |
| `tab_spearman.tex` | Rank correlation between monthly and daily results | `tab_spearman.py` |
| `tab_inference_{monthly,daily}.tex` | Joint inference against EW | `tab_inference.py` |
| `tab_ranks_{monthly,daily}.tex` | Ranks within universes and mean ranks | `tab_ranks.py` |
| `tab_fallbacks.tex` | Optimiser failures and dates at 1/N | `tab_fallbacks.py` |
| `tab_compute.tex` | Computing cost | `tab_compute.py` |
| `tab_main_ranks.tex`, `tab_main_inference.tex` | Compact tables of the main text | `tab_main_ranks.py`, `tab_main_inference.py` |

The diagnostics of the article are run on a finished run and write to
`results/main/diagnostics/`:

| Output | Content | Script in `scripts/diagnostics/` |
|---|---|---|
| Figure 1: `fig_lambda_sweep.pdf` (and `.png`), `lambda_sweep_MONTHLY.csv`, `lambda_sweep_summary.csv` | Out-of-sample Sharpe ratio of MV against its risk-aversion parameter | `lambda_sweep.py` |
| Table 10: `tab_dmvy_collapse.tex`, `dmvy_collapse_by_grid.csv` | DMVY: calibration dates at 1/N by grid combination | `dmvy_collapse.py` |
| Table 11: `tab_ssd_sorting.tex`, `ssd_sorting.csv` | SSD with chronological and with sorted cumulative sums | `ssd_sorting.py` |
| `return_floor.csv` | Return-floor family: infeasibility of the hard floor, concentration of the soft floor | `return_floor.py` |
| `concentration_semiv_mvn.csv` | SemiV and MVN: dates with one asset above 99% and coincidence with GMR | `concentration.py` |
| `mvn_total_vs_excess.csv`, `mvn_total_vs_excess_summary.csv` (`_exact` with `--solver exact`) | MVN with the moments of total returns, and the joint test with it in the place of MVN | `mvn_total_moments.py` |

```bash
python scripts/diagnostics/lambda_sweep.py
python scripts/diagnostics/dmvy_collapse.py
python scripts/diagnostics/ssd_sorting.py
python scripts/diagnostics/return_floor.py
python scripts/diagnostics/concentration.py
python scripts/diagnostics/mvn_total_moments.py
```

`mvn_total_moments.py` runs MVN again with total-return moments as the main version is run
(lambda recalibrated in the static pass, differential evolution), which takes several hours;
`--solver exact` is a quick alternative that solves both versions at the lambda of the run by a
parametric search on the efficient frontier.

Each table script can also be run on its own, for example
`python scripts/tables/tab_ranks.py --run-dir results/main --freq MONTHLY`. The underlying
CSV files are in `results/main/tables/`, `inference/` and `summary/`.

## Computing time

The full experiment took 39.4 CPU hours on a 13th Gen Intel Core i7-13700HX (16 physical
cores, 32 GB of memory, Windows 11): 3.6 hours for twenty strategies (0.7 hours of
wall-clock time) and 35.8 hours for MVN (2.4 hours of wall-clock time), with 16 worker
processes and one linear-algebra thread per process. `--workers N` sets the number of
processes; it does not change any result.

MVN is solved by a stochastic global search, and its weights can differ between two runs
by amounts of the order of 1e-7 in the universes with ten or more assets. The other
twenty strategies give identical series from run to run in the pinned environment.

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the article.

## Contact

Jorge Rafael Sarmiento-Laya, jorgesarmientol@gmail.com
(institutional address: jrsarmientolaya@al.uloyola.es)
