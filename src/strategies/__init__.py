"""Strategy registry: code used in the output files -> strategy class."""
from src.strategies.advanced import (
    DiversifiedMeanVariance, DMVReturn, DMVVars, DMVYager, EWMV_Mixture,
)
from src.strategies.basic import (
    EquallyWeighted, GlobalMaximumReturn, GlobalMinimumVariance, MaxDiversification,
)
from src.strategies.dominance import MAD, FirstOrderSD, MinCVaR, SecondOrderSD
from src.strategies.markowitz import (
    MeanSemivariance, MeanSquaredVariance, MeanVariance, WeightConstrainedMV,
    WeightLowerBoundConstrainedMV, WeightUpperBoundConstraint,
)
from src.strategies.ml_robust import HierarchicalRiskParity, RiskParity

RANDOM_SEED = 42

# Codes and the names used in the article: MSV = MVN, WUBC_Alt = WCMV,
# MD = MaxDiv, FSD = P-Dom, SSD = SSD-R, DMV_Yager = DMVY, DMV_Return = DMVR,
# DMV_Vars = DMVV.
STRATEGY_MAP = {
    # benchmarks and single-moment strategies
    "EW": EquallyWeighted,
    "GMV": GlobalMinimumVariance,
    "GMR": GlobalMaximumReturn,
    "MD": MaxDiversification,
    # mean-variance family
    "MV": MeanVariance,
    "MSV": MeanSquaredVariance,
    "SemiV": MeanSemivariance,
    "WUBC": WeightUpperBoundConstraint,
    "WLBC": WeightLowerBoundConstrainedMV,
    "WUBC_Alt": WeightConstrainedMV,
    # stochastic dominance
    "FSD": FirstOrderSD,
    "SSD": SecondOrderSD,
    # diversification-penalised mean-variance
    "DMV": DiversifiedMeanVariance,
    "DMV_Yager": DMVYager,
    "DMV_Return": DMVReturn,
    "DMV_Vars": DMVVars,
    "EWMV": EWMV_Mixture,
    # risk budgeting and hierarchical allocation
    "RP": RiskParity,
    "HRP": HierarchicalRiskParity,
    # alternative risk measures
    "CVaR": MinCVaR,
    "MAD": MAD,
}
