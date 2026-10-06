"""Configuration loading and project paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Output layout (tear-sheet convention). Paths are relative to results/figures and results/tables.
FIGURES = {
    # tearsheet: primary performance and factor tear-sheet
    "cumulative_returns": "tearsheet/cumulative_returns.png",
    "factor_decay": "tearsheet/factor_decay.png",
    "factor_correlation": "tearsheet/factor_correlation.png",
    "market_beta_exposure": "tearsheet/market_beta_exposure.png",
    "out_of_sample_returns": "tearsheet/out_of_sample_returns.png",
    # diagnostics: audit, sensitivity and distribution detail
    "data_quality": "diagnostics/data_quality.png",
    "factor_distributions": "diagnostics/factor_distributions.png",
    "ic_distribution_1d": "diagnostics/ic_distribution_1d.png",
    "ic_distribution_5d": "diagnostics/ic_distribution_5d.png",
    "ic_timeseries": "diagnostics/ic_timeseries.png",
    "drawdown_underwater": "diagnostics/drawdown_underwater.png",
    "rolling_sharpe": "diagnostics/rolling_sharpe.png",
    "portfolio_turnover": "diagnostics/portfolio_turnover.png",
    "strategy_comparison": "diagnostics/strategy_comparison.png",
    "cost_sensitivity_validation": "diagnostics/cost_sensitivity_validation.png",
    "long_short_contribution": "diagnostics/long_short_contribution.png",
}
SUMMARY_TABLES = {
    "performance_full", "beta_and_neutrality", "factor_decay_research", "factor_summary_research",
    "in_sample_vs_out_of_sample_sharpe", "ridge_coefficients", "portfolio_performance_test",
}


def figure_relpath(key: str) -> str:
    """Path of a figure relative to results/figures."""
    return FIGURES[key]


def table_relpath(name: str, ext: str = "csv") -> str:
    """Path of a table relative to results/tables: summary/ for headline tables, else diagnostics/."""
    return f"{'summary' if name in SUMMARY_TABLES else 'diagnostics'}/{name}.{ext}"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve every entry in `paths` to an absolute path."""
    path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    for key in ("raw_dir", "processed_dir", "figures_dir", "tables_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    for sub in ("tearsheet", "diagnostics"):
        (cfg["paths"]["figures_dir"] / sub).mkdir(exist_ok=True)
    for sub in ("summary", "diagnostics"):
        (cfg["paths"]["tables_dir"] / sub).mkdir(exist_ok=True)
    return cfg
