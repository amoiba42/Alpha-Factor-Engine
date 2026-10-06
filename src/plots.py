"""Research figures (matplotlib, static PNG).

Colors follow a fixed entity -> color mapping so every figure shows a given factor or
strategy in the same color. Palette: validated categorical reference palette (light mode).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"

COLORS = {  # fixed order: slot 1..5 of the categorical palette
    "ensemble": "#2a78d6",
    "equal_weight": "#eb6834",
    "reversal": "#1baf7a",
    "momentum": "#eda100",
    "volatility": "#e87ba4",
}
LABELS = {"ensemble": "Ridge ensemble", "equal_weight": "Equal-weight composite",
          "reversal": "Mean reversion", "momentum": "Momentum", "volatility": "Low volatility"}
PERIOD_SHADE = {"train": "#f3f2ee", "validation": "#e9eef7", "test": "#fbeee8"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.6, "font.size": 9, "figure.dpi": 110,
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shade_periods(ax, periods: dict[str, tuple[str, str]]):
    for name, (s, e) in periods.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=PERIOD_SHADE.get(name, "#f5f5f5"), zorder=0, lw=0)
        ax.text(pd.Timestamp(s), 1.0, f" {name}", transform=ax.get_xaxis_transform(), va="top",
                fontsize=8, color=INK_2)


def _end_labels(ax, series: dict[str, pd.Series]):
    """Direct labels at the right end of each line (identity not by color alone)."""
    for k, s in series.items():
        s = s.dropna()
        if len(s):
            ax.annotate(LABELS.get(k, k), (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7.5, color=INK_2, va="center")


def data_quality(universe_count: pd.Series, members_with_data: pd.Series, members_total: pd.Series,
                 returns: pd.DataFrame, quality: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(members_total.index, members_total, color=NEUTRAL, lw=1.2, label="S&P 500 members (PIT)")
    ax.plot(members_with_data.index, members_with_data, color=COLORS["equal_weight"], label="Members with price data")
    ax.plot(universe_count.index, universe_count, color=COLORS["ensemble"], label="Tradable universe")
    ax.set_title("Universe size over time")
    ax.set_ylabel("Stocks")
    ax.legend(loc="lower right")
    ax = axes[1]
    r = returns.stack().dropna().clip(-0.2, 0.2)
    ax.hist(r, bins=200, color=COLORS["ensemble"], alpha=0.9)
    ax.set_yscale("log")
    ax.set_title("Daily stock returns (clipped at ±20% for display)")
    ax.set_xlabel("Daily return")
    ax = axes[2]
    q = quality.sort_values("count")
    ax.barh(q.index, q["count"], color=COLORS["ensemble"], height=0.6)
    for i, v in enumerate(q["count"]):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=7.5, color=INK_2)
    ax.set_xscale("symlog")
    ax.set_title("Raw data-quality flags (rows)")
    _save(fig, path)


def factor_distributions(raw: dict[str, pd.DataFrame], std: dict[str, pd.DataFrame], dates, path: Path):
    names = list(raw)
    fig, axes = plt.subplots(2, len(names), figsize=(4.5 * len(names), 6.5))
    for j, n in enumerate(names):
        x = raw[n].reindex(dates).stack().dropna()
        lo, hi = x.quantile([0.005, 0.995])
        axes[0, j].hist(x.clip(lo, hi), bins=120, color=COLORS[n])
        axes[0, j].set_title(f"{LABELS[n]}: raw factor")
        z = std[n].reindex(dates).stack().dropna()
        axes[1, j].hist(z, bins=120, color=COLORS[n])
        axes[1, j].set_title(f"{LABELS[n]}: standardized (winsorized z-score)")
    _save(fig, path)


def correlation_heatmaps(mats: dict[str, pd.DataFrame], path: Path):
    fig, axes = plt.subplots(1, len(mats), figsize=(4.8 * len(mats), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (title, m) in zip(axes, mats.items()):
        ax.imshow(m.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
        labels = [LABELS.get(c, c) for c in m.columns]
        ax.set_xticks(range(len(m)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(m)), labels)
        ax.grid(False)
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=9, color=INK)
        ax.set_title(title)
    _save(fig, path)


def ic_distribution(ics: dict[str, pd.Series], horizon: int, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(-0.6, 0.6, 121)
    for n, ic in ics.items():
        ic = ic.dropna()
        ax.hist(ic, bins=bins, histtype="step", lw=1.6, color=COLORS[n],
                label=f"{LABELS[n]} (mean {ic.mean():+.4f})")
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_title(f"Distribution of daily cross-sectional Rank IC, {horizon}D horizon")
    ax.set_xlabel("Rank IC")
    ax.set_ylabel("Days")
    ax.legend()
    _save(fig, path)


def ic_timeseries(ics: dict[str, pd.Series], window: int, periods, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    _shade_periods(ax, periods)
    roll = {n: ic.rolling(window, min_periods=window // 2).mean() for n, ic in ics.items()}
    for n, s in roll.items():
        ax.plot(s.index, s, color=COLORS[n], label=LABELS[n])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_title(f"Rolling {window}-day mean Rank IC (5D horizon)")
    ax.set_ylabel("Mean Rank IC")
    ax.legend(loc="lower left", ncol=3)
    _save(fig, path)


def factor_decay(ic_tab: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n, g in ic_tab.groupby("factor"):
        g = g.sort_values("horizon")
        se = g["mean_ic"] / g["t_stat_nw"].replace(0, np.nan)
        ax.errorbar(g["horizon"], g["mean_ic"], yerr=1.96 * se.abs(), color=COLORS[n], marker="o", ms=5,
                    capsize=3, label=LABELS[n])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(sorted(ic_tab["horizon"].unique()))
    ax.set_xlabel("Forward-return horizon (trading days)")
    ax.set_ylabel("Mean Rank IC (±1.96 Newey-West s.e.)")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def cumulative_returns(rets: dict[str, pd.Series], periods, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    _shade_periods(ax, periods)
    curves = {k: (1 + r.fillna(0)).cumprod() - 1 for k, r in rets.items()}
    for k, c in curves.items():
        ax.plot(c.index, c, color=COLORS.get(k, NEUTRAL), label=LABELS.get(k, k))
    _end_labels(ax, curves)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative net return")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper left", ncol=3)
    _save(fig, path)


def drawdowns(dds: dict[str, pd.Series], periods, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    _shade_periods(ax, periods)
    for k, d in dds.items():
        ax.plot(d.index, d, color=COLORS.get(k, NEUTRAL), label=LABELS.get(k, k), lw=1.3)
    ax.set_title("Drawdown of net returns")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", ncol=3)
    _save(fig, path)


def rolling_sharpe(rs: dict[str, pd.Series], window: int, periods, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    _shade_periods(ax, periods)
    for k, s in rs.items():
        ax.plot(s.index, s, color=COLORS.get(k, NEUTRAL), label=LABELS.get(k, k), lw=1.3)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_title(f"Rolling {window}-day Sharpe ratio (net)")
    ax.legend(loc="lower left", ncol=3)
    _save(fig, path)


def turnover(tos: dict[str, pd.Series], window: int, periods, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    _shade_periods(ax, periods)
    for k, s in tos.items():
        ax.plot(s.index, s.rolling(window).mean() * 252, color=COLORS.get(k, NEUTRAL), label=LABELS.get(k, k), lw=1.3)
    ax.set_title(f"Annualized turnover (rolling {window}-day mean of daily traded notional / capital)")
    ax.set_ylabel("Turnover per year (x capital)")
    ax.legend(loc="upper left", ncol=3)
    _save(fig, path)


def sharpe_by_period(tab: pd.DataFrame, path: Path, title: str):
    """Grouped bars: strategy x period Sharpe."""
    periods = list(tab.columns)
    strategies = list(tab.index)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(strategies)
    x = np.arange(len(periods))
    for i, s in enumerate(strategies):
        vals = tab.loc[s].to_numpy(dtype=float)
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92, color=COLORS.get(s, NEUTRAL),
                      label=LABELS.get(s, s))
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7, color=INK_2)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, periods)
    ax.set_ylabel("Sharpe ratio (net)")
    ax.set_title(title)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, path)


def exposure_beta(exp: pd.DataFrame, beta_roll: pd.Series, ex_ante: pd.Series, periods, path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax = axes[0]
    _shade_periods(ax, periods)
    ax.plot(exp.index, exp["long_exposure"], color=COLORS["ensemble"], label="Long exposure", lw=1.2)
    ax.plot(exp.index, exp["short_exposure"], color=COLORS["equal_weight"], label="Short exposure", lw=1.2)
    ax.plot(exp.index, exp["net_exposure"], color=INK_2, label="Net exposure", lw=1.2)
    ax.set_title("Ridge ensemble: dollar exposure (end of day, after drift)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.73), ncol=3)
    ax = axes[1]
    _shade_periods(ax, periods)
    ax.plot(beta_roll.index, beta_roll, color=COLORS["ensemble"], label="Realized beta to SPY (rolling 126d)", lw=1.2)
    ax.plot(ex_ante.index, ex_ante, color=COLORS["reversal"], label="Ex-ante beta (holdings x trailing stock betas)", lw=1.0)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_title("Market beta exposure (measured, not assumed)")
    ax.legend(loc="lower left", ncol=2)
    _save(fig, path)


def long_short_contribution(daily: pd.DataFrame, periods, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    _shade_periods(ax, periods)
    for col, color, lab in [("long_contribution", COLORS["ensemble"], "Long leg"),
                            ("short_contribution", COLORS["equal_weight"], "Short leg"),
                            ("gross_return", INK_2, "Total (gross)")]:
        ax.plot(daily.index, daily[col].cumsum(), color=color, label=lab, lw=1.3)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_title("Ridge ensemble: cumulative (additive) long vs short contribution, gross of costs")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper left")
    _save(fig, path)


def cost_sensitivity(tab: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s in tab.columns:
        ax.plot(tab.index, tab[s], marker="o", ms=5, color=COLORS.get(s, NEUTRAL), label=LABELS.get(s, s))
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("Total cost per dollar traded (bps, one-way)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe vs. execution-cost assumption")
    ax.legend()
    _save(fig, path)
