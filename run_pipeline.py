"""End-to-end research pipeline.

    python run_pipeline.py               # train + validation only (test locked by config)
    python run_pipeline.py --run-test    # final, one-time out-of-sample evaluation

Outputs: results/tables/{summary,diagnostics}/*.csv|md, results/figures/{tearsheet,diagnostics}/*.png,
results/RESULTS.md,
data/processed/*.parquet (intermediate series used by the notebooks).
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from src import plots
from src.backtest import rebalance_schedule, run_backtest
from src.config import figure_relpath, load_config, table_relpath
from src.data import build_panel
from src.evaluation import (decay_table, ic_by_year, ic_concentration, ic_summary, ic_table,
                            mean_cs_correlation, period_dates, rank_ic)
from src.factors import FACTORS, build_factor, candidate_windows
from src.metrics import drawdown, performance_summary, rolling_sharpe
from src.models import equal_weight_composite, fit_ridge, predict_composite, select_ridge_alpha, select_window
from src.portfolio import ex_ante_beta, long_short_weights
from src.preprocessing import (build_universe, daily_returns, forward_returns, neutralize_cs, rank_gauss_cs,
                               rolling_beta, zscore_cs)

log = logging.getLogger("pipeline")
STRATEGIES = ["reversal", "momentum", "volatility", "ensemble", "equal_weight"]


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def save_table(df: pd.DataFrame, name: str, cfg: dict, floatfmt: str = ".4f") -> str:
    d = cfg["paths"]["tables_dir"]
    df.to_csv(d / table_relpath(name, "csv"))
    md = md_table(df, floatfmt)
    (d / table_relpath(name, "md")).write_text(md + "\n")
    return md


def headline_summary(perf: pd.DataFrame, ccfg: dict, cost_bps: float) -> str:
    """Headline measured values for the Ridge ensemble over the test period."""
    t = perf.loc["test"]
    e = t.loc["ensemble"]
    best = t["sharpe"].idxmax()
    rows = [
        ("Annualized Sharpe ratio",
         f"{e['sharpe']:.2f} net / {e['sharpe_gross']:.2f} gross (best single strategy: {best} {t.loc[best, 'sharpe']:.2f})"),
        ("Information Ratio",
         f"{e['ir_vs_SPY']:.2f} vs SPY; {e['ir_vs_equal_weight_composite']:.2f} vs equal-weight composite "
         "(the second figure is the gain over a simple average of the signals)"),
        ("Out-of-sample protocol", "test 2020-2025, evaluated once after decisions were frozen on train/validation"),
        ("Market beta", f"{e['market_beta']:+.3f} (HAC t = {e['market_beta_t']:.2f}) on daily net returns vs SPY"),
        ("Dollar neutrality",
         f"mean net exposure {e['avg_net_exposure']:+.4f}, max abs net {e['max_abs_net_exposure']:.3f} (drift between rebalances)"),
        ("Execution costs",
         f"{ccfg['slippage_bps']} bps slippage + {cost_bps} bps commission/spread per dollar traded; "
         f"annual cost drag {e['ann_cost_drag'] * 100:.2f}%"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Measured"]).to_markdown(index=False)


def main(run_test: bool) -> None:
    t0 = time.time()
    cfg = load_config()
    np.random.seed(cfg["seed"])
    P = cfg["paths"]
    fig = P["figures_dir"]
    FIG = lambda key: fig / figure_relpath(key)  # noqa: E731
    IMG = lambda key: f"![](figures/{figure_relpath(key)})"  # noqa: E731
    sp = cfg["splits"]
    periods = {"train": tuple(sp["train"]), "validation": tuple(sp["validation"])}
    if run_test:
        periods["test"] = tuple(sp["test"])
    last_date = pd.Timestamp(periods[list(periods)[-1]][1])
    h_sel = cfg["evaluation"]["selection_horizon"]
    lag = cfg["portfolio"]["execution_lag"]
    purge = h_sel + lag
    report: list[str] = []

    # ---------------------------------------------------------------- data
    log.info("loading data")
    panel = build_panel(cfg)
    wide = panel["wide"]
    adj = wide["adj_close"]
    rets = daily_returns(adj)
    bench = panel["benchmark_returns"]
    universe = build_universe(wide, panel["membership"], cfg["universe"])
    dates = adj.index

    quality = pd.DataFrame({"raw": pd.Series(panel["quality_raw"]), "clean": pd.Series(panel["quality_clean"])})
    md_quality = save_table(quality, "data_quality", cfg)
    md_actions = save_table(pd.Series(panel["cleaning_actions"], name="count").to_frame(), "cleaning_actions", cfg)
    flags = pd.Series({k: v for k, v in panel["quality_raw"].items()
                       if k in ("duplicate_rows", "missing_adj_close", "missing_volume", "zero_volume",
                                "non_positive_prices", "high_below_low", "close_outside_range",
                                "abs_return_gt_25pct", "abs_return_gt_50pct", "tickers_lt_252_obs")}, name="count")
    members_total = pd.read_csv(P["raw_dir"] / "sp500_membership.csv", parse_dates=["date"]).set_index("date")["tickers"] \
        .str.split(",").str.len().reindex(dates, method="ffill")
    plots.data_quality(universe.sum(axis=1), (panel["membership"] & adj.notna()).sum(axis=1), members_total,
                       rets.where(universe), flags.to_frame(), FIG("data_quality"))
    uni_stats = universe.sum(axis=1).groupby(dates.year).agg(["mean", "min", "max"]).round(0)
    md_uni = save_table(uni_stats, "universe_size_by_year", cfg, ".0f")

    # ---------------------------------------------------------------- forward returns
    horizons = cfg["evaluation"]["horizons"]
    fwd = {h: forward_returns(adj, h).where(universe) for h in horizons}
    target_fwd = forward_returns(adj, h_sel, lag=lag).where(universe)  # what the strategy trades

    # ---------------------------------------------------------------- factor windows (train/validation only)
    log.info("selecting factor lookbacks on validation")
    chosen, factor_frames, sel_tables = {}, {}, []
    for name in FACTORS:
        cands = {w: build_factor(name, w, adj, rets, universe, cfg) for w in candidate_windows(cfg["factors"])[name]}
        w, tab = select_window({w: f["signal"] for w, f in cands.items()}, target_fwd,
                               periods["train"], periods["validation"], purge)
        chosen[name] = w
        factor_frames[name] = cands[w]
        sel_tables.append(tab.assign(factor=name, chosen=lambda d, w=w: d.index == w))
        del cands
    sel = pd.concat(sel_tables).reset_index().set_index(["factor", "window"])
    md_sel = save_table(sel, "lookback_selection", cfg)
    signals = {n: factor_frames[n]["signal"] for n in FACTORS}
    raw = {n: factor_frames[n]["raw"] for n in FACTORS}
    std = {n: factor_frames[n]["standardized"] for n in FACTORS}

    # ---------------------------------------------------------------- factor research
    log.info("factor IC analysis")
    research_dates = period_dates(dates, periods["train"][0], periods["validation"][1])
    plots.factor_distributions(raw, std, research_dates, FIG("factor_distributions"))

    ic_tabs = {}
    for pname, (s, e) in periods.items():
        ic_tabs[pname] = ic_table(signals, fwd, period_end_purge=(s, e)).assign(period=pname)
    ic_tabs["train+validation"] = ic_table(signals, fwd, period_end_purge=(periods["train"][0], periods["validation"][1])) \
        .assign(period="train+validation")
    ic_all = pd.concat(ic_tabs.values(), ignore_index=True)
    save_table(ic_all.set_index(["period", "factor", "horizon"]), "ic_by_period_horizon", cfg)

    rs = ic_tabs["train+validation"]
    best_h = rs.loc[rs.groupby("factor")["ic_ir"].idxmax()].set_index("factor")["horizon"]
    fs = rs[rs["horizon"] == h_sel].set_index("factor")
    factor_summary = pd.DataFrame({
        "Window": pd.Series(chosen),
        f"Mean IC ({h_sel}D)": fs["mean_ic"], "Median IC": fs["median_ic"], "IC Std": fs["ic_std"],
        "IC IR": fs["ic_ir"], "Positive IC %": fs["pct_positive"] * 100, "NW t-stat": fs["t_stat_nw"],
        "Best Horizon (IC IR)": best_h.map(lambda h: f"{h}D"),
    }).reindex(FACTORS)
    md_fsum = save_table(factor_summary, "factor_summary_research", cfg)
    md_decay = save_table(decay_table(rs), "factor_decay_research", cfg)
    md_decay_ir = save_table(decay_table(rs, "ic_ir"), "factor_decay_icir_research", cfg)
    plots.factor_decay(rs, FIG("factor_decay"), "Factor decay: mean Rank IC by horizon (train+validation)")

    ic_series = {n: rank_ic(signals[n], fwd[h_sel]).reindex(period_dates(dates, dates[0], last_date, purge=h_sel))
                 for n in FACTORS}
    # labels must end inside train+validation: purge the last h dates of the research window
    research_h = period_dates(dates, periods["train"][0], periods["validation"][1], purge=h_sel)
    research_1 = period_dates(dates, periods["train"][0], periods["validation"][1], purge=1)
    ic1 = {n: rank_ic(signals[n], fwd[1]).reindex(research_1) for n in FACTORS}
    plots.ic_distribution(ic1, 1, FIG("ic_distribution_1d"))
    plots.ic_distribution({n: s.reindex(research_h) for n, s in ic_series.items()}, h_sel,
                          FIG("ic_distribution_5d"))
    plots.ic_timeseries(ic_series, 126, periods, FIG("ic_timeseries"))
    pd.DataFrame(ic_series).to_parquet(P["processed_dir"] / "ic_series_5d.parquet")

    yearly = pd.concat({n: ic_by_year(s.reindex(research_h))["mean_ic"] for n, s in ic_series.items()}, axis=1)
    md_yearly = save_table(yearly, "ic_by_year_research", cfg)
    conc = pd.DataFrame({n: ic_concentration(s.reindex(research_h)) for n, s in ic_series.items()}).T
    conc["mean_ic"] = pd.Series({n: s.reindex(research_h).mean() for n, s in ic_series.items()})
    conc["years_with_positive_mean_ic"] = (yearly > 0).sum()
    conc["years"] = yearly.notna().sum()
    md_conc = save_table(conc, "ic_robustness_research", cfg)

    # ---------------------------------------------------------------- factor relationships
    raw_corr = mean_cs_correlation(raw, research_dates, "spearman")
    sig_corr = mean_cs_correlation(signals, research_dates, "pearson")

    # ---------------------------------------------------------------- design variants (validation only)
    log.info("evaluating design variants on validation")
    pcfg, ccfg = cfg["portfolio"], cfg["costs"]
    cost_bps = ccfg["commission_bps"] + ccfg["spread_bps"]
    stock_beta = rolling_beta(rets, bench)
    target = rank_gauss_cs(target_fwd)
    train_fit_dates = period_dates(dates, *periods["train"], purge=purge)
    val_end = pd.Timestamp(periods["validation"][1])

    def final_signals(beta_neutral: bool) -> dict[str, pd.DataFrame]:
        if not beta_neutral:
            return signals
        b = stock_beta.where(universe)
        return {n: zscore_cs(neutralize_cs(signals[n], b)) for n in FACTORS}

    var_rows = []
    for bn in pcfg["beta_neutral_candidates"]:
        sig_v = final_signals(bn)
        a_v, _ = select_ridge_alpha(sig_v, target, target_fwd, periods["train"], periods["validation"],
                                    cfg["model"]["ridge_alphas"], purge)
        comp_v = predict_composite(fit_ridge(sig_v, target, train_fit_dates, a_v), sig_v)
        w_v = long_short_weights(comp_v, pcfg["quantile"], pcfg["weighting"], pcfg["gross_exposure"])
        for reb in pcfg["rebalance_every_candidates"]:
            sch = rebalance_schedule(dates, pd.Timestamp(periods["train"][0]), reb)
            d_v, _ = run_backtest(w_v, rets, sch, lag, cost_bps, ccfg["slippage_bps"], end=val_end)
            row = {"beta_neutral": bn, "rebalance_every": reb, "ridge_alpha": a_v}
            for pname in ("train", "validation"):
                ps = performance_summary(d_v.loc[slice(*periods[pname])], {}, bench)
                row.update({f"{pname}_sharpe": ps["sharpe"], f"{pname}_sharpe_gross": ps["sharpe_gross"],
                            f"{pname}_turnover": ps["ann_turnover"], f"{pname}_beta": ps["market_beta"]})
            var_rows.append(row)
    variants = pd.DataFrame(var_rows).set_index(["beta_neutral", "rebalance_every"])
    best = variants["validation_sharpe"].idxmax()
    beta_neutral, rebalance_every = bool(best[0]), int(best[1])
    variants["chosen"] = [ix == best for ix in variants.index]
    md_variants = save_table(variants, "design_variant_selection", cfg)
    signals = final_signals(beta_neutral)
    final_corr = mean_cs_correlation(signals, research_dates, "pearson")

    # ---------------------------------------------------------------- ensemble
    log.info("ridge ensemble")
    alpha, alpha_tab = select_ridge_alpha(signals, target, target_fwd, periods["train"], periods["validation"],
                                          cfg["model"]["ridge_alphas"], purge)
    md_alpha = save_table(alpha_tab, "ridge_alpha_selection", cfg, ".6f")
    m_train = fit_ridge(signals, target, train_fit_dates, alpha)
    comp_train = predict_composite(m_train, signals)
    ensemble = comp_train
    coefs = {"train_model": dict(zip(FACTORS, m_train.coef_)), "alpha": alpha}
    if run_test:
        tv_dates = period_dates(dates, periods["train"][0], periods["validation"][1], purge=purge)
        m_final = fit_ridge(signals, target, tv_dates, alpha) if cfg["model"]["refit_on_train_plus_validation"] else m_train
        comp_final = predict_composite(m_final, signals)
        test_start = pd.Timestamp(periods["test"][0])
        # train-period model for train/validation dates, train+validation refit for test dates
        ensemble = pd.concat([comp_train.loc[comp_train.index < test_start], comp_final.loc[comp_final.index >= test_start]])
        coefs["final_model"] = dict(zip(FACTORS, m_final.coef_))
    coef_tab = pd.DataFrame({k: v for k, v in coefs.items() if isinstance(v, dict)})
    md_coef = save_table(coef_tab, "ridge_coefficients", cfg, ".5f")

    all_signals = {**signals, "ensemble": ensemble, "equal_weight": equal_weight_composite(signals)}
    ens_ic_rows = []
    for pname, (s, e) in periods.items():
        for k in ("ensemble", "equal_weight", *FACTORS):
            ic = rank_ic(all_signals[k], target_fwd).reindex(period_dates(dates, s, e, purge=purge))
            ens_ic_rows.append({"period": pname, "signal": k, **ic_summary(ic, h_sel)})
    md_ens_ic = save_table(pd.DataFrame(ens_ic_rows).set_index(["period", "signal"]), "signal_ic_execution_aligned", cfg)

    # ---------------------------------------------------------------- backtests
    log.info("backtesting")
    sched = rebalance_schedule(dates, pd.Timestamp(periods["train"][0]), rebalance_every)
    daily, holdings, weights = {}, {}, {}
    for k in STRATEGIES:
        w = long_short_weights(all_signals[k], pcfg["quantile"], pcfg["weighting"], pcfg["gross_exposure"])
        d, hld = run_backtest(w, rets, sched, lag, cost_bps, ccfg["slippage_bps"], end=last_date)
        daily[k], holdings[k], weights[k] = d, hld, w
        d.to_parquet(P["processed_dir"] / f"backtest_{k}.parquet")

    ew_ret = daily["equal_weight"]["net_return"]
    benchmarks = {"SPY": bench, "equal_weight_composite": ew_ret}

    perf_rows = []
    for pname, (s, e) in periods.items():
        for k in STRATEGIES:
            d = daily[k].loc[s:e]
            perf_rows.append({"period": pname, "strategy": k, **performance_summary(d, benchmarks, bench)})
    perf = pd.DataFrame(perf_rows).set_index(["period", "strategy"])
    save_table(perf, "performance_full", cfg)

    def perf_table(pname: str) -> pd.DataFrame:
        p = perf.loc[pname]
        return pd.DataFrame({
            "Ann. Return": p["ann_return"] * 100, "Volatility": p["ann_volatility"] * 100, "Sharpe": p["sharpe"],
            "Sharpe (gross)": p["sharpe_gross"], "Max Drawdown": p["max_drawdown"] * 100,
            "IR vs SPY": p["ir_vs_SPY"], "IR vs EW composite": p["ir_vs_equal_weight_composite"],
            "Turnover (x/yr)": p["ann_turnover"], "Win Rate": p["win_rate"] * 100,
            "Beta (SPY)": p["market_beta"], "Beta t": p["market_beta_t"],
        })

    md_perf = {pname: save_table(perf_table(pname), f"portfolio_performance_{pname}", cfg, ".2f") for pname in periods}
    sharpe_tab = perf["sharpe"].unstack("period").reindex(STRATEGIES)[list(periods)]
    sharpe_tab.columns = [f"{c.title()} Sharpe" for c in sharpe_tab.columns]
    md_sharpe = save_table(sharpe_tab, "in_sample_vs_out_of_sample_sharpe", cfg, ".2f")

    # factor-return correlation (gross daily L/S returns, research period)
    fr = pd.DataFrame({n: daily[n]["gross_return"] for n in FACTORS}).loc[research_dates[0]:research_dates[-1]]
    fret_corr = fr.corr()
    corr_md = {
        "raw": save_table(raw_corr, "factor_correlation_raw_spearman", cfg, ".3f"),
        "signal": save_table(sig_corr, "signal_correlation_pearson", cfg, ".3f"),
        "final": save_table(final_corr, "final_signal_correlation_pearson", cfg, ".3f"),
        "returns": save_table(fret_corr, "factor_return_correlation", cfg, ".3f"),
    }
    plots.correlation_heatmaps({"Raw factors (mean daily Spearman)": raw_corr,
                                "Directional signals (mean daily Pearson)": sig_corr,
                                f"Final signals{' (beta-neutral)' if beta_neutral else ''} (Pearson)": final_corr,
                                "Factor L/S returns (daily, gross)": fret_corr}, FIG("factor_correlation"))

    # cost sensitivity (analytic: net = gross - turnover * bps)
    cs_rows = {}
    val_s, val_e = periods["validation"]
    for bps in ccfg["sensitivity_total_bps"]:
        cs_rows[bps] = {k: (lambda d: (d["gross_return"] - d["turnover"] * bps * 1e-4))(daily[k].loc[val_s:val_e])
                        for k in STRATEGIES}
    cs = pd.DataFrame({bps: {k: (r.mean() / r.std() * np.sqrt(252)) for k, r in v.items()} for bps, v in cs_rows.items()}).T
    cs.index.name = "total_bps"
    md_cs = save_table(cs, "cost_sensitivity_validation_sharpe", cfg, ".2f")
    plots.cost_sensitivity(cs, FIG("cost_sensitivity_validation"))
    if run_test:
        ts, te = periods["test"]
        cs_t = pd.DataFrame({bps: {k: (lambda r: r.mean() / r.std() * np.sqrt(252))(
            daily[k].loc[ts:te, "gross_return"] - daily[k].loc[ts:te, "turnover"] * bps * 1e-4) for k in STRATEGIES}
            for bps in ccfg["sensitivity_total_bps"]}).T
        cs_t.index.name = "total_bps"
        md_cs_test = save_table(cs_t, "cost_sensitivity_test_sharpe", cfg, ".2f")

    # execution-lag sensitivity for the ensemble (robustness only, not used for selection)
    w_ens = weights["ensemble"]
    lag_rows = []
    for L in (0, 1, 2):
        d, _ = run_backtest(w_ens, rets, sched, L, cost_bps, ccfg["slippage_bps"], end=last_date)
        for pname, (s, e) in periods.items():
            lag_rows.append({"execution_lag": L, "period": pname, "sharpe": performance_summary(d.loc[s:e], {})["sharpe"]})
    md_lag = save_table(pd.DataFrame(lag_rows).pivot(index="execution_lag", columns="period", values="sharpe")[list(periods)],
                        "execution_lag_sensitivity_ensemble", cfg, ".2f")

    # beta: realized rolling and ex-ante
    ens_d = daily["ensemble"]
    roll_cov = ens_d["net_return"].rolling(126).cov(bench.reindex(ens_d.index))
    roll_beta = roll_cov / bench.reindex(ens_d.index).rolling(126).var()
    exante = ex_ante_beta(holdings["ensemble"], stock_beta)
    beta_rows = []
    for pname, (s, e) in periods.items():
        for k in STRATEGIES:
            beta_rows.append({"period": pname, "strategy": k,
                              "realized_beta": perf.loc[(pname, k), "market_beta"],
                              "realized_beta_t": perf.loc[(pname, k), "market_beta_t"],
                              "mean_ex_ante_beta": ex_ante_beta(holdings[k].loc[s:e], stock_beta).mean(),
                              "mean_net_exposure": daily[k].loc[s:e, "net_exposure"].mean(),
                              "max_abs_net_exposure": daily[k].loc[s:e, "net_exposure"].abs().max()})
    md_beta = save_table(pd.DataFrame(beta_rows).set_index(["period", "strategy"]), "beta_and_neutrality", cfg)

    # exposure / positions summary
    exp_rows = []
    for pname, (s, e) in periods.items():
        for k in STRATEGIES:
            d = daily[k].loc[s:e]
            exp_rows.append({"period": pname, "strategy": k, "gross": d["gross_exposure"].mean(),
                             "net": d["net_exposure"].mean(), "long": d["long_exposure"].mean(),
                             "short": d["short_exposure"].mean(), "positions": d["n_positions"].mean(),
                             "daily_turnover": d["turnover"].mean()})
    md_exp = save_table(pd.DataFrame(exp_rows).set_index(["period", "strategy"]), "exposure_summary", cfg)

    # ---------------------------------------------------------------- figures
    net = {k: daily[k]["net_return"] for k in STRATEGIES}
    plots.cumulative_returns(net, periods, FIG("cumulative_returns"),
                             "Cumulative net returns, dollar-neutral long/short (after costs + slippage)")
    plots.drawdowns({k: drawdown(r) for k, r in net.items()}, periods, FIG("drawdown_underwater"))
    plots.rolling_sharpe({k: rolling_sharpe(net[k], 252) for k in ("ensemble", "equal_weight", *FACTORS)},
                         252, periods, FIG("rolling_sharpe"))
    plots.turnover({k: daily[k]["turnover"] for k in STRATEGIES}, 63, periods, FIG("portfolio_turnover"))
    plots.sharpe_by_period(sharpe_tab.rename(columns=lambda c: c.replace(" Sharpe", "")), FIG("strategy_comparison"),
                           "Net Sharpe by strategy and period: individual factors vs combined")
    plots.exposure_beta(ens_d, roll_beta, exante, periods, FIG("market_beta_exposure"))
    plots.long_short_contribution(ens_d, periods, FIG("long_short_contribution"))
    if run_test:
        ts, te = periods["test"]
        plots.cumulative_returns({k: net[k].loc[ts:te] for k in STRATEGIES}, {"test": (ts, te)},
                                 FIG("out_of_sample_returns"), "Out-of-sample (test) cumulative net returns")

    # ---------------------------------------------------------------- report
    manifest = json.loads((P["raw_dir"] / "prices_yfinance.manifest.json").read_text())
    report += [
        "# Alpha-Factor-Engine: Research Results (auto-generated)",
        "",
        f"Generated by `run_pipeline.py{' --run-test' if run_test else ''}`. "
        f"Test period evaluated: {'yes' if run_test else 'no (locked)'}.",
        "",
        f"Data: Yahoo Finance daily OHLCV snapshot downloaded {manifest['downloaded_at_utc']} "
        f"({manifest['received']} of {manifest['requested']} historical S&P 500 tickers available), "
        f"{cfg['data']['start']} to {cfg['data']['end']}. Benchmark: {cfg['data']['benchmark']}.",
        "",
        "Periods: " + ", ".join(f"{k} {v[0]} to {v[1]}" for k, v in periods.items()) + ".",
        "",
        f"Execution: signals at close t, trades at close t+{lag}, rebalance every {rebalance_every} days, "
        f"beta-neutral signals: {beta_neutral}, "
        f"top/bottom {int(pcfg['quantile'] * 100)}% {pcfg['weighting']}-weighted, gross {pcfg['gross_exposure']}. "
        f"Costs per dollar traded: {cost_bps} bps commission+spread + {ccfg['slippage_bps']} bps slippage.",
        "",
        "## 1. Data quality", "", md_quality, "", "Cleaning actions:", "", md_actions, "",
        "Tradable universe size by year:", "", md_uni, "", IMG("data_quality"), "",
        "## 2. Lookback selection (train IC reported, chosen by validation IC; execution-aligned "
        f"{h_sel}D target, purge {purge} days)", "", md_sel, "",
        "## 3. Factor summary (train+validation, research sample)", "", md_fsum, "",
        IMG("factor_distributions"), "",
        "### Factor decay: mean Rank IC by horizon", "", md_decay, "",
        "### Factor decay: IC IR by horizon", "", md_decay_ir, "", IMG("factor_decay"), "",
        f"### Robustness of the {h_sel}D IC", "", md_conc, "", "Mean IC by year:", "", md_yearly, "",
        IMG("ic_distribution_1d"), "", IMG("ic_distribution_5d"), "",
        IMG("ic_timeseries"), "",
        "## 4. Factor independence (train+validation)", "",
        "Raw factors, mean daily Spearman:", "", corr_md["raw"], "",
        "Directional standardized signals (before any beta neutralization), mean daily Pearson:", "", corr_md["signal"], "",
        "Final signals used in portfolios, mean daily Pearson:", "", corr_md["final"], "",
        "Daily gross long/short factor returns:", "", corr_md["returns"], "", IMG("factor_correlation"), "",
        "## 5. Design variants (chosen by validation net Sharpe of the ensemble)", "", md_variants, "",
        "## 6. Ridge ensemble", "", f"Chosen alpha: {alpha}", "", md_alpha, "", "Coefficients:", "", md_coef, "",
        "Execution-aligned IC of every signal by period:", "", md_ens_ic, "",
        "## 7. Portfolio performance (net of costs)", "",
    ]
    for pname in periods:
        report += [f"### {pname}", "", md_perf[pname], ""]
    report += [
        "### In-sample vs out-of-sample Sharpe", "", md_sharpe, "",
        IMG("cumulative_returns"), "", IMG("drawdown_underwater"), "",
        IMG("rolling_sharpe"), "", IMG("portfolio_turnover"), "",
        IMG("strategy_comparison"), "",
    ]
    if run_test:
        report += [IMG("out_of_sample_returns"), ""]
    report += [
        "## 8. Exposure, neutrality and beta", "", md_exp, "", md_beta, "",
        IMG("market_beta_exposure"), "", IMG("long_short_contribution"), "",
        "## 9. Sensitivity", "", "Validation Sharpe vs total cost per dollar traded:", "", md_cs, "",
    ]
    if run_test:
        report += ["Test Sharpe vs total cost per dollar traded:", "", md_cs_test, ""]
    report += [IMG("cost_sensitivity_validation"), "",
               "Ensemble Sharpe vs execution lag (robustness only):", "", md_lag, ""]
    if run_test:
        report += ["## 10. Headline summary (Ridge ensemble, test period)", "",
                   headline_summary(perf, ccfg, cost_bps), ""]
    P["report"].write_text("\n".join(report))
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-test", action="store_true", help="evaluate the locked test period (final run only)")
    args = ap.parse_args()
    cfg_flag = load_config()["splits"]["run_test"]
    main(run_test=args.run_test or cfg_flag)
