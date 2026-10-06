# Alpha-Factor-Engine: executive research report

* Universe: historical S&P 500 members (Yahoo Finance daily OHLCV). Benchmark: SPY.
* Splits: train 2007-2015, validation 2016-2019, out-of-sample test 2020-2025.
* Execution: signals at the close of day $t$, trades at the close of day $t+1$, rebalanced every 20 trading days.
* Risk and costs: dollar neutral, signals beta-neutralized cross-sectionally, 6 bps per dollar traded (3 bps commission and spread plus 3 bps slippage).

## Summary

This study tests whether three classic price-volume signals predict returns of liquid US large caps well enough to pay for trading them. The signals are 5-day short-term mean reversion, 120-day momentum that skips the most recent 5 days, and 60-day realized volatility.

Main findings:

1. Five-day mean reversion predicts returns with statistical significance in the research sample (Rank IC 0.016, Newey-West $t = 3.39$, positive in 11 of 13 years). Momentum and volatility add little on their own in this universe.
2. The portfolio carried almost no market exposure. Signals were residualized on trailing 252-day stock betas, and the measured out-of-sample beta to SPY was $+0.006$ ($t = 0.50$).
3. Trading costs decide the outcome. The Ridge ensemble earned a gross Sharpe ratio of 0.22 in the 2020-2025 test period. With turnover of $16.5\times$ capital a year, the 6 bps cost takes about 1% a year and leaves a net Sharpe of 0.06.
4. Changing the portfolio construction did not help. A post-hoc study (notebook 05) tried 5-day rebalancing, concentration in the top and bottom 5-10%, and a turnover buffer. The buffer cut turnover by 23-37% against the comparable 10% book, but every variant still had a negative net Sharpe. With a composite Rank IC near 0.01, costs grew faster than the return spread.

## 1. Factor diagnostics (2007-2019 research period)

Each factor was winsorized at 1%/99% and z-scored within each day's universe, with z-scores clipped at $\pm 3$. The ICs below use these directional signals before beta neutralization, measured with Spearman Rank IC at several horizons.

| Factor | Lookback | 5D Rank IC | Newey-West $t$ | Positive years | Mean IC at 1D, 5D, 20D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Short-term reversal | 5-day return, reversed | 0.0160 | 3.39 ($p < 0.001$) | 11 of 13 | 0.012, 0.016, 0.014 (slow decay) |
| Momentum | 120 days, skipping the last 5 | 0.0054 | 0.76 (not significant) | 6 of 13 | 0.008, 0.005, -0.003 (fast decay) |
| Realized volatility | 60-day annualized std | 0.0046 | 0.52 (not significant) | 7 of 13 | 0.010, 0.005, 0.004 (fast decay) |

Reversal is consistent across years and keeps most of its predictive power out to 20 days. The momentum and volatility averages come mostly from a few extreme periods, such as the 2009 momentum crash: their top 5% of days contribute 4 to 6 times the entire IC sum.

## 2. Factor independence and the Ridge combination

The three beta-neutralized signals are nearly uncorrelated cross-sectionally ($\rho \approx 0.00$ between reversal and the other two, $\rho = 0.14$ between momentum and volatility). Their long/short returns are less independent: momentum and low volatility have a return correlation of $+0.49$, which points to a shared exposure beyond market beta.

The Ridge penalty made no practical difference. With about 800,000 panel rows and only 3 features, validation IC was identical for every $\alpha$ from 0.01 to 1000. The final model, refit on train and validation, is therefore effectively a fitted linear weighting:

$$\text{Composite} = 0.0098 \cdot F_{\text{rev}} + 0.0074 \cdot F_{\text{mom}} + 0.0059 \cdot F_{\text{vol}}$$

## 3. Out-of-sample performance (2020-2025 test period)

The portfolio held the top and bottom quintiles in equal weights (about 93 stocks long and 93 short, 186 positions) and rebalanced every 20 trading days. Trades executed at the close of $t+1$, so no signal used prices it could not have seen.

| Strategy | Ann. net return | Volatility | Net Sharpe | Gross Sharpe | Max drawdown | Beta to SPY | Ann. turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ridge ensemble | +0.40% | 6.28% | 0.06 | 0.22 | -14.69% | +0.006 ($t = 0.50$) | 16.5x |
| Equal-weight composite | -1.51% | 6.78% | -0.22 | -0.10 | -17.60% | -0.008 ($t = -0.64$) | 14.3x |
| Reversal alone | +1.57% | 6.39% | 0.25 | 0.43 | -15.82% | +0.029 ($t = 2.69$) | 19.8x |
| Momentum alone | -1.47% | 8.64% | -0.17 | -0.11 | -24.02% | -0.029 ($t = -1.99$) | 9.1x |
| Volatility alone | -0.83% | 6.00% | -0.14 | -0.06 | -17.01% | -0.040 ($t = -3.97$) | 7.4x |

Against the equal-weight composite, the Ridge ensemble had an Information Ratio of 0.95, so its weighting did better than a simple average of the three signals over this period. Reversal alone still did better than the ensemble.

The ensemble breaks even at about 4 bps per dollar traded on validation and about 8.5 bps on test. At the assumed 6 bps, its net Sharpe is 0.06.

## 4. Portfolio-construction extensions (notebook 05)

Three changes to portfolio construction were tested to see whether they were holding back the baseline. These variants were defined after the test results had been seen, so they are ranked on validation and the test column is descriptive only.

| Variant | Validation net Sharpe | Test net Sharpe (post-hoc) | Annual turnover (validation, test) | Turnover vs baseline |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (20d, 20% quintiles) | -0.04 | +0.06 | 12.0x, 16.5x | 1x |
| A (5d rebalance, 20%) | -0.43 | -0.40 | 41.3x, 61.5x | about 3.5x |
| B (5d rebalance, 10% deciles) | -0.37 | -0.25 | 49.0x, 71.8x | about 4x |
| B5 (5d rebalance, 5% ventiles) | -0.21 | -0.30 | 55.1x, 78.5x | about 4.5x |
| C (5d, 10% entry / 20% exit buffer) | -0.10 | -0.23 | 30.8x, 55.4x | 2.5x to 3.4x |

What the variants showed:

1. Rebalancing every 5 days added cost without adding return. Reversal decays slowly out to 20 days, so faster rebalancing did not raise the gross spread. It multiplied trading by about 3.5 times and added 2.5% to 3.7% a year in costs.
2. Concentration added noise. Extreme ranks change faster, and cutting each leg from about 93 stocks to about 23 more than doubled maximum drawdown and raised volatility from about 5-6% to 9-12%, without reliably widening the return spread.
3. The turnover buffer helped but not enough. Entering at the top 10% and exiting below the top 20% cut turnover by 23-37% against variant B and lifted validation net Sharpe from $-0.37$ to $-0.10$. The IC is still too small to clear 6 bps of cost.

## 5. Methodology caveats

Survivorship: Yahoo Finance serves only currently listed symbols, so most delisted or bankrupt former index members are missing. Coverage of the point-in-time index rises from about 56% in 2006 to 97% in 2025, so the bias is strongest in the early years.

Execution: the pipeline models a one-day execution lag, dollar neutrality and turnover-based costs. Borrow fees and non-linear market impact at larger position sizes are not modelled.

Post-hoc work: the notebook 05 variants were designed after the baseline test period had been inspected. The baseline metrics are genuinely out-of-sample; the extension results are descriptive sensitivity checks.

## Conclusion

The strategy did not earn a meaningful return after costs, because the signal is too weak to pay for trading it.

1. Short-term mean reversion has a measurable edge (Rank IC about 0.016), but in large-cap US equities it does not cover realistic trading costs.
2. Buffers and concentration change how much the portfolio trades, and leave the signal itself as weak as before.
3. Making this work would need a stronger signal, for example from other data sources or intraday execution, or much cheaper execution than assumed here.
