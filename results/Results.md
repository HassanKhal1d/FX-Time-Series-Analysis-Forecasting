# Results

Full experimental results for the Financial Time Series Analysis capstone project. All outputs were produced on the 4-hour resampled OHLCV data (2018-01-01 to 2021-01-16), with a 70/30 train/test split. The test period runs from approximately 2020-02-11 to 2021-01-16.

---

## 1. Data Sanity Check (Notebook 01)

| Check | AUD/USD | EUR/USD |
|---|---|---|
| Shape (4h bars) | (4,861, 5) | (4,861, 5) |
| Missing values | 0 | 0 |
| Duplicate timestamps | 0 | 0 |
| Timestamp gaps (total) | 1,806 | 1,806 |
| Weekend/holiday gaps | 93% of gaps | 93% of gaps |
| Weekday gaps | 126 (Christmas, New Year, Easter) | 126 (same dates) |
| Zero-volume bars | 0 | 0 |
| OHLC violations | 129 rows | 194 rows |
| OHLC violation magnitude | Mean 0.001, max 0.012 | Mean 0.002, max 0.032 |
| Price outliers (Z > 3) | 21 bars | 31 bars |
| Single-bar spikes (data errors) | 2 bars | 4 bars |
| Volume spikes (Z > 5) | 24 bars | 11 bars |
| Stale price runs (>= 5 bars) | 46 runs, longest 10 bars | 31 runs, longest 15 bars |

**Fixes applied:**

| Issue | Action |
|---|---|
| OHLC violations | Enforced `high = max(o,h,l,c)`, `low = min(o,h,l,c)` -- violations were genuine rounding errors from resampling, not floating point (magnitudes up to 0.032) |
| Timestamp gaps | No action -- 93% are weekends, remainder are confirmed market holidays |
| Price outlier spikes | Dropped 6 confirmed single-bar data errors; retained 46 genuine macro-event outliers including the March 2020 AUD/USD crash to 0.556 |
| Stale price runs | Short runs (<= 3 bars) interpolated via time method; long runs forward-filled and flagged with `stale_flag = True` |
| Volume spikes | No action -- flagged in commentary as likely news events |

---

## 2. Stationarity Testing (Notebook 02)

### Raw Close Price

| Test | AUD/USD | EUR/USD |
|---|---|---|
| ADF statistic | -1.7689 | -2.4036 |
| ADF p-value | 0.3960 | 0.1407 |
| ADF conclusion | Non-stationary | Non-stationary |
| KPSS statistic | 5.0031 | 3.4928 |
| KPSS p-value | 0.0100 | 0.0100 |
| KPSS conclusion | Non-stationary | Non-stationary |
| Reconciliation | ARIMA required | ARIMA required |

### First Difference

| Test | AUD/USD | EUR/USD |
|---|---|---|
| ADF statistic | -13.2117 | -26.3087 |
| ADF p-value | 0.0000 | 0.0000 |
| ADF conclusion | Stationary | Stationary |
| KPSS statistic | 0.3880 | 0.1632 |
| KPSS p-value | 0.0823 | 0.1000 |
| KPSS conclusion | Stationary | Stationary |
| Integration order d | **1** | **1** |

### Seasonality Check (ACF at candidate lags after d=1 differencing)

| Lag | Meaning | AUD/USD ACF | Significant? | EUR/USD ACF | Significant? |
|---|---|---|---|---|---|
| 6 | ~1 trading day | -0.0198 | No | -0.0098 | No |
| 12 | ~2 trading days | 0.0148 | No | -0.0118 | No |
| 24 | ~4 trading days | -0.0278 | No | -0.0019 | No |
| 42 | ~1 trading week | 0.0011 | No | -0.0124 | No |

**Verdict: No seasonal component detected in either pair. ARIMA (not SARIMA) recommended.**

---

## 3. Model Selection (Notebook 02)

### Grid Search Results (p, q in range 0-4, d=1)

**AUD/USD -- AIC and BIC disagree:**

| Rank | Model | AIC | BIC |
|---|---|---|---|
| Best AIC | ARIMA(3,1,2) | -45,711.74 | -45,673.02 |
| Best BIC | ARIMA(2,1,0) | -45,701.51 | **-45,682.15** |
| 2nd BIC | ARIMA(0,1,2) | -45,701.50 | -45,682.14 |
| 3rd BIC | ARIMA(3,1,0) | -45,704.61 | -45,678.80 |

**EUR/USD -- AIC and BIC disagree:**

| Rank | Model | AIC | BIC |
|---|---|---|---|
| Best AIC | ARIMA(3,1,2) | -43,366.18 | -43,327.46 |
| Best BIC | ARIMA(0,1,1) | -43,363.28 | **-43,350.37** |
| 2nd BIC | ARIMA(1,1,0) | -43,363.19 | -43,350.28 |
| 3rd BIC | ARIMA(0,1,2) | -43,362.74 | -43,343.38 |

**Selected models (BIC preferred for better generalisability):**

| Pair | Selected Model | BIC | Rationale |
|---|---|---|---|
| AUD/USD | ARIMA(2,1,0) | -45,682.15 | BIC winner; AR(2) captures weak mean-reversion in price changes |
| EUR/USD | ARIMA(0,1,1) | -43,350.37 | BIC winner; MA(1) captures one-period shock propagation |

---

## 4. In-Sample ARIMA Fit (Notebook 03)

### AUD/USD -- ARIMA(2,1,0)

| Parameter | Coefficient | Std Error | z-stat | p-value |
|---|---|---|---|---|
| ar.L1 | -0.0536 | 0.014 | -3.860 | 0.000 |
| ar.L2 | -0.0016 | 0.016 | -0.099 | 0.921 |
| sigma2 | 2.209e-06 | 2.86e-08 | 77.268 | 0.000 |

| Metric | Value |
|---|---|
| AIC | -33,431.19 |
| BIC | -33,412.90 |
| Log-likelihood | 16,718.60 |
| Ljung-Box failures | 0 / 40 lags |
| Training bars | 3,284 |

**Note:** AR(L2) is not significant (p = 0.921). The model is parsimonious at AR(1) in practice; AR(2) was retained as BIC selection.

### EUR/USD -- ARIMA(0,1,1)

| Parameter | Coefficient | Std Error | z-stat | p-value |
|---|---|---|---|---|
| ma.L1 | -0.0134 | 0.016 | -0.863 | 0.388 |
| sigma2 | 3.974e-06 | 4.93e-08 | 80.649 | 0.000 |

| Metric | Value |
|---|---|
| AIC | -32,637.83 |
| BIC | -32,625.56 |
| Log-likelihood | 16,320.91 |
| Ljung-Box failures | 0 / 40 lags |
| Training bars | 3,402 |

**Note:** MA(L1) is not significant (p = 0.388), suggesting EUR/USD price changes are very close to a pure random walk. The coefficient is retained as selected by BIC.

---

## 5. Walk-Forward Forecast Metrics (Notebook 03)

| Metric | AUD/USD ARIMA(2,1,0) | EUR/USD ARIMA(0,1,1) |
|---|---|---|
| Test bars | 1,407 | 1,458 |
| MAE (price units) | 0.001586 | 0.001857 |
| RMSE (price units) | 0.002516 | 0.002989 |
| MAPE | 0.2344% | 0.1372% |
| Directional Accuracy | 36.96% | 36.84% |
| Bars with valid direction | 1,369 / 1,407 | 1,455 / 1,458 |

**Directional accuracy at ~37% is below chance (50%).** This result, combined with the near-zero and statistically insignificant model coefficients, is consistent with weak-form market efficiency -- the ARIMA conditional mean captures minimal predictive information about price direction at the 4-hour frequency.

---

## 6. ARCH Effect Diagnostics (Notebook 03.1)

### AUD/USD ARIMA(2,1,0) Residuals

| ARCH-LM Lags | LM Statistic | p-value | Conclusion |
|---|---|---|---|
| 5 | 33.08 | 3.62e-06 | ARCH effects present |
| 10 | 42.08 | 7.27e-06 | ARCH effects present |
| 20 | 47.54 | 4.93e-04 | ARCH effects present |

| Statistic | Value |
|---|---|
| Excess kurtosis | 61.48 (extreme fat tails) |
| Skewness | -2.09 (negative skew) |

### EUR/USD ARIMA(0,1,1) Residuals

| ARCH-LM Lags | LM Statistic | p-value | Conclusion |
|---|---|---|---|
| 5 | 7.36 | 1.95e-01 | No ARCH effects |
| 10 | 98.83 | 9.33e-17 | ARCH effects present |
| 20 | 139.49 | 6.33e-20 | ARCH effects present |

| Statistic | Value |
|---|---|
| Excess kurtosis | 9.89 (fat tails) |
| Skewness | 0.29 (slight positive skew) |

**Verdict: Significant volatility clustering in both pairs. A GARCH model for the variance process is justified.**

---

## 7. GARCH Model Selection (Notebook 03.1)

### AUD/USD -- Selection Grid

| Model | AIC | LB(10) p | LB Pass | Persistence | Converged |
|---|---|---|---|---|---|
| EGARCH(1,1) Student-t | -1,701.50 | 0.019 | No | 1.033 | Yes |
| **GARCH(1,1) Student-t** | **-1,675.54** | **0.199** | **Yes** | **1.000** | **Yes** |
| GJR-GARCH(1,1) Student-t | -1,609.80 | 0.449 | Yes | 0.975 | Yes |
| GARCH(2,1) Student-t | -1,607.92 | 0.033 | No | 0.975 | Yes |
| EGARCH(1,1) Normal | -1,150.94 | 0.004 | No | 1.013 | Yes |
| **GARCH(1,1) Normal** | **-1,141.69** | **0.101** | **Yes** | **0.995** | **Yes** |
| GJR-GARCH(1,1) Normal | -1,140.65 | 0.045 | No | 0.998 | Yes |
| GARCH(2,1) Normal | -1,133.95 | 0.094 | Yes | 0.986 | Yes |

### EUR/USD -- Selection Grid

| Model | AIC | LB(10) p | LB Pass | Persistence | Converged |
|---|---|---|---|---|---|
| EGARCH(1,1) Student-t | -3,945.55 | 0.001 | No | 1.039 | Yes |
| GJR-GARCH(1,1) Student-t | -3,723.87 | 0.000 | No | 0.973 | Yes |
| GARCH(1,1) Student-t | -3,698.91 | 0.000 | No | 0.970 | Yes |
| GARCH(2,1) Student-t | -3,638.53 | 0.000 | No | 0.975 | Yes |
| EGARCH(1,1) Normal | -3,051.87 | 0.000 | No | 1.013 | Yes |
| GJR-GARCH(1,1) Normal | -2,972.17 | 0.000 | No | 0.976 | Yes |
| **GARCH(1,1) Normal** | **-2,969.64** | **0.000** | **No** | **0.980** | **Yes** |
| GARCH(2,1) Normal | -2,966.08 | 0.000 | No | 0.975 | Yes |

**Selected models:**

| Pair | Model | Rationale |
|---|---|---|
| AUD/USD | GARCH(1,1) Normal | Lowest AIC among LB-passing models; symmetric variance appropriate (no significant leverage effect at 4H) |
| EUR/USD | GARCH(1,1) Normal | No variant achieves LB pass due to near-IGARCH dynamics (structural limitation noted); GARCH(1,1) Normal selected as most interpretable |

**Note on Student-t for AUD/USD:** The t-distribution GARCH collapses to a boundary solution (alpha=1, beta=0), losing all temporal dynamics. Normal GARCH correctly captures the clustering structure.

---

## 8. In-Sample GARCH(1,1) Fit (Notebook 03.1)

### AUD/USD

| Parameter | Value |
|---|---|
| AIC | -1,141.69 |
| BIC | -1,123.40 |
| omega | 0.000211 |
| alpha[1] | 0.007535 |
| beta[1] | 0.987537 |
| Persistence (alpha + beta) | 0.9951 (near-integrated -- shocks decay slowly) |
| Mean cond vol (annualised) | 7.93% |
| Min cond vol (annualised) | 5.54% |
| Max cond vol (annualised) | 11.63% |

### EUR/USD

| Parameter | Value |
|---|---|
| AIC | -2,969.64 |
| BIC | -2,951.35 |
| omega | 0.000483 |
| alpha[1] | 0.010000 |
| beta[1] | 0.970000 |
| Persistence (alpha + beta) | 0.9800 (near-integrated -- shocks decay slowly) |
| Mean cond vol (annualised) | 6.03% |
| Min cond vol (annualised) | 5.23% |
| Max cond vol (annualised) | 8.14% |

---

## 9. Walk-Forward GARCH Volatility (Notebook 03.1)

| Metric | AUD/USD | EUR/USD |
|---|---|---|
| Test bars | 1,408 | 1,408 |
| Mean cond vol (annualised) | 12.31% | 7.52% |
| Max cond vol (annualised) | 50.56% | 19.19% |
| Correlation with realised vol | **0.9563** | **0.9692** |

The high correlation between GARCH-forecast and realised volatility (>0.95 for both pairs) confirms the model generalises out-of-sample and is capturing genuine volatility clustering rather than overfitting.

---

## 10. ARIMA-GARCH Ensemble Results (Notebook 03.1)

### Enhancement 1: Dynamic vs Static Confidence Intervals

| Pair | CI Type | Width | Hit Rate | Closer to 95%? |
|---|---|---|---|---|
| AUD/USD | Static (ARIMA) | 0.005826 | 82.37% | No |
| AUD/USD | Dynamic (GARCH) | 0.008414 | 93.89% | Yes |
| EUR/USD | Static (ARIMA) | 0.007953 | 86.43% | No |
| EUR/USD | Dynamic (GARCH) | 0.010231 | 93.25% | Yes |

**Dynamic CI is better calibrated for both pairs.** The static ARIMA CI significantly under-covers (82-86% vs nominal 95%), confirming heteroscedasticity. GARCH dynamic CIs reach 93-94% -- closer to nominal but not exact, consistent with near-IGARCH dynamics.

### Enhancement 2: Volatility-Filtered Directional Accuracy

**AUD/USD** (baseline DA = 36.96%)

| Filter Percentile | Vol Threshold | DA | Bars Kept | Change vs Baseline |
|---|---|---|---|---|
| 50th | 0.2026 | 30.56% | 72 (6%) | -6.41pp |
| 65th | 0.2102 | 28.00% | 100 (8%) | -8.96pp |
| 80th | 0.2191 | 32.93% | 164 (13%) | -4.03pp |

**EUR/USD** (baseline DA = 38.29%)

| Filter Percentile | Vol Threshold | DA | Bars Kept | Change vs Baseline |
|---|---|---|---|---|
| 50th | 0.1547 | 32.98% | 282 (20%) | -5.31pp |
| 65th | 0.1589 | 35.35% | 396 (28%) | -2.94pp |
| 80th | 0.1641 | 34.39% | 535 (38%) | -3.90pp |

### Enhancement 3: Vol-Scaled Signal Strength

| Pair | Baseline DA | Vol-Scaled DA | Signal Strength Range |
|---|---|---|---|
| AUD/USD | 36.96% | 36.96% | -5.97 to +5.88 |
| EUR/USD | 38.29% | 38.29% | -8.60 to +8.51 |

Scaling by volatility does not change directional accuracy (sign is preserved), confirming the ARIMA direction is the primary signal driver.

### Ensemble Verdict

| Metric | AUD/USD | EUR/USD |
|---|---|---|
| Vol filter improves DA? | No (best 32.93% vs baseline 36.96%) | No (best 35.35% vs baseline 38.29%) |
| Dynamic CI better calibrated? | Yes (93.89% vs static 82.37%) | Yes (93.25% vs static 86.43%) |

The GARCH component adds genuine value for confidence interval calibration but does not improve directional accuracy. The primary limitation is that the ARIMA directional signal itself is weak (DA ~37%), so filtering a weak signal produces a filtered weak signal. The key insight is structural: ARIMA coefficients are statistically insignificant (or borderline significant) for both pairs, consistent with near-efficient FX markets at 4-hour frequency.

---

## 11. Strategy Performance (Notebook 04)

### Signal Logic

```python
signal          = np.where(predicted_price > close, 1, -1)
strategy_returns = signal.shift(1) * close.pct_change()
equity_curve    = (1 + strategy_returns).cumprod()
```

### AUD/USD Strategy -- Pyfolio Tearsheet

| Metric | Value |
|---|---|
| Backtest start | 2020-06-10 |
| Backtest end | 2021-01-15 |
| Total months | 10 |
| Annual return | **4.795%** |
| Cumulative return | 4.174% |
| Annual volatility | 7.861% |
| Sharpe ratio | **0.63** |
| Calmar ratio | 0.66 |
| Stability | 0.54 |
| Max drawdown | **-7.273%** |
| Omega ratio | 1.13 |
| Sortino ratio | 0.96 |
| Skew | 0.21 |
| Kurtosis | 1.04 |
| Tail ratio | 1.04 |
| Daily VaR | -0.971% |

### EUR/USD Strategy -- Pyfolio Tearsheet

| Metric | Value |
|---|---|
| Backtest start | 2020-06-10 |
| Backtest end | 2021-01-15 |
| Total months | 10 |
| Annual return | **1.622%** |
| Cumulative return | 1.415% |
| Annual volatility | 5.805% |
| Sharpe ratio | **0.31** |
| Calmar ratio | 0.33 |
| Stability | 0.52 |
| Max drawdown | **-4.973%** |
| Omega ratio | 1.06 |
| Sortino ratio | 0.44 |
| Skew | -0.01 |
| Kurtosis | 3.10 |
| Tail ratio | 1.08 |
| Daily VaR | -0.724% |

### Strategy Summary Comparison

| Metric | AUD/USD | EUR/USD |
|---|---|---|
| Annual return | +4.795% | +1.622% |
| Sharpe ratio | 0.63 | 0.31 |
| Sortino ratio | 0.96 | 0.44 |
| Max drawdown | -7.273% | -4.973% |
| Calmar ratio | 0.66 | 0.33 |
| Omega ratio | 1.13 | 1.06 |

Both strategies are net positive on an annualised basis, with AUD/USD showing a materially stronger risk-adjusted result (Sharpe 0.63 vs 0.31). The positive Omega ratios (>1) confirm both strategies generate more weighted gains than losses. The Sortino ratio of 0.96 for AUD/USD indicates the downside risk is well-compensated by returns.

---

## 12. Model Pipeline Summary

| Stage | AUD/USD | EUR/USD |
|---|---|---|
| Integration order | d = 1 | d = 1 |
| Seasonality | None detected | None detected |
| Selected ARIMA | (2,1,0) | (0,1,1) |
| AR(1) significant? | Yes (p < 0.001) | N/A |
| AR(2) significant? | No (p = 0.921) | N/A |
| MA(1) significant? | N/A | No (p = 0.388) |
| ARCH effects in residuals | Yes (all lags) | Yes (lags 10+) |
| Selected GARCH | GARCH(1,1) Normal | GARCH(1,1) Normal |
| GARCH persistence | 0.9951 | 0.9800 |
| OOS vol correlation | 0.9563 | 0.9692 |
| ARIMA directional accuracy | 36.96% | 36.84% |
| GARCH CI improvement | +11.5pp hit rate | +6.8pp hit rate |
| Strategy annual return | +4.795% | +1.622% |
| Strategy Sharpe | 0.63 | 0.31 |

---

## 13. Limitations

The following limitations are documented for completeness and academic honesty:

**ARIMA directional accuracy (~37%) is below the 50% random baseline.** Both AR/MA coefficients are small and at least one is statistically insignificant in each pair. The strategy produces positive returns despite this because even a slightly biased coin, when applied consistently at high frequency, can accumulate a small positive drift.

**GARCH volatility filter does not improve directional accuracy.** Filtering a weak signal reduces trade count without improving hit rate. The baseline DA would need to be above 50% for a volatility-regime filter to meaningfully improve performance.

**Near-IGARCH dynamics limit variance forecasting.** Persistence of 0.98-0.995 means volatility shocks decay extremely slowly. The GARCH forecast behaves almost like an exponential smoother of past variance, limiting its ability to predict sharp regime transitions.

**10-month pyfolio backtest window is short.** The tearsheet covers only 2020-06-10 to 2021-01-15, which excludes the extreme COVID volatility of March-April 2020. Results may overstate performance stability.

**No transaction costs modelled.** FX spread costs of approximately 1 pip round-trip (0.0001) would reduce returns, particularly for the high-frequency signal which generates a large number of position changes.

---

*Full notebook outputs, figures, and CSV exports are saved in `results/figures/` and `results/metrics/`. See [DataSheet.md](../DataSheet.md) for data provenance.*
