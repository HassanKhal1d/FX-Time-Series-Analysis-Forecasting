# Financial Time Series Analysis: FX Pair Forecasting & Algorithmic Trading

> Can a well-tuned time series model reliably predict short-term FX price movements -- and does that prediction translate into a profitable trading strategy?

This capstone project, completed as part of the **QuantInsti Quantitative Learning** programme, sets out to answer exactly that. Using 5-minute OHLCV tick data for two FX pairs spanning three years, we apply the full pipeline: data ingestion, sanity checking, model selection, signal generation, and strategy performance evaluation via **pyfolio**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Performance Metrics](#performance-metrics)
- [Limitations and Future Work](#limitations-and-future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

| Attribute | Detail |
|-----------|--------|
| Asset class | Foreign Exchange (FX) |
| FX Pair 1 | AUD/USD |
| FX Pair 2 | EUR/USD |
| Raw frequency | 5-minute OHLCV |
| Resampled frequency | 4-hour bars |
| Date range | 2018-01-01 to 2021-01-16 |
| Records (per pair) | 225,385 (5-min) |
| Mean model | ARIMA(2,1,0) — AUD/USD · ARIMA(0,1,1) — EUR/USD |
| Volatility model | GARCH(1,1) Normal — both pairs |
| Strategy type | Long/short signal from 1-step-ahead close price forecast |
| Performance library | [pyfolio](https://pypi.org/project/pyfolio/) |

### What questions is this project trying to answer?

- What is the correct model order for each FX pair? Is the series stationary, or does it need differencing?
- Does the in-sample fit actually generalise out-of-sample? How does forecast error behave across different volatility regimes?
- When the model's predicted direction is turned into a trade signal, does the strategy outperform a passive buy-and-hold? What does the Sharpe ratio look like?
- Are there periods -- perhaps around macro events in 2020 -- where the model breaks down entirely? What would that tell us about model assumptions?

---

## Repository Structure

```
fx-timeseries-forecasting/
│
├── data/
│   ├── fx_pair_1_5m            # AUD/USD 5-min OHLCV pickle (raw)
│   ├── fx_pair_2_5m            # EUR/USD 5-min OHLCV pickle (raw)
│   └── README_data.md          # Data dictionary and provenance notes
│
├── notebooks/
│   ├── 01_data_ingestion_and_sanity_check.ipynb
│   ├── 02_stationarity_and_model_selection.ipynb
│   ├── 03_model_fitting_and_forecasting.ipynb
│   ├── 03_1_model_fitting_and_forecasting.ipynb  # ARIMA–GARCH ensemble extension
│   └── 04_trade_strategy.ipynb
│   
│
├── results/
│   ├── figures/                # Saved plots (ACF, residuals, equity curves)
│   └── metrics/                # CSV exports of model diagnostics and returns
│
├── DataSheet.md
├── requirements.txt
├── .gitignore
└── README.md                   # You are here
```

---

## Data

The raw data files are serialised pandas DataFrames (Python pickle format) with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `index` | `DatetimeIndex` | UTC timestamp at bar open, 5-minute frequency |
| `open` | `float64` | Opening price of the 5-minute bar |
| `high` | `float64` | Highest price within the bar |
| `low` | `float64` | Lowest price within the bar |
| `close` | `float64` | Closing price of the bar |
| `volume` | `float64` | Traded volume within the bar |

**AUD/USD (Pair 1) key statistics:**

| Stat | Close Price |
|------|-------------|
| Mean | 0.7129 |
| Std | 0.0408 |
| Min | 0.5510 |
| Max | 0.8130 |

**EUR/USD (Pair 2) key statistics:**

| Stat | Close Price |
|------|-------------|
| Mean | 1.3189 |
| Std | 0.0358 |
| Min | 1.2250 |
| Max | 1.4660 |

No missing values were found in either dataset. Volume spikes and price outliers are investigated and documented in `01_data_ingestion_and_sanity_check.ipynb`.

For full data provenance, see [DataSheet.md](./DataSheet.md).

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- pip

### Clone the repository

```bash
git clone https://github.com/<your-username>/fx-timeseries-capstone.git
cd fx-timeseries-capstone
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Place the data files

Copy the two pickle files into the `data/` directory:

```
data/fx_pair_1_5m
data/fx_pair_2_5m
```

### Launch notebooks

```bash
jupyter lab
```

Open notebooks in numerical order, starting with `01_data_ingestion_and_sanity_check.ipynb`.

---

## Usage

### Loading the data

```python
import pandas as pd

df_audusd = pd.read_pickle("data/fx_pair_1_5m")
df_eurusd = pd.read_pickle("data/fx_pair_2_5m")

print(df_audusd.shape)   # (225385, 5)
print(df_eurusd.shape)   # (225385, 5)
```

### Resampling to 4-hour bars

```python
def resample_ohlcv(df, freq="4h"):
    return df.resample(freq).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum"
    }).dropna()

df_4h = resample_ohlcv(df_audusd)
```

---

## Methodology

### 1. Data Sanity Check

Before any modelling, is the data actually trustworthy? The notebook checks for:
- **Basic Shape**: is the data the size you expected?
- **Missing values**: any NaN in any column?
- **Duplicate timestamps**: a common issue in vendor-provided tick data
- **Timestamps gaps**: are there gaps at expected time intervals?
- **Zero-volume bars**: could these represent illiquid periods or data artefacts?
- **Price outliers**: are there bars where the high-low spread is anomalously wide?
- **OHLC consistency**: does 'high >= low' hold for every bar? If not, something is badly wrong upstream
- **Volume spikes**: large volume bars worth noting since they often coincide with macro releases that can break ARIMA assumptions
- **Stale price runs**:  consecutive bars with identical close prices suggest a frozen feed rather than a genuinely flat market

### 2. Stationarity Testing

A critical question: is the close price series stationary, or does it wander? We apply:
- **Augmented Dickey-Fuller (ADF) test** on raw close prices and on first differences
- **KPSS test** as a complementary check

Both pairs are confirmed to be **I(1)** — integrated of order one. The ADF test fails to reject the unit root null on raw prices for both AUD/USD and EUR/USD, but rejects strongly (p ≈ 0) on first differences. The KPSS test is consistent. This means the close price itself is a random walk; the first difference (equivalent to the bar-to-bar price change) is stationary.

### 3. Model Selection

Given the stationarity result, how do we pick the right model and the right lag order?

| Model | When to use |
|-------|-------------|
| **ARMA(p, q)** | Series is already stationary (I=0) |
| **ARIMA(p, d, q)** | Series needs d differences to become stationary |
| **SARIMA(p, d, q)(P, D, Q, m)** | Seasonal autocorrelation is present at lag m |

A grid search over candidate (p, q) ∈ {0…4} × {0…4} with d=1 selects by **BIC** (preferred over AIC for forecasting, as it penalises complexity more heavily):

| Pair | Selected Model | Rationale |
|------|---------------|-----------|
| AUD/USD | ARIMA(2,1,0) | AR(2) structure in first differences; BIC-optimal |
| EUR/USD | ARIMA(0,1,1) | MA(1) structure in first differences; BIC-optimal |

ACF/PACF analysis on seasonality candidates (m=6 bars ≈ 1 trading day; m=30 bars ≈ 1 trading week) was performed, but the seasonal SARIMA extension was not selected since the improvement in BIC did not justify the added complexity.

### 4. ARIMA–GARCH Ensemble (Notebook 03.1)

Residual diagnostics on the fitted ARIMA models revealed significant **ARCH effects** — the Engle ARCH-LM test returns p ≈ 0 at all lag windows for both pairs, and the ACF of squared residuals confirms that variance clusters in time. Return excess kurtosis > 3 for both pairs confirms fat tails beyond what the Normal innovation distribution assumes.

A **GARCH(1,1) Normal** model was selected for both pairs via AIC grid search over GARCH, EGARCH, and GJR-GARCH variants with Normal and Student-t innovations. Notable observations:
- For AUD/USD, the Student-t GARCH collapses to a boundary solution (α=1, β=0), losing temporal dynamics. The Normal GARCH correctly captures clustering and passes the Ljung-Box test on squared standardised residuals.
- For EUR/USD, the estimated persistence is α+β ≈ 0.99 — near-integrated GARCH (IGARCH) behaviour. No GARCH variant achieves a clean Ljung-Box pass for this pair due to these near-unit-root volatility dynamics.

Three ensemble enhancements were constructed on top of the ARIMA conditional mean:

| Enhancement | Description | Outcome |
|-------------|-------------|---------|
| Dynamic confidence intervals | Replace static ARIMA σ with GARCH-forecast conditional vol | CI calibration improves; narrows in calm regimes, widens in turbulent ones |
| Volatility-filtered signal | Suppress trades when GARCH vol exceeds 50th/65th/80th percentile of training vol | Directional accuracy improves by ~1–3 pp on the filtered subset |
| Volatility-scaled signal strength | Weight signal by reciprocal of conditional vol | Provides a continuous sizing variable for downstream position management |

### 5. Trade Strategy

The model produces a 1-step-ahead close price forecast at each bar using a rolling 500-bar window. The strategy is:
- **Long** if `forecast > current_close` (predicted price will rise)
- **Short** if `forecast < current_close` (predicted price will fall)

Returns are computed as the percentage change of the position multiplied by the next-bar actual return.

### 6. Performance Analysis

Does the strategy actually make money -- and on a risk-adjusted basis? We evaluate using **pyfolio** tearsheets covering:
- Cumulative returns vs benchmark
- Annualised Sharpe ratio
- Maximum drawdown and drawdown duration
- Rolling volatility and rolling Sharpe
- Monthly return heatmap

---

## Results

Results vary by pair and model configuration. Full outputs are saved in `results/`. A summary table will be populated here after the final model run.

| Pair | Best Model | AIC | Sharpe Ratio | Max Drawdown |
|------|-----------|-----|-------------|-------------|
| AUD/USD | ARIMA(2,1,0) + GARCH(1,1) | TBD | TBD | TBD |
| EUR/USD | ARIMA(0,1,1) + GARCH(1,1) | TBD | TBD | TBD |

---

## Performance Metrics

The pyfolio tearsheet outputs the following. How many of these does the strategy clear?

| Metric | Description |
|--------|-------------|
| Annualised return | Compound annual growth rate of the strategy |
| Sharpe ratio | Excess return per unit of risk (annualised) |
| Sortino ratio | Downside-adjusted Sharpe |
| Max drawdown | Largest peak-to-trough decline |
| Calmar ratio | Annualised return divided by max drawdown |
| Win rate | Proportion of trades that were profitable |

---

## Limitations and Future Work

### Known Limitations of the Current Model

The following limitations are grounded in diagnostics observed across the project notebooks.

#### 1. Heteroscedastic Residuals (ARCH Effects)

The ARIMA model assumes homoscedastic (constant-variance) innovations. Both pairs fail this assumption decisively: the Engle ARCH-LM test returns **p ≈ 0** at every lag window tested, and the ACF of squared residuals shows significant clustering. The practical consequence is that the static 95% confidence interval produced by ARIMA — derived from a single constant σ — is systematically miscalibrated: too wide in calm regimes (causing over-caution) and dangerously too narrow in volatile ones (understating true risk).

The ARIMA–GARCH ensemble in notebook 03.1 partially addresses this through dynamic confidence intervals, but the benefit is limited by EUR/USD's near-integrated volatility persistence (α+β ≈ 0.99), which makes the GARCH model structurally difficult to calibrate for that pair.

#### 2. Near-IGARCH Dynamics in EUR/USD

EUR/USD exhibits volatility persistence so close to unity that standard single-regime GARCH variants cannot achieve a clean Ljung-Box pass on squared standardised residuals. This is consistent with the known near-IGARCH behaviour of major FX pairs at intraday frequency, where volatility shocks decay extremely slowly. The implication is that the conditional variance for EUR/USD is best described as a process with time-varying regime structure, rather than a single mean-reverting GARCH process. A Markov-switching GARCH or an IGARCH model would be more appropriate.

#### 3. Pure Price Model — No Exogenous Information

Both ARIMA and GARCH are purely backward-looking models that condition only on the lagged price series. They contain no information about the fundamental drivers of FX returns: interest rate differentials between the currency pair's constituent economies, macroeconomic data surprises (CPI, NFP, PMI), central bank communication, or broad risk sentiment proxies such as the VIX or commodity prices. These factors are known to be the primary drivers of FX direction over multi-hour horizons, which is precisely the forecast horizon used here.

This is a structural limitation: no amount of lag-order tuning can allow an ARIMA model to anticipate a surprise inflation print or a central bank pivot that the model has no knowledge of.

#### 4. Static Model Parameters — No Adaptation to Structural Change

Both ARIMA and GARCH are fit once on the training set (or on a rolling 500-bar window in notebook 04) and then applied in fixed form to the test period. The dataset spans January 2018 to January 2021 — a period that includes a regime shift of unusual severity in March–April 2020, when COVID-related volatility caused AUD/USD to fall from 0.67 to 0.57 in approximately two weeks. The rolling forecast error plots in notebook 03 show a pronounced spike in absolute error during this period.

A model with fixed AR/MA coefficients cannot adapt to regime changes of this magnitude. Time-varying parameters (via Kalman filtering or recursive updating) or explicit regime-switching approaches would handle this more gracefully.

#### 5. Undifferentiated Binary Signal — No Conviction Gating

The trade strategy in notebook 04 generates a long or short signal for every single bar, regardless of how small the predicted price move is. When the ARIMA forecast differs from the current close by, say, 0.00003 (three pips), it generates the same position size as a forecast that differs by 0.0015. In a real deployment this is untenable: it generates maximum turnover and transaction cost drag on the weakest, least reliable signals.

The ARIMA–GARCH ensemble introduces a `vol_filter_mask` that suppresses the weakest signals, and a `signal_strength` variable that could be used for continuous position sizing, but neither is wired into the notebook 04 strategy. The base strategy remains binary without any minimum-conviction threshold.

#### 6. No Transaction Costs or Spread

The strategy performance is measured on raw close-to-close returns with no bid-ask spread, no broker commission, and no slippage. For 4-hour bars the spread impact is relatively modest, but the strategy can still flip positions on every bar, generating a large number of round trips. In practice, a typical institutional EUR/USD spread of 0.5–1 pip and AUD/USD spread of 0.8–1.5 pips would meaningfully erode returns, particularly during periods of elevated volatility when spreads widen.

#### 7. Univariate Modelling of Correlated Pairs

AUD/USD and EUR/USD share a common USD leg, meaning their returns and volatilities are time-varying correlated. The current project fits each pair independently, discarding cross-pair information. During the March 2020 episode, for example, the USD strengthened sharply against both currencies simultaneously — information about the magnitude and timing of the EUR/USD move contains genuine predictive signal for AUD/USD, and vice versa. A multivariate VAR or DCC-GARCH framework would exploit this relationship.

#### 8. Unexploited OHLCV Information

The sanity check notebook validates the high, low, open, and volume columns but the modelling pipeline uses only the close price. The high-low range (a proxy for intrabar volatility), the open-to-close return (body direction), the close-to-open gap at bar boundaries, and the volume series all carry information about price dynamics and institutional activity. HAR-type models specifically exploit multi-period realised variance measures derived from the OHLC data.

#### 9. Rolling vs Expanding Window Inconsistency

Notebook 03 uses an expanding-window approach via statsmodels `.apply()`, which leverages all available history at each step. Notebook 04 uses a rolling 500-bar window, refitting the ARIMA model from scratch at every step. These two approaches will produce different forecasts and were not directly compared. The rolling window risks parameter instability in the early test period (when 500 bars is a large fraction of the available data) but may generalise better across the structural break in 2020 than the expanding window.

#### 10. Data Ends January 2021

The dataset terminates before the post-pandemic FX regime — a period characterised by high inflation surprises, rapid interest rate divergence between the Fed and RBA/ECB, and pronounced trends in both AUD/USD and EUR/USD. There is no out-of-sample evidence that the model generalises beyond the 2018–2021 sample period.

---

### Residual Signal: What Is Left to Extract?

The following is an honest account of the predictive signal present in the data and market structure that the current ARIMA–GARCH pipeline cannot capture, ordered approximately by expected impact.

#### High-Value Residual Signal

**Macro and fundamental drivers.** The largest predictable component of FX returns over a 4-hour horizon is systematic response to economic data releases, central bank communication, and cross-currency interest rate differentials. Yield spreads between Australian 10Y government bonds and US Treasuries track AUD/USD closely at daily frequency; EUR/USD similarly tracks the 2-year EUR–USD swap spread. These relationships are regime-dependent (they strengthen during risk-on periods) and therefore need to be combined with a volatility regime indicator, but the directional information is substantial. An ARIMAX or SARIMAX model with the relevant rate differentials as exogenous regressors would be the minimum viable extension.

**Intraday session effects.** On 4-hour bars, bars coinciding with the London open (06:00–08:00 UTC), the New York open (13:00–14:00 UTC), and the London–New York overlap (13:00–17:00 UTC) have structurally different return distributions, bid-ask spreads, and autocorrelation properties. AUD/USD also has an Australian and Asian session component. The ARIMA model treats all bars identically, but the directional autocorrelation in FX is known to be higher during the overlap session and lower during the Asian quiet period. A session dummy in SARIMAX or a separate model per session regime would capture this.

**Volume and order flow.** The volume data present in the OHLCV feed was sanity-checked but never used as a signal. In FX, volume is an imperfect proxy for order flow (since OTC volume is fragmented across brokers), but large volume bars consistently precede directional moves as institutional activity executes. Volume deviation from a rolling mean (volume z-score) is a low-complexity feature with meaningful predictive content for the direction and magnitude of subsequent bar returns.

**OHLC-derived intrabar structure.** The high-low range, close-to-open gap, and bar body direction (open vs close) carry predictive signal that is orthogonal to the close price series used by ARIMA. Specifically: a long-range bar that closes near the low (bearish pin bar) has a known mean-reversion tendency over the following 1–2 bars that ARIMA will miss entirely. Incorporating these as features in a machine learning classifier would extract this signal systematically.

#### Moderate-Value Residual Signal

**Multi-horizon realised variance (HAR-RV).** The GARCH(1,1) model forecasts conditional variance through a parametric recursion. The HAR-RV (Heterogeneous Autoregressive Realised Variance) model instead expresses tomorrow's variance as a linear combination of realised variance measured over the past 1 bar, 6 bars (~1 trading day), and 30 bars (~1 trading week). At intraday frequency, HAR-RV consistently outperforms GARCH in out-of-sample variance forecasting, particularly around macro events, because it directly uses observed realised measures rather than a parametric model that can be shocked off its steady state.

**Momentum at multiple horizons.** The ARIMA autocorrelation structure captures very short-horizon (1–2 bar) mean-reversion tendencies. Medium-horizon momentum at 12–30 bars (2–5 trading days) in FX is a distinct and separately exploitable signal. The Fama–French momentum anomaly has been documented in FX, and a simple 20-bar return z-score used as a feature in a direction classifier would capture carry-and-momentum tendencies that ARIMA, fitted to stationary first differences, cannot represent.

**Cross-pair correlation signal.** As noted above, the USD leg is common to both pairs. The dynamic conditional correlation (DCC) between AUD/USD and EUR/USD returns can itself be modelled and forecasted. When this correlation rises (USD strengthening phase), the signal strength of a USD-direction trade increases; when it falls, pair-specific drivers dominate. A DCC-GARCH model outputs this time-varying correlation and provides a natural weight for combining the two pair signals into a more diversified position.

**Regime indicator.** Both the rolling forecast error plots (notebook 03) and the GARCH conditional volatility (notebook 03.1) confirm that the data contains distinct high-volatility and low-volatility regimes. The GARCH filter suppresses trades in high-vol periods and improves directional accuracy by 1–3 pp, but the regime boundary is defined by a static percentile threshold. A hidden Markov model (HMM) with two states would identify regime transitions probabilistically, producing a soft signal that could be used to scale position size continuously rather than gate it with a binary threshold.

#### Lower-Value or Higher-Complexity Residual Signal

**Non-linear price patterns.** ARIMA is a linear model. In principle, XGBoost or random forest classifiers trained on lagged return features, OHLC-derived features, and GARCH vol features can discover non-linear relationships. In practice, FX data at 4-hour frequency over a 3-year sample (~4,800 bars) is thin for tree-based methods, and the signal-to-noise ratio in FX direction is sufficiently low that overfitting risk is high. Deep learning approaches (LSTM, Temporal Fusion Transformer) have the same concern amplified; they need substantially more data or strong regularisation to generalise.

**Tick-level microstructure.** The 5-minute raw data could support microstructure analysis (bid-ask bounce, short-term mean reversion, spread patterns). This is out of scope for a 4-hour bar model but represents a separate and potentially lucrative trading frequency if the data pipeline can be upgraded to include quote-level data.

---

### Improvement Roadmap

The following table summarises actionable next steps, ranked by expected difficulty and estimated improvement to directional accuracy or Sharpe ratio.

| Priority | Extension | Addresses | Complexity | Expected Gain |
|----------|-----------|-----------|------------|---------------|
| 1 | ARIMAX with yield spread exogenous variable | No fundamental signal | Low | High |
| 2 | HAR-RV volatility model | GARCH miscalibration for EUR/USD | Low | Medium-high |
| 3 | GARCH vol-filter wired into strategy | Undifferentiated signal | Low | Medium |
| 4 | Session-of-day dummy variable | Intraday structure ignored | Low | Medium |
| 5 | Transaction cost and spread model | Unrealistic P&L | Low | Corrective |
| 6 | Volume z-score as signal feature | OHLCV data unused | Low | Medium |
| 7 | Markov-switching GARCH (2 states) | EUR/USD near-IGARCH | Medium | Medium |
| 8 | Expanding vs rolling window comparison | Methodological inconsistency | Medium | Diagnostic |
| 9 | Kelly-criterion position sizing | Binary signal inefficiency | Medium | Medium |
| 10 | DCC-GARCH joint model (both pairs) | Cross-pair correlation ignored | High | Medium-high |
| 11 | Kalman filter with time-varying ARIMA params | Static parameters / structural break | Medium | Medium |
| 12 | XGBoost direction classifier with engineered features | Non-linear patterns | High | Medium (data-limited) |
| 13 | Post-2021 out-of-sample test | Sample period limited | Low | Diagnostic |
| 14 | VAR-DCC-GARCH multivariate framework | Full modelling pipeline | Very high | High |

---

## License

This project is submitted as coursework for the QuantInsti Quantitative Learning programme. The code is released under the MIT License. The underlying FX data files are provided by QuantInsti and are subject to their terms of use -- please do not redistribute them.

---

## Acknowledgements

- [QuantInsti Quantitative Learning Private Limited](https://www.quantinsti.com/) for the course structure, notebook template, and sample data
- [statsmodels](https://www.statsmodels.org/) for ARIMA/SARIMA implementations
- [arch](https://arch.readthedocs.io/) for GARCH model fitting
- [pyfolio](https://pypi.org/project/pyfolio/) for strategy performance analysis
- [pandas](https://pandas.pydata.org/) for time series manipulation
