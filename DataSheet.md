# DataSheet: FX Pair OHLCV Time Series Dataset

This datasheet documents the two FX price datasets used in the Financial Time Series Analysis capstone project. It follows the datasheet framework proposed by Gebru et al. (2018) and is intended to help anyone who works with this data understand what it contains, where it came from, and where it should and should not be applied.

---

## Motivation

**For what purpose was the dataset created?**

The dataset was created to support a supervised capstone project on financial time series modelling. The specific aim is to provide learners with realistic, high-frequency FX price data on which to practise stationarity testing, ARMA/ARIMA/SARIMA model selection, 1-step-ahead forecasting, and algorithmic strategy construction. The dataset is intentionally scoped to a defined date range so that learners can study a period containing a range of market conditions -- including the extreme volatility of early 2020 -- without having to source and clean vendor data themselves.

**Who created the dataset?**

The dataset was compiled and provided by **QuantInsti Quantitative Learning Private Limited** as supplementary learning material for their algorithmic trading programme. The raw underlying price data originates from FX market data providers; QuantInsti prepared and serialised the files in their current format.

**Who funded the creation of the dataset?**

QuantInsti Quantitative Learning Private Limited funded the collection, preparation, and distribution of the dataset as part of their course infrastructure.

---

## Composition

**What do the instances represent?**

Each instance (row) in the dataset represents a single **5-minute OHLCV bar** for a foreign exchange currency pair. A bar aggregates all trades that occurred within a 5-minute window into five values: the opening price, the highest price reached, the lowest price reached, the closing price, and total traded volume.

**How many FX pairs are there, and how many instances per pair?**

| Dataset | FX Pair | Instances | Date Range |
|---------|---------|-----------|------------|
| `fx_pair_1_5m` | AUD/USD (Australian Dollar / US Dollar) | 225,385 | 2018-01-01 to 2021-01-16 |
| `fx_pair_2_5m` | EUR/USD (Euro / US Dollar) | 225,385 | 2018-01-01 to 2021-01-16 |

Both datasets share the same temporal extent and bar count, which is consistent with the FX market operating 24 hours a day, 5 days a week across the full date range.

**What columns does each dataset contain?**

| Column | Type | Description |
|--------|------|-------------|
| `index` | `DatetimeIndex` (UTC, no tz-info) | Bar open timestamp, 5-minute frequency |
| `open` | `float64` | Price at the start of the 5-minute bar |
| `high` | `float64` | Highest traded price within the bar |
| `low` | `float64` | Lowest traded price within the bar |
| `close` | `float64` | Price at the end of the 5-minute bar |
| `volume` | `float64` | Aggregate traded volume within the bar |

**Summary statistics:**

AUD/USD (`fx_pair_1_5m`):

| Statistic | Open | High | Low | Close | Volume |
|-----------|------|------|-----|-------|--------|
| Mean | 0.7130 | 0.7131 | 0.7129 | 0.7129 | 692,598 |
| Std dev | 0.0408 | 0.0408 | 0.0408 | 0.0408 | 625,706 |
| Min | 0.5520 | 0.5520 | 0.5520 | 0.5510 | 1,000 |
| 25th pct | 0.6870 | 0.6870 | 0.6870 | 0.6870 | 317,000 |
| Median | 0.7120 | 0.7120 | 0.7120 | 0.7120 | 545,000 |
| 75th pct | 0.7370 | 0.7370 | 0.7370 | 0.7370 | 874,000 |
| Max | 0.8130 | 0.8130 | 0.8130 | 0.8130 | 11,485,000 |

EUR/USD (`fx_pair_2_5m`):

| Statistic | Open | High | Low | Close | Volume |
|-----------|------|------|-----|-------|--------|
| Mean | 1.3190 | 1.3191 | 1.3190 | 1.3189 | 760,130 |
| Std dev | 0.0358 | 0.0358 | 0.0359 | 0.0358 | 695,360 |
| Min | 1.2250 | 1.2260 | 1.2240 | 1.2250 | 1,000 |
| 25th pct | 1.3020 | 1.3020 | 1.3020 | 1.3020 | 309,000 |
| Median | 1.3180 | 1.3180 | 1.3180 | 1.3180 | 578,000 |
| 75th pct | 1.3330 | 1.3330 | 1.3330 | 1.3330 | 986,000 |
| Max | 1.4660 | 1.4660 | 1.4660 | 1.4660 | 20,486,000 |

**Is there any missing data?**

No missing values (NaN) were detected in any column for either dataset. However, consumers should note that "no NaN" does not rule out all data quality issues -- stale prices, zero-volume bars, or timestamp gaps where the market was briefly illiquid may still be present and should be investigated during the sanity check phase.

**Does the dataset contain confidential data?**

No. FX price data is publicly observable market data. No personal data, non-public communications, or legally privileged information is present.

---

## Collection Process

**How was the data acquired?**

The data was sourced from institutional FX market data feeds and aggregated at a 5-minute bar frequency. The specific data vendor or exchange connectivity used by QuantInsti to source the raw ticks is not publicly documented. The files were serialised as pandas DataFrames using Python's `pickle` protocol.

**Is this a sample of a larger dataset?**

Yes. The 5-minute bar data is itself an aggregation of underlying tick-by-tick trade data. The three-year window provided (2018-2021) is a subset of the full historical depth available for both currency pairs.

**Over what time frame was the data collected?**

Both datasets cover the period from **2018-01-01 00:00 UTC** to **2021-01-16 00:00 UTC**, a span of approximately three years and two weeks. This window is notable because it encompasses several distinct market regimes: relatively calm trending conditions through 2018-2019, and the extreme volatility event of Q1 2020 associated with global COVID-19 uncertainty.

---

## Preprocessing and Cleaning

**Was any preprocessing done before the files were distributed?**

The data has been aggregated from tick-level trades to 5-minute OHLCV bars. This is the primary transformation applied. Beyond aggregation, the data appears to have been lightly cleaned -- there are no NaN values in the distributed files -- though the specific cleaning steps applied by QuantInsti are not documented.

**What preprocessing is expected of the consumer?**

The capstone project guidelines specify that consumers should **resample the 5-minute bars to 4-hour bars** before modelling. This reduces noise, improves stationarity properties, and makes model fitting computationally tractable. The recommended resampling aggregation is:

```python
df.resample("4h").agg({
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum"
}).dropna()
```

Consumers should also apply their own sanity checks, including tests for outliers, stale-price runs, and anomalous volume spikes.

**Is the raw data preserved alongside the processed data?**

The distributed files represent the raw 5-minute bars. Any further derived datasets (4-hour resampled series, log-return series, differenced series) are created locally by the consumer and are not included in the repository.

---

## Uses

**What other tasks could this dataset be used for?**

Beyond the capstone use case, the dataset could reasonably be used for:

- Exploratory analysis of FX market microstructure at the 5-minute frequency
- Testing alternative forecasting approaches -- neural networks, GARCH models for volatility, or factor-based models
- Pairs trading or cointegration analysis between AUD/USD and EUR/USD
- Regime detection: can an unsupervised model identify the March 2020 volatility shift without being told about it?
- Benchmarking new time series libraries or model implementations

**Is there anything about the data that might affect future uses or create risks?**

A few things are worth noting:

- The data covers a very specific three-year window that includes an unusual macro shock (COVID-19 in 2020). Models trained on this window may behave differently when applied to calmer or structurally different regimes. Whether that is a limitation or a feature depends on the downstream use case.
- Volume figures in FX are indicative rather than authoritative: because FX trading is decentralised across multiple venues, no single provider captures all market activity. Volume values reflect the provider's own order flow and should not be interpreted as total global FX volume.
- Price series for both pairs have a long-term drift component. Close prices are not stationary in levels; any model that assumes stationarity must difference the series first. Applying ARMA directly to the raw close price series without checking this assumption would be incorrect.
- The minimum volume recorded is 1,000 units, and some bars may reflect very low-liquidity periods (e.g. late Friday evening or early Monday morning). These bars may behave differently from bars during active trading hours.

**Are there tasks for which this dataset should not be used?**

The dataset should not be used for:

- Live trading decisions without independent data validation and risk controls. The data is provided for learning purposes and has not been independently audited.
- Drawing conclusions about total FX market volume or liquidity, given that the volume figures are provider-specific.
- Any use that involves redistributing the data to third parties, as it is subject to QuantInsti's terms of use.

---

## Distribution

**How has the dataset already been distributed?**

The dataset is distributed directly to enrolled students of the QuantInsti algorithmic trading programme as part of the capstone project unit. It is not publicly available through a data registry or open-data platform.

**Is the dataset subject to copyright or IP restrictions?**

Yes. The dataset was compiled and is owned by QuantInsti Quantitative Learning Private Limited. Students may use it for coursework and personal learning but should not redistribute it. Any publication of results derived from this data should credit QuantInsti as the data source.

---

## Maintenance

**Who maintains the dataset?**

The dataset is maintained by **QuantInsti Quantitative Learning Private Limited**. Updates, corrections, or extended versions of the dataset would be distributed through the same course platform.

For questions about data provenance, licensing, or corrections, contact QuantInsti directly via [https://www.quantinsti.com/](https://www.quantinsti.com/).

---

*Datasheet prepared following the framework proposed in: Gebru, T. et al. (2018). Datasheets for Datasets. arXiv:1803.09010.*
