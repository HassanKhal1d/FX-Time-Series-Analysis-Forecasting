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
| Modelling approach | ARMA / ARIMA / SARIMA selection via AIC/BIC |
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
fx-timeseries-capstone/
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
│   ├── 04_trade_strategy.ipynb
│   └── 05_performance_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py           # Loading, resampling, sanity checks
│   ├── model_selection.py      # ACF/PACF, ADF test, AIC/BIC grid search
│   ├── forecasting.py          # ARMA/ARIMA/SARIMA fit and predict wrappers
│   ├── strategy.py             # Signal generation and position sizing
│   └── performance.py          # Pyfolio tearsheet helpers
│
├── results/
│   ├── figures/                # Saved plots (ACF, residuals, equity curves)
│   └── metrics/                # CSV exports of model diagnostics and returns
│
├── tests/
│   └── test_data_utils.py
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

### Running the full pipeline via src modules

```python
from src.data_utils import load_and_resample, run_sanity_checks
from src.model_selection import select_model_order
from src.forecasting import fit_and_forecast
from src.strategy import generate_signals, compute_returns
from src.performance import run_tearsheet

df = load_and_resample("data/fx_pair_1_5m", freq="4h")
run_sanity_checks(df)

order, model_type = select_model_order(df["close"])
forecasts = fit_and_forecast(df["close"], order=order, model_type=model_type)

returns = compute_returns(generate_signals(df["close"], forecasts))
run_tearsheet(returns)
```

---

## Methodology

### 1. Data Sanity Check

Before any modelling, is the data actually trustworthy? The notebook checks for:
- **Missing timestamps**: are there gaps at expected bar intervals?
- **Zero-volume bars**: could these represent illiquid periods or data artefacts?
- **Price outliers**: are there bars where the high-low spread is anomalously wide?
- **Duplicate timestamps**: a common issue in vendor-provided tick data

### 2. Stationarity Testing

A critical question: is the close price series stationary, or does it wander? We apply:
- **Augmented Dickey-Fuller (ADF) test** on raw close prices and on first differences
- **KPSS test** as a complementary check

The null hypothesis of the ADF test is that the series has a unit root. Rejecting it means we can proceed with ARMA; failing to reject means the series needs differencing, pointing to ARIMA.

### 3. Model Selection

Given the stationarity result, how do we pick the right model and the right lag order?

| Model | When to use |
|-------|-------------|
| **ARMA(p, q)** | Series is already stationary (I=0) |
| **ARIMA(p, d, q)** | Series needs d differences to become stationary |
| **SARIMA(p, d, q)(P, D, Q, m)** | Seasonal autocorrelation is present at lag m |

We use a grid search over candidate (p, q) pairs and select by **AIC** and **BIC**. ACF and PACF plots guide the initial candidate range.

### 4. Trade Strategy

The model produces a 1-step-ahead close price forecast at each bar. The strategy is:
- **Long** if `forecast > current_close` (predicted price will rise)
- **Short** if `forecast < current_close` (predicted price will fall)
- **Flat** if the predicted move is below a configurable threshold (to avoid trading noise)

Returns are computed as the log-return of the position multiplied by the actual next-bar return.

### 5. Performance Analysis

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
| AUD/USD | TBD | TBD | TBD | TBD |
| EUR/USD | TBD | TBD | TBD | TBD |

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

Some things worth asking:
- The model is refit on an expanding window -- would a rolling window produce different results, and if so, why?
- Could exogenous variables (macro releases, volatility regime indicators) improve forecast accuracy? That would point toward ARIMAX or SARIMAX.
- The strategy currently uses equal position sizing. Would Kelly-criterion sizing change the risk profile?
- The data ends in January 2021. Would the model generalise to the post-pandemic FX regime?

---

## License

This project is submitted as coursework for the QuantInsti Quantitative Learning programme. The code is released under the MIT License. The underlying FX data files are provided by QuantInsti and are subject to their terms of use -- please do not redistribute them.

---

## Acknowledgements

- [QuantInsti Quantitative Learning Private Limited](https://www.quantinsti.com/) for the course structure, notebook template, and sample data
- [statsmodels](https://www.statsmodels.org/) for ARIMA/SARIMA implementations
- [pyfolio](https://pypi.org/project/pyfolio/) for strategy performance analysis
- [pandas](https://pandas.pydata.org/) for time series manipulation
