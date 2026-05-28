# Data Directory

Place the two raw data files here before running any notebooks:

| Filename | FX Pair | Format |
|----------|---------|--------|
| `fx_pair_1_5m` | AUD/USD | pandas pickle |
| `fx_pair_2_5m` | EUR/USD | pandas pickle |

Both files contain 225,385 rows of 5-minute OHLCV bars from 2018-01-01 to 2021-01-16.

See the root-level `DataSheet.md` for full data documentation.

**Note:** These files are excluded from version control via `.gitignore` as they are subject to QuantInsti's terms of use.
