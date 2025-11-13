# Data Preprocessing Pipeline Documentation

**Project:** GDP Nowcasting for G7 Countries  
**Phase:** Data Preprocessing  
**Date:** November 2025  
**Status:** ✅ Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Feature Engineering](#feature-engineering)
5. [Output Files](#output-files)
6. [Technical Specifications](#technical-specifications)

---

## Overview

### Purpose

This preprocessing pipeline transforms raw economic data into analysis-ready quarterly datasets for GDP nowcasting models.

### Scope

- **Countries:** G7 (USA, Canada, UK, Japan, Germany, France, Italy)
- **Frequency:** Quarterly (1980-2025)
- **Features:** 80+ engineered features per country
- **Output:** Normalized and unnormalized datasets

### Data Sources

- **OECD:** Monthly economic indicators (2000-2025)
- **FRED:** Historical data (1980-2000)
- **Bloomberg:** Financial indicators (1987-2025)
- **Yahoo Finance:** Stock market data (USA)

---

## Pipeline Architecture
```
Raw Data Sources
    ↓
[1. Load & Merge]
    ↓ (Historical + Current + Financial)
[2. Resample to Quarterly]
    ↓ (Quarter-end alignment)
[3. Impute Missing Values]
    ↓ (Intelligent strategies by indicator type)
[4. Feature Engineering]
    ↓ (Growth rates, lags, ratios, volatility)
[5. Normalize]
    ↓ (Z-score per country)
[6. Save]
    ↓
Processed Datasets (Normalized + Unnormalized)
```

---

## Preprocessing Steps

### Step 1: Data Loading and Merging

**Process:**
1. Load current OECD data (2000-2025, monthly)
2. Load historical FRED data (1980-2000, quarterly/monthly)
3. Load Bloomberg financial data (1987-2025, monthly)
4. Merge with priority: Current > Historical

**Merge Strategy:**
- Historical data fills 1980-2000 period
- Current data overwrites overlaps (2000-2025)
- Financial indicators added as separate columns

### Step 2: Frequency Harmonization

**Objective:** Resample all data to quarterly frequency (quarter-end).

**Method:**
- **Monthly → Quarterly:** `resample('QE').last()` - take end-of-quarter value
- **Annual → Quarterly:** Cubic spline interpolation
- **Alignment:** All quarters end on last day (Mar 31, Jun 30, Sep 30, Dec 31)

**Quarter Definition:**
- Q1: Jan 1 - Mar 31
- Q2: Apr 1 - Jun 30
- Q3: Jul 1 - Sep 30
- Q4: Oct 1 - Dec 31

### Step 3: Missing Data Imputation

**Strategy:** Indicator-specific imputation methods.

| Indicator Type | Method | Examples |
|----------------|--------|----------|
| Policy Rates | Forward-fill | Interest rates, overnight rates |
| Stock Indices | Forward-fill | Stock market indices |
| Economic Volumes | Cubic interpolation | GDP, consumption, investment |
| Rates/Percentages | Linear/cubic interpolation | Unemployment, inflation, CPI |
| Trade Indicators | Cubic interpolation | Exports, imports, trade balance |

**Rationale:**
- **Forward-fill for rates:** Central bank rates persist until policy change
- **Interpolation for volumes:** Economic activity changes smoothly
- **Cubic spline:** Preserves trends when sufficient data points exist (>3)

### Step 4: Feature Engineering

**Categories Created:**

#### A. Growth Rates (Year-over-Year %)

| Original | Engineered | Formula |
|----------|------------|---------|
| `gdp_real` | `gdp_growth_yoy` | `pct_change(4) * 100` |
| `industrial_production_index` | `ip_growth` | `pct_change(4) * 100` |
| `employment_level` | `employment_growth` | `pct_change(4) * 100` |
| `household_consumption` | `consumption_growth` | `pct_change(4) * 100` |
| `capital_formation` | `investment_growth` | `pct_change(4) * 100` |
| `exports_volume` | `exports_growth` | `pct_change(4) * 100` |
| `imports_volume` | `imports_growth` | `pct_change(4) * 100` |

**Why 4-quarter lag?**
- Removes seasonal effects
- Year-over-year comparison standard
- Comparable across quarters

#### B. Lagged Features (t-1, t-2, t-4)

**Indicators lagged:**
- GDP (real/constant)
- Unemployment rate
- CPI annual growth
- Industrial production
- Interest rates

**Lags created:**
- `t-1`: 1 quarter ago
- `t-2`: 2 quarters ago
- `t-4`: 4 quarters ago (year-ago)

**Total:** 9 indicators × 3 lags = 27 lag features

#### C. Economic Ratios

| Ratio | Formula |
|-------|---------|
| `trade_gdp_ratio` | `(trade_balance / gdp_real) * 100` |
| `gov_gdp_ratio` | `(government_spending / gdp_real) * 100` |

**Purpose:** Normalize for economy size, capture structural characteristics.

#### D. Moving Averages (4-Quarter)

**Indicators smoothed:**
- `gdp_growth_yoy`
- `unemployment_rate`
- `cpi_annual_growth`
- `inflation`

**Window:** 4 quarters (annual smoothing)

**Purpose:** Remove volatility while preserving cycles.

#### E. First Differences (Δ)

**Indicators differenced:**
- `gdp_real` → `gdp_real_diff`
- `employment_level` → `employment_level_diff`
- `industrial_production_index` → `industrial_production_index_diff`

**Purpose:** Convert non-stationary levels to stationary changes.

#### F. Regime Detection Features

**Volatility Indicators:**
- `gdp_volatility_4q`: Rolling 4-quarter std of GDP growth
- `stock_volatility_4q`: Rolling 4-quarter std of stock returns

**Momentum Indicators:**
- `gdp_momentum`: Change in GDP growth vs 4 quarters ago
- `inflation_momentum`: Change in inflation vs 4 quarters ago

**Regime Flags:**
- `high_inflation_regime`: Binary (1 if inflation > 2.5%)
- `high_volatility_regime`: Binary (1 if volatility > 75th percentile)

**Interaction Terms:**
- `inflation_x_volatility`: Inflation × GDP volatility
- `unemployment_x_inflation`: Unemployment × Inflation

#### G. Financial Indicators (V7)

**Yield Curve:**
- `yield_curve_slope`: 10Y - 2Y government bond spread
- `yield_curve_curvature`: 10Y - 2×5Y + 2Y
- `yield_curve_inverted`: Binary flag for inversion

**Credit Markets:**
- `credit_spread`: Corporate - Government bond spread
- `credit_spread_change`: Quarter-over-quarter change

**Stock Market:**
- `stock_returns_1q`: 1-quarter stock return (%)
- `stock_volatility_4q`: 4-quarter rolling volatility
- `risk_adjusted_returns`: Returns / volatility

**Composite Indices:**
- `financial_conditions_index`: Z-scored combination of yield curve, credit spread, stock returns
- `real_activity_index`: Z-scored combination of building permits, consumer sentiment

**Total Financial Features:** ~20 per country

### Step 5: Normalization (Z-Score)

**Formula:**
```
normalized_value = (value - mean) / std
```

**Scope:** Per-country normalization (each country has own mean/std).

**Rationale:**
- Different scales (USA GDP in trillions, others in billions)
- Different currencies (USD, EUR, JPY, GBP, CAD)
- Preserves country-specific patterns
- Prevents large economies from dominating models

**Exclusions:** Country identifier column not normalized.

**Statistics Saved:** Mean and standard deviation for each feature saved to `{country}_normalization_stats.csv` for inverse transformation.

### Step 6: Output Generation

**Files Created Per Country:**
1. `{country}_processed_unnormalized.csv` - Original scale values
2. `{country}_processed_normalized.csv` - Z-scored values
3. `{country}_normalization_stats.csv` - Mean/std for inverse transform

**Logs Created:**
- `preprocessing_log_{Country}.txt` - Detailed processing log

**Figures Created:**
- `stationarity_{country}.png` - Rolling mean/std plots

---

## Feature Engineering

### Feature Count by Country

| Country | Original | Engineered | Total |
|---------|----------|------------|-------|
| USA | 20 | 60+ | 80+ |
| Canada | 31 | 50+ | 81+ |
| UK | 30 | 50+ | 80+ |
| Japan | 31 | 50+ | 81+ |
| Germany | 26 | 50+ | 76+ |
| France | 26 | 50+ | 76+ |
| Italy | 26 | 50+ | 76+ |

### Feature Types

**Core Economic (V1-V5):**
- Leading indicators: Industrial production, stock market, interest rates
- Coincident indicators: Employment, unemployment, trade
- Regime features: Volatility, momentum, regime flags

**Financial (V7):**
- Yield curve indicators
- Credit spreads
- Stock market features
- Composite indices

**Temporal:**
- Lagged values (t-1, t-2, t-4)
- Moving averages (4-quarter)
- Growth rates (YoY)
- First differences

---

## Output Files

### Directory Structure
```
data_preprocessing/
├── preprocessing_pipeline.py           # Main pipeline script
│
├── extension_scripts/                  # Data extension scripts
│   ├── extend_canada_data.py
│   ├── extend_uk_data.py
│   ├── extend_japan_data.py
│   ├── extend_germany_data.py
│   ├── extend_france_data.py
│   ├── extend_italy_data.py
│   ├── usa_fix_sp500.py
│   └── utils/
│       ├── fred_series_finder.py
│       └── audit_existing_data.py
│
├── outputs/
│   ├── processed_data/                 # Final datasets
│   │   ├── canada_processed_normalized.csv
│   │   ├── canada_processed_unnormalized.csv
│   │   ├── uk_processed_normalized.csv
│   │   ├── uk_processed_unnormalized.csv
│   │   ├── japan_processed_normalized.csv
│   │   ├── japan_processed_unnormalized.csv
│   │   ├── germany_processed_normalized.csv
│   │   ├── germany_processed_unnormalized.csv
│   │   ├── france_processed_normalized.csv
│   │   ├── france_processed_unnormalized.csv
│   │   ├── italy_processed_normalized.csv
│   │   ├── italy_processed_unnormalized.csv
│   │   ├── usa_processed_normalized.csv
│   │   └── usa_processed_unnormalized.csv
│   │
│   ├── stats/                          # Normalization statistics
│   │   ├── canada_normalization_stats.csv
│   │   ├── uk_normalization_stats.csv
│   │   ├── japan_normalization_stats.csv
│   │   ├── germany_normalization_stats.csv
│   │   ├── france_normalization_stats.csv
│   │   ├── italy_normalization_stats.csv
│   │   └── usa_normalization_stats.csv
│   │
│   ├── logs/                           # Processing logs
│   │   ├── preprocessing_log_Canada.txt
│   │   ├── preprocessing_log_UK.txt
│   │   ├── preprocessing_log_Japan.txt
│   │   ├── preprocessing_log_Germany.txt
│   │   ├── preprocessing_log_France.txt
│   │   ├── preprocessing_log_Italy.txt
│   │   └── preprocessing_log_USA.txt
│   │
│   └── figures/                        # Stationarity plots
│       ├── stationarity_canada.png
│       ├── stationarity_uk.png
│       ├── stationarity_japan.png
│       ├── stationarity_germany.png
│       ├── stationarity_france.png
│       ├── stationarity_italy.png
│       └── stationarity_usa.png
│
├── archive/                            # Old folder structure
│   └── old_structure/
│
└── README.md                           # This document
```

### File Descriptions

#### Unnormalized CSV Files

**Format:** `{country}_processed_unnormalized.csv`

**Contents:**
- Original scale values
- All engineered features
- Date index (quarterly)
- Country identifier column

**Use Case:**
- Visualization
- Human interpretation
- Inverse transformation reference

#### Normalized CSV Files

**Format:** `{country}_processed_normalized.csv`

**Contents:**
- Z-score standardized values (mean=0, std=1)
- All engineered features
- Date index (quarterly)
- Country identifier column

**Use Case:**
- Machine learning model training
- Model prediction

#### Normalization Statistics

**Format:** `{country}_normalization_stats.csv`

**Columns:**
- `feature_name`: Feature identifier
- `mean`: Country-specific mean
- `std`: Country-specific standard deviation

**Use Case:**
- Inverse transformation: `original_value = (normalized_value * std) + mean`
- Model interpretation
- Prediction conversion to original scale

#### Processing Logs

**Format:** `preprocessing_log_{Country}.txt`

**Contents:**
- Step-by-step execution log
- Imputation details per feature
- Feature engineering summary
- Error messages (if any)
- Timestamps

**Use Case:**
- Debugging
- Audit trail
- Reproducibility verification

#### Stationarity Plots

**Format:** `stationarity_{country}.png`

**Contents:**
- Original time series (blue line)
- 4-quarter rolling mean (red line)
- ±1 standard deviation band (shaded)
- Multiple indicators per country

**Use Case:**
- Visual stationarity assessment
- Feature transformation decisions
- Model diagnostics

---

## Technical Specifications

### Data Dimensions

| Country | Quarters | Features | Date Range |
|---------|----------|----------|------------|
| USA | 184 | 82 | 1980 Q1 - 2025 Q4 |
| Canada | 184 | 82 | 1980 Q1 - 2025 Q4 |
| UK | 184 | 82 | 1980 Q1 - 2025 Q4 |
| Japan | 184 | 82 | 1980 Q1 - 2025 Q4 |
| Germany | 183 | 76 | 1991 Q1 - 2025 Q3 |
| France | 139 | 76 | 1991 Q1 - 2025 Q3 |
| Italy | 139 | 76 | 1991 Q1 - 2025 Q3 |

### Data Types

| Column Type | Data Type | Example |
|-------------|-----------|---------|
| Date Index | `datetime64[ns]` | `2025-03-31` |
| Economic Indicators | `float64` | `2.5` (GDP growth %) |
| Binary Flags | `int64` | `1` (high inflation regime) |
| Country Identifier | `object` | `'Canada'` |

### Missing Data Handling

**Before Preprocessing:**
- USA: ~186 missing values
- Canada: ~270 missing values
- European countries: ~300-450 missing values

**After Preprocessing:**
- Core economic indicators: 0% missing
- Lagged features: ~4% missing (first 4 quarters)
- Moving averages: ~3% missing (first 3 quarters)

**Acceptable Missing:**
- Lag features lose first n quarters by design
- MA features lose first (window-1) quarters by design

### Computation Time

**Per Country:**
- Data loading: <1 second
- Resampling: <1 second
- Imputation: 1-2 seconds
- Feature engineering: 2-3 seconds
- Normalization: <1 second
- **Total:** ~5-10 seconds per country

**Full Pipeline (7 countries):** ~1 minute

### Dependencies

**Python Version:** 3.9+

**Required Libraries:**
```
pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
fredapi >= 0.5.0
yfinance >= 0.1.70
openpyxl >= 3.0.9
```

### Memory Requirements

**Per Country:**
- Raw data: ~5 MB
- Processed data: ~10 MB
- Peak memory: ~50 MB

**Full Pipeline:** ~500 MB peak memory usage

---

## Usage

### Running the Pipeline

**Single Country:**
```python
# Edit preprocessing_pipeline.py
COUNTRIES_TO_PROCESS = ['Canada']

# Run
python preprocessing_pipeline.py
```

**Multiple Countries:**
```python
COUNTRIES_TO_PROCESS = ['Canada', 'UK', 'Japan']
python preprocessing_pipeline.py
```

**All G7:**
```python
COUNTRIES_TO_PROCESS = ['USA', 'Canada', 'UK', 'Japan', 'Germany', 'France', 'Italy']
python preprocessing_pipeline.py
```

### Loading Processed Data
```python
import pandas as pd

# Load unnormalized (for interpretation)
df = pd.read_csv('outputs/processed_data/canada_processed_unnormalized.csv',
                 index_col=0, parse_dates=True)

# Load normalized (for modeling)
df_norm = pd.read_csv('outputs/processed_data/canada_processed_normalized.csv',
                      index_col=0, parse_dates=True)

# Load normalization stats (for inverse transform)
stats = pd.read_csv('outputs/stats/canada_normalization_stats.csv',
                    index_col=0)
```

### Inverse Transformation
```python
# Get stats for target variable
mean = stats.loc['gdp_growth_yoy', 'mean']
std = stats.loc['gdp_growth_yoy', 'std']

# Convert normalized prediction to original scale
prediction_original = (prediction_normalized * std) + mean
```
