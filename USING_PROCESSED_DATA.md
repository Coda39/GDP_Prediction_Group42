# Using Processed Feature Engineering Data

Quick reference guide for loading and using the forecasting and nowcasting datasets.

---

## Dataset Locations

```
Data_v2/processed/
├── forecasting/
│   └── usa_forecasting_features.csv      # 3.5 MB | 3,593 rows × 52 cols
├── nowcasting/
│   └── usa_nowcasting_features.csv       # 2.2 MB | 3,593 rows × 31 cols
└── metadata.json                          # Feature lists and metadata
```

---

## Loading the Data

### Basic Loading

```python
import pandas as pd

# Load forecasting dataset
df_forecast = pd.read_csv('Data_v2/processed/forecasting/usa_forecasting_features.csv', index_col=0, parse_dates=True)

# Load nowcasting dataset
df_nowcast = pd.read_csv('Data_v2/processed/nowcasting/usa_nowcasting_features.csv', index_col=0, parse_dates=True)

print(f"Forecasting shape: {df_forecast.shape}")   # (3593, 52)
print(f"Nowcasting shape: {df_nowcast.shape}")     # (3593, 31)
print(f"Date range: {df_forecast.index.min()} to {df_forecast.index.max()}")
```

### With Metadata

```python
import json

with open('Data_v2/processed/metadata.json', 'r') as f:
    metadata = json.load(f)

# Check what features are in each dataset
forecast_features = metadata['forecasting']['feature_list']
nowcast_features = metadata['nowcasting']['feature_list']

print(f"Forecasting features ({len(forecast_features)}):")
for feat in forecast_features[:10]:
    print(f"  - {feat}")
```

---

## Understanding the Datasets

### Forecasting Dataset

**Purpose**: Predict GDP growth 6-18 months ahead

**Key Features**:
- Target: `GDPC1` (Real GDP in billions, chained 2012 dollars)
- 51 leading indicators and derived features
- Leading indicators move before GDP (economic signals)
- Appropriate for quarterly GDP forecasting

**Example Features**:
```python
# Raw leading indicators
'ICSA'                    # Initial Jobless Claims (weekly)
'PERMIT'                  # Building Permits
'HOUST'                   # Housing Starts
'UMCSENT'                 # Consumer Sentiment
'VIXCLS'                  # VIX (Market Volatility)

# Derived features
'ICSA_ma_3'              # 3-month moving average
'PERMIT_growth_3m'        # 3-month growth rate
'T10Y2Y'                  # 10Y-2Y yield spread
'financial_stress_index'  # Combination of spreads
'USEPUINDXD_wavelet'      # Wavelet-decomposed EPU
```

**Use Cases**:
- Multi-step ahead forecasting (6-18 months)
- Leading economic indicator models
- Recession prediction
- Forward-looking analysis

### Nowcasting Dataset

**Purpose**: Estimate current quarter's GDP growth (real-time)

**Key Features**:
- Target: `GDPC1` (Real GDP)
- 30 coincident indicators and derived features
- Coincident indicators move with GDP (contemporary economic data)
- Appropriate for current-quarter GDP nowcasting

**Example Features**:
```python
# Raw coincident indicators
'INDPRO'                  # Industrial Production Index
'PAYEMS'                  # Total Nonfarm Payroll
'UNRATE'                  # Unemployment Rate
'RSXFS'                   # Retail Sales
'M2SL'                    # Money Supply M2
'CPIAUCSL'                # Consumer Price Index

# Derived features
'INDPRO_growth_3m'        # 3-month growth rate
'PAYEMS_ma_6'             # 6-month moving average
'UNRATE_lag1'             # Lagged unemployment
'trade_balance'           # Exports - Imports
'IMPGS_growth_mom'        # Month-over-month growth
```

**Use Cases**:
- Current-quarter GDP estimation
- Real-time monitoring
- Nowcasting models (Kalman filters, bridge equations)
- Confirmation of forecasts
- High-frequency economic tracking

---

## Data Characteristics

### Time Series Properties

```python
# Check data properties
df_forecast.info()          # Data types, non-null counts
df_forecast.describe()      # Basic statistics

# Date range
print(f"Forecasting: {df_forecast.shape[0]} observations")
print(f"Daily frequency effective after 2016-01-10")
print(f"3,593 fully-aligned observations (all features present)")
```

### Normalization

All features are **normalized using regime-aware Z-score scaling**:

```python
# Normalized: mean ≈ 0, std ≈ 1 (within regime)
print(df_forecast.mean())   # Should be close to 0
print(df_forecast.std())    # Should be close to 1

# Separately normalized for:
# - Normal economic regime (70% of data)
# - Crisis economic regime (30% of data)
```

This means:
- Values centered around 0 with unit variance
- Crisis periods preserve their relative magnitude
- No lookback bias in normalization
- Ready for any standardization-sensitive model

---

## Common Workflows

### 1. Time Series Train-Test Split

```python
# Standard time-series split (no look-ahead)
from sklearn.model_selection import TimeSeriesSplit

df = df_forecast
y = df['GDPC1']
X = df.drop('GDPC1', axis=1)

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"Test:  {X_test.index[0]} to {X_test.index[-1]}")
```

### 2. Basic Forecasting Model

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv('Data_v2/processed/forecasting/usa_forecasting_features.csv',
                  index_col=0, parse_dates=True)

# Prepare
X = df.drop('GDPC1', axis=1)
y = df['GDPC1']

# Split
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²:  {r2_score(y_test, y_pred):.3f}")
```

### 3. Feature Importance Analysis

```python
import matplotlib.pyplot as plt

# Get feature importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

# Plot top 20 features
plt.figure(figsize=(10, 6))
plt.title('Top 20 Most Important Features')
plt.bar(range(20), importances[indices[:20]])
plt.xticks(range(20), features[indices[:20]], rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### 4. Comparing Forecasting vs Nowcasting

```python
# Load both datasets
df_forecast = pd.read_csv('Data_v2/processed/forecasting/usa_forecasting_features.csv',
                          index_col=0, parse_dates=True)
df_nowcast = pd.read_csv('Data_v2/processed/nowcasting/usa_nowcasting_features.csv',
                         index_col=0, parse_dates=True)

# Note: Different feature counts
print(f"Forecasting features: {df_forecast.shape[1]}")  # 52
print(f"Nowcasting features: {df_nowcast.shape[1]}")   # 31

# Train separate models
model_forecast = train_model(df_forecast)
model_nowcast = train_model(df_nowcast)

# Compare performance
print("\nForecast Model R²:", evaluate(model_forecast, df_forecast))
print("Nowcast Model R²:", evaluate(model_nowcast, df_nowcast))
```

### 5. Feature Inspection

```python
# What features are available?
print("Forecasting features:")
print(df_forecast.columns.tolist())

# Which are raw indicators vs. derived?
raw = [col for col in df_forecast.columns if '_' not in col]
derived = [col for col in df_forecast.columns if '_' in col]

print(f"\nRaw indicators: {len(raw)}")
print(f"Derived features: {len(derived)}")

# Check data availability
missing = df_forecast.isnull().sum()
print(f"\nMissing values per feature:")
print(missing[missing > 0])  # Should be 0 (data is clean)
```

---

## Key Statistics

### Forecasting Dataset

| Property | Value |
|----------|-------|
| Shape | 3,593 × 52 |
| Date Range | 2016-01-10 to 2025-11-10 |
| Features | 52 (51 features + GDPC1 target) |
| Data Type | float64 (normalized) |
| Missing Values | None (cleaned) |
| File Size | 3.5 MB |

### Nowcasting Dataset

| Property | Value |
|----------|-------|
| Shape | 3,593 × 31 |
| Date Range | 2016-01-10 to 2025-11-10 |
| Features | 31 (30 features + GDPC1 target) |
| Data Type | float64 (normalized) |
| Missing Values | None (cleaned) |
| File Size | 2.2 MB |

---

## Performance Tips

### 1. Appropriate Model Selection

```python
# For forecasting (leading indicators, longer horizon):
# Good: LSTM, GRU, XGBoost, Random Forest
# Also good: VAR, ARIMA with exogenous variables

# For nowcasting (coincident indicators, current quarter):
# Good: Ridge/Lasso, Elastic Net, XGBoost
# Also good: Bridge equations, Nowcasting models (Kalman filter)
```

### 2. Time Series Specific Validation

```python
# Always use time-series aware cross-validation
# Never shuffle - temporal order matters!

from sklearn.model_selection import TimeSeriesSplit

# Good ✓
tscv = TimeSeriesSplit(n_splits=5)
for train, test in tscv.split(X):
    model.fit(X.iloc[train], y.iloc[train])
    score = model.score(X.iloc[test], y.iloc[test])

# Bad ✗
from sklearn.model_selection import cross_val_score
# This shuffles by default - don't use for time series!
```

### 3. Handle Regime Changes

```python
# Optional: Train separate models for normal vs. crisis regimes
# since data is normalized separately by regime

# Detect regime from target variable
gdp_returns = df['GDPC1'].pct_change()
volatility = gdp_returns.rolling(60).std()
is_crisis = volatility > volatility.median() * 1.5

# Train on normal regime, test on crisis (or vice versa)
model_normal = train_on_subset(X[~is_crisis], y[~is_crisis])
model_crisis = train_on_subset(X[is_crisis], y[is_crisis])
```

### 4. Feature Scaling (Already Done!)

```python
# Data is already normalized - typically no need for additional scaling
# But if you add new features, scale them consistently:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training data only (prevent look-ahead bias)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Troubleshooting

### Issue: "File not found"
```python
# Make sure you're in the project root directory
import os
os.chdir('/Users/Mateo/School/Fall_2025/CS_4485/GDP_Prediction_Group42')
```

### Issue: Date parsing
```python
# Index should be datetime, not string
df = pd.read_csv('...', index_col=0, parse_dates=True)
print(type(df.index))  # Should be: DatetimeIndex
```

### Issue: NaN values appear after operations
```python
# Data is clean, but operations may create NaNs
# Forward fill or drop as appropriate
df.fillna(method='ffill', inplace=True)  # Forward fill
df.dropna(inplace=True)                   # Drop rows with any NaN
```

---

## Next Steps

1. **Load the data** using provided code snippets
2. **Explore features** - understand what each one represents
3. **Train models** - start with simple models (linear regression, XGBoost)
4. **Evaluate performance** - use time-series aware metrics
5. **Iterate** - try different features, models, hyperparameters
6. **Interpret results** - which features matter most?

---

**Created**: November 11, 2025
**Data Ready**: ✅ Yes
**Next**: Start model training!
