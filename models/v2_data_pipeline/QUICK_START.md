# V2 Data Pipeline - Quick Start Guide

## What You Have

A complete, research-based data collection and feature engineering pipeline ready for GDP forecasting and nowcasting. All code is organized and documented, with data automatically saved to the new `Data_v2/` folder.

## Getting Started in 5 Steps

### Step 1: Set Up FRED API Key

```bash
# Visit https://fredaccount.stlouisfed.org/login to get free API key

# Add to your environment
export FRED_API_KEY="your_api_key_here"

# Or add to ~/.bashrc or ~/.zshrc for persistence
echo 'export FRED_API_KEY="your_api_key_here"' >> ~/.bashrc
```

### Step 2: Collect FRED Data (USA)

```python
from models.v2_data_pipeline.data_collectors.fred_collector import FREDCollector

# Initialize collector
collector = FREDCollector()

# Collect 50+ US economic indicators
data = collector.collect_all_indicators(
    start_date='2000-01-01',
    end_date='2025-12-31'
)

# Save to Data_v2/raw/fred/
collector.save_collected_data()

print(collector.get_summary())
```

**Output**: All data saved to `Data_v2/raw/fred/` organized by frequency (daily, monthly, quarterly)

### Step 3: Collect OECD Data (G7 Countries)

```python
from models.v2_data_pipeline.data_collectors.oecd_collector import OECDCollector

# Initialize collector (no API key required)
collector = OECDCollector(start_date='2000', end_date='2025')

# Collect data for all G7 countries
data = collector.collect_all_g7_data()

# Save metadata
collector.save_metadata()

print(collector.get_summary())
```

**Output**: All data saved to `Data_v2/raw/oecd/` organized by indicator type (gdp, employment, trade, etc.)

### Step 4: Create Features (Forecasting or Nowcasting)

#### For Forecasting (6-18 months ahead):

```python
import pandas as pd
from feature_engineering import (
    HardDataFeatures, SoftDataFeatures, FinancialFeatures,
    AlternativeDataFeatures, InteractionFeatures, SignalProcessing
)
from feature_selection import FeatureClassifier

# Load raw data
df = pd.read_csv('Data_v2/raw/fred/GDPC1.csv')

# Create all features
df = HardDataFeatures.create_all_hard_features(df)
df = SoftDataFeatures.create_all_soft_features(df)
df = FinancialFeatures.create_all_financial_features(df)
df = AlternativeDataFeatures.create_all_alternative_features(df)
df = InteractionFeatures.create_all_interaction_features(df)
df = SignalProcessing.create_all_signal_processing_features(df)

# Get forecasting features (leading indicators only)
forecast_features, _ = FeatureClassifier.get_features_for_task(
    'forecasting',
    df.columns.tolist()
)

# Keep only leading indicators
df_forecast = df[forecast_features]

# Save
df_forecast.to_csv('Data_v2/processed/forecasting/usa_forecasting_features.csv')
```

#### For Nowcasting (current quarter):

```python
# Same as above, but at the end:

# Get nowcasting features (coincident indicators only)
nowcast_features, _ = FeatureClassifier.get_features_for_task(
    'nowcasting',
    df.columns.tolist()
)

# Keep only coincident indicators
df_nowcast = df[nowcast_features]

# Save
df_nowcast.to_csv('Data_v2/processed/nowcasting/usa_nowcasting_features.csv')
```

### Step 5: Use Data in v4 Models

```python
# In v4 forecasting pipeline
df_forecast = pd.read_csv('Data_v2/processed/forecasting/usa_forecasting_features.csv')

# Train your forecasting model
model = train_model(df_forecast)

# Or for nowcasting
df_nowcast = pd.read_csv('Data_v2/processed/nowcasting/usa_nowcasting_features.csv')
nowcast_model = train_nowcasting_model(df_nowcast)
```

---

## Data Flow

```
FRED API → Data_v2/raw/fred/
OECD API → Data_v2/raw/oecd/
    ↓
Raw Data Merged → Data_v2/combined/
    ↓
[Feature Engineering]
    ├─ Hard Data Features (production, labor, consumption, housing, trade)
    ├─ Soft Data Features (PMI, sentiment)
    ├─ Financial Features (yield curve, spreads, equity, VIX)
    ├─ Alternative Features (policy uncertainty, shipping)
    ├─ Interaction Features (stress signals, dynamics)
    └─ Signal Processing (wavelet, cycles, momentum)
    ↓
[Task-Specific Selection]
    ├─ FORECASTING: Leading indicators only (60-80 features)
    └─ NOWCASTING: Coincident indicators only (50-70 features)
    ↓
Output → Data_v2/processed/forecasting/ and nowcasting/
             ↓
          v4 Models
```

---

## Feature Categories Included

### Hard Data (Official Statistics)
- Industrial Production Index
- Non-farm Payrolls, Hours, Claims
- Retail Sales, Personal Income
- Building Permits, Housing Starts
- Exports, Imports, Trade Balance

### Soft Data (Surveys - Forward-Looking)
- ISM Manufacturing & Services PMI
- New Orders Index (KEY leading indicator)
- Consumer Sentiment (UMich, Conference Board)

### Financial Markets (Real-Time)
- **Yield Curve** (10Y-2Y spread) - Most powerful recession predictor
- Credit Spreads (High-yield bonds)
- S&P 500 Index, Returns, Volatility
- VIX (Fear Index)
- Term Structure Factors (Level, Slope, Curvature)

### Alternative Data (Big Data)
- Economic Policy Uncertainty Index
- World Uncertainty Index
- Shipping/Maritime data (framework)
- Google Trends (framework)

### Advanced
- Interaction Features (stress signals, labor dynamics, demand-supply)
- Wavelet Decomposition (trend + cycle separation)
- Momentum & Acceleration Signals
- Business Cycle Frequency Extraction

---

## Key Features of This Pipeline

✅ **Research-Based**: Implements academic best practices from GDP forecasting literature

✅ **Separate Forecasting/Nowcasting**: Different feature sets prevent lookahead bias

✅ **Comprehensive**: 50+ FRED indicators + G7 countries from OECD

✅ **Advanced Features**: Wavelet decomposition, interaction effects, signal processing

✅ **Production-Ready**: Error handling, validation, logging, metadata tracking

✅ **Easy Integration**: Data organized in `Data_v2/` for seamless v4 model use

✅ **Well-Documented**: README, quick start, implementation status, code comments

---

## Files Reference

| File | Purpose |
|------|---------|
| `fred_collector.py` | Download from Federal Reserve |
| `oecd_collector.py` | Download from OECD for G7 |
| `hard_data_features.py` | Create production, labor, consumption features |
| `soft_data_features.py` | Create PMI, sentiment features |
| `financial_features.py` | Create yield curve, spreads, equity, VIX features |
| `alternative_features.py` | Create policy uncertainty, shipping features |
| `interaction_features.py` | Create combined economic stress signals |
| `signal_processing.py` | Create wavelet, cycle, momentum features |
| `leading_lagging_classifier.py` | Select features by timing (forecasting vs nowcasting) |

---

## Troubleshooting

### "FRED_API_KEY not set"
```bash
export FRED_API_KEY="your_key_here"
# Check: echo $FRED_API_KEY
```

### "No observations returned for [indicator]"
- Indicator may not exist or may not have data in the specified date range
- Check FRED website to verify indicator is available

### "OECD API timeout"
- Network issue or OECD server busy
- Built-in retry logic will attempt 3 times with exponential backoff

### Feature size doesn't match
- Ensure you're using the right feature set (forecasting vs nowcasting)
- Some features may be dropped if input data is missing

---

## Next Steps

1. **Immediate**: Collect data using Steps 1-3 above
2. **Short-term**: Create features using Step 4
3. **Integration**: Use processed data in v4 models (Step 5)
4. **Validation**: Compare v2 features with v1 data on model performance
5. **Monitoring**: Track data quality and regime changes over time

---

## Where Data Gets Saved

```
Data_v2/
├── raw/fred/           ← FRED downloads go here
├── raw/oecd/           ← OECD downloads go here
├── combined/           ← Merged raw data
├── processed/
│   ├── forecasting/    ← Features for 6-18 month forecasts
│   ├── nowcasting/     ← Features for current quarter
│   └── quality_reports/← Data quality analysis
└── logs/               ← Collection and processing logs
```

**All data is automatically organized and ready to use!**

---

## Questions?

Refer to:
- **Full Documentation**: `V2_DATA_PIPELINE_README.md`
- **Implementation Status**: `IMPLEMENTATION_STATUS.md`
- **Code Comments**: Each module has detailed docstrings

**Created**: November 2025
**Version**: 2.0
