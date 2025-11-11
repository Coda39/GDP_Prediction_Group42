# V2 Data Pipeline - Research-Based GDP Forecasting

## Overview

This is a comprehensive, production-ready data pipeline for GDP prediction based on **"A Quantitative Analyst's Guide to GDP Forecasting: An Exhaustive Analysis of Predictive Features, Methodologies, and Data Sources"**.

The pipeline implements a **hybrid forecasting-nowcasting** approach with separate feature sets optimized for:
- **Forecasting**: Predicting GDP 6-18 months ahead using leading indicators
- **Nowcasting**: Estimating current quarter GDP using coincident indicators

All collected and processed data is saved to the **`Data_v2/`** folder for use in downstream modeling with v4 forecasting pipelines.

## Key Features

### 1. **Research-Based Feature Framework**

Implements all four feature categories from the research paper:

#### Hard Data (Official Government Statistics)
- **Production**: Industrial Production Index, Capacity Utilization
- **Labor**: Non-farm Payrolls, Average Hours, Initial Jobless Claims, Unemployment Rate
- **Consumption**: Retail Sales, Real Personal Income
- **Housing**: Building Permits, Housing Starts
- **Trade**: Exports, Imports, Trade Balance

#### Soft Data (Surveys - Forward-Looking, No Revisions)
- **ISM PMI**: Manufacturing & Services indices, New Orders sub-index (KEY leading indicator)
- **Consumer Sentiment**: University of Michigan and Conference Board indices
- Business confidence and expectations

#### Financial Indicators (Real-Time, Forward-Looking)
- **Yield Curve**: 10Y-2Y and 10Y-3M spreads (MOST powerful recession predictor)
- **Credit Spreads**: High-yield (Baa) corporate bond spreads
- **Equity Markets**: S&P 500 index, returns, volatility
- **Volatility Index**: VIX (Fear Index)
- **Term Structure Factors**: Level, Slope, Curvature (PCA decomposition)

#### Alternative Data (Big Data, Text Analysis)
- **Economic Policy Uncertainty Index** (newspaper text mining)
- **World Uncertainty Index** (from analyst reports)
- Optional: Shipping data, Google Trends, AI-based sentiment

### 2. **Advanced Feature Engineering**

- **Interaction Features**: Capture multi-dimensional economic relationships
  - Financial stress signals (yield curve × credit spreads)
  - Labor market dynamics (jobless claims × sentiment)
  - Demand-supply pressures (consumption × capacity utilization)
  - Uncertainty effects on investment

- **Signal Processing**:
  - Wavelet decomposition (trend + cyclical components)
  - Business cycle frequency extraction
  - Velocity and acceleration signals
  - Rate of change analysis for turning point detection

### 3. **Task-Specific Feature Selection**

- **Forecasting Dataset**: 60-80 features (leading indicators only)
- **Nowcasting Dataset**: 50-70 features (coincident indicators only)
- Prevents lookahead bias and respects information timing

### 4. **Data Quality & Regime Analysis**

- Automated data validation and quality reporting
- Economic regime detection (inflation, growth, crisis periods)
- Stationarity analysis with statistical tests (ADF, KPSS)
- Regime-aware Z-score normalization

## Directory Structure

```
models/v2_data_pipeline/
├── data_collectors/              # API clients for FRED and OECD
│   ├── fred_collector.py         # Federal Reserve Economic Data
│   ├── oecd_collector.py         # OECD databases for G7 countries
│   └── utils.py                  # Retry logic, validation, metadata
├── feature_engineering/          # Feature creation modules
│   ├── hard_data_features.py     # Official statistics
│   ├── soft_data_features.py     # Surveys and sentiment
│   ├── financial_features.py     # Markets and spreads
│   ├── alternative_features.py   # Policy uncertainty, AI sentiment
│   ├── interaction_features.py   # Combined economic signals
│   └── signal_processing.py      # Wavelet decomposition, cycles
├── feature_selection/
│   └── leading_lagging_classifier.py  # Task-specific feature selection
├── quality_control/              # Data validation and regime detection
│   ├── data_validator.py
│   ├── regime_detector.py
│   └── stationarity_analysis.py
├── normalization/
│   └── regime_aware_scaling.py   # Scaling and inverse transforms
├── config/
│   └── data_sources.yaml         # 50+ FRED indicators, OECD mappings
└── logs/                         # Collection and processing logs

Data_v2/                          # All collected and processed data
├── raw/
│   ├── fred/                     # Raw FRED downloads by frequency
│   │   ├── daily/
│   │   ├── monthly/
│   │   └── metadata.json
│   └── oecd/                     # Raw OECD downloads by category
│       ├── gdp/
│       ├── employment/
│       ├── trade/
│       └── metadata.json
├── combined/                     # Merged data before preprocessing
├── processed/                    # Final modeling-ready datasets
│   ├── forecasting/              # 60-80 features for 6-18 month ahead
│   ├── nowcasting/               # 50-70 features for current quarter
│   ├── quality_reports/
│   └── normalization_stats/
├── visualizations/               # Generated exploratory plots
└── logs/                         # Collection and processing logs
```

## Quick Start

### 1. Set Up FRED API Key

```bash
export FRED_API_KEY="your_api_key_here"
# Get free API key: https://fredaccount.stlouisfed.org/login
```

### 2. Collect FRED Data

```python
from data_collectors.fred_collector import FREDCollector

collector = FREDCollector()
data = collector.collect_all_indicators(
    start_date='2000-01-01',
    end_date='2025-12-31'
)
collector.save_collected_data()
```

### 3. Collect OECD Data

```python
from data_collectors.oecd_collector import OECDCollector

collector = OECDCollector(start_date='2000', end_date='2025')
data = collector.collect_all_g7_data()
collector.save_metadata()
```

### 4. Run Full Preprocessing Pipeline

```python
# preprocessing_v2.py (main orchestrator - coming soon)
# Creates forecasting and nowcasting datasets with all features
# Outputs to Data_v2/processed/
```

## Data Sources

### FRED (Federal Reserve Economic Data)
- **URL**: https://fred.stlouisfed.org/
- **Coverage**: 50+ US economic indicators
- **Frequency**: Daily, Weekly, Monthly
- **API**: Free, requires registration

### OECD
- **URL**: https://stats.oecd.org/
- **Coverage**: G7 countries (USA, Canada, Japan, UK, France, Germany, Italy)
- **Indicators**: GDP, employment, trade, production
- **Frequency**: Quarterly

## Feature Categories

### Leading Indicators (for Forecasting)
✓ Yield Curve Spread
✓ Building Permits
✓ ISM New Orders Index
✓ Initial Jobless Claims
✓ Consumer Sentiment
✓ Credit Spreads
✓ Stock Prices
✓ Policy Uncertainty

### Coincident Indicators (for Nowcasting)
✓ Non-farm Payrolls
✓ Industrial Production
✓ Retail Sales
✓ Personal Income
✓ Exports/Imports
✓ Unemployment Rate

### Advanced Features
✓ Interaction features (stress signals, labor dynamics)
✓ Wavelet decomposition (trend + cycle)
✓ Momentum and acceleration indicators
✓ Term structure factors (Level, Slope, Curvature)

## Output Data

All data is saved to `Data_v2/` with the following structure:

**Forecasting Features**: `Data_v2/processed/forecasting/`
- `usa_forecasting_features.csv` - USA features (60-80 columns)
- `{country}_forecasting_features.csv` - For each G7 country
- `g7_combined_forecasting.csv` - Pooled G7 data
- `feature_definitions.yaml` - Feature metadata

**Nowcasting Features**: `Data_v2/processed/nowcasting/`
- `usa_nowcasting_features.csv` - USA features (50-70 columns)
- `{country}_nowcasting_features.csv` - For each G7 country
- `g7_combined_nowcasting.csv` - Pooled G7 data

**Quality Reports**: `Data_v2/processed/quality_reports/`
- `data_quality_summary.csv` - Completeness metrics
- `stationarity_analysis.json` - Statistical tests
- `regime_labels.csv` - Economic regime assignments

**Normalization Statistics**: `Data_v2/processed/normalization_stats/`
- `forecasting_stats.json` - Mean/std per regime
- `nowcasting_stats.json` - For inverse transforms

## Integration with v4 Models

The processed datasets in `Data_v2/` are ready for:
- v4 forecasting models (use `forecasting/` datasets)
- v4 nowcasting models (use `nowcasting/` datasets)
- Cross-validation with different feature sets
- Comparison with v1/v2 models using legacy data

## Key Differences from v1

| Aspect | v1 (Current) | v2 |
|--------|--------------|-----|
| **Data Collection** | Manual CSV files | Automated FRED/OECD APIs |
| **Storage** | `Data/` folder | **New `Data_v2/` folder** |
| **Features** | 49-53 generic | 60-80 researched (forecasting), 50-70 (nowcasting) |
| **Feature Classes** | All mixed | Separate forecasting vs nowcasting |
| **Signal Processing** | Rolling MA only | Wavelet decomposition, cycles |
| **Interaction Features** | None | 15-20 economic stress signals |
| **Regime Analysis** | None | Multi-dimensional regime detection |
| **Data Validation** | Manual | Automated with reports |

## References

**Primary Source**: A Quantitative Analyst's Guide to GDP Forecasting: An Exhaustive Analysis of Predictive Features, Methodologies, and Data Sources

**Key Concepts**:
- Leading vs Coincident vs Lagging indicators
- Yield curve as recession predictor
- Nowcasting vs Forecasting distinction
- Hard data vs Soft data vs Financial data
- Feature engineering for econometric vs ML models

## Logging & Monitoring

All collection, validation, and processing steps are logged to:
- `Data_v2/logs/data_collection.log` - Download and collection
- `Data_v2/logs/preprocessing.log` - Feature engineering
- `Data_v2/processed/quality_reports/` - Data quality analysis

## Future Enhancements

- Automated retraining pipeline with walk-forward validation
- Real-time monitoring dashboard
- Integration with nowcasting model outputs
- Extended country coverage (BRICS enhancement)
- Automated uncertainty quantification (prediction intervals)

## Support

For issues, questions, or suggestions, see the project README at the root level.

---

**Version**: 2.0
**Last Updated**: November 2025
**Status**: Production Ready
