# GDP Prediction Group 42 - Codebase Exploration Summary

**Project:** GDP Nowcasting & Forecasting for G7 Countries  
**Version Explored:** v3 (Latest)  
**Date:** October 2025  
**Location:** `/Users/Mateo/School/Fall_2025/CS_4485/GDP_Prediction_Group42/`

---

## 1. PROJECT STRUCTURE OVERVIEW

```
GDP_Prediction_Group42/
├── Data/                          # Raw economic data for 12 countries
│   ├── G7: USA, Canada, Japan, UK, France, Germany, Italy
│   └── BRICS: Brazil, Russia, India, China, South Africa
│
├── data_preprocessing/            # Data cleaning & feature engineering
│   ├── preprocessing_pipeline.py  # Main preprocessing module
│   ├── resampled_data/           # Output: processed quarterly data
│   └── preprocessing_figures/    # Visualization outputs
│
├── data_viz/                      # Exploratory data analysis
│   ├── exploratory_visualization.py  # EDA module
│   ├── VISUALIZATION_FINDINGS.md
│   └── figures/                   # Generated plots
│
├── models/                        # Forecasting pipelines
│   ├── quarterly_forecasting/      # v1: Baseline model
│   │   └── forecasting_pipeline.py
│   ├── quarterly_forecasting_v2/   # v2: Feature selection + regularization
│   │   └── forecasting_pipeline_v2.py
│   └── quarterly_forecasting_v3/   # v3: Advanced distribution shift handling
│       ├── forecasting_pipeline_v3.py (969 lines)
│       ├── FORECASTING_V3_RESULTS.md (787 lines)
│       ├── saved_models/           # 50 trained models (pkl files)
│       ├── results/                # CSV results files
│       ├── comparison_plots/       # Performance comparisons
│       └── figures/                # Individual model visualizations
│
└── README.md
```

---

## 2. QUARTERLY FORECASTING v3 MODEL LOCATIONS & STRUCTURE

### Main Pipeline File
**Location:** `/Users/Mateo/School/Fall_2025/CS_4485/GDP_Prediction_Group42/models/quarterly_forecasting_v3/forecasting_pipeline_v3.py`

**Size:** 969 lines | **Language:** Python | **Framework:** scikit-learn, XGBoost, joblib

### Model Configuration
- **Countries:** USA, Canada, Japan, UK (4 countries)
- **Horizons:** 1-quarter & 4-quarter ahead forecasts
- **Models per Country/Horizon:** 6-7 models
  - Ridge Regression
  - LASSO
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - Regime-Switching (NEW in v3)
  - Ensemble (weighted by validation R²)

### Saved Models Location
**Path:** `/Users/Mateo/School/Fall_2025/CS_4485/GDP_Prediction_Group42/models/quarterly_forecasting_v3/saved_models/`

**Format:** Pickle files (.pkl) serialized with joblib

**Naming Convention:** `{country}_h{horizon}_{model_name}_v3.pkl`

**Example Files:**
- `canada_h1_ridge_v3.pkl` (672 bytes - small linear model)
- `canada_h1_random_forest_v3.pkl` (725 KB - tree ensemble)
- `canada_h4_xgboost_v3.pkl` (197 KB - gradient boosting)
- `usa_h1_regime_switching_v3.pkl` (1.2 KB - regime-switching)

**Total Models:** 50 trained models across all countries/horizons/algorithms

---

## 3. QUARTERLY FORECASTING v3: KEY CAPABILITIES & IMPROVEMENTS

### A. Core Features of v3

#### 1. **Reversible Instance Normalization (RevIN)**
- **Purpose:** Address distribution shift between training (2001-2018) and test (2022-2025) periods
- **Implementation:** `RevIN` class (lines 90-129)
- **Mechanism:**
  ```
  normalize(x):   mean = avg(x), std = std(x)
                  return (x - mean) / std
  
  denormalize(y_pred): return y_pred * std + mean
  ```
- **Benefit:** Removes non-stationary statistics from input, restores them in predictions
- **Applied to:** All features after feature selection

#### 2. **Data Augmentation via Window Slicing**
- **Function:** `augment_by_window_slicing()` (lines 135-186)
- **Mechanism:**
  - Original: 69 training samples
  - Creates overlapping 2-quarter, 3-quarter, 4-quarter windows
  - Result: 270 augmented samples (3.9x expansion)
  - Weighted: Original samples weight 2.0, augmented samples weight 1.0
- **Purpose:** Reduce overfitting by providing more diverse training data
- **Benefit:** Canada Ridge improved from R²=0.222 → 0.458 on 1Q forecast

#### 3. **Regime-Switching Models**
- **Class:** `ThresholdRegimeSwitcher` (lines 192-252)
- **Mechanism:**
  - Trains separate models for different economic regimes
  - Threshold: Inflation > 3.0% = high inflation regime
  - Automatically selects appropriate model based on current inflation
  - Fallback: If regime has insufficient training data, uses full-dataset model
  
  ```python
  def predict(X_test, inflation_test):
      for x, inflation in zip(X_test, inflation_test):
          model = model_low if inflation <= 3.0 else model_high
          pred = model.predict(x)
  ```
- **Challenge:** Training period (2001-2018) entirely in low-inflation regime (0 high-inflation samples)
- **Performance:** Best for 4Q forecasts
  - USA 4Q: R² = -0.045 (vs v2 best -0.109)
  - Canada 4Q: R² = -0.081 (vs v2 best -0.487)
  - UK 1Q: R² = 0.485 (close to v2 Ridge 0.513)

#### 4. **Walk-Forward Validation**
- **Class:** `WalkForwardValidator` (lines 258-293)
- **Mechanism:** Expanding window validation simulating real-world deployment
  - Min training size: 52 quarters (13 years)
  - Step forward: 4 quarters (1 year)
  - Tests on future periods incrementally
- **Purpose:** More robust performance evaluation

#### 5. **Feature Selection**
- **Method:** LASSO-based feature selection (lines 424-447)
- **Process:**
  1. Train LASSO with alpha=10.0 on training data
  2. Rank features by coefficient magnitude
  3. Select top 15 features by importance
  4. Filter all datasets to selected features
- **Benefit:** Reduces dimensionality, improves model stability
- **Output:** Selected features saved to `selected_features_h{horizon}_v3.csv`

#### 6. **Ensemble with Dynamic Weighting**
- **Function:** `create_ensemble()` (lines 573-590)
- **Weighting:** Based on validation R² scores
- **Formula:**
  ```
  weight[i] = max(0, R²[i]) / sum(max(0, R²[j]))
  ```
- **Prediction:**
  ```
  y_pred_ensemble = Σ(weight[i] * model[i].predict(X))
  ```
- **Benefit:** Combines strengths of multiple models

---

## 4. PREDICTION OUTPUT & CONFIDENCE INTERVALS

### A. Prediction Mechanism

**Single Predictions:**
```python
# Point predictions (no confidence intervals currently)
y_test_pred = model.predict(X_test)  # Returns numpy array
```

**Ensemble Predictions:**
```python
y_test_pred_ensemble = np.zeros(len(y_test))
for model, weight in zip(model_list, ensemble_weights):
    y_test_pred_ensemble += weight * model.predict(X_test_norm)
```

### B. Current Output Format

**Results CSV Structure:** (lines 749-765)
```
country,horizon,model,version,n_features,use_revin,use_augmentation,use_regime_switching,
train_rmse,val_rmse,test_rmse,train_mae,val_mae,test_mae,train_r2,val_r2,test_r2
```

**Example Row:**
```
canada,1,Ridge,v3,15,True,True,False,0.284,2.366,0.426,0.225,1.545,0.323,0.859,-0.207,-0.488
```

**Files Generated:**
- `all_countries_h1_v3_results.csv` (1-quarter forecasts)
- `all_countries_h4_v3_results.csv` (4-quarter forecasts)
- `selected_features_h{horizon}_v3.csv` (feature importance)

### C. Confidence Interval Capability

**Current Status:** ❌ **NOT IMPLEMENTED**

**Available Information for CIs:**
1. **Residual-based:** Can calculate prediction intervals from test residuals
2. **Ensemble spread:** Distance between best/worst model predictions
3. **Validation R²:** Proxy for prediction reliability

**Implementation Opportunity:**
```python
# Pseudo-code for bootstrap confidence intervals
residuals = y_test - y_test_pred
std_residual = np.std(residuals)
ci_95_lower = y_pred - 1.96 * std_residual
ci_95_upper = y_pred + 1.96 * std_residual
```

---

## 5. DATA STRUCTURES FOR PREDICTIONS

### A. Input Data Structure

**Raw Data Format:** CSV files per country
```
Location: /Users/Mateo/School/Fall_2025/CS_4485/GDP_Prediction_Group42/data_preprocessing/resampled_data/
Files: {country}_processed_normalized.csv
```

**Example: Canada**
```python
df = pd.read_csv('canada_processed_normalized.csv', index_col=0, parse_dates=True)
# Columns: 49+ economic indicators
# Index: Quarterly dates (2001-Q1 to 2025-Q4)
# Target: gdp_growth_yoy (year-over-year %)
```

### B. Feature Matrix Structure

**Before Selection:**
- Shape: (n_samples, 49) features
- Types: Float64 (normalized to [0,1])
- Missing values: Forward-filled + interpolated

**After Selection (v3 Standard):**
- Shape: (n_samples, 15) features
- Selected via LASSO importance ranking
- Normalized by RevIN (mean=0, std=1)

**Example Feature Set for Canada 1Q:**
```
Top 5 Selected Features:
1. [Feature ranked by LASSO importance]
2. [Feature ranked by LASSO importance]
3. [Feature ranked by LASSO importance]
4. [Feature ranked by LASSO importance]
5. [Feature ranked by LASSO importance]
(Full list: selected_features_h1_v3.csv)
```

### C. Prediction Output Structure

**Point Predictions:**
```python
y_test_pred = np.array([
    0.45,   # Q1 2022 prediction
    1.23,   # Q2 2022 prediction
    2.15,   # Q3 2022 prediction
    ...
])  # Shape: (n_test_periods,), dtype=float64
```

**Ensemble Weighted Output:**
```python
y_test_pred_ensemble = np.zeros(len(y_test))
# Weighted combination of 6 models
# Each weight: 0-1, sums to 1.0
```

**Evaluation Metrics Dictionary:**
```python
metrics = {
    'train_rmse': 0.284,     # Root Mean Squared Error on training
    'val_rmse': 2.366,       # RMSE on validation
    'test_rmse': 0.426,      # RMSE on test set
    'train_mae': 0.225,      # Mean Absolute Error
    'val_mae': 1.545,
    'test_mae': 0.323,
    'train_r2': 0.859,       # R² coefficient of determination
    'val_r2': -0.207,
    'test_r2': -0.488
}
```

### D. Regime-Switching Specific Data

**Input to Regime Model:**
```python
inflation_train = df['cpi_annual_growth'].values[:len(X_train)]
inflation_test = df['cpi_annual_growth'].values[-len(X_test):]
```

**Model State:**
```python
regime_model.model_low   # Ridge trained on low-inflation samples
regime_model.model_high  # Ridge trained on high-inflation samples
regime_model.threshold   # 3.0 (inflation threshold)
regime_model.n_low_samples   # 69
regime_model.n_high_samples  # 0 (limitation)
```

---

## 6. VISUALIZATION & GRAPHING UTILITIES

### A. Exploratory Data Visualization Module
**Location:** `/Users/Mateo/School/Fall_2025/CS_4485/GDP_Prediction_Group42/data_viz/exploratory_visualization.py`

**Visualization Functions:**

| Function | Purpose | Output |
|----------|---------|--------|
| `plot_gdp_timeseries()` | GDP trends G7 + BRICS | Time series line plot |
| `plot_gdp_growth_rates()` | Year-over-year growth % | Growth rate trends |
| `plot_correlation_heatmaps()` | Indicator correlations (3 countries) | Correlation matrices |
| `plot_missing_data_patterns()` | Data completeness by indicator | Heatmap of missingness |
| `plot_indicator_distributions()` | Boxplots of key indicators | Distribution comparisons |
| `plot_leading_indicators()` | Leading vs lagging indicators (USA) | Normalized overlay plots |
| `plot_cross_country_synchronization()` | Economic cycle alignment | 4Q rolling average |
| `plot_trade_balance_analysis()` | Trade trends G7 | Balance over time |

**Output:** Saved to `/data_viz/figures/` with naming:
- `01_gdp_timeseries_all_countries.png`
- `02_gdp_growth_rates.png`
- `03_correlation_heatmaps.png`
- `04_missing_data_patterns.png`
- etc.

### B. Forecasting Pipeline Visualizations
**Location:** `/models/quarterly_forecasting_v3/`

**Built-in Visualization Functions:**

| Function | Purpose | Output |
|----------|---------|--------|
| `create_v3_comparison_plots()` | v1 vs v2 vs v3 performance | Model comparison charts |
| Subplot: Test R² by model/version | Model rankings across versions | Bar charts per country |
| Subplot: Overall improvement summary | Average R² by version | Grouped bar chart |
| Subplot: Incremental improvements | v1→v2 vs v2→v3 gains | Stacked improvement bars |

**Outputs Saved:**
- `test_r2_comparison_v1_v2_v3_h1.png` - 1Q forecast comparison
- `test_r2_comparison_v1_v2_v3_h4.png` - 4Q forecast comparison
- `overall_improvement_v1_v2_v3_h1.png` - Summary improvement
- `overall_improvement_v1_v2_v3_h4.png` - Summary improvement
- `incremental_improvements_h1.png` - Component improvements
- `incremental_improvements_h4.png` - Component improvements

### C. Visualization Patterns & Libraries

**Libraries Used:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
```

**Common Patterns:**

1. **Time Series Plots:**
   ```python
   ax.plot(df.index, df['column'], label=country, linewidth=2, alpha=0.7)
   ax.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--')  # Event markers
   ```

2. **Heatmaps:**
   ```python
   sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, vmin=-1, vmax=1, ax=ax, square=True)
   ```

3. **Bar Charts (Model Comparison):**
   ```python
   x = np.arange(len(models))
   width = 0.25
   for i, version in enumerate(['v1', 'v2', 'v3']):
       ax.bar(x + (i-1)*width, data[version], width, label=version)
   ```

4. **Boxplots (Distributions):**
   ```python
   ax.boxplot(data_list, labels=country_list, patch_artist=True,
              boxprops=dict(facecolor='lightblue', alpha=0.7),
              medianprops=dict(color='red', linewidth=2))
   ```

5. **Styling:**
   ```python
   plt.style.use('seaborn-v0_8-darkgrid')
   sns.set_palette("husl")
   fig.suptitle('Title', fontsize=16, fontweight='bold')
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig(filename, dpi=300, bbox_inches='tight')
   ```

### D. Key Visualization Outputs

**Location:** `/models/quarterly_forecasting_v3/comparison_plots/`

1. **test_r2_comparison_v1_v2_v3_h{horizon}.png**
   - 2x2 grid (one per country)
   - Bar chart: Models on X-axis, R² on Y-axis
   - 3 bars per model (v1, v2, v3)
   - Reference line at R²=0 (mean baseline)

2. **overall_improvement_v1_v2_v3_h{horizon}.png**
   - 4 countries on X-axis
   - Average R² by version
   - Shows progression: v1 → v2 → v3

3. **incremental_improvements_h{horizon}.png**
   - Shows v1→v2 improvement, v2→v3 improvement, total improvement
   - Stacked bar chart by country
   - Identifies which techniques added value

---

## 7. v3 PERFORMANCE SUMMARY

### A. 1-Quarter Ahead Forecasting Results

| Country | Best Model v3      | R² Score | RMSE    | vs v2 Best |
|---------|-------------------|----------|---------|-----------|
| **USA**     | LASSO           | -0.114   | 0.369%  | ⚠️ Worse (v2: -0.006) |
| **Canada**  | **Ridge** ⭐      | 0.458    | 0.446%  | ✅ Better (v2: 0.433) |
| **Japan**   | Ensemble        | -0.056   | 0.342%  | → Mixed |
| **UK**      | **Regime-Switching** | 0.485 | 0.170%  | → Close (v2: 0.513) |

### B. 4-Quarter Ahead Forecasting Results

| Country | Best Model v3        | R² Score | RMSE    | vs v2 Best |
|---------|----------------------|----------|---------|-----------|
| **USA**     | **Regime-Switching** | -0.045   | 0.370%  | ✅ Better (v2: -0.109) |
| **Canada**  | **Regime-Switching** | -0.081   | 0.301%  | ✅ Better (v2: -0.487) |
| **Japan**   | XGBoost             | 0.070    | 0.350%  | → Similar |
| **UK**      | Random Forest (v2)  | -0.601   | 0.125%  | → v2 still better |

### C. Key Improvements in v3

**What Worked:**
✅ Ridge + RevIN + Augmentation: Canada 1Q (R² +0.236)  
✅ Regime-Switching: USA/Canada 4Q (R² +0.064, +0.406)  
✅ Data Augmentation: Tree models stabilized (Japan Ensemble +0.280)  
✅ No catastrophic failures: All RMSE < 1%

**What Didn't Work:**
❌ RevIN for Ridge: Removed scale information (Japan Ridge -0.720)  
❌ Tree models for 4Q: Canada GB worse (-2.128)  
❌ Universal improvement: Country-specific results varied

---

## 8. CRITICAL LIMITATIONS & GAPS

### A. Confidence Intervals
**Status:** ❌ Not implemented
**Current:** Point predictions only
**Options for Implementation:**
- Bootstrap resampling on residuals
- Ensemble standard deviation
- Quantile regression

### B. Training Data Regime Diversity
**Issue:** All training period (2001-2018) in low-inflation regime
**Impact:** Regime-switching can't learn distinct high-inflation patterns
**Solution:** Extend to 1980s-1990s data

### C. Model Persistence & Loading
**Format:** Pickle files (security consideration)
**No:** Confidence intervals, prediction explanation, uncertainty quantification

---

## 9. QUICK REFERENCE: FILE LOCATIONS

| Artifact | Location |
|----------|----------|
| **Main v3 Pipeline** | `/models/quarterly_forecasting_v3/forecasting_pipeline_v3.py` |
| **v3 Results Doc** | `/models/quarterly_forecasting_v3/FORECASTING_V3_RESULTS.md` |
| **Trained Models** | `/models/quarterly_forecasting_v3/saved_models/*.pkl` |
| **Results CSV** | `/models/quarterly_forecasting_v3/results/all_countries_h{1\|4}_v3_results.csv` |
| **Comparison Plots** | `/models/quarterly_forecasting_v3/comparison_plots/*.png` |
| **EDA Visualizations** | `/data_viz/figures/*.png` |
| **Preprocessed Data** | `/data_preprocessing/resampled_data/{country}_processed_normalized.csv` |
| **Feature Selection** | `/models/quarterly_forecasting_v3/results/selected_features_h{1\|4}_v3.csv` |

---

## 10. RECOMMENDATIONS FOR NEXT STEPS

### Immediate (Implementation Ready)
1. Extract prediction intervals from ensemble model variance
2. Create explanation module for regime-switching decisions
3. Build production deployment wrapper with error handling

### Short-Term (1-2 weeks)
1. Extend training data to 1980-2000 for regime diversity
2. Implement multi-threshold regime detection (low/moderate/high)
3. Test selective v3 application per country

### Medium-Term (1-2 months)
1. Walk-forward validation with dynamic retraining
2. ML-based regime detection (beyond inflation threshold)
3. Hybrid v2/v3 production pipeline

### Long-Term (3+ months)
1. Automated performance monitoring dashboard
2. Ensemble combining ML + econometric models
3. Explainable AI (SHAP values for feature importance)

