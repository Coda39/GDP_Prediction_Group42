# GDP Prediction Group 42 - Codebase Exploration Index

This directory contains comprehensive documentation of the quarterly forecasting v3 models and the entire GDP prediction project structure.

## Documentation Files

### 1. CODEBASE_EXPLORATION_SUMMARY.md (519 lines)
**Start here for comprehensive technical understanding**

Complete overview of the project including:
- Project structure overview
- Quarterly Forecasting v3 model locations and architecture
- 6 core v3 capabilities (RevIN, augmentation, regime-switching, walk-forward, feature selection, ensemble)
- Prediction output mechanisms and confidence interval status
- Data structures used throughout the pipeline
- Visualization utilities and graphing patterns
- Performance summary tables and comparisons
- Critical limitations and gaps
- Quick reference file locations
- Recommendations for next steps

**Best for:** Understanding system architecture, technical details, implementation patterns

---

### 2. QUICK_START_GUIDE.md (210 lines)
**Start here to actually use the models**

Practical guide for working with trained models:
- How to load pre-trained pickle models
- How to list and explore available models
- How to get selected features for each country
- Model performance quick reference tables
- Explanation of key techniques (RevIN, regime-switching, augmentation)
- Current prediction output format (point predictions only)
- Workaround code for confidence intervals
- Directory structure reference
- Complete working example (load → predict → evaluate)
- Known limitations
- Improvement recommendations

**Best for:** Getting predictions from models, code examples, quick reference

---

### 3. FORECASTING_V3_RESULTS.md (787 lines)
**Already in: quarterly_forecasting_v3/FORECASTING_V3_RESULTS.md**

Detailed results and analysis document including:
- Executive summary
- Methodology changes from v2 to v3
- Detailed results by horizon (1Q and 4Q ahead)
- Country-specific analysis
- Performance comparisons (v1 vs v2 vs v3)
- Regime-switching analysis
- Production recommendations
- Next steps for improvements

**Best for:** Understanding why v3 was created, detailed performance analysis

---

### 4. quarterly_forecasting_v3/forecasting_pipeline_v3.py (969 lines)
**Already in: quarterly_forecasting_v3/forecasting_pipeline_v3.py**

The actual implementation:
- RevIN class
- Data augmentation function
- ThresholdRegimeSwitcher class
- WalkForwardValidator class
- GDPForecastingPipelineV3 main class
- Model training functions
- Ensemble creation and evaluation
- Visualization and comparison functions

**Best for:** Understanding actual implementation, modifying behavior

---

## Quick Navigation

### I want to...

**Understand the project structure**
→ Read Section 1 of CODEBASE_EXPLORATION_SUMMARY.md

**Use a trained model to make predictions**
→ Read QUICK_START_GUIDE.md sections 1-3 and "Complete Example"

**Understand how predictions are made**
→ Read Section 4 of CODEBASE_EXPLORATION_SUMMARY.md + Section 4 of QUICK_START_GUIDE.md

**Learn about the data structures**
→ Read Section 5 of CODEBASE_EXPLORATION_SUMMARY.md

**Understand visualization utilities**
→ Read Section 6 of CODEBASE_EXPLORATION_SUMMARY.md

**See how well the models perform**
→ Read Section 7 of CODEBASE_EXPLORATION_SUMMARY.md OR QUICK_START_GUIDE.md "Model Performance Quick Reference"

**Know the limitations**
→ Read Section 8 of CODEBASE_EXPLORATION_SUMMARY.md OR QUICK_START_GUIDE.md "Known Limitations"

**Find a specific file or function**
→ Use Section 9 of CODEBASE_EXPLORATION_SUMMARY.md "Quick Reference: File Locations"

**Plan improvements or next steps**
→ Read Section 10 of CODEBASE_EXPLORATION_SUMMARY.md OR QUICK_START_GUIDE.md "Next Steps for Improvement"

---

## Key Facts At A Glance

### Quarterly Forecasting v3 Models
- **Location:** `quarterly_forecasting_v3/`
- **Trained Models:** 50 pickle files in `saved_models/`
- **Countries:** USA, Canada, Japan, UK
- **Forecasting Horizons:** 1-quarter and 4-quarter ahead
- **Algorithms:** Ridge, LASSO, Random Forest, XGBoost, Gradient Boosting, Regime-Switching, Ensemble

### Best Performers
**1-Quarter Ahead:**
- Canada Ridge: R² = 0.458 (RMSE 0.446%)
- UK Regime-Switching: R² = 0.485 (RMSE 0.170%)

**4-Quarter Ahead:**
- USA Regime-Switching: R² = -0.045 (RMSE 0.370%)
- Canada Regime-Switching: R² = -0.081 (RMSE 0.301%)

### v3 Improvements Over v2
1. **RevIN** - Handles distribution shift between training and test periods
2. **Data Augmentation** - 69 samples → 270 samples (3.9x)
3. **Regime-Switching** - Separate models for different economic regimes
4. **Walk-Forward Validation** - Realistic time-series validation
5. **Enhanced Feature Selection** - LASSO-based top 15 features

### Critical Limitation
**Confidence intervals are NOT implemented** - models output point predictions only

---

## File Locations Quick Reference

```
GDP_Prediction_Group42/
├── models/                          (You are here)
│   ├── CODEBASE_EXPLORATION_SUMMARY.md      (Comprehensive guide)
│   ├── QUICK_START_GUIDE.md                  (Practical examples)
│   ├── quarterly_forecasting_v3/
│   │   ├── forecasting_pipeline_v3.py        (Main code - 969 lines)
│   │   ├── FORECASTING_V3_RESULTS.md         (Detailed results)
│   │   ├── saved_models/                     (50 trained models)
│   │   ├── results/                          (CSV results files)
│   │   │   ├── all_countries_h1_v3_results.csv
│   │   │   ├── all_countries_h4_v3_results.csv
│   │   │   ├── selected_features_h1_v3.csv
│   │   │   └── selected_features_h4_v3.csv
│   │   └── comparison_plots/                 (6 PNG comparison plots)
│   ├── quarterly_forecasting_v2/
│   └── quarterly_forecasting/
├── data_preprocessing/
│   ├── preprocessing_pipeline.py
│   └── resampled_data/              (Processed CSV files)
└── data_viz/
    ├── exploratory_visualization.py
    └── figures/                      (8 EDA plots)
```

---

## Example: Loading and Using a Model

```python
import joblib
import pandas as pd
from pathlib import Path

# 1. Load the model
model_path = Path('models/quarterly_forecasting_v3/saved_models/canada_h1_ridge_v3.pkl')
model = joblib.load(model_path)

# 2. Load features
features_df = pd.read_csv('models/quarterly_forecasting_v3/results/selected_features_h1_v3.csv')
features = features_df[features_df['country'] == 'canada'].sort_values('rank')['feature'].tolist()

# 3. Load data
data_path = Path('data_preprocessing/resampled_data/canada_processed_normalized.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
X_latest = df[features].iloc[-1:].values  # Latest quarter

# 4. Make prediction
forecast = model.predict(X_latest)[0]
print(f"Canada 1Q GDP Growth Forecast: {forecast:.2f}%")
```

See QUICK_START_GUIDE.md for more detailed examples.

---

## Document Statistics

| Document | Lines | Size | Focus |
|----------|-------|------|-------|
| CODEBASE_EXPLORATION_SUMMARY.md | 519 | 18 KB | Comprehensive technical overview |
| QUICK_START_GUIDE.md | 210 | 6.7 KB | Practical code examples |
| FORECASTING_V3_RESULTS.md | 787 | 30 KB | Detailed performance analysis |
| forecasting_pipeline_v3.py | 969 | 37 KB | Implementation code |

---

## Contact & Updates

**Last Updated:** November 5, 2025
**Documentation Version:** 1.0
**Python Version:** 3.x
**Key Libraries:** scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn, joblib

---

**Start with:** 
- QUICK_START_GUIDE.md if you want to use the models immediately
- CODEBASE_EXPLORATION_SUMMARY.md if you need to understand the architecture
