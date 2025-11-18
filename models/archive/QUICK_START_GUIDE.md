# Quarterly Forecasting v3 - Quick Start Guide

## Loading and Using Trained Models

### 1. Load a Pre-trained Model
```python
import joblib
from pathlib import Path

# Load Canada Ridge model for 1-quarter ahead
models_dir = Path(__file__).parent / 'quarterly_forecasting_v3' / 'saved_models'
model = joblib.load(models_dir / 'canada_h1_ridge_v3.pkl')

# Make a prediction
import numpy as np
X_new = np.array([...])  # 15-dimensional feature vector (after feature selection)
prediction = model.predict(X_new.reshape(1, -1))[0]
print(f"Predicted GDP growth: {prediction:.2f}%")
```

### 2. List All Available Models
```python
from pathlib import Path

models_dir = Path('models/quarterly_forecasting_v3/saved_models')
all_models = sorted(models_dir.glob('*.pkl'))

# Organize by country and horizon
for model_file in all_models:
    print(f"{model_file.name}")
    # Output: canada_h1_ridge_v3.pkl
    #         canada_h1_lasso_v3.pkl
    #         canada_h1_random_forest_v3.pkl
    #         etc.
```

### 3. Load Model Metadata
```python
import pandas as pd
from pathlib import Path

results_dir = Path('models/quarterly_forecasting_v3/results')

# 1-quarter ahead results
results_h1 = pd.read_csv(results_dir / 'all_countries_h1_v3_results.csv')

# 4-quarter ahead results
results_h4 = pd.read_csv(results_dir / 'all_countries_h4_v3_results.csv')

# Best model for Canada 1Q
best_canada_1q = results_h1[results_h1['country'] == 'canada'].nlargest(1, 'test_r2')
print(best_canada_1q[['country', 'model', 'test_r2', 'test_rmse']])
```

### 4. Get Selected Features for a Country
```python
import pandas as pd
from pathlib import Path

features_dir = Path('models/quarterly_forecasting_v3/results')

# Features for 1-quarter ahead
features_h1 = pd.read_csv(features_dir / 'selected_features_h1_v3.csv')

# Get top features for Canada
canada_features = features_h1[features_h1['country'] == 'canada'].sort_values('rank')
print(canada_features['feature'].values)
```

## Model Performance Quick Reference

### Best Performers - 1 Quarter Ahead
- **Canada Ridge**: R² = 0.458 (RMSE 0.446%) - BEST OVERALL 1Q
- **UK Regime-Switching**: R² = 0.485 (RMSE 0.170%) - BEST CONFIDENCE
- **USA LASSO**: R² = -0.114 (RMSE 0.369%)
- **Japan Ensemble**: R² = -0.056 (RMSE 0.342%)

### Best Performers - 4 Quarters Ahead
- **USA Regime-Switching**: R² = -0.045 (RMSE 0.370%) - BEST USA 4Q
- **Canada Regime-Switching**: R² = -0.081 (RMSE 0.301%) - BEST CANADA 4Q
- **Japan Random Forest (v2)**: R² = 0.081 (RMSE 0.348%) - ONLY POSITIVE 4Q

## Understanding the Models

### RevIN (Reversible Instance Normalization)
- Normalizes features to mean=0, std=1
- Removes distribution shift between training and test periods
- Applied automatically to all features
- **Impact**: Helps models generalize to new economic conditions

### Regime-Switching
- Detects economic regime based on inflation rate
- Uses different models for low-inflation (<=3%) vs high-inflation (>3%) periods
- Best for 4-quarter ahead forecasting
- **Limitation**: Training period only contains low-inflation regime

### Data Augmentation
- Creates synthetic samples using overlapping windows
- Increases training data from 69 to 270 samples (3.9x)
- Helps prevent overfitting
- **Best for**: Ridge regression and stable economies (Canada, UK)

## Prediction Output Format

### What you get:
```python
model.predict(X)  # Returns numpy array
# Example: array([0.456, 1.234, 2.156, ...])
```

### What you DON'T get (yet):
- Confidence intervals
- Uncertainty bounds
- Feature importance explanations
- Regime detection results

### Workaround - Estimate Confidence Intervals:
```python
import numpy as np

# Using ensemble model variance
predictions = []
for model in [ridge, lasso, rf, xgb, gb]:
    predictions.append(model.predict(X_test))

pred_mean = np.mean(predictions, axis=0)
pred_std = np.std(predictions, axis=0)

# 95% confidence interval
ci_lower = pred_mean - 1.96 * pred_std
ci_upper = pred_mean + 1.96 * pred_std
```

## Directory Structure for Models

```
quarterly_forecasting_v3/
├── forecasting_pipeline_v3.py        # Main code
├── FORECASTING_V3_RESULTS.md         # Detailed analysis
├── saved_models/
│   ├── canada_h1_ridge_v3.pkl
│   ├── canada_h1_lasso_v3.pkl
│   ├── canada_h4_ridge_v3.pkl
│   └── ... (50 models total)
├── results/
│   ├── all_countries_h1_v3_results.csv
│   ├── all_countries_h4_v3_results.csv
│   ├── selected_features_h1_v3.csv
│   └── selected_features_h4_v3.csv
└── comparison_plots/
    ├── test_r2_comparison_v1_v2_v3_h1.png
    ├── overall_improvement_v1_v2_v3_h1.png
    └── incremental_improvements_h1.png
```

## Complete Example: Making Predictions with Canada Model

```python
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load the model
model_path = Path('models/quarterly_forecasting_v3/saved_models/canada_h1_ridge_v3.pkl')
model = joblib.load(model_path)

# 2. Load preprocessed data
data_path = Path('data_preprocessing/resampled_data/canada_processed_normalized.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# 3. Get selected features
features_df = pd.read_csv('models/quarterly_forecasting_v3/results/selected_features_h1_v3.csv')
selected_features = features_df[features_df['country'] == 'canada'].sort_values('rank')['feature'].tolist()

# 4. Prepare features
X_latest = df[selected_features].iloc[-1:].values  # Latest quarter, 15 features

# 5. Make prediction
forecast = model.predict(X_latest)[0]

# 6. Get model performance metrics
results_df = pd.read_csv('models/quarterly_forecasting_v3/results/all_countries_h1_v3_results.csv')
metrics = results_df[(results_df['country'] == 'canada') & (results_df['model'] == 'Ridge')].iloc[0]

print(f"Canada 1-Quarter Ahead GDP Growth Forecast: {forecast:.2f}%")
print(f"Model R² on Test Set: {metrics['test_r2']:.3f}")
print(f"Model RMSE on Test Set: {metrics['test_rmse']:.3f}%")
```

## Known Limitations

1. **No Confidence Intervals**: Models output point predictions only
2. **Regime Diversity Limited**: Training data doesn't include high-inflation periods
3. **UK 4Q Forecasting**: All models struggle (R² < -0.5)
4. **USA 1Q Forecasting**: Mixed results, all models negative R²
5. **Japan**: Unique deflationary dynamics don't fit inflation-based regimes

## Next Steps for Improvement

1. **Add Confidence Intervals**: Use ensemble standard deviation or bootstrap
2. **Extend Training Data**: Include 1980s-1990s for regime diversity
3. **Multi-Regime Switching**: Low/moderate/high inflation thresholds
4. **Feature Importance**: Add SHAP explanations
5. **Monitoring**: Track real-world prediction accuracy

---

**For detailed information:** See `CODEBASE_EXPLORATION_SUMMARY.md`  
**For complete results:** See `FORECASTING_V3_RESULTS.md`
