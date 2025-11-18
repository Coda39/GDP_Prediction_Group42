# Quarterly GDP Forecasting Models

## Overview

This directory contains quarterly GDP forecasting models for all G7 countries (USA, Canada, UK, Japan, Germany, France, Italy). Each country has models that forecast GDP growth at 1, 2, 3, and 4 quarter horizons.

For each country, use these specific models:

- USA: XGBoost for all horizons
- Canada: XGBoost (h1-h3), Random Forest (h4) - AVOID ensemble
- UK: XGBoost for all horizons
- Japan: Gradient Boosting (h1), LASSO (h2), XGBoost (h3-h4)
- Germany: Random Forest (h1, h4), Gradient Boosting (h2-h3)
- France: XGBoost for all horizons
- Italy: Ensemble (h1, h3, h4), Random Forest (h2)

**Total Models:** 140 (7 countries × 4 horizons × 5 models per horizon)

## Key Features

### Multi-Horizon Forecasting

- **h=1**: 1 quarter ahead forecasting
- **h=2**: 2 quarters ahead forecasting
- **h=3**: 3 quarters ahead forecasting
- **h=4**: 4 quarters ahead forecasting

### Model Ensemble

Each horizon uses 5 models:

1. **Ridge Regression** - L2 regularization for stability
2. **LASSO Regression** - L1 regularization with automatic feature selection
3. **Random Forest** - Tree ensemble for non-linear patterns
4. **XGBoost** - Gradient boosting with advanced regularization
5. **Gradient Boosting** - Alternative ensemble approach

### Weighted Ensemble Approach

- Models are combined using validation R² as weights
- Higher performing models receive greater weight
- Ensemble predictions account for both individual model uncertainty AND model disagreement

### Uncertainty Quantification

- **Bootstrap Confidence Intervals**: 80% and 95% prediction intervals
- **Method**: Residual resampling (1000 iterations)
- **Ensemble Uncertainty**: Combines individual model variance with model disagreement
- **Formula**: Total Variance = Σ(w_i² × var_i) + Σ(w_i × (pred_i - ensemble_mean)²)

### Data Leakage Prevention

Uses ONLY exogenous features (no GDP-dependent variables):

- Labor market indicators (unemployment, employment)
- Inflation (CPI)
- Monetary policy (interest rates, yield curve)
- Production (industrial production)
- Trade volumes (exports, imports)
- Financial markets (stocks, bonds, credit spreads)
- Consumption and investment (NOT as GDP ratios)

## Directory Structure

```
quarterly_forecasting/
├── README.md                    # This file
├── {country}/                   # For each G7 country
│   └── current/
│       ├── quarterly_forecasting_pipeline.py
│       ├── figures/             # 8 plots per country
│       │   ├── {country}_h1_predictions.png
│       │   ├── {country}_h1_feature_importance.png
│       │   ├── {country}_h2_predictions.png
│       │   ├── {country}_h2_feature_importance.png
│       │   ├── {country}_h3_predictions.png
│       │   ├── {country}_h3_feature_importance.png
│       │   ├── {country}_h4_predictions.png
│       │   └── {country}_h4_feature_importance.png
│       ├── results/             # 4 CSV files per country
│       │   ├── {country}_h1_results.csv
│       │   ├── {country}_h2_results.csv
│       │   ├── {country}_h3_results.csv
│       │   └── {country}_h4_results.csv
│       └── saved_models/        # 20 models per country
│           ├── {country}_h1_ridge.pkl
│           ├── {country}_h1_lasso.pkl
│           ├── {country}_h1_random_forest.pkl
│           ├── {country}_h1_xgboost.pkl
│           ├── {country}_h1_gradient_boosting.pkl
│           ├── {country}_h2_*.pkl
│           ├── {country}_h3_*.pkl
│           └── {country}_h4_*.pkl
```

## Output Format

### Results CSV Structure

Each `{country}_h{horizon}_results.csv` contains:

- **Model Performance Metrics**: RMSE, MAE, R², MAPE for train/val/test
- **Confidence Interval Coverage**: 80% and 95% CI coverage percentages
- **Ensemble Row**: Weighted ensemble performance with uncertainty metrics

### Front-End Ready Features

All outputs are designed for dynamic visualization:

- ✓ Point forecasts
- ✓ Lower and upper confidence bounds (80% and 95%)
- ✓ Model weights (based on validation performance)
- ✓ Performance metadata (RMSE, R², coverage)
- ✓ Individual model predictions for comparison
- ✓ Ensemble predictions with combined uncertainty

## Running the Models

### Single Country

```bash
cd models/quarterly_forecasting/usa/current
python3 quarterly_forecasting_pipeline.py
```

### All Countries

```bash
for country in usa canada uk japan germany france italy; do
  cd models/quarterly_forecasting/$country/current
  python3 quarterly_forecasting_pipeline.py
  cd -
done
```

## Technical Details

### Train/Val/Test Split

- **Train**: 1980-2018 (or 1991-2018 for Germany/France/Italy)
- **Validation**: 2019-2021
- **Test**: 2022-2024

### Hyperparameter Tuning

- Uses `TimeSeriesSplit` (3 folds) for cross-validation
- Prevents look-ahead bias
- Conservative hyperparameters to reduce overfitting

### Bootstrap Configuration

- **Iterations**: 1000
- **Method**: Residual resampling
- **Seed**: 42 (reproducibility)

## Country-Specific Notes

### Full Historical Data (1980-2024)

- **USA**: 176 quarters after horizon shifting
- **Canada**: 184 quarters
- **UK**: 184 quarters
- **Japan**: 184 quarters

### Limited Historical Data (1991-2024)

- **Germany**: 140 quarters (post-reunification)
- **France**: 140 quarters (post-EU formation)
- **Italy**: 140 quarters (post-EU formation)

## Model Performance Expectations

Based on forecasting literature and v4 archive results:

- **1Q ahead**: R² 0.10-0.30 (short-term predictability)
- **2Q ahead**: R² 0.05-0.20 (moderate predictability)
- **3Q ahead**: R² 0.00-0.15 (declining predictability)
- **4Q ahead**: R² -0.10-0.10 (long-term uncertainty)

Note: Negative R² indicates model performs worse than naive mean prediction, which is common for long-horizon forecasting.

## Comparison to Nowcasting

| Aspect          | Nowcasting                     | Quarterly Forecasting     |
| --------------- | ------------------------------ | ------------------------- |
| **Target**      | Current quarter GDP            | h quarters ahead          |
| **Horizon**     | t (present)                    | t+1, t+2, t+3, t+4        |
| **Features**    | Leading + coincident           | Exogenous only            |
| **Models**      | 7 (includes Linear + Stacking) | 5 (focused ensemble)      |
| **Expected R²** | 0.10-0.50                      | -0.10-0.30                |
| **Uncertainty** | N/A                            | Bootstrap 80% & 95% CI    |
| **Ensemble**    | Stacking                       | Weighted by validation R² |

## Usage for Front-End

### Loading Predictions

```python
import pandas as pd

# Load results for a specific horizon
results = pd.read_csv('models/quarterly_forecasting/usa/current/results/usa_h1_results.csv')

# Ensemble predictions are in the row where Model == 'Ensemble'
ensemble = results[results['Model'] == 'Ensemble']
```

### Loading Models

```python
import joblib

# Load a specific model
model = joblib.load('models/quarterly_forecasting/usa/current/saved_models/usa_h1_xgboost.pkl')

# Make predictions
predictions = model.predict(new_data)
```

### Accessing Uncertainty Bands

The pipeline stores detailed predictions in the results. For programmatic access to confidence intervals, you'll need to:

1. Load the saved model
2. Re-run the bootstrap procedure on new data
3. Or store predictions in a separate pickle file (can be added to pipeline)

## Future Enhancements

Potential improvements for V2:

- [ ] Add ARIMAX/SARIMAX time series models
- [ ] Implement regime-switching models
- [ ] Add neural network forecasters (LSTM, Transformer)
- [ ] Create combined nowcast + forecast pipeline
- [ ] Add real-time data updating capability
- [ ] Implement rolling window retraining
- [ ] Add model explanability (SHAP values)

## References

- **Data Source**: OECD, FRED, Bloomberg (processed via data_preprocessing/)
- **Methodology**: Bootstrap residual resampling (Efron & Tibshirani, 1993)
- **Ensemble Weighting**: Validation R²-based soft voting
- **Uncertainty Quantification**: Combined individual variance + model disagreement

## Version History

- **V1** (November 2025): Initial implementation
  - 140 models across 7 G7 countries
  - 4 forecast horizons per country
  - Bootstrap confidence intervals
  - Weighted ensemble approach
  - Exogenous features only (no data leakage)

---

**Last Updated**: November 15, 2025
**Author**: GDP Prediction Group 42
**Course**: CS 4485 - Fall 2025
