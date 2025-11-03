# GDP Quarterly Forecasting Results

**Project:** GDP Nowcasting & Prediction for G7 Countries
**Phase:** Quarterly Forecasting (1Q and 4Q Ahead)
**Date:** October 2025
**Status:** ✅ Complete

---

## Executive Summary

This document presents results from training 6 machine learning models to forecast GDP growth for 4 G7 countries (USA, Canada, Japan, UK) at two forecasting horizons:

- **1 Quarter Ahead (1Q)**: Short-term forecast
- **4 Quarters Ahead (4Q)**: One-year forecast

### Key Findings

⚠️ **Critical Issue Identified:** All models show **severe overfitting** with negative test R² scores, indicating the models perform worse than a naive mean baseline.

**Best Performing Models (by Test R²):**

| Horizon | Country | Best Model    | Test R² | Test RMSE |
| ------- | ------- | ------------- | ------- | --------- |
| **1Q**  | Canada  | Random Forest | 0.412   | 1.19%     |
| **1Q**  | UK      | Random Forest | 0.284   | 0.89%     |
| **1Q**  | USA     | XGBoost       | -0.100  | 0.77%     |
| **1Q**  | Japan   | Random Forest | -0.226  | 0.90%     |
| **4Q**  | Japan   | XGBoost       | 0.032   | 0.87%     |
| **4Q**  | Canada  | XGBoost       | -0.322  | 0.85%     |
| **4Q**  | UK      | Random Forest | -2.167  | 0.78%     |
| **4Q**  | USA     | XGBoost       | -10.789 | 2.61%     |

**Interpretation:**

- Only **Canada (1Q)** and **UK (1Q)** show modest positive test R² (0.41 and 0.28)
- **All other country-horizon combinations** have negative R², meaning predictions are worse than simply predicting the mean
- Longer forecast horizon (4Q) performs significantly worse than 1Q

---

## Table of Contents

1. [Methodology](#methodology)
2. [Model Performance Summary](#model-performance-summary)
3. [Detailed Results by Country](#detailed-results-by-country)
4. [Critical Analysis](#critical-analysis)
5. [Root Cause Analysis](#root-cause-analysis)
6. [Recommendations](#recommendations)
7. [Technical Details](#technical-details)

---

## Methodology

### Models Trained (6 Total)

| Model                 | Type               | Hyperparameters                                           | Purpose                  |
| --------------------- | ------------------ | --------------------------------------------------------- | ------------------------ |
| **Linear Regression** | Baseline           | None                                                      | Simplest baseline        |
| **Ridge**             | Regularized Linear | alpha: [0.01-100] (CV)                                    | L2 regularization        |
| **LASSO**             | Regularized Linear | alpha: [0.001-10] (CV)                                    | L1 + feature selection   |
| **Random Forest**     | Ensemble Tree      | n_estimators: [100-200], max_depth: [5-15]                | Non-linear relationships |
| **XGBoost**           | Gradient Boosting  | n_estimators: [100-200], max_depth: [3-7], lr: [0.01-0.1] | Advanced boosting        |
| **Gradient Boosting** | Ensemble Tree      | n_estimators: [100-200], max_depth: [3-5], lr: [0.01-0.1] | Sklearn boosting         |

### Data Split (Temporal)

- **Training**: 2001 Q1 - 2018 Q4 (72 quarters)
- **Validation**: 2019 Q1 - 2021 Q4 (12 quarters) [includes COVID-19]
- **Test**: 2022 Q1 - 2024/2025 (11-15 quarters) [recent post-COVID recovery]

### Features Used

- **Original indicators**: 20 economic variables (after dropping M2)
- **Engineered features**: Growth rates, lags (t-1, t-2, t-4), ratios, moving averages
- **Total features**: 49 features per country
- **Feature selection**: LASSO selected 13-32 features depending on country

### Evaluation Metrics

| Metric   | Formula                                | Interpretation                                                              |
| -------- | -------------------------------------- | --------------------------------------------------------------------------- |
| **RMSE** | √(mean((actual - pred)²))              | Prediction error in percentage points                                       |
| **MAE**  | mean(\|actual - pred\|)                | Average absolute error                                                      |
| **R²**   | 1 - (SS_res / SS_tot)                  | Variance explained (1.0 = perfect, 0 = mean baseline, <0 = worse than mean) |
| **MAPE** | mean(\|actual - pred\| / actual) × 100 | Percentage error                                                            |

---

## Model Performance Summary

### 1-Quarter Ahead Forecasting

#### USA (Horizon: 1Q)

| Model             | Train RMSE | Val RMSE | **Test RMSE** | Train R² | Val R² | **Test R²**  |
| ----------------- | ---------- | -------- | ------------- | -------- | ------ | ------------ |
| Linear Regression | 0.250      | 12.349   | **20.529**    | 0.975    | -6.470 | **-783.052** |
| Ridge             | 0.415      | 4.898    | **5.262**     | 0.931    | -0.175 | **-50.505**  |
| LASSO             | 0.501      | 6.078    | **10.161**    | 0.900    | -0.810 | **-191.080** |
| Random Forest     | 0.341      | 3.694    | **0.884**     | 0.954    | 0.332  | **-0.452**   |
| **XGBoost** ⭐    | 0.781      | 3.988    | **0.769**     | 0.757    | 0.221  | **-0.100**   |
| Gradient Boosting | 0.000      | 3.587    | **1.232**     | 1.000    | 0.370  | **-1.824**   |

**Analysis:**

- XGBoost performs best on test set (R² = -0.10, RMSE = 0.77%)
- Gradient Boosting shows perfect training fit (R² = 1.0) → severe overfitting
- Linear models catastrophically fail on test set (RMSE > 5%)

#### Canada (Horizon: 1Q)

| Model                | Train RMSE | Val RMSE | **Test RMSE** | Train R² | Val R² | **Test R²**   |
| -------------------- | ---------- | -------- | ------------- | -------- | ------ | ------------- |
| Linear Regression    | 0.242      | 11.261   | **23.050**    | 0.976    | -2.589 | **-220.582**  |
| Ridge                | 0.420      | 5.398    | **65.811**    | 0.928    | 0.176  | **-1805.253** |
| LASSO                | 0.363      | 6.436    | **8.741**     | 0.946    | -0.172 | **-30.867**   |
| **Random Forest** ⭐ | 0.439      | 4.966    | **1.187**     | 0.921    | 0.302  | **0.412**     |
| XGBoost              | 0.779      | 5.454    | **1.349**     | 0.752    | 0.158  | **0.241**     |
| Gradient Boosting    | 0.605      | 5.349    | **1.422**     | 0.850    | 0.190  | **0.156**     |

**Analysis:**

- ✅ **Random Forest** achieves positive test R² (0.41) - best result across all models!
- XGBoost and Gradient Boosting also show reasonable performance (R² = 0.24, 0.16)
- Tree-based models significantly outperform linear models

#### Japan (Horizon: 1Q)

| Model                | Train RMSE | Val RMSE | **Test RMSE** | Train R² | Val R² | **Test R²**  |
| -------------------- | ---------- | -------- | ------------- | -------- | ------ | ------------ |
| Linear Regression    | 0.430      | 4.509    | **12.334**    | 0.962    | -0.231 | **-230.739** |
| Ridge                | 0.734      | 5.630    | **9.151**     | 0.890    | -0.918 | **-126.552** |
| LASSO                | 0.555      | 4.712    | **3.272**     | 0.937    | -0.344 | **-15.303**  |
| **Random Forest** ⭐ | 0.739      | 3.851    | **0.897**     | 0.889    | 0.102  | **-0.226**   |
| XGBoost              | 0.737      | 3.654    | **1.144**     | 0.890    | 0.192  | **-0.995**   |
| Gradient Boosting    | 0.005      | 3.661    | **0.976**     | 1.000    | 0.189  | **-0.450**   |

**Analysis:**

- Random Forest has lowest test RMSE (0.90%) but still negative R² (-0.23)
- Gradient Boosting shows perfect training fit → overfitting
- All models struggle to generalize

#### UK (Horizon: 1Q)

| Model                | Train RMSE | Val RMSE | **Test RMSE** | Train R² | Val R² | **Test R²** |
| -------------------- | ---------- | -------- | ------------- | -------- | ------ | ----------- |
| Linear Regression    | 0.175      | 10.672   | **7.844**     | 0.991    | 0.169  | **-54.620** |
| Ridge                | 0.340      | 14.690   | **1.688**     | 0.968    | -0.574 | **-1.576**  |
| LASSO                | 0.284      | 13.141   | **3.777**     | 0.977    | -0.260 | **-11.892** |
| **Random Forest** ⭐ | 0.380      | 11.008   | **0.890**     | 0.960    | 0.116  | **0.284**   |
| XGBoost              | 0.962      | 11.504   | **0.926**     | 0.740    | 0.035  | **0.225**   |
| Gradient Boosting    | 0.003      | 11.013   | **1.063**     | 1.000    | 0.115  | **-0.021**  |

**Analysis:**

- ✅ Random Forest achieves positive test R² (0.28)
- XGBoost close behind (R² = 0.23)
- Tree-based models consistently outperform linear models

---

### 4-Quarter Ahead Forecasting

#### USA (Horizon: 4Q)

| Model             | Train RMSE | Val RMSE | **Test RMSE** | Train R² | Val R²  | **Test R²**   |
| ----------------- | ---------- | -------- | ------------- | -------- | ------- | ------------- |
| Linear Regression | 0.253      | 18.331   | **35.299**    | 0.974    | -15.474 | **-2157.349** |
| Ridge             | 0.443      | 11.650   | **32.155**    | 0.919    | -5.654  | **-1790.034** |
| LASSO             | 0.790      | 8.110    | **2.932**     | 0.743    | -2.225  | **-13.895**   |
| Random Forest     | 0.471      | 5.077    | **4.085**     | 0.909    | -0.264  | **-27.903**   |
| **XGBoost** ⭐    | 0.809      | 4.726    | **2.609**     | 0.731    | -0.095  | **-10.789**   |
| Gradient Boosting | 0.676      | 4.831    | **3.316**     | 0.813    | -0.144  | **-18.043**   |

**Analysis:**

- All models fail catastrophically on test set (R² < -10)
- XGBoost has lowest test RMSE (2.61%) but R² = -10.8
- 4Q forecasting much harder than 1Q

#### Canada (Horizon: 4Q)

| Model             | Train RMSE | Val RMSE | **Test RMSE** | Train R² | Val R² | **Test R²**         |
| ----------------- | ---------- | -------- | ------------- | -------- | ------ | ------------------- |
| Linear Regression | 0.243      | 17.489   | **21.348**    | 0.976    | -7.074 | **-828.249**        |
| Ridge             | 0.281      | 13.772   | **35860.052** | 0.967    | -4.006 | **-2339925136.250** |
| LASSO             | 0.517      | 10.654   | **20.989**    | 0.890    | -1.996 | **-800.642**        |
| Random Forest     | 0.766      | 5.782    | **1.781**     | 0.758    | 0.118  | **-4.770**          |
| **XGBoost** ⭐    | 0.769      | 6.101    | **0.852**     | 0.756    | 0.018  | **-0.322**          |
| Gradient Boosting | 0.632      | 6.107    | **1.167**     | 0.836    | 0.016  | **-1.477**          |

**Analysis:**

- Ridge completely fails with extreme RMSE (35,860%!) - numerical instability
- XGBoost performs best (RMSE = 0.85%) but still negative R² (-0.32)

#### Japan (Horizon: 4Q)

| Model             | Train RMSE | Val RMSE | **Test RMSE** | Train R² | Val R² | **Test R²**  |
| ----------------- | ---------- | -------- | ------------- | -------- | ------ | ------------ |
| Linear Regression | 0.476      | 3.410    | **11.639**    | 0.954    | 0.302  | **-172.197** |
| Ridge             | 1.082      | 7.563    | **5.682**     | 0.764    | -2.434 | **-40.286**  |
| LASSO             | 0.632      | 3.156    | **2.314**     | 0.919    | 0.402  | **-5.849**   |
| Random Forest     | 0.668      | 3.848    | **1.241**     | 0.910    | 0.111  | **-0.969**   |
| **XGBoost** ⭐    | 1.299      | 4.052    | **0.870**     | 0.659    | 0.014  | **0.032**    |
| Gradient Boosting | 0.900      | 3.722    | **0.994**     | 0.837    | 0.168  | **-0.262**   |

**Analysis:**

- ✅ XGBoost achieves small positive R² (0.032) - rare success for 4Q horizon
- Test RMSE (0.87%) is reasonable
- Only positive test R² in entire 4Q forecasting exercise

#### UK (Horizon: 4Q)

| Model                | Train RMSE | Val RMSE | **Test RMSE** | Train R² | Val R² | **Test R²**  |
| -------------------- | ---------- | -------- | ------------- | -------- | ------ | ------------ |
| Linear Regression    | 0.327      | 13.051   | **11.068**    | 0.970    | -0.234 | **-641.474** |
| Ridge                | 0.650      | 15.258   | **4.516**     | 0.880    | -0.687 | **-105.940** |
| LASSO                | 0.510      | 15.592   | **7.886**     | 0.926    | -0.762 | **-325.181** |
| **Random Forest** ⭐ | 0.577      | 12.014   | **0.777**     | 0.906    | -0.046 | **-2.167**   |
| XGBoost              | 0.881      | 11.822   | **1.096**     | 0.780    | -0.013 | **-5.297**   |
| Gradient Boosting    | 0.704      | 11.759   | **1.543**     | 0.860    | -0.002 | **-11.492**  |

**Analysis:**

- Random Forest has lowest RMSE (0.78%) but R² = -2.17
- All models fail to generalize for 4Q forecasts

---

## Detailed Results by Country

### Overall Performance Ranking

**1-Quarter Ahead (Best to Worst by Test R²):**

1. **Canada** - Random Forest (R² = 0.412, RMSE = 1.19%) ✅
2. **UK** - Random Forest (R² = 0.284, RMSE = 0.89%) ✅
3. **USA** - XGBoost (R² = -0.100, RMSE = 0.77%) ⚠️
4. **Japan** - Random Forest (R² = -0.226, RMSE = 0.90%) ❌

**4-Quarter Ahead (Best to Worst by Test R²):**

1. **Japan** - XGBoost (R² = 0.032, RMSE = 0.87%) ⚠️
2. **Canada** - XGBoost (R² = -0.322, RMSE = 0.85%) ❌
3. **UK** - Random Forest (R² = -2.167, RMSE = 0.78%) ❌
4. **USA** - XGBoost (R² = -10.789, RMSE = 2.61%) ❌

### Key Observations

1. **Only 3 models achieve positive test R²** (out of 48 model-country-horizon combinations):

   - Canada 1Q - Random Forest (0.412)
   - UK 1Q - Random Forest (0.284)
   - Japan 4Q - XGBoost (0.032)

2. **Tree-based models outperform linear models** consistently:

   - Random Forest and XGBoost dominate top performers
   - Linear regression shows extreme overfitting

3. **Shorter horizon performs better**:

   - 1Q forecasts more accurate than 4Q
   - 4Q forecasts almost universally fail

4. **Canada and UK** are most predictable for 1Q horizon

---

## Critical Analysis

### What Went Wrong?

#### Issue #1: Severe Overfitting

**Evidence:**

- Train R² = 0.90-1.00 (excellent)
- Val R² = -0.5 to 0.4 (poor to moderate)
- Test R² = -800 to 0.4 (catastrophic)

**Pattern:**

- Perfect training fit (especially Gradient Boosting with R² = 1.0)
- Reasonable validation performance
- Complete failure on test set

**Interpretation:**
Models memorized training data patterns that don't generalize to recent economic conditions (2022-2025).

#### Issue #2: Non-Stationary Test Period

**Context:**

- Training: 2001-2018 (pre-pandemic, relatively stable)
- Validation: 2019-2021 (includes COVID-19 shock)
- Test: 2022-2025 (post-COVID recovery, inflation surge, rate hikes)

**Problem:**
Test period has fundamentally different economic dynamics:

- High inflation (first time in 40 years)
- Rapid interest rate increases
- Supply chain disruptions
- Energy price shocks (Ukraine war)

Models trained on 2001-2018 "normal" conditions cannot predict 2022-2025 "abnormal" conditions.

#### Issue #3: Small Sample Size

**Numbers:**

- Training: 72 quarters
- Features: 49 variables
- Ratio: 72/49 = 1.47 samples per feature

**Problem:**
Insufficient data for high-dimensional modeling, especially for complex models like Random Forest and XGBoost.

#### Issue #4: Data Leakage Potential

**Concern:**
Using `gdp_growth_yoy` (which includes current quarter GDP) to predict future GDP may create information leakage through lagged features.

**Example:**
If `gdp_growth_yoy_lag1` contains information about current quarter, it indirectly contains future information when predicting 1Q ahead.

#### Issue #5: Feature Multicollinearity

**Warning Signs:**

- Ridge warnings: "Ill-conditioned matrix"
- Linear regression catastrophic failures
- LASSO selecting only 13-32 of 49 features

**Problem:**
High correlation between features (e.g., GDP level, GDP growth, GDP lags) creates unstable coefficient estimates.

---

## Root Cause Analysis

### Why Did Models Fail on Test Set?

**Hypothesis 1: Distribution Shift** ⭐ (Most Likely)

The 2022-2025 test period represents a **regime change**:

- **Training period (2001-2018)**: Low inflation (0-3%), gradual rate changes, stable growth
- **Test period (2022-2025)**: High inflation (5-10%), rapid rate hikes, volatile growth

**Evidence:**

- Models perform reasonably on validation (2019-2021) which is closer to training distribution
- Test errors are orders of magnitude larger than validation errors

**Solution:**

- Include more diverse economic regimes in training (1980s high inflation, 1990s recession)
- Use regime-switching models
- Add external indicators (oil prices, policy shocks)

**Hypothesis 2: Overfitting** ⭐ (Contributing Factor)

**Evidence:**

- Perfect training fit (R² = 1.0 for Gradient Boosting)
- Negative test R² (worse than mean baseline)
- High model complexity (49 features, 72 samples)

**Solution:**

- Stronger regularization (higher alpha for Ridge/LASSO)
- Feature selection (use top 10-15 features only)
- Simpler models (linear with strong regularization)
- Larger training set (include 1980s-1990s data if available)

**Hypothesis 3: Insufficient Features** (Possible)

**Current features** are mostly economic indicators (GDP components, rates, production).

**Missing:**

- **Policy shocks**: Fiscal stimulus, trade wars, regulatory changes
- **Expectations**: Consumer confidence, business sentiment, inflation expectations
- **External factors**: Oil prices, global trade volume, exchange rates vs basket
- **Financial conditions**: Credit spreads, volatility indices, lending standards

**Solution:**

- Add qualitative indicators (survey data)
- Include commodity prices
- Incorporate global economic indicators

**Hypothesis 4: Inappropriate Target** (Structural Issue)

**Current target**: `gdp_growth_yoy` (year-over-year % change)

**Problem:**
YoY growth is volatile and includes seasonal effects even after adjustment.

**Alternative targets:**

- **GDP level (in differences)**: More stable, easier to predict
- **Detrended GDP**: Remove long-term trend, predict deviations
- **GDP components separately**: Predict C, I, G, NX individually then sum

---

## Recommendations

### Immediate Actions (Quick Wins)

#### 1. Reduce Model Complexity ⚠️ Priority: HIGH

**Current**: 49 features, 72 samples → overfitting

**Action:**

- Use LASSO to select top 10-15 most important features
- Retrain with reduced feature set
- Expected improvement: Better generalization, positive test R²

**Top Features from LASSO** (example from USA):

- `gdp_real_lag1`, `gdp_real_lag4`
- `unemployment_rate`
- `industrial_production_index`
- `capital_formation`
- `interest_rate_short_term`

#### 2. Stronger Regularization ⚠️ Priority: HIGH

**Current**: Ridge alpha = 10-100, LASSO alpha = 0.01-10

**Action:**

- Increase Ridge alpha to 500-1000
- Increase LASSO alpha to 10-100
- Expected improvement: Reduce overfitting, smoother predictions

#### 3. Ensemble with Mean Baseline ⚠️ Priority: MEDIUM

**Observation**: Test R² < 0 means mean baseline outperforms ML models

**Action:**

- Create ensemble: 50% ML model + 50% historical mean
- Weighted by validation performance
- Expected improvement: Never worse than baseline

#### 4. Walk-Forward Validation ⚠️ Priority: HIGH

**Current**: Single train/val/test split

**Action:**

- Expanding window: Train on 2001-2015, test 2016; train 2001-2016, test 2017; etc.
- Average test errors across multiple splits
- Expected improvement: More robust performance estimate

### Medium-Term Improvements

#### 5. Add External Indicators

**Candidates:**

- **Oil prices** (Brent crude, WTI)
- **VIX** (volatility index)
- **Consumer confidence indices**
- **Global PMI** (Purchasing Managers' Index)
- **Central bank policy rates** (ECB, BOJ, BOE for context)

#### 6. Regime-Switching Models

**Approach:**

- Detect high-inflation vs low-inflation regimes
- Train separate models for each regime
- Switch models based on current inflation level

#### 7. Longer Training Period

**Current**: 2001-2018 (18 years)

**Ideal**:

- Extend back to 1980s-1990s if data available
- Include multiple inflation/recession cycles
- More diverse training distribution

### Long-Term Strategy

#### 8. Hybrid Models

**Combine:**

- **ML models** (capture non-linear relationships)
- **Econometric models** (ARIMA, VAR - capture time series structure)
- **Expert forecasts** (IMF, OECD predictions)

**Ensemble**: Weight by recent validation performance

#### 9. Focus on Nowcasting

**Observation**: 1Q forecasts outperform 4Q forecasts

**Action:**

- Prioritize nowcasting (current quarter GDP estimation)
- Use high-frequency indicators (monthly industrial production, employment)
- Delay forecasting to after nowcasting is perfected

#### 10. Explainable AI

**Current**: Black-box models (XGBoost, Random Forest)

**Add:**

- SHAP values for feature importance
- Partial dependence plots
- Counterfactual explanations ("if unemployment increases by 1%, GDP growth decreases by X%")

---

## Technical Details

### Files Generated

**Models** (saved_models/)

- 48 model files (.pkl): 4 countries × 2 horizons × 6 models

**Results** (results/)

- `usa_h1_results.csv`, `usa_h4_results.csv`
- `canada_h1_results.csv`, `canada_h4_results.csv`
- `japan_h1_results.csv`, `japan_h4_results.csv`
- `uk_h1_results.csv`, `uk_h4_results.csv`
- `all_countries_h1_results.csv`
- `all_countries_h4_results.csv`

**Figures** (figures/)

- Predictions vs Actuals plots (8 plots: 4 countries × 2 horizons)
- Feature importance plots (8 plots: 4 countries × 2 horizons)

### Computational Details

**Training Time:**

- Total: ~10-15 minutes (all models, all countries, all horizons)
- Per model: ~30-60 seconds
- Grid search CV: 3-fold time series cross-validation

**Warnings Encountered:**

- `Ill-conditioned matrix`: Feature multicollinearity
- `Singular matrix`: Ridge regression numerical instability
- `Convergence warnings`: LASSO did not fully converge (acceptable for CV)

---

## Conclusion

### Summary of Findings

✅ **Successes:**

1. Successfully trained 6 models across 4 countries and 2 horizons (48 models total)
2. Canada and UK showed modest positive performance for 1Q forecasting (R² = 0.28-0.41)
3. Tree-based models (Random Forest, XGBoost) consistently outperformed linear models
4. Feature importance analysis revealed key predictors (GDP lags, unemployment, industrial production)

❌ **Failures:**

1. **Severe overfitting**: 45/48 models showed negative test R² (worse than mean baseline)
2. **Distribution shift**: Test period (2022-2025) fundamentally different from training (2001-2018)
3. **4Q forecasts failed universally**: Only 1/24 models achieved positive R² for 4Q horizon
4. **Linear models catastrophic**: Linear regression, Ridge often had RMSE > 10%

### Best Use Cases for Current Models

**DO USE:**

- **Canada 1Q forecast** (Random Forest, R² = 0.41, RMSE = 1.19%)
- **UK 1Q forecast** (Random Forest, R² = 0.28, RMSE = 0.89%)

**DO NOT USE:**

- Any 4Q forecasts (all negative R²)
- USA forecasts (high volatility, overfitting)
- Japan forecasts (negative R² even for 1Q)
- Any linear models (catastrophic failures)

### Next Steps

**Priority 1: Fix Overfitting**

1. Reduce features to top 10-15 (use LASSO selection)
2. Increase regularization strength (Ridge alpha = 500-1000)
3. Implement walk-forward validation

**Priority 2: Add Data**

1. Extend training period back to 1980s if available
2. Include external indicators (oil prices, VIX, policy shocks)
3. Add high-frequency monthly data for nowcasting

**Priority 3: Alternative Approaches**

1. Nowcasting pipeline (separate task - higher priority than forecasting)
2. Regime-switching models (detect high/low inflation periods)
3. Hybrid models (combine ML + econometric + expert forecasts)

---

**Document Status:** ✅ Complete
**Last Updated:** October 2025
**Prepared By:** GDP Forecasting Pipeline
**For:** GDP Nowcasting & Forecasting Project - Phase 3 Complete
