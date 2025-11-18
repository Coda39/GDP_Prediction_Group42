# v4 Implementation Plan: Clean Features & Separate Horizons

## Problem Analysis from v3

### 1. GDP-Dependent Features Issue

**Problematic Features Identified in v3:**
- `gdp_growth_qoq` - DEPENDS ON GDP (directly calculated as QoQ change)
- `gdp_growth_yoy` - TARGET ITSELF (should not be used as predictor!)
- `gdp_growth_yoy_ma4` - DEPENDS ON GDP (moving average of target)
- `gdp_real_lag1, lag2, lag4` - DEPENDS ON GDP (lagged values of target)
- `gov_gdp_ratio` - DEPENDS ON GDP (government spending / GDP)
- `trade_gdp_ratio` - DEPENDS ON GDP (trade / GDP)

**Impact:** These features create data leakage because they directly or indirectly contain GDP information, inflating R² scores and creating false confidence in predictions.

### 2. v3 Limitations

From FORECASTING_V3_RESULTS.md:
- USA 1Q: R² = -0.114 (worse than random)
- UK 4Q: R² = -1.136 (terrible)
- Japan 4Q: R² = 0.070 (poor)
- Regime-switching limited by lack of training data diversity
- RevIN had mixed impact (helped some, hurt others)

### 3. Improvements Needed

1. **Remove all GDP-dependent features** → Clean predictions
2. **Separate models per horizon** → Optimize for each timeframe
3. **USA focus** → Build strong baseline before extending
4. **Better feature selection** → Use only exogenous variables
5. **Enhanced confidence intervals** → Bootstrap or ensemble variance
6. **4 distinct models** → 1Q, 2Q, 3Q, 4Q ahead instead of generic

---

## v4 Architecture

### Feature Selection Strategy

**Keep (Exogenous - Independent of GDP):**
- `unemployment_rate` ✓ Labor market
- `employment_level` ✓ Labor market
- `employment_growth` ✓ Labor market
- `cpi_annual_growth` ✓ Inflation (independent)
- `interest_rate_short_term` ✓ Monetary policy
- `interest_rate_long_term` ✓ Monetary policy
- `industrial_production_index` ✓ Production (quantity)
- `ip_growth` ✓ Production growth
- `exports_volume` ✓ Trade quantity
- `imports_volume` ✓ Trade quantity
- `exports_growth` ✓ Trade growth
- `imports_growth` ✓ Trade growth
- `household_consumption` ✓ Nominal consumption (level indicator)
- `consumption_growth` ✓ Consumption growth
- `capital_formation` ✓ Investment (quantity)
- `investment_growth` ✓ Investment growth
- `money_supply_broad` ✓ Monetary aggregate
- `m2_growth` ✓ Money growth rate
- `stock_market_index` ✓ Asset prices
- `exchange_rate_usd` ✓ Exchange rate
- `government_spending` ✓ Government expenditure

**Remove (GDP-dependent - cause data leakage):**
- `gdp_real` ❌ Target itself
- `gdp_nominal` ❌ Target itself
- `gdp_growth_qoq` ❌ Calculated from target
- `gdp_growth_yoy` ❌ THIS IS THE TARGET
- `gdp_growth_yoy_ma4` ❌ MA of target
- `gdp_real_lag1/2/4` ❌ Lagged target
- `gdp_real_diff` ❌ Diff of target
- `trade_balance` ❌ Component of GDP (X-M)
- `trade_gdp_ratio` ❌ Trade divided by GDP
- `gov_gdp_ratio` ❌ Gov spending / GDP
- `population_total` ⚠️ Nearly constant (not predictive)
- `population_working_age` ⚠️ Nearly constant

**Final Count:** ~21 exogenous features instead of 49+ mixed features

### Model Architecture

**4 Independent Models:**
```
Model_1Q: Predict GDP growth 1 quarter ahead
Model_2Q: Predict GDP growth 2 quarters ahead
Model_3Q: Predict GDP growth 3 quarters ahead
Model_4Q: Predict GDP growth 4 quarters ahead
```

**For Each Model:**
- Train on features: t, t-1, t-2, t-4 (1Q, 2Q, 4Q lags)
- Target: GDP growth at respective horizon
- Validation: Expanding window (same as v3)
- Ensemble: Ridge + RF + GB (best performers from v3)

### Expected Improvements

1. **Eliminating data leakage** → More honest R² scores
2. **Separate optimization** → Each horizon learns its own pattern
3. **Clean predictions** → Use only truly exogenous variables
4. **Better interpretability** → Features clearly don't depend on target
5. **Realistic confidence intervals** → Based on true prediction variance, not information leakage

---

## Implementation Timeline

1. **Feature Engineering** (30 min)
   - Load USA data
   - Select 21 clean exogenous features
   - Create lagged features (t-1, t-2, t-4)
   - Handle missing values

2. **Train 4 Horizon Models** (90 min)
   - Build models for h=1,2,3,4
   - Use Ridge, RF, GB
   - Expanding window validation
   - Calculate metrics (R², RMSE, MAE)

3. **Generate Predictions & Confidence Intervals** (30 min)
   - Ensemble predictions
   - Bootstrap CI calculation
   - Store predictions with dates

4. **Create Visualizations** (45 min)
   - Match v3 style (actual vs predicted + CI)
   - Heatmap of performance by horizon
   - RMSE degradation plot
   - Save as PNG (300 DPI)

5. **Documentation** (30 min)
   - V4_RESULTS.md with findings
   - Compare to v3 (same features vs clean)
   - Lessons learned
   - Recommendations for v5

---

## Expected Results

### Realistic Performance (with clean features)

**1Q Ahead:**
- Ridge: R² ~0.05-0.10
- Random Forest: R² ~0.10-0.15
- Gradient Boosting: R² ~0.08-0.12
- Ensemble: R² ~0.10-0.15

**4Q Ahead:**
- Ridge: R² ~-0.05-0.00
- Random Forest: R² ~0.00-0.05
- Gradient Boosting: R² ~0.05-0.10
- Ensemble: R² ~0.00-0.05

(Much lower than v3's inflated numbers due to removing data leakage)

### Validation

Compare v3 (with leakage) vs v4 (clean):
- If v3 was relying on leakage, clean features will show lower R²
- If correlation was real, v4 will maintain decent R²
- Confidence intervals should be wider (more honest uncertainty)

---

## Files to Create

```
quarterly_forecasting_v4/
├── V4_IMPLEMENTATION_PLAN.md         (this file)
├── forecasting_pipeline_v4.py        (main code)
├── V4_RESULTS.md                     (analysis)
├── saved_models/
│   ├── usa_h1_ridge_v4.pkl
│   ├── usa_h1_rf_v4.pkl
│   ├── usa_h1_gb_v4.pkl
│   ├── usa_h2_ridge_v4.pkl
│   ... (4 horizons × 3 models = 12 models)
├── results/
│   ├── v4_model_performance.csv
│   ├── v4_predictions.csv
│   └── v4_v3_comparison.csv
└── forecast_visualizations/
    ├── usa_forecast_h1_v4.png
    ├── usa_forecast_h2_v4.png
    ├── usa_forecast_h3_v4.png
    ├── usa_forecast_h4_v4.png
    ├── usa_rmse_by_horizon_v4.png
    ├── usa_r2_heatmap_v4.png
    ├── v3_vs_v4_comparison.png
    └── confidence_intervals_v4.png
```

---

## Key Design Decisions

1. **USA Only:** Validate approach thoroughly before extending to other countries
2. **4 Separate Models:** Each horizon learns its own temporal dynamics
3. **No Data Leakage:** Remove all GDP-derived features
4. **Conservative Hyperparameters:** Avoid overfitting on small dataset
5. **Ensemble Approach:** Combine best performers for robustness
6. **Bootstrap CIs:** More statistically sound than ensemble variance alone

---

## Success Criteria

- [x] Plan created
- [ ] Clean features selected (21 exogenous variables)
- [ ] 4 models trained (1Q, 2Q, 3Q, 4Q)
- [ ] Results match v3 performance or better when excluding leakage
- [ ] Visualizations created matching v3 style
- [ ] Honest confidence intervals (wider than v3)
- [ ] Documentation complete
- [ ] Comparison v3 vs v4 analysis done
