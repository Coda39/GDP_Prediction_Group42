"""
GDP Nowcasting Pipeline - V6 (Financial Indicators)
====================================================
This script trains and evaluates multiple models for nowcasting current quarter GDP growth.

VERSION 6 CHANGES (November 1, 2025):
- Added financial market indicators (yield curve, credit spreads)
- Tested real activity indicators (housing, capacity, sentiment) - most failed
- Extended training data from 72 quarters (2000-2018) to 152 quarters (1981-2018)
- These are NOT GDP components - they're forward-looking market prices

V6 FEATURE TESTING RESULTS:
✅ WORKING Features (Final V6):
Financial Indicators:
  - credit_spread: 25% importance (RF), 48% importance (GB) - DOMINANT FEATURE
  - credit_spread_change: Captures financial stress dynamics
  - yield_curve_slope: 7% importance - proven recession predictor
  - yield_curve_curvature: 4% importance - marginal contribution

❌ FAILED Features (Tested but removed):
Real Activity Indicators:
  - consumer_sentiment: Dominated at 64% importance, caused massive overfitting
  - capacity_utilization: Dominated at 31%, crowded out credit spreads
  - building_permits: Low predictive power alone, redundant with housing_starts
  
⚠️ UNCERTAIN Features (Under investigation):
  - housing_starts: R² = 0.348 alone, but 51% dominance raises overfitting concerns
  - Interaction terms: Currently being tested one-by-one

PERFORMANCE RESULTS:
- V5 Baseline: Test R² = 0.009 (152 quarters, no financial indicators)
- V6 Conservative (credit spreads only): Test R² = 0.19 (Random Forest)
- V6 Optimistic (credit spreads + housing_starts): Test R² = 0.348 (Stacking - may be overfitting)

KEY LEARNINGS:
1. Financial market indicators (credit spreads, yield curve) generalize to post-COVID period
2. Historical economic indicators (sentiment, capacity) DON'T generalize - cause overfitting
3. Credit spread is the single most important nowcasting feature (48% GB importance)
4. Fewer, higher-quality features > kitchen sink approach
5. Training on 1981-2018 works better than 2010-2018 (more regime diversity)

Models:
1. Linear Regression (Baseline)
2. Ridge Regression (L2 Regularization)
3. LASSO Regression (L1 Regularization, Feature Selection)
4. Random Forest Regressor (BEST: R² = 0.19)
5. XGBoost Regressor
6. Gradient Boosting Regressor
7. Stacking Ensemble (experimental)

Target: Current quarter GDP Growth (YoY %)
Training Data: 1981-2018 (152 quarters)
Validation Data: 2019-2021 (12 quarters)
Test Period: 2022-2025 (12 quarters)
Countries: USA (focus)

Current Status: Conservative model (R² = 0.19) established. Testing interaction terms.
Reference: See technical analysis document and v6_testing_log.md for full rationale
"""