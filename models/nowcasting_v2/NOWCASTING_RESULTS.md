Nowcasting V2 - Feature Cleaning Results
Date: October 29, 2025
Status: Intermediate step - performance degraded as expected

Changes from V1
Features Removed (11 total)
GDP Components (Bad Practice):

investment_growth - Part of GDP formula (I)
consumption_growth - Part of GDP formula (C)
household_consumption - Part of GDP formula (C)

Redundant Growth Rates:

employment_growth - Already have employment_level
exports_growth - Already have exports_volume
imports_growth - Already have imports_volume

Other:

interest_rate_long_term - Removed for simplification

Features Kept (10 core)
Leading Indicators (4):

industrial_production_index
stock_market_index
interest_rate_short_term
capital_formation

Coincident Indicators (6):

employment_level
unemployment_rate
cpi_annual_growth
exports_volume
imports_volume
trade_balance

Feature Count: 21 → 10 core

USA Results
MetricV1V2ChangeBest ModelXGBoostRandom Forest-Test R²0.089-0.037-0.126 (worse)Test RMSE0.7820.834+0.052 (worse)

Visual Analysis
Time Series Predictions
V1 (XGBoost):

Predictions vary from 2.1% to 2.7%
Orange line has movement, tracks general trend
Misses magnitude but captures some dynamics

V2 (Random Forest):

Predictions COMPLETELY FLAT at ~2.6%
Zero movement across entire test period
Model just predicts the mean value

Why V2 is Worse
Root Cause: Removing investment_growth eliminated the model's primary signal

In V1, investment_growth had 35% feature importance
Even though it was a "bad" feature (GDP component causing circular prediction)
It was the only feature providing strong signal to the model
Removing it exposed that model has no real predictive features

Current State: Model has nothing to learn from

10 remaining features don't capture short-term GDP dynamics
Model defaults to predicting mean (~2.6%)
R² = -0.037 means we're worse than just predicting the mean


Why This Was Expected
Following Teammate's Recommendation
From VISUALIZATION_FINDINGS.md (line 203):

"GDP components (C+I+G) cause circular prediction"

The Process:

✅ Remove bad features (GDP components) → V2
❌ Performance drops (expected!)
➡️ Add good features (regime indicators) → V3

Similar to Teammate's Journey
Teammate went: 49 features → 15 features (with better feature engineering)
We're going: 21 features → 10 features → 17 features (with regime features)

Learnings
What We Confirmed ✅

GDP components are problematic - Removing them was the right call
investment_growth was carrying the model - V1's R² of 0.089 was artificial
Need replacement features - Can't just remove bad features, must add good ones

What We Learned ❌

V2 is not a final solution - It's an intermediate step
10 core features insufficient - Need additional signal for nowcasting
Model has no regime awareness - Can't detect 2022-2025 high-inflation period

Why V2 Still Matters ✓

Clean foundation for V3
Removed circular prediction problem
Ready to add regime features that actually help


Next Steps: V3 (Regime Features)
The Problem
Training Period (2001-2018): Low inflation, stable economy
Test Period (2022-2025): High inflation (>5% in 2022-2023), volatile
Solution: Add features that detect regime changes
Regime Features to Add (7 total)
Volatility Indicators:

gdp_volatility_4q - Rolling 4Q std dev of GDP growth
stock_volatility_4q - Rolling 4Q std dev of stock returns

Momentum Indicators:
3. gdp_momentum - Change in GDP growth from 4Q ago
4. inflation_momentum - Change in inflation from 4Q ago
Regime Flags:
5. high_inflation_regime - Binary flag (>2.5% inflation)
6. high_volatility_regime - Binary flag (top 25% volatility)
Interaction Terms:
7. inflation_x_volatility - Interaction term
Expected V3 Outcome

Target: R² between 0.15-0.30
Why: Regime features will capture 2022-2025 volatility
Benchmark: Beat both V1 (0.089) and V2 (-0.037)


Technical Notes
Model Selection Change

V1 chose XGBoost
V2 chose Random Forest
This is automatic based on cross-validation performance
Shows models are struggling with different feature sets

Feature Importance (Not Available for V2)

V2 predictions are flat, so feature importance is meaningless
All features would show near-zero importance
Will be meaningful again in V3 with regime features


Conclusion
V2 Status: ⚠️ Expected intermediate step, not production-ready
Success Criteria Met:

✅ Removed GDP components (following best practices)
✅ Created clean feature foundation
✅ Documented why performance degraded

Success Criteria NOT Met:

❌ Improved performance (expected to fail at this step)
❌ Production-ready model

Path Forward:
V2 → V3 (add regime features) → Expected R² > 0.15

Version: 2.0 (Feature Cleaning)
Created: October 29, 2025
Next: V3 with regime detection features