V4: Critical Analysis (Proving the Leakage)
Date: October 29, 2025
Features: 15 total (V3 minus 2 leaky base features)
Surgical Removal - Only 2 Features Removed:

❌ gdp_momentum (contains current GDP)
❌ gdp_volatility_4q (rolling window includes current quarter)

Kept Everything Else (Including Derived Features):

✅ high_volatility_regime (derived from gdp_volatility_4q)
✅ inflation_x_volatility (uses gdp_volatility_4q)
✅ All other regime features
✅ All core indicators

Purpose:
Minimal removal to prove that V3's success depended entirely on those 2 leaky features. If derived features were useful on their own, performance would remain. If not, it collapses.
Results:

USA: R² = -0.016, RMSE = 0.826
All models: Negative R² (worse than mean baseline)

What This Proves:

✅ V3's success was entirely from the 2 removed features
✅ Derived features (high_volatility_regime, inflation_x_volatility) are useless without their source
✅ The data leakage hypothesis is confirmed
✅ Without GDP-derived features, current approach has no signal

Key Insight:
"We removed ONLY 2 features - those with confirmed data leakage. We kept their derived features to show they don't work independently. The complete collapse proves those 2 features were doing all the work through data leakage."
Conclusion: ✅ Successfully demonstrated V3 was invalid
Status: Critique/validation version - proved the hypothesis