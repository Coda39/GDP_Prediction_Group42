# Exploratory Data Visualization - Findings & Takeaways

**Project:** GDP Nowcasting & Prediction for G7 and BRICS Countries
**Phase:** Initial Data Exploration
**Date:** October 2025

---

## Executive Summary

This document summarizes key findings from the exploratory visualization phase and provides actionable recommendations for subsequent preprocessing and modeling phases.

---

## 1. Data Quality Assessment

### G7 Countries (Quarterly Data)

**Overall Observations:**
- **USA**: 104 quarterly observations (2000-2025), 8.5% missing data
  - Daily data was successfully resampled to quarterly
  - Most complete G7 dataset due to FRED's comprehensive coverage

- **Other G7 Countries**: 308 monthly observations each (2000-2025)
  - **High missing data**: 42.7% (Japan) to 54.7% (France)
  - Missing data is **systematic**, not random - indicators update at different frequencies
  - Quarterly indicators (GDP, employment) only populate every 3 months
  - Monthly indicators (interest rates, CPI) more complete

**Key Finding:** Missing data percentages are inflated because monthly rows exist for quarterly indicators. Actual data quality is better than percentages suggest.

### BRICS Countries (Annual Data)

**Overall Observations:**
- 25 annual observations per country (2000-2024)
- **Excellent completeness**: 0.2% to 1.6% missing data
- Missing data limited to specific unavailable indicators (e.g., Russia missing imports)

**Key Finding:** BRICS data is nearly complete within available indicators. Missing indicators are entirely absent (not partially available).

---

## 2. Economic Patterns & Insights

### GDP Growth Characteristics

**G7 Average Annual Growth (2000-2025):**
- **USA**: 2.12% (strongest G7 growth)
- **Canada**: 0.65%
- **UK**: 0.54%
- **France**: 0.43%
- **Germany**: 0.34%
- **Japan**: 0.22% (weakest growth, persistent stagnation)
- **Italy**: 0.16% (near-zero growth over 25 years)

**BRICS Average Annual Growth (2000-2024):**
- **China**: 8.18% (highest growth, moderating over time)
- **India**: 6.36% (second highest, accelerating recently)
- **Russia**: 3.14% (volatile, sanctions impact)
- **Brazil**: 2.30%
- **South Africa**: 2.13% (slowest BRICS growth)

**Volatility (Standard Deviation):**
- **Highest volatility**: Russia (3.86%), India (3.11%), UK (2.98%)
- **Lowest volatility**: Japan (1.50%), Germany (1.67%)
- **Implication**: Emerging markets show higher growth but greater uncertainty

### Crisis Impact Analysis

**2008 Financial Crisis:**
- **All G7 countries** experienced severe, synchronized contraction
- **Deepest impact**: USA, UK (financial sector exposure)
- **Moderate impact**: Germany, France (exported recession)
- **Delayed recovery**: Italy (never fully recovered pre-crisis growth)
- **BRICS**: China/India experienced slowdown but maintained positive growth

**2020 COVID-19 Pandemic:**
- **Most severe shock**: All countries contracted simultaneously
- **Sharpest drops**: UK, France, Italy (lockdown measures)
- **Fastest recovery**: USA (fiscal stimulus), China (rapid containment)
- **Persistent effects**: Service-heavy economies slower to rebound

**Key Finding:** Economic cycles are highly synchronized across G7. BRICS show more independence but still affected by global shocks.

---

## 3. Correlation Insights

### Strong Positive Correlations with GDP:
1. **Household Consumption** (r > 0.95) - strongest predictor
2. **Employment Level** (r > 0.85)
3. **Capital Formation** (r > 0.80)
4. **Government Spending** (r > 0.75)
5. **Industrial Production Index** (r > 0.70)

### Negative Correlations with GDP:
1. **Unemployment Rate** (r < -0.60) - inverse relationship as expected
2. **Interest Rates** (moderate negative, -0.30 to -0.50) - tightening slows growth

### Weak/Variable Correlations:
1. **Exchange Rates** (country-dependent)
2. **Stock Market** (leads GDP but noisy)
3. **Trade Balance** (varies by export dependency)

**Key Finding:** GDP components (C+I+G) are highly collinear with GDP by accounting definition. Must use carefully to avoid circular prediction.

---

## 4. Leading vs Lagging Indicators

### Leading Indicators (Change Before GDP):
1. **Stock Market Index** - forward-looking expectations
2. **Capital Formation (Investment)** - businesses invest ahead of expansion
3. **Industrial Production** - manufacturing precedes broader economy
4. **Short-term Interest Rates** - monetary policy impacts with lag

### Coincident Indicators (Move With GDP):
1. **Employment Level** - moves simultaneously with output
2. **Household Consumption** - reflects current conditions
3. **CPI/Inflation** - demand-driven, tracks growth

### Lagging Indicators (Change After GDP):
1. **Unemployment Rate** - businesses hire after confirming recovery
2. **Long-term Interest Rates** - adjust slowly to structural changes

**Key Finding:** For **nowcasting**, use coincident/leading indicators. For **forecasting**, emphasize leading indicators with proper lags.

---

## 5. Missing Data Patterns

### Systematic Missingness:
- **G7 monthly files**: Only ~25% of rows have GDP data (quarterly updates)
- **Solution**: Resample to quarterly frequency OR forward-fill appropriately

### Indicator-Specific Gaps:
- **Money Supply (M2)**: Missing for Germany, France, Italy
- **Industrial Production**: Some gaps in recent years
- **Trade Components**: Occasional missing months

### USA Daily Data Challenge:
- 6,813 daily rows but most indicators update monthly/quarterly
- **82.5% missing** before quarterly resampling
- After resampling: only 8.5% missing (excellent)

**Key Finding:** Frequency harmonization is critical. Quarterly is optimal baseline for G7. BRICS annual data limits modeling sophistication.

---

## 6. Trade Balance Observations

**Persistent Deficits:**
- **USA**: Large, growing deficit (-$400B+ range)
- **UK**: Moderate deficit
- **France**: Fluctuating around balanced

**Persistent Surpluses:**
- **Germany**: Large surplus ("export champion")
- **Japan**: Consistent surplus (manufacturing exports)
- **Canada**: Resource-driven surpluses

**BRICS:**
- **China**: Massive surpluses (export-driven growth)
- **Brazil**: Commodity-dependent, volatile
- **Russia**: Energy export surpluses

**Key Finding:** Trade patterns reflect structural economic models. Net exports contribute differently to GDP across countries.

---

## 7. Recommendations for Preprocessing Phase

### 1. Frequency Harmonization
- [x] **Primary approach**: Resample all data to **quarterly frequency**
- [ ] USA: Already resampled (use `.resample('QE').last()`)
- [ ] G7: Filter to quarter-end dates OR interpolate monthly → quarterly
- [ ] BRICS: Interpolate annual → quarterly (cubic spline) OR model separately

### 2. Missing Data Strategy
**By Indicator Type:**
- **Policy rates (interest rates)**: Forward-fill (rates persist until changed)
- **Stock indices**: Forward-fill (carry last observation)
- **Economic indicators (GDP, production)**: Linear/cubic interpolation
- **High-missing indicators (>70%)**: Consider dropping or using as binary "available/not"

**Country-Specific:**
- **Germany/France/Italy**: Drop M2 money supply (not available)
- **Russia**: Imports entirely missing - use exports only or drop trade features

### 3. Feature Engineering Priorities
**Create These Features:**
1. **Growth rates**: YoY % change for GDP, consumption, investment
2. **Lags**: t-1, t-2, t-3, t-4 for leading indicators
3. **Moving averages**: 4-quarter MA to smooth volatility
4. **Ratios**:
   - Trade balance / GDP
   - Government spending / GDP
   - Debt / GDP (BRICS)
5. **Differences**: First differences for non-stationary series
6. **Seasonal dummies**: If quarterly seasonality detected

**Avoid:**
- Using GDP components (C+I+G+NX) as direct features when predicting GDP (perfect multicollinearity)
- Mixing frequencies without proper alignment

### 4. Outlier Handling
**Identified Outliers:**
- 2008 Q4 - 2009 Q2: Financial crisis extreme values
- 2020 Q2 - Q3: COVID-19 shock
- Russia 2014-2015: Sanctions/oil price collapse

**Strategy:**
- **DO NOT remove** - these are real economic events
- **Consider**: Dummy variables for crisis periods
- **Alternative**: Robust scaling (median/IQR instead of mean/std)

### 5. Stationarity Considerations
**Non-Stationary Series (require differencing):**
- GDP levels (all countries) - use growth rates instead
- CPI levels - use inflation rate (already provided)
- Employment levels - use growth or unemployment rate
- Stock market indices - use returns

**Stationary Series (use as-is):**
- Unemployment rate
- Inflation rate (CPI growth)
- Interest rates
- Growth rates

---

## 8. Recommendations for Modeling Phase

### Model Selection Guidance

**For Nowcasting (Current Quarter GDP):**
- **Best features**: Industrial production, employment, stock market (available earlier than GDP)
- **Suggested models**:
  - Random Forest (handles non-linearity)
  - XGBoost (excellent for mixed-frequency data)
  - LASSO regression (interpretable, feature selection)

**For Forecasting (1-4 quarters ahead):**
- **Best features**: Lagged GDP, investment, interest rates, stock market
- **Suggested models**:
  - LSTM/GRU (captures temporal dependencies)
  - Vector Autoregression (VAR) - multi-country system
  - ARIMA + ML hybrid

### Country Grouping Strategy

**Option 1: Pooled Model**
- Train single model on all countries
- Add country fixed effects (dummy variables)
- **Pros**: More data, cross-country learning
- **Cons**: Assumes similar economic structures

**Option 2: Country-Specific Models**
- Separate model per country
- **Pros**: Captures unique dynamics
- **Cons**: Less training data, especially for BRICS (n=25)

**Option 3: Tiered Approach**
- **Tier 1**: G7 pooled model (similar economies)
- **Tier 2**: BRICS pooled model (emerging markets)
- **Tier 3**: Individual models for outliers (China, USA)

**Recommendation**: Start with Option 3 (tiered approach)

### Feature Set Recommendations

**Minimal Feature Set (Start Here):**
1. Lagged GDP (t-1, t-2, t-4)
2. Industrial Production Index
3. Employment Level / Unemployment Rate
4. CPI Annual Growth (Inflation)
5. Short-term Interest Rate
6. Capital Formation (Investment)

**Extended Feature Set (After Baseline):**
7. Household Consumption
8. Government Spending
9. Exports + Imports (or Trade Balance)
10. Stock Market Index
11. Exchange Rate
12. Long-term Interest Rate

**BRICS-Specific Features:**
- FDI Inflows
- External Debt
- Urban Population Growth
- Manufacturing vs Services Value Added

### Train/Validation/Test Split

**Recommended Split (Temporal):**
- **Training**: 2000 Q1 - 2018 Q4 (76 quarters, ~75%)
- **Validation**: 2019 Q1 - 2021 Q4 (12 quarters, ~12%)
- **Test**: 2022 Q1 - 2025 Q3 (15 quarters, ~13%)

**Rationale:**
- Validation includes pre-COVID baseline (2019)
- Test includes post-COVID recovery and recent data
- Never shuffle - maintain temporal ordering

**Alternative (Expanding Window):**
- Train on 2000-2015, test on 2016
- Train on 2000-2016, test on 2017
- Etc. (walk-forward validation)

---

## 9. Key Insights Summary

### ✅ Data Strengths
1. **Long time horizon**: 25 years captures multiple business cycles
2. **Crisis periods included**: 2008 financial crisis, 2020 pandemic provide stress-test data
3. **BRICS near-complete**: Minimal missing data for annual indicators
4. **Standardized indicators**: Same variables across G7 enables comparison

### ⚠️ Data Challenges
1. **High G7 missingness**: 42-55% due to monthly storage of quarterly data
2. **Frequency mismatch**: Daily (USA) vs quarterly (G7) vs annual (BRICS)
3. **Recent data gaps**: 2024-2025 data incomplete for some indicators
4. **Structural breaks**: China growth model shift, Brexit impact on UK

### 🎯 Critical Success Factors
1. **Proper resampling**: Quarterly frequency is non-negotiable
2. **Thoughtful imputation**: Use economic logic (forward-fill rates, interpolate volumes)
3. **Feature engineering**: Growth rates and lags are essential
4. **Validation strategy**: Temporal splits only, no data leakage
5. **Model interpretation**: Understand which indicators drive predictions

---

## 10. Next Steps

### Immediate (Preprocessing Phase):
1. **Resample all data to quarterly frequency**
2. **Implement missing data imputation** (forward-fill + interpolation)
3. **Create engineered features** (growth rates, lags, ratios)
4. **Scale/normalize data** (per-country standardization)
5. **Perform stationarity tests** (visual + statistical)
6. **Save processed datasets** in consistent format

### After Preprocessing (Modeling Phase):
1. **Establish baseline model** (simple linear regression)
2. **Implement minimal feature set** (6 core indicators)
3. **Train tiered models** (G7 vs BRICS vs individual)
4. **Evaluate nowcast vs forecast** performance separately
5. **Iterate with extended features** based on results

---

## Appendix: Generated Visualizations

All visualizations saved to `data_viz/figures/`:

1. **01_gdp_timeseries_all_countries.png** - Raw GDP levels over time
2. **02_gdp_growth_rates.png** - Year-over-year growth rates
3. **03_correlation_heatmaps.png** - Indicator correlations (USA, Canada, Japan)
4. **04_missing_data_patterns.png** - Heatmap of missing data by country/indicator
5. **05_indicator_distributions.png** - Boxplots of key indicators across G7
6. **06_leading_indicators_usa.png** - Leading indicators vs GDP (USA)
7. **07_cross_country_synchronization.png** - Economic cycle alignment
8. **08_trade_balance_trends.png** - Trade dynamics across G7

---

**Document Prepared By:** Exploratory Analysis Pipeline
**For:** GDP Nowcasting & Forecasting Project
**Status:** Phase 1 Complete ✓
