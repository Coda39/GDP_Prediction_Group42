# v4 USA GDP Models - Input/Output Summary

## Quick Overview

```
                        ┌──────────────────────┐
                        │  v4 GDP Model        │
                        │  (12 variations)     │
                        └──────────────────────┘
                              ↓
                ┌─────────────────────────────┐
                │   INPUT: 84 Features        │
                │ (Normalized numeric values) │
                └─────────────────────────────┘
                              ↓
                ┌─────────────────────────────┐
                │   PROCESS: Neural Network   │
                │   (Ridge/RF/GB)             │
                └─────────────────────────────┘
                              ↓
                ┌─────────────────────────────┐
                │  OUTPUT: Single Float       │
                │  GDP Growth YoY (%)         │
                └─────────────────────────────┘
```

---

## INPUT (What you feed in)

### Size: 84 Features

**Breakdown:**
- 21 economic indicators
- Each with 4 time lags (current, -1Q, -2Q, -4Q)
- **21 × 4 = 84 features**

### The 21 Indicators

```
LABOR MARKET (3)
├─ unemployment_rate: % of population unemployed
├─ employment_level: Total employed (thousands)
└─ employment_growth: Quarterly change (%)

INFLATION (1)
└─ cpi_annual_growth: Year-over-year inflation (%)

MONETARY POLICY (2)
├─ interest_rate_short_term: Fed Funds Rate (%)
└─ interest_rate_long_term: 10-year Treasury (%)

PRODUCTION (2)
├─ industrial_production_index: IP index (base=100)
└─ ip_growth: Quarterly change (%)

TRADE (4)
├─ exports_volume: Total exports (billions USD)
├─ imports_volume: Total imports (billions USD)
├─ exports_growth: Quarterly change (%)
└─ imports_growth: Quarterly change (%)

CONSUMPTION (2)
├─ household_consumption: Personal spending (billions USD)
└─ consumption_growth: Quarterly change (%)

INVESTMENT (2)
├─ capital_formation: Fixed investment (billions USD)
└─ investment_growth: Quarterly change (%)

MONEY SUPPLY (2)
├─ money_supply_broad: M2 (billions USD)
└─ m2_growth: Quarterly change (%)

ASSETS (2)
├─ stock_market_index: S&P 500 level
└─ exchange_rate_usd: Trade-weighted USD

GOVERNMENT (1)
└─ government_spending: Federal spending (billions USD)
```

### Example Input Values

```python
# Single quarter of data (array of 84 values)
input_array = [
    # unemployment_rate: current, lag1, lag2, lag4
    3.5,      # Q3 2024 = 3.5%
    3.4,      # Q2 2024 = 3.4%
    3.6,      # Q1 2024 = 3.6%
    3.8,      # Q3 2023 = 3.8%

    # employment_level: current, lag1, lag2, lag4
    132500,   # Q3 2024 = 132.5M people
    132300,   # Q2 2024 = 132.3M
    132100,   # Q1 2024 = 132.1M
    131500,   # Q3 2023 = 131.5M

    # employment_growth: current, lag1, lag2, lag4
    0.15,     # Q3 2024 = +0.15%
    0.10,     # Q2 2024 = +0.10%
    0.05,     # Q1 2024 = +0.05%
    -0.05,    # Q3 2023 = -0.05%

    # ... continue for 21 features × 4 lags = 84 total
]
```

### Input Properties

| Property | Value |
|----------|-------|
| **Shape** | (1, 84) |
| **Data Type** | numpy array or list of floats |
| **Normalization** | REQUIRED (StandardScaler) |
| **Missing Values** | NOT ALLOWED |
| **NaN/Inf** | NOT ALLOWED |

---

## OUTPUT (What you get back)

### Size: Single Float Value

```python
prediction = 2.15
# ↓
# Interpretation: USA GDP will grow 2.15% year-over-year
```

### Output Format

```python
# Raw output
model.predict(X_input)
# Returns: numpy array with one element
# Example: array([2.15])

# Extract scalar
prediction = model.predict(X_input)[0]
# Returns: float
# Example: 2.15
```

### Output Range (Expected)

```
1Q ahead:  -1.0% to +4.0%  (most reliable)
2Q ahead:  -1.0% to +4.0%
3Q ahead:  -2.0% to +5.0%
4Q ahead:  -2.0% to +5.0%  (least reliable)
```

### Output Interpretation

```python
prediction = 2.15

# If prediction > 0:
#   GDP is growing
#   Value = growth rate

# If prediction = 0:
#   GDP is flat
#   No growth or contraction

# If prediction < 0:
#   GDP is contracting (recession)
#   Negative value = magnitude of contraction

# Example outputs:
 3.5  → GDP growth of 3.5% YoY
 0.8  → GDP growth of 0.8% YoY
 0.0  → No GDP growth (stagnation)
-1.2  → GDP contraction of 1.2% YoY
```

---

## 12 Available Models

### Organized by Horizon and Algorithm

```
1-QUARTER AHEAD
├─ usa_h1_ridge_v4.pkl
├─ usa_h1_randomforest_v4.pkl
└─ usa_h1_gradientboosting_v4.pkl

2-QUARTER AHEAD
├─ usa_h2_ridge_v4.pkl
├─ usa_h2_randomforest_v4.pkl
└─ usa_h2_gradientboosting_v4.pkl

3-QUARTER AHEAD
├─ usa_h3_ridge_v4.pkl
├─ usa_h3_randomforest_v4.pkl
└─ usa_h3_gradientboosting_v4.pkl

4-QUARTER AHEAD
├─ usa_h4_ridge_v4.pkl
├─ usa_h4_randomforest_v4.pkl
└─ usa_h4_gradientboosting_v4.pkl
```

### Which Model to Use?

| Use Case | Recommendation |
|----------|-----------------|
| **Accuracy** | Ensemble (average all 3) |
| **Speed** | Ridge (fastest) |
| **Complex patterns** | Random Forest or GB |
| **1Q prediction** | Any model (most reliable) |
| **4Q prediction** | Ensemble (most robust) |

---

## Complete Example

### Step-by-Step

```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# STEP 1: Load Model
# ─────────────────────────────────────────

ridge_h1 = joblib.load('saved_models/usa_h1_ridge_v4.pkl')
rf_h1 = joblib.load('saved_models/usa_h1_randomforest_v4.pkl')
gb_h1 = joblib.load('saved_models/usa_h1_gradientboosting_v4.pkl')

# ─────────────────────────────────────────
# STEP 2: Prepare Raw Data
# ─────────────────────────────────────────

raw_features = np.array([[
    # 84 feature values from your data source
    # Unemployment (4 values): current, lag1, lag2, lag4
    3.5, 3.4, 3.6, 3.8,

    # Employment level (4 values)
    132500, 132300, 132100, 131500,

    # ... continue with all 84 values
]])

# ─────────────────────────────────────────
# STEP 3: Normalize (CRITICAL!)
# ─────────────────────────────────────────

scaler = StandardScaler()
# Note: In production, use previously fitted scaler
normalized_features = scaler.fit_transform(raw_features)

# ─────────────────────────────────────────
# STEP 4: Get Predictions
# ─────────────────────────────────────────

pred_ridge = ridge_h1.predict(normalized_features)[0]
pred_rf = rf_h1.predict(normalized_features)[0]
pred_gb = gb_h1.predict(normalized_features)[0]

# ─────────────────────────────────────────
# STEP 5: Ensemble (Average)
# ─────────────────────────────────────────

ensemble_pred = (pred_ridge + pred_rf + pred_gb) / 3

# ─────────────────────────────────────────
# STEP 6: Use Prediction
# ─────────────────────────────────────────

print(f"Ridge: {pred_ridge:.2f}%")
print(f"RF: {pred_rf:.2f}%")
print(f"GB: {pred_gb:.2f}%")
print(f"Ensemble (recommended): {ensemble_pred:.2f}%")

# Example output:
# Ridge: 2.10%
# RF: 2.25%
# GB: 2.05%
# Ensemble (recommended): 2.13%
```

---

## Common Mistakes & How to Fix

### ❌ Mistake 1: Not Normalizing

```python
# WRONG - model trained on normalized data
prediction = model.predict(raw_features)  # ❌ Bad results

# RIGHT - use StandardScaler
scaler = StandardScaler()
normalized = scaler.fit_transform(raw_features)
prediction = model.predict(normalized)  # ✓ Good results
```

### ❌ Mistake 2: Wrong Input Shape

```python
# WRONG - 2D shape required
X = np.array([1, 2, 3, ...])  # Shape: (84,) ❌

# RIGHT
X = np.array([[1, 2, 3, ...]])  # Shape: (1, 84) ✓
```

### ❌ Mistake 3: Including GDP Data

```python
# WRONG - model doesn't use these features
features = [
    unemployment, employment, cpi,
    gdp_growth,        # ❌ Don't include GDP!
    gdp_real,          # ❌ Don't include GDP!
    trade_balance,     # ❌ Component of GDP!
]

# RIGHT - only the 21 exogenous features
features = [
    unemployment, employment, employment_growth,
    cpi_annual_growth,
    interest_rate_short, interest_rate_long,
    ip_index, ip_growth,
    exports_vol, imports_vol, exports_growth, imports_growth,
    consumption, consumption_growth,
    capital_formation, investment_growth,
    money_supply, m2_growth,
    stock_market, exchange_rate,
    government_spending,
    # REPEAT each 4 times for lags (t, t-1, t-2, t-4)
]
```

### ❌ Mistake 4: Predicting Recession

```python
# WRONG - model predicts GDP growth, not recession risk
prediction = 2.15  # ✓ Model output

# Don't interpret as: "Recession coming" ❌
# Do interpret as: "GDP will grow 2.15% YoY" ✓

# If prediction < 0, THEN you can say GDP will contract
prediction = -0.5  # ✓ GDP contraction of 0.5%
```

---

## For Production/Hosting

### API Endpoint Design

```
POST /predict
{
    "horizon": 1,           # 1, 2, 3, or 4
    "model": "ensemble",    # "ridge", "rf", "gb", "ensemble"
    "features": [...]       # 84 normalized values
}

Response:
{
    "prediction": 2.15,
    "horizon": "1Q",
    "unit": "GDP growth YoY (%)",
    "confidence": "±0.8%"   # Optional
}
```

### Key Requirements

1. **Input validation** - Check 84 features present
2. **Normalization** - Apply StandardScaler before prediction
3. **Range checking** - Warn if output outside -2% to +5%
4. **Error handling** - Return meaningful error messages
5. **Logging** - Log all predictions for audit trail

---

## Model Metadata

```python
# Model properties you might need
HORIZONS = [1, 2, 3, 4]  # Quarters ahead
MODELS = ['ridge', 'randomforest', 'gradientboosting']
FEATURES_COUNT = 84
BASE_FEATURES = 21
LAGS = [0, 1, 2, 4]  # Current + quarter lags

# Expected performance (validation set: 2022-2025)
EXPECTED_R2 = {
    1: (0.08, 0.13),    # 1Q: R² between 8-13%
    2: (0.04, 0.09),    # 2Q: R² between 4-9%
    3: (0.00, 0.07),    # 3Q: R² between 0-7%
    4: (-0.02, 0.05),   # 4Q: R² between -2% to 5%
}

# Input/output specs
INPUT_SHAPE = (1, 84)
INPUT_TYPE = "float32"
OUTPUT_SHAPE = (1,)
OUTPUT_TYPE = "float32"
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Input** | 84 normalized numerical features |
| **Output** | Single float (GDP growth %) |
| **Models** | 12 (4 horizons × 3 algorithms) |
| **Recommended** | Ensemble (average of 3) |
| **Horizons** | 1Q, 2Q, 3Q, 4Q ahead |
| **Accuracy (1Q)** | R² = 0.08-0.13 |
| **Accuracy (4Q)** | R² = -0.02-0.05 |
| **Normalization** | Required (StandardScaler) |
| **Data Source** | Federal Reserve, BLS, BEA, Census |

---

**Ready to deploy!** See MODEL_API_GUIDE.md for full implementation examples.
