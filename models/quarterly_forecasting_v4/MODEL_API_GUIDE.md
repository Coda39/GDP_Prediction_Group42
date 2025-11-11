# v4 USA GDP Prediction Models - API Guide

## Overview

The v4 models predict **USA GDP growth (Year-over-Year percentage)** for 4 different forecast horizons.

**12 trained models** are available, allowing you to choose:
- **Horizon:** 1Q, 2Q, 3Q, or 4Q ahead
- **Algorithm:** Ridge Regression, Random Forest, or Gradient Boosting
- **Recommendation:** Use the Ensemble (average of all 3)

---

## Models Available

### Model Files

```
saved_models/
├── usa_h1_ridge_v4.pkl                  (1Q Ridge)
├── usa_h1_randomforest_v4.pkl           (1Q Random Forest)
├── usa_h1_gradientboosting_v4.pkl       (1Q Gradient Boosting)
├── usa_h2_ridge_v4.pkl                  (2Q Ridge)
├── usa_h2_randomforest_v4.pkl           (2Q Random Forest)
├── usa_h2_gradientboosting_v4.pkl       (2Q Gradient Boosting)
├── usa_h3_ridge_v4.pkl                  (3Q Ridge)
├── usa_h3_randomforest_v4.pkl           (3Q Random Forest)
├── usa_h3_gradientboosting_v4.pkl       (3Q Gradient Boosting)
├── usa_h4_ridge_v4.pkl                  (4Q Ridge)
├── usa_h4_randomforest_v4.pkl           (4Q Random Forest)
└── usa_h4_gradientboosting_v4.pkl       (4Q Gradient Boosting)
```

### Model Dimensions

| Aspect | Details |
|--------|---------|
| **Input Shape** | (1, 84) - single sample with 84 features |
| **Feature Count** | 84 (21 exogenous features × 4 lags each) |
| **Output** | Single float - GDP growth YoY (%) |
| **Data Type** | numpy array or pandas DataFrame |

---

## Input Features (84 Total)

### Feature Engineering

Each of the **21 base features** is expanded to 4 versions:
- **Current value** (t)
- **1-quarter lag** (t-1)
- **2-quarter lag** (t-2)
- **4-quarter lag** (t-4)

**Math:** 21 features × 4 lags = 84 input features

### The 21 Base Features

```python
CLEAN_FEATURES = [
    # Labor Market (3)
    'unemployment_rate',              # % unemployed
    'employment_level',               # Total employed (thousands)
    'employment_growth',              # QoQ employment change (%)

    # Inflation (1)
    'cpi_annual_growth',              # Year-over-year inflation (%)

    # Monetary Policy (2)
    'interest_rate_short_term',       # Fed Funds Rate (%)
    'interest_rate_long_term',        # 10-year Treasury (%)

    # Production (2)
    'industrial_production_index',    # IP index (base year = 100)
    'ip_growth',                      # QoQ IP growth (%)

    # Trade Volumes (4)
    'exports_volume',                 # Exports (billions USD)
    'imports_volume',                 # Imports (billions USD)
    'exports_growth',                 # QoQ export change (%)
    'imports_growth',                 # QoQ import change (%)

    # Consumption (2)
    'household_consumption',          # Personal consumption (billions USD)
    'consumption_growth',             # QoQ consumption change (%)

    # Investment (2)
    'capital_formation',              # Fixed capital investment (billions USD)
    'investment_growth',              # QoQ investment change (%)

    # Monetary Aggregates (2)
    'money_supply_broad',             # M2 money supply (billions USD)
    'm2_growth',                      # QoQ M2 growth (%)

    # Asset Prices (2)
    'stock_market_index',             # S&P 500 or similar (index points)
    'exchange_rate_usd',              # Trade-weighted USD index

    # Government (1)
    'government_spending',            # Gov expenditure (billions USD)
]
```

### Feature List (84 features in order)

```
Position  Feature Name                      Lag
1-4       unemployment_rate                 t, t-1, t-2, t-4
5-8       employment_level                  t, t-1, t-2, t-4
9-12      employment_growth                 t, t-1, t-2, t-4
13-16     cpi_annual_growth                 t, t-1, t-2, t-4
17-20     interest_rate_short_term          t, t-1, t-2, t-4
21-24     interest_rate_long_term           t, t-1, t-2, t-4
25-28     industrial_production_index       t, t-1, t-2, t-4
29-32     ip_growth                         t, t-1, t-2, t-4
33-36     exports_volume                    t, t-1, t-2, t-4
37-40     imports_volume                    t, t-1, t-2, t-4
41-44     exports_growth                    t, t-1, t-2, t-4
45-48     imports_growth                    t, t-1, t-2, t-4
49-52     household_consumption             t, t-1, t-2, t-4
53-56     consumption_growth                t, t-1, t-2, t-4
57-60     capital_formation                 t, t-1, t-2, t-4
61-64     investment_growth                 t, t-1, t-2, t-4
65-68     money_supply_broad                t, t-1, t-2, t-4
69-72     m2_growth                         t, t-1, t-2, t-4
73-76     stock_market_index                t, t-1, t-2, t-4
77-80     exchange_rate_usd                 t, t-1, t-2, t-4
81-84     government_spending               t, t-1, t-2, t-4
```

---

## Output

### Output Format

```python
prediction = model.predict(X_input)
# Returns: float (single value for one sample)
# Value: GDP growth rate, Year-over-Year (%)
```

### Output Interpretation

**Example outputs:**
- `2.5` = USA GDP will grow 2.5% year-over-year
- `0.8` = USA GDP will grow 0.8% year-over-year
- `-0.3` = USA GDP will contract 0.3% year-over-year

**Expected ranges** (from validation):
- **1Q ahead:** -1% to 4%
- **2Q ahead:** -1% to 4%
- **3Q ahead:** -2% to 5%
- **4Q ahead:** -2% to 5%

---

## How to Use (Python Code)

### Basic Usage

```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Load a model
model = joblib.load('saved_models/usa_h1_ridge_v4.pkl')

# Step 2: Prepare your input data
# Shape: (1, 84) - one sample, 84 features
X_input = np.array([[
    # unemployment_rate (t, t-1, t-2, t-4)
    3.5, 3.4, 3.6, 3.8,

    # employment_level (t, t-1, t-2, t-4)
    132500, 132300, 132100, 131500,

    # employment_growth (t, t-1, t-2, t-4)
    0.15, 0.10, 0.05, -0.05,

    # cpi_annual_growth (t, t-1, t-2, t-4)
    2.8, 2.9, 3.0, 2.5,

    # interest_rate_short_term (t, t-1, t-2, t-4)
    4.25, 4.50, 4.75, 3.50,

    # interest_rate_long_term (t, t-1, t-2, t-4)
    4.10, 4.30, 4.50, 3.80,

    # industrial_production_index (t, t-1, t-2, t-4)
    103.5, 103.2, 102.8, 100.5,

    # ip_growth (t, t-1, t-2, t-4)
    0.3, 0.4, 0.5, 0.2,

    # exports_volume (t, t-1, t-2, t-4)
    1850, 1840, 1820, 1750,

    # imports_volume (t, t-1, t-2, t-4)
    2420, 2410, 2400, 2350,

    # exports_growth (t, t-1, t-2, t-4)
    0.5, 1.1, 0.9, -0.2,

    # imports_growth (t, t-1, t-2, t-4)
    0.4, 0.9, 1.0, 0.1,

    # household_consumption (t, t-1, t-2, t-4)
    10250, 10200, 10100, 9900,

    # consumption_growth (t, t-1, t-2, t-4)
    0.5, 0.8, 1.0, 0.7,

    # capital_formation (t, t-1, t-2, t-4)
    2900, 2880, 2850, 2750,

    # investment_growth (t, t-1, t-2, t-4)
    0.7, 1.0, 1.2, 0.5,

    # money_supply_broad (t, t-1, t-2, t-4)
    21500, 21400, 21200, 20500,

    # m2_growth (t, t-1, t-2, t-4)
    0.5, 0.6, 0.7, 2.0,

    # stock_market_index (t, t-1, t-2, t-4)
    5200, 5150, 5100, 4800,

    # exchange_rate_usd (t, t-1, t-2, t-4)
    105.5, 105.0, 104.5, 100.0,

    # government_spending (t, t-1, t-2, t-4)
    7850, 7840, 7820, 7700,
]])

# Step 3: Make prediction
prediction = model.predict(X_input)[0]
print(f"Predicted 1Q GDP growth: {prediction:.2f}%")
```

### With Ensemble (Recommended)

```python
import joblib
import numpy as np

# Load all 3 models for a horizon
ridge = joblib.load('saved_models/usa_h1_ridge_v4.pkl')
rf = joblib.load('saved_models/usa_h1_randomforest_v4.pkl')
gb = joblib.load('saved_models/usa_h1_gradientboosting_v4.pkl')

# Prepare input (same as above)
X_input = np.array([[...]])  # 84 features

# Get predictions from each model
pred_ridge = ridge.predict(X_input)[0]
pred_rf = rf.predict(X_input)[0]
pred_gb = gb.predict(X_input)[0]

# Ensemble (average)
ensemble_pred = (pred_ridge + pred_rf + pred_gb) / 3

print(f"Ridge prediction: {pred_ridge:.2f}%")
print(f"RF prediction: {pred_rf:.2f}%")
print(f"GB prediction: {pred_gb:.2f}%")
print(f"Ensemble prediction: {ensemble_pred:.2f}%")
```

---

## Feature Normalization

**IMPORTANT:** The models were trained on normalized features using StandardScaler.

### Correct way to normalize:

```python
from sklearn.preprocessing import StandardScaler

# Fit on training data (DO ONCE)
scaler = StandardScaler()
scaler.fit(X_train_raw)  # Your historical data

# Apply to new input (DO FOR EACH PREDICTION)
X_normalized = scaler.transform(X_input_raw)

# Then predict
prediction = model.predict(X_normalized)
```

### Common mistake to avoid:

```python
# ❌ WRONG - Don't fit and transform on the same data
X_normalized = scaler.fit_transform(X_input_raw)

# ✅ RIGHT - Use previously fitted scaler
X_normalized = scaler.transform(X_input_raw)
```

---

## Data Sources

Here's where to get the 21 features:

| Feature | Source | Notes |
|---------|--------|-------|
| unemployment_rate | BLS | Bureau of Labor Statistics |
| employment_level | BLS | Total non-farm employment |
| employment_growth | BLS | Calculated from above |
| cpi_annual_growth | BLS | Consumer Price Index |
| interest_rate_short_term | Federal Reserve | Fed Funds Rate |
| interest_rate_long_term | Federal Reserve | 10-year Treasury |
| industrial_production_index | Federal Reserve | IP Index |
| ip_growth | Federal Reserve | Calculated from above |
| exports_volume | Census Bureau | Merchandise exports |
| imports_volume | Census Bureau | Merchandise imports |
| exports_growth | Census Bureau | Calculated from above |
| imports_growth | Census Bureau | Calculated from above |
| household_consumption | BEA | Personal consumption expenditures |
| consumption_growth | BEA | Calculated from above |
| capital_formation | BEA | Gross capital formation |
| investment_growth | BEA | Calculated from above |
| money_supply_broad | Federal Reserve | M2 Money Supply |
| m2_growth | Federal Reserve | Calculated from above |
| stock_market_index | FRED/Yahoo Finance | S&P 500 or similar |
| exchange_rate_usd | Federal Reserve | Trade-weighted USD |
| government_spending | BEA | Government expenditures |

---

## Hosting the Model

### Option 1: Python Flask API

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models
models = {
    'h1_ridge': joblib.load('saved_models/usa_h1_ridge_v4.pkl'),
    'h1_rf': joblib.load('saved_models/usa_h1_randomforest_v4.pkl'),
    'h1_gb': joblib.load('saved_models/usa_h1_gradientboosting_v4.pkl'),
    # ... load all 12 models
}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON:
    {
        "horizon": 1,
        "model": "ensemble",
        "features": [84-element array]
    }
    """
    data = request.json
    horizon = data['horizon']  # 1, 2, 3, or 4
    model_type = data['model']  # 'ridge', 'rf', 'gb', 'ensemble'
    features = np.array(data['features']).reshape(1, -1)

    if model_type == 'ensemble':
        # Load all 3 models for this horizon
        ridge = models[f'h{horizon}_ridge']
        rf = models[f'h{horizon}_rf']
        gb = models[f'h{horizon}_gb']

        # Get predictions
        pred = (
            ridge.predict(features)[0] +
            rf.predict(features)[0] +
            gb.predict(features)[0]
        ) / 3
    else:
        # Single model
        model = models[f'h{horizon}_{model_type}']
        pred = model.predict(features)[0]

    return jsonify({
        'horizon': f'{horizon}Q',
        'prediction': float(pred),
        'unit': 'GDP growth YoY (%)'
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

### Option 2: FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class PredictionRequest(BaseModel):
    horizon: int  # 1, 2, 3, or 4
    model_type: str  # 'ridge', 'rf', 'gb', 'ensemble'
    features: list  # 84 features

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Same logic as Flask above
    pass
```

### Option 3: AWS Lambda

```python
import json
import boto3
import joblib
import numpy as np

s3 = boto3.client('s3')
models = {}

def lambda_handler(event, context):
    # Load model from S3 (first time only)
    if not models:
        obj = s3.get_object(Bucket='your-bucket', Key='usa_h1_ridge_v4.pkl')
        models['ridge'] = joblib.load(obj['Body'])

    # Get input
    body = json.loads(event['body'])
    features = np.array(body['features']).reshape(1, -1)

    # Predict
    prediction = models['ridge'].predict(features)[0]

    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction': float(prediction),
            'unit': 'GDP growth YoY (%)'
        })
    }
```

---

## Example API Request/Response

### Request
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "horizon": 1,
    "model": "ensemble",
    "features": [3.5, 132500, 0.15, 2.8, 4.25, 4.10, 103.5, 0.3, 1850, 2420, 0.5, 0.4, 10250, 0.5, 2900, 0.7, 21500, 0.5, 5200, 105.5, 7850, 3.4, 132300, 0.10, 2.9, 4.50, 4.30, 103.2, 0.4, 1840, 2410, 1.1, 0.9, 10200, 0.8, 2880, 1.0, 21400, 0.6, 5150, 105.0, 7840, 3.6, 132100, 0.05, 3.0, 4.75, 4.50, 102.8, 0.5, 1820, 2400, 0.9, 1.0, 10100, 1.0, 2850, 1.2, 21200, 0.7, 5100, 104.5, 7820, 3.8, 131500, -0.05, 2.5, 3.50, 3.80, 100.5, 0.2, 1750, 2350, -0.2, 0.1, 9900, 0.7, 2750, 0.5, 20500, 2.0, 4800, 100.0, 7700]
  }'
```

### Response
```json
{
  "horizon": "1Q",
  "prediction": 2.15,
  "unit": "GDP growth YoY (%)"
}
```

---

## Performance Expectations

| Horizon | Expected Accuracy | Use Case |
|---------|-------------------|----------|
| **1Q ahead** | R² = 0.08-0.13 | Good for nowcasting |
| **2Q ahead** | R² = 0.04-0.09 | Moderate predictions |
| **3Q ahead** | R² = 0.00-0.07 | Weak signals |
| **4Q ahead** | R² = -0.02-0.05 | Very uncertain |

**Note:** These are the expected ranges from validation on 2022-2025 test data. Actual performance may vary based on economic conditions.

---

## Summary

**Input:**
- 84 normalized numerical features (21 economic indicators × 4 lags)
- Shape: (1, 84) - one sample, 84 features
- Must be normalized using StandardScaler before prediction

**Output:**
- Single float value
- Represents GDP growth rate, Year-over-Year (%)
- Range: typically -2% to +5%

**Models:**
- 12 trained scikit-learn models (Ridge, RandomForest, GradientBoosting × 4 horizons)
- Recommendation: Use ensemble (average of 3 algorithms)
- For 1Q horizon: Expected accuracy R² = 0.08-0.13

**Hosting:**
- Can be hosted on Flask, FastAPI, AWS Lambda, or similar
- Just load pickle files and call `.predict(X_normalized)`
- Return prediction as JSON with horizon and percentage value
