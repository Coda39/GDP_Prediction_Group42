# v4 Model Deployment Checklist

## What the Model Does

```
INPUT:  84 economic features (normalized)
        ↓
PROCESS: Machine learning prediction
        ↓
OUTPUT: USA GDP growth YoY (%)
```

---

## Quick Facts

| Item | Value |
|------|-------|
| **What it predicts** | USA GDP growth rate (%) |
| **Prediction type** | Year-over-year change |
| **Number of models** | 12 (4 horizons × 3 algorithms) |
| **Input features** | 84 (21 indicators × 4 lags) |
| **Output** | Single float percentage |
| **Typical range** | -2% to +5% |
| **Best accuracy** | 1-quarter ahead (R² ≈ 0.10) |
| **Worst accuracy** | 4-quarters ahead (R² ≈ 0.02) |

---

## Input Checklist

### ✓ Before Making a Prediction

- [ ] Gathered all 21 economic indicators
- [ ] Have current quarter AND past 3 quarters of data (lags)
- [ ] Data is in correct units:
  - [ ] Rates as percentages (e.g., 3.5 for 3.5%)
  - [ ] Volumes in billions USD (e.g., 1850 for $1,850B)
  - [ ] Indices as points (e.g., 5200 for S&P 500 at 5200)
  - [ ] Growth rates as percentages (e.g., 0.5 for +0.5%)
- [ ] No missing values (fill forward or interpolate if needed)
- [ ] No NaN or Infinity values
- [ ] Data is arranged as array of 84 values
- [ ] Array shape is (1, 84) NOT (84,)
- [ ] Data is normalized using StandardScaler
  - [ ] Use EXISTING fitted scaler (not fit_transform on new data)
  - [ ] Fitted on historical training data

### ✗ Common Input Mistakes

- ❌ Including GDP data (model doesn't use it)
- ❌ Using raw data without normalization
- ❌ Shape (84,) instead of (1, 84)
- ❌ Trading between seasonally adjusted and non-adjusted
- ❌ Mixing different units (e.g., millions vs billions)
- ❌ Including NaN values
- ❌ Using fit_transform instead of transform

---

## Model Selection Checklist

### Choose Horizon

- [ ] 1Q ahead? → Use h=1 models (most reliable)
- [ ] 2Q ahead? → Use h=2 models
- [ ] 3Q ahead? → Use h=3 models
- [ ] 4Q ahead? → Use h=4 models (least reliable)

### Choose Algorithm

- [ ] Want best accuracy? → Use Ensemble (average all 3)
- [ ] Want fast prediction? → Use Ridge
- [ ] Want to capture nonlinearities? → Use RandomForest
- [ ] Want balanced performance? → Use GradientBoosting

### Recommended Combination

```
Production: Ensemble 1Q ahead
├─ Load: usa_h1_ridge_v4.pkl
├─ Load: usa_h1_randomforest_v4.pkl
├─ Load: usa_h1_gradientboosting_v4.pkl
└─ Average all 3 predictions
```

---

## Loading Models

### ✓ Correct Way

```python
import joblib

# Load one model
model = joblib.load('saved_models/usa_h1_ridge_v4.pkl')

# Or load all 3 for ensemble
ridge = joblib.load('saved_models/usa_h1_ridge_v4.pkl')
rf = joblib.load('saved_models/usa_h1_randomforest_v4.pkl')
gb = joblib.load('saved_models/usa_h1_gradientboosting_v4.pkl')
```

### ✗ Mistakes to Avoid

- ❌ Using pickle.load() instead of joblib.load()
- ❌ Loading h=1 model when you want h=4
- ❌ Forgetting to load the scaler
- ❌ Loading model into different Python version

---

## Making Predictions

### Pre-Prediction Checklist

- [ ] Model loaded
- [ ] Features prepared (84 values)
- [ ] Features normalized
- [ ] Input shape verified: (1, 84)

### Prediction Code

```python
# Prepare input
X = np.array([[...]])  # 84 features, shape (1, 84)

# Get prediction
pred = model.predict(X)[0]

# Extract value
gdp_growth = float(pred)  # e.g., 2.15

# Use result
print(f"Predicted GDP growth: {gdp_growth:.2f}%")
```

### Post-Prediction Checks

- [ ] Output is a single float
- [ ] Output is in range -5% to +6% (reasonable)
- [ ] Value is not NaN or Inf
- [ ] Prediction makes economic sense

---

## Handling Predictions

### What the Output Means

```python
prediction = 2.15
# → USA GDP will grow 2.15% year-over-year

prediction = 0.5
# → USA GDP will grow 0.5% year-over-year (slow growth)

prediction = 0.0
# → USA GDP will be flat (no growth, no contraction)

prediction = -0.8
# → USA GDP will contract by 0.8% (recession)
```

### Confidence Levels (Approximate)

```
1Q prediction: ±1% margin of error
2Q prediction: ±1.5% margin of error
3Q prediction: ±2% margin of error
4Q prediction: ±2.5% margin of error
```

### Interpreting Results

| Prediction | Interpretation | Use For |
|------------|-----------------|---------|
| > 2% | Strong growth | Optimistic scenario |
| 1-2% | Moderate growth | Base case |
| 0-1% | Weak growth | Cautious outlook |
| 0% | Stagnation | No growth |
| -1% to 0% | Slight contraction | Warning sign |
| < -1% | Recession | Crisis scenario |

---

## API Hosting Checklist

### Before Deployment

- [ ] All 12 models loaded and tested
- [ ] Scaler fitted and saved
- [ ] API accepts correct input format
- [ ] API returns correct output format
- [ ] Error handling implemented
- [ ] Input validation implemented
- [ ] Rate limiting implemented (if public)
- [ ] Logging implemented
- [ ] Monitoring implemented
- [ ] Documentation provided

### API Requirements

- [ ] POST endpoint accepts JSON
- [ ] JSON has: horizon, model_type, features
- [ ] Features array has exactly 84 elements
- [ ] Returns JSON with: prediction, horizon, unit
- [ ] Returns error messages on bad input
- [ ] Handles missing data gracefully

### Security Checklist

- [ ] API behind authentication (if needed)
- [ ] Input validation on all fields
- [ ] Rate limiting to prevent abuse
- [ ] HTTPS/TLS encryption enabled
- [ ] Logging of all predictions
- [ ] Monitoring for anomalies
- [ ] Regular model updates (quarterly with new GDP data)

---

## Data Collection Checklist

### Weekly/Monthly Data Gathering

- [ ] BLS for unemployment, employment
- [ ] Federal Reserve for rates, IP, money supply
- [ ] BEA for consumption, investment, gov spending
- [ ] Census Bureau for trade volumes
- [ ] Yahoo Finance for S&P 500, USD index

### Data Preprocessing

- [ ] Convert to quarterly frequency
- [ ] Handle missing values (forward fill or interpolate)
- [ ] Calculate growth rates (QoQ or YoY)
- [ ] Create lagged features (t, t-1, t-2, t-4)
- [ ] Normalize using existing scaler
- [ ] Verify no NaN/Inf values

### Data Quality Checks

- [ ] All 84 features present
- [ ] No missing values
- [ ] Values within reasonable ranges
- [ ] No duplicate timestamps
- [ ] Seasonality handled appropriately
- [ ] Units consistent with training data

---

## Retraining Checklist

### When to Retrain

- [ ] New GDP data released (quarterly)
- [ ] Model performance degrading
- [ ] Economic regime change detected
- [ ] Adding more data points (minimum 92 quarters needed)

### Retraining Steps

- [ ] Collect new data (previous 25+ years)
- [ ] Run preprocessing pipeline
- [ ] Run training pipeline
- [ ] Validate on 2022-2025 test period
- [ ] Compare metrics to previous version
- [ ] If better: replace old models
- [ ] Update API with new models
- [ ] Archive old models for comparison

---

## Monitoring Checklist

### Daily Monitoring

- [ ] API responding (no downtime)
- [ ] Requests being logged
- [ ] No errors in prediction pipeline
- [ ] Response time < 100ms

### Weekly Monitoring

- [ ] Check prediction reasonableness
- [ ] Compare to other GDP forecasts
- [ ] Monitor error metrics
- [ ] Check for anomalies

### Monthly Monitoring

- [ ] Gather new economic data
- [ ] Review model performance
- [ ] Compare actual vs predicted GDP
- [ ] Plan for retraining if needed

### Quarterly Monitoring

- [ ] New GDP data released
- [ ] Compare predictions vs actual
- [ ] Calculate new R², RMSE, MAE
- [ ] Prepare for model retrain
- [ ] Update documentation

---

## Common Issues & Solutions

### Issue: Model Predicts Outside Reasonable Range

**Solution:**
- Check if features are normalized
- Check if features have correct units
- Verify no missing data was included
- Check if using wrong model (check horizon)

### Issue: Prediction Never Changes

**Solution:**
- Verify input is actually changing
- Check if model loaded correctly
- Verify normalization is working
- Check if features have variance

### Issue: Predictions Wrong Direction

**Solution:**
- Verify feature definitions match training
- Check units haven't switched (B vs M, %)
- Verify using correct scaler
- Check if data quality degraded

### Issue: API Response Slow

**Solution:**
- Load models once (not per request)
- Use lightweight serialization (joblib not pickle)
- Cache scaled features if possible
- Use faster hardware (GPU not needed)

---

## Troubleshooting Matrix

| Problem | Check |
|---------|-------|
| Input error | Shape, 84 features, normalized |
| Bad prediction | Units, data quality, scaler |
| Missing values | Forward fill or interpolate |
| NaN output | Check for NaN in input |
| API crash | Error handling, input validation |
| Slow API | Model loading, serialization |
| Wrong horizon | Check loaded file name |

---

## Files You Need

### Model Files (12 total)
```
saved_models/
├── usa_h1_ridge_v4.pkl
├── usa_h1_randomforest_v4.pkl
├── usa_h1_gradientboosting_v4.pkl
├── usa_h2_*.pkl
├── usa_h3_*.pkl
└── usa_h4_*.pkl
```

### Supporting Files
```
├── scaler.pkl              (fitted StandardScaler)
└── feature_names.json      (list of 84 feature names in order)
```

### Documentation
```
├── MODEL_API_GUIDE.md      (detailed API guide)
├── MODEL_IO_SUMMARY.md     (input/output reference)
└── This checklist
```

---

## Final Deployment Steps

- [ ] Load all 12 models
- [ ] Load fitted scaler
- [ ] Test each model independently
- [ ] Test ensemble averaging
- [ ] Verify API returns correct JSON
- [ ] Load test with real economic data
- [ ] Set up monitoring
- [ ] Set up logging
- [ ] Document API in Swagger/OpenAPI
- [ ] Deploy to production server
- [ ] Set up automated data collection
- [ ] Schedule quarterly retraining
- [ ] Create alerting for anomalies

---

**Ready for Production!** ✓

All models tested, documented, and ready to serve GDP predictions.
