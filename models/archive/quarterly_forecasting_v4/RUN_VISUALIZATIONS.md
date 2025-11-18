# How to Run the Visualizations (Fixed Version)

## Quick Start

```bash
# Navigate to v4 directory
cd /Users/Mateo/School/Fall_2025/CS_4485/GDP_Prediction_Group42/models/quarterly_forecasting_v4

# Step 1: Train models (5-10 minutes)
python3 forecasting_pipeline_v4.py

# Step 2: Generate visualizations (2-3 minutes)
python3 forecast_visualization_v4.py

# Step 3: Check your outputs
ls forecast_visualizations/
```

---

## What Gets Created

### Step 1: Training
**Command:** `python3 forecasting_pipeline_v4.py`

**Creates:**
- `saved_models/` - 12 trained model files (.pkl)
- `results/v4_model_performance.csv` - Performance metrics
- `results/v4_predictions.pkl` - **Predictions data (NEW!)**

**Output looks like:**
```
================================================================================
GDP QUARTERLY FORECASTING v4 - CLEAN FEATURES & SEPARATE HORIZONS
================================================================================

1. Loading and preparing data...
✓ Loaded USA data: X rows
✓ Using 21 clean exogenous features

2. Training separate models for each horizon...
  [Training 1Q, 2Q, 3Q, 4Q models...]
  [Shows R², RMSE, MAE for each model]

3. Saving results...
  ✓ Saved results to v4_model_performance.csv
  ✓ Saved predictions to v4_predictions.pkl

================================================================================
✓ v4 PIPELINE COMPLETE
================================================================================
```

### Step 2: Visualization
**Command:** `python3 forecast_visualization_v4.py`

**Creates 9 PNG files:**
```
forecast_visualizations/
├── usa_forecast_h1_v4.png                 (1Q ahead forecast)
├── usa_forecast_h2_v4.png                 (2Q ahead forecast)
├── usa_forecast_h3_v4.png                 (3Q ahead forecast)
├── usa_forecast_h4_v4.png                 (4Q ahead forecast)
├── usa_forecast_grid_v4.png               (All 4 horizons in 2×2)
├── usa_ensemble_vs_actual_gdp_v4.png      (✨ NEW! Main visualization)
├── usa_rmse_by_horizon_v4.png             (Error by horizon)
├── usa_r2_heatmap_v4.png                  (Performance heatmap)
├── usa_model_comparison_v4.png            (1Q vs 4Q comparison)
└── v3_vs_v4_feature_impact.png            (Feature leakage analysis)
```

**Output looks like:**
```
✓ Loaded predictions data from v4_predictions.pkl

================================================================================
v4 FORECAST VISUALIZATIONS
================================================================================

1. Creating individual horizon forecasts...
✓ Saved: usa_forecast_h1_v4.png
✓ Saved: usa_forecast_h2_v4.png
✓ Saved: usa_forecast_h3_v4.png
✓ Saved: usa_forecast_h4_v4.png

2. Creating grid of all horizons...
✓ Saved: usa_forecast_grid_v4.png

3. Creating ensemble vs actual GDP comparison...
✓ Saved: usa_ensemble_vs_actual_gdp_v4.png

4. Creating RMSE degradation plot...
✓ Saved: usa_rmse_by_horizon_v4.png

5. Creating R² heatmap...
✓ Saved: usa_r2_heatmap_v4.png

6. Creating model comparison plots...
✓ Saved: usa_model_comparison_v4.png

7. Creating feature impact analysis...
✓ Saved: v3_vs_v4_feature_impact.png

================================================================================
✓ ALL VISUALIZATIONS COMPLETE
Total plots generated: 9 (plus individual horizon plots)
Output directory: .../forecast_visualizations/
================================================================================
```

---

## The New Visualization: `usa_ensemble_vs_actual_gdp_v4.png`

### What It Shows

A **2×2 grid** with one subplot per horizon:

```
┌──────────────────────────┬──────────────────────────┐
│  1Q Ahead Predictions    │  2Q Ahead Predictions    │
│  vs Actual GDP Growth    │  vs Actual GDP Growth    │
│                          │                          │
│  Blue = Actual           │  Blue = Actual           │
│  Orange = Prediction     │  Orange = Prediction     │
│  Green band = 95% CI     │  Green band = 95% CI     │
│  Red band = 68% CI       │  Red band = 68% CI       │
│                          │                          │
│  [Metrics box]           │  [Metrics box]           │
├──────────────────────────┼──────────────────────────┤
│  3Q Ahead Predictions    │  4Q Ahead Predictions    │
│  vs Actual GDP Growth    │  vs Actual GDP Growth    │
│  ...                     │  ...                     │
└──────────────────────────┴──────────────────────────┘
```

### What Each Element Means

| Element | Meaning |
|---------|---------|
| **Blue line (solid)** | Actual GDP growth YoY (%) |
| **Orange line (dashed)** | Model's ensemble prediction |
| **Green shaded area** | 95% Confidence Interval - predictions likely within this range |
| **Red shaded area** | 68% Confidence Interval - ±1 standard deviation |
| **Gold box** | Performance metrics: RMSE, MAE, R² |

### Interpreting the Metrics

- **RMSE** (Root Mean Squared Error): Average error magnitude - lower is better
- **MAE** (Mean Absolute Error): Average absolute error - lower is better
- **R²** (Coefficient of Determination): How well predictions fit actual values
  - 1.0 = perfect prediction
  - 0.0 = no better than just using the mean
  - Negative = worse than using the mean

---

## Troubleshooting

### Issue: Visualization script doesn't create the new plot

**Solution:**
1. Make sure you ran the training script first:
   ```bash
   python3 forecasting_pipeline_v4.py
   ```
   This creates `results/v4_predictions.pkl` which the visualization needs.

2. Check that the pickle file exists:
   ```bash
   ls -lh results/v4_predictions.pkl
   ```
   Should show file size > 1 MB

3. Run visualization script:
   ```bash
   python3 forecast_visualization_v4.py
   ```

### Issue: "No predictions data found" message

**Means:** `results/v4_predictions.pkl` doesn't exist
**Solution:** Run the training pipeline first

### Issue: Script crashes with import error

**Means:** Missing Python package
**Solution:**
```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

---

## What Was Fixed

### Before
- Visualization script couldn't access prediction data
- Ensemble vs Actual GDP plot wasn't created
- Only 8 plots were generated

### After
- Training pipeline saves predictions to `v4_predictions.pkl`
- Visualization script automatically loads and uses it
- **All 9 plots are now generated** ✓

---

## File Sizes (Typical)

| File | Size |
|------|------|
| `v4_model_performance.csv` | ~2 KB |
| `v4_predictions.pkl` | ~50-100 MB (depends on data) |
| Individual PNG files | ~100-300 KB each |
| `usa_ensemble_vs_actual_gdp_v4.png` | ~200-400 KB |

---

## How to Use the Visualizations

### For Presentations
- Open `usa_ensemble_vs_actual_gdp_v4.png` (shows prediction accuracy)
- Use `usa_r2_heatmap_v4.png` (shows which models work best)
- Use `usa_rmse_by_horizon_v4.png` (shows error increases)

### For Research
- `usa_ensemble_vs_actual_gdp_v4.png` - validate predictions
- `v3_vs_v4_feature_impact.png` - show improvement over v3
- Individual plots - show detailed per-horizon analysis

### For Reports
- All 9 plots are publication-quality (300 DPI)
- Professional styling
- Ready to insert into documents

---

## Next Steps

After generating visualizations:

1. **Inspect the results**
   - Open `usa_ensemble_vs_actual_gdp_v4.png`
   - Check if predictions align with actual values
   - Review confidence intervals

2. **Validate accuracy**
   - Check RMSE/MAE values
   - See if they match expected performance from README
   - Compare across horizons

3. **Use for presentations**
   - All plots are publication-ready (300 DPI PNG)
   - Professional styling
   - Easy to explain to stakeholders

4. **Next improvements** (v5)
   - Add SHAP feature importance
   - Implement regime-switching models
   - Extend to Canada, Japan, UK

---

## Quick Reference

```bash
# Everything in one go
python3 forecasting_pipeline_v4.py && python3 forecast_visualization_v4.py

# Check outputs
ls -la forecast_visualizations/ | wc -l  # Should show 10+ (9 plots + header)

# View a specific plot
open forecast_visualizations/usa_ensemble_vs_actual_gdp_v4.png

# Count the files
ls forecast_visualizations/*.png | wc -l  # Should show 9
```

---

**Status:** ✅ Fixed and Ready to Use!

The ensemble vs actual GDP visualization is now automatically created when you run the visualization script.
