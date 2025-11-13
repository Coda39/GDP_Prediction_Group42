# Nowcasting Models

## Overview

GDP nowcasting models organized **by country**. Each country's version numbering starts from their baseline (v0).

**Structure:**
- `current/` - Active working version (EDITABLE)
- `archive/v0/` - Pristine baseline (NEVER EDIT - restore point)
- `archive/v1+/` - Future improved versions

**Purpose:** Predict **current quarter** GDP growth using available leading indicators.

---

## ⚠️ IMPORTANT: File Paths Need Manual Update

After running the reorganization script, you MUST update file paths in each country's `current/nowcasting_pipeline.py`:

### Required Path Changes:
```python
# OLD (in nowcasting_v7):
DATA_DIR = Path(__file__).parent.parent.parent / 'data_preprocessing' / 'outputs' / 'processed_data'
OUTPUT_DIR = Path(__file__).parent / 'results'
MODEL_DIR = Path(__file__).parent / 'saved_models'
FIG_DIR = Path(__file__).parent / 'figures'

# NEW (in country/current/):
DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data_preprocessing' / 'outputs' / 'processed_data'
OUTPUT_DIR = Path(__file__).parent / 'outputs' / 'results'
MODEL_DIR = Path(__file__).parent / 'outputs' / 'saved_models'
FIG_DIR = Path(__file__).parent / 'outputs' / 'figures'
```

**Changes:**
1. `parent.parent.parent` → `parent.parent.parent.parent` (one extra level)
2. Output folders now in `outputs/` subdirectory

### Update Process:

1. **Test with Canada first:**
```bash
   cd nowcasting/canada/current
   # Edit nowcasting_pipeline.py with new paths (lines 77-80)
   python nowcasting_pipeline.py  # Test!
```

2. **If Canada works, copy changes to other countries:**
   - UK, Japan, Germany, France, Italy, USA

3. **DO NOT edit `archive/v0/`** - keep pristine!

---

## Folder Structure
```
nowcasting/
├── canada/
│   ├── current/              # EDITABLE v0 baseline
│   └── archive/
│       └── v0/               # PRISTINE v0 baseline (restore point)
├── uk/
│   ├── current/
│   └── archive/
│       └── v0/
├── japan/
│   ├── current/
│   └── archive/
│       └── v0/
├── germany/
│   ├── current/
│   └── archive/
│       └── v0/
├── france/
│   ├── current/
│   └── archive/
│       └── v0/
├── italy/
│   ├── current/
│   └── archive/
│       └── v0/
└── usa/
    ├── current/              # v7 baseline (not updating now)
    └── archive/              # Historical versions
        ├── v0_original/
        ├── v1/
        ├── v2/
        ├── v3/
        ├── v4/
        ├── v5/
        └── v6/
        (No v7 - it's current!)
```

---

## Version Numbering Logic

### Non-USA Countries (Canada, UK, Japan, Germany, France, Italy):
- **v0** = Their baseline (global v7 features)
- **v1+** = Country-specific improvements

### USA:
- **v0-v6** = Historical versions (archived)
- **v7** = Current baseline (in `current/`)
- Not actively updating USA right now

**Why different?** Each country's version numbering reflects THEIR development path, not global version numbers. USA has more history because it was the original test country.

---

## Current Baseline (All Countries Start Here)

**Training Data:** 1980-2024 (180 quarters)  
**Test Period:** 2022-2025  
**Features:** ~80 engineered features  
**Models:** Linear, Ridge, LASSO, RF, XGBoost, GB, Stacking

### Baseline Results

| Country | Best Model | Test R² | Version |
|---------|------------|---------|---------|
| UK | XGBoost | 0.482 | v0 (current) |
| Italy | Stacking | 0.072 | v0 (current) |
| Canada | XGBoost | 0.054 | v0 (current) |
| Japan | Linear Regression | -0.015 | v0 (current) |
| France | Gradient Boosting | -0.133 | v0 (current) |
| Germany | XGBoost | -0.487 | v0 (current) |

**Priority optimization:** Germany, France, Japan

---

## Workflow

### Making Changes to a Country:

1. **Work in `current/`** (never edit `archive/v0/`)
```bash
   cd canada/current
   # Edit nowcasting_pipeline.py
   python nowcasting_pipeline.py
```

2. **If changes break something, restore:**
```bash
   cp ../archive/v0/nowcasting_pipeline.py ./nowcasting_pipeline.py
```

### Creating New Version (When You Improve):

When you get better results and want to archive:
```bash
# From models/nowcasting/canada/
# Archive current as v1
cp -r current archive/v1

# Now continue editing current/
cd current
# Make more improvements...

# Later, when satisfied, archive as v2
cp -r current ../archive/v2
```

### Archive Rules:
- **v0 is sacred** - Never edit, never delete (restore point)
- **v1+** - Your improvement versions
- **current/** - Always the active working version

---

## Usage

### Running Models:
```bash
# Canada
cd canada/current
python nowcasting_pipeline.py

# Germany (needs optimization)
cd germany/current
python nowcasting_pipeline.py
```

### Loading Saved Models:
```python
import joblib

# Load Canada v0 baseline model
model = joblib.load('canada/current/outputs/saved_models/canada_nowcast_v7_xgboost.pkl')

# Make prediction
prediction = model.predict(X_new)
```

---

## Country-Specific Notes

### 🇺🇸 USA (Not Updating)
- Current version: v7 baseline
- Has complete history (v0-v6 archived)
- Not actively optimizing right now

### 🇨🇦 Canada (Active)
- Starting: v0 baseline (R² = 0.054)
- Target: v1 optimization
- Good baseline - room for improvement

### 🇬🇧 UK (Active)
- **Best performer** (R² = 0.482)
- Starting: v0 baseline
- Use as reference for other countries

### 🇯🇵 Japan (Active - Priority)
- Starting: v0 baseline (R² = -0.015)
- Target: Japan-specific features
- Near-zero, needs work

### 🇩🇪 Germany (Active - Priority #1)
- Starting: v0 baseline (R² = -0.487)
- **Needs urgent work**
- Missing industrial production

### 🇫🇷 France (Active - Priority)
- Starting: v0 baseline (R² = -0.133)
- Target: Eurozone features
- Negative R²

### 🇮🇹 Italy (Active)
- Starting: v0 baseline (R² = 0.072)
- Slight positive, can improve
- Stacking works best

---

## Optimization Strategy

### Phase 1 - Fix Negatives:
1. **Germany** (R² = -0.487) - Add industrial production
2. **France** (R² = -0.133) - Eurozone indicators
3. **Japan** (R² = -0.015) - Japan-specific features

### Phase 2 - Improve Positives:
4. **Canada** (R² = 0.054) - Fine-tune hyperparameters
5. **Italy** (R² = 0.072) - Feature selection

### Phase 3 - Maintain Best:
6. **UK** (R² = 0.482) - Document best practices

---

## Data Location

**Preprocessed data:** `../../data_preprocessing/outputs/processed_data/`

**Files per country:**
- `{country}_processed_normalized.csv`
- `{country}_processed_unnormalized.csv`
- `{country}_normalization_stats.csv`

---

**Last Updated:** November 13, 2025  
**Organization:** By Country  
**Version Scheme:** Country-specific (v0 = baseline)  
**Status:** ✅ Ready for optimization (update paths first!)
