"""
Reorganize nowcasting models by COUNTRY
- Non-USA countries: archive/v0/ (their baseline)
- USA: archive/ has v0-v6 only (v7 is current)
Paths need manual update after running!
"""

from pathlib import Path
import shutil

MODELS_DIR = Path('.')
NOWCASTING_DIR = MODELS_DIR / 'nowcasting'

# Non-USA countries get v0 baseline in archive
NON_USA_COUNTRIES = ['canada', 'uk', 'japan', 'germany', 'france', 'italy']

print("=" * 80)
print("REORGANIZING NOWCASTING MODELS BY COUNTRY")
print("=" * 80)
print("\n⚠️  NOTE: This script DOES NOT update file paths!")
print("⚠️  You will manually update paths in each country's current/nowcasting_pipeline.py")
print("=" * 80)

# Create country folders for non-USA countries
print("\n[1/4] Creating folder structure for Canada, UK, Japan, Germany, France, Italy...")
for country in NON_USA_COUNTRIES:
    country_dir = NOWCASTING_DIR / country
    # Create current/
    (country_dir / 'current' / 'outputs' / 'figures').mkdir(parents=True, exist_ok=True)
    (country_dir / 'current' / 'outputs' / 'results').mkdir(parents=True, exist_ok=True)
    (country_dir / 'current' / 'outputs' / 'saved_models').mkdir(parents=True, exist_ok=True)
    # Create archive/v0/ (their baseline)
    (country_dir / 'archive' / 'v0' / 'outputs' / 'figures').mkdir(parents=True, exist_ok=True)
    (country_dir / 'archive' / 'v0' / 'outputs' / 'results').mkdir(parents=True, exist_ok=True)
    (country_dir / 'archive' / 'v0' / 'outputs' / 'saved_models').mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created {country}/ (current + archive/v0)")

# Create USA folder structure
print("\n[2/4] Creating USA folder structure...")
usa_dir = NOWCASTING_DIR / 'usa'
(usa_dir / 'current' / 'outputs' / 'figures').mkdir(parents=True, exist_ok=True)
(usa_dir / 'current' / 'outputs' / 'results').mkdir(parents=True, exist_ok=True)
(usa_dir / 'current' / 'outputs' / 'saved_models').mkdir(parents=True, exist_ok=True)
(usa_dir / 'archive').mkdir(exist_ok=True)
print(f"  ✓ Created usa/ (current + archive)")

# Extract country files from v7
print("\n[3/4] Extracting country files from v7...")
v7_dir = MODELS_DIR / 'nowcasting_v7'

if v7_dir.exists():
    # Non-USA countries: copy to both current/ and archive/v0/
    for country in NON_USA_COUNTRIES:
        print(f"\n  Processing {country}...")
        
        # Copy to BOTH current/ and archive/v0/
        for dest_type in ['current', 'archive/v0']:
            dest_dir = NOWCASTING_DIR / country / dest_type
            
            # Copy pipeline script
            if (v7_dir / 'nowcasting_pipeline.py').exists():
                shutil.copy2(v7_dir / 'nowcasting_pipeline.py',
                            dest_dir / 'nowcasting_pipeline.py')
            
            # Extract country-specific figures
            if (v7_dir / 'figures').exists():
                for fig_file in (v7_dir / 'figures').glob(f'{country}_*'):
                    shutil.copy2(fig_file, dest_dir / 'outputs' / 'figures' / fig_file.name)
            
            # Extract country-specific results
            if (v7_dir / 'results').exists():
                for res_file in (v7_dir / 'results').glob(f'{country}_*'):
                    shutil.copy2(res_file, dest_dir / 'outputs' / 'results' / res_file.name)
                
                # Copy all_countries results
                if (v7_dir / 'results' / 'all_countries_nowcast_V7_results.csv').exists():
                    shutil.copy2(v7_dir / 'results' / 'all_countries_nowcast_V7_results.csv',
                                dest_dir / 'outputs' / 'results' / 'all_countries_nowcast_V7_results.csv')
            
            # Extract country-specific saved models
            if (v7_dir / 'saved_models').exists():
                for model_file in (v7_dir / 'saved_models').glob(f'{country}_*'):
                    shutil.copy2(model_file, dest_dir / 'outputs' / 'saved_models' / model_file.name)
        
        print(f"    ✓ Copied to current/ (editable)")
        print(f"    ✓ Copied to archive/v0/ (pristine baseline)")
    
    # USA: copy to current/ only (not to archive - v7 is current)
    print(f"\n  Processing usa...")
    dest_dir = usa_dir / 'current'
    
    # Copy pipeline script
    if (v7_dir / 'nowcasting_pipeline.py').exists():
        shutil.copy2(v7_dir / 'nowcasting_pipeline.py',
                    dest_dir / 'nowcasting_pipeline.py')
    
    # Extract USA-specific figures
    if (v7_dir / 'figures').exists():
        for fig_file in (v7_dir / 'figures').glob('usa_*'):
            shutil.copy2(fig_file, dest_dir / 'outputs' / 'figures' / fig_file.name)
    
    # Extract USA-specific results
    if (v7_dir / 'results').exists():
        for res_file in (v7_dir / 'results').glob('usa_*'):
            shutil.copy2(res_file, dest_dir / 'outputs' / 'results' / res_file.name)
        
        # Copy all_countries results
        if (v7_dir / 'results' / 'all_countries_nowcast_V7_results.csv').exists():
            shutil.copy2(v7_dir / 'results' / 'all_countries_nowcast_V7_results.csv',
                        dest_dir / 'outputs' / 'results' / 'all_countries_nowcast_V7_results.csv')
    
    # Extract USA-specific saved models
    if (v7_dir / 'saved_models').exists():
        for model_file in (v7_dir / 'saved_models').glob('usa_*'):
            shutil.copy2(model_file, dest_dir / 'outputs' / 'saved_models' / model_file.name)
    
    print(f"    ✓ Copied to current/ (v7 baseline - not updating now)")

# Archive USA old versions (v0-v6 only, NOT v7)
print("\n[4/4] Archiving USA old versions (v0-v6)...")
usa_archive = usa_dir / 'archive'

old_usa_versions = {
    'nowcasting': 'v0_original',
    'nowcasting_v1': 'v1',
    'nowcasting_v2': 'v2',
    'nowcasting_v3': 'v3',
    'nowcasting_v4': 'v4',
    'nowcasting_v5': 'v5',
    'nowcasting_v6': 'v6'
}

for old_dir, new_name in old_usa_versions.items():
    src = MODELS_DIR / old_dir
    if src.exists():
        dst = usa_archive / new_name
        dst.mkdir(exist_ok=True)
        
        # Copy pipeline
        if (src / 'nowcasting_pipeline.py').exists():
            shutil.copy2(src / 'nowcasting_pipeline.py', dst / 'nowcasting_pipeline.py')
        
        # Copy USA figures
        if (src / 'figures').exists():
            (dst / 'figures').mkdir(exist_ok=True)
            for fig_file in (src / 'figures').glob('usa_*'):
                shutil.copy2(fig_file, dst / 'figures' / fig_file.name)
        
        # Copy USA results
        if (src / 'results').exists():
            (dst / 'results').mkdir(exist_ok=True)
            for res_file in (src / 'results').glob('usa_*'):
                shutil.copy2(res_file, dst / 'results' / res_file.name)
        
        # Copy USA models
        if (src / 'saved_models').exists():
            (dst / 'saved_models').mkdir(exist_ok=True)
            for model_file in (src / 'saved_models').glob('usa_*'):
                shutil.copy2(model_file, dst / 'saved_models' / model_file.name)
        
        print(f"  ✓ Archived {old_dir} → usa/archive/{new_name}/")

# Create README
print("\n[5/5] Creating README.md...")

readme_content = """# Nowcasting Models

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
"""

with open(NOWCASTING_DIR / 'README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
print("  ✓ Created README.md")

# Delete old version folders
print("\n[6/6] Cleaning up old folders...")
old_folders = ['nowcasting', 'nowcasting_v1', 'nowcasting_v2', 'nowcasting_v3',
               'nowcasting_v4', 'nowcasting_v5', 'nowcasting_v6', 'nowcasting_v7']

for folder in old_folders:
    folder_path = MODELS_DIR / folder
    if folder_path.exists():
        shutil.rmtree(folder_path)
        print(f"  ✓ Removed {folder}/")

# Summary
print("\n" + "=" * 80)
print("✅ REORGANIZATION COMPLETE!")
print("=" * 80)

print("\n📁 Structure created:")
print("\nCanada, UK, Japan, Germany, France, Italy:")
print("  current/           (EDITABLE v0 baseline)")
print("  archive/v0/        (PRISTINE v0 baseline)")

print("\nUSA:")
print("  current/           (v7 baseline - not updating)")
print("  archive/v0-v6/     (historical versions)")

print("\n" + "=" * 80)
print("⚠️  NEXT STEPS - UPDATE FILE PATHS:")
print("=" * 80)
print("\n1. Test with Canada first:")
print("   cd canada/current")
print("   # Edit nowcasting_pipeline.py lines 77-80")
print("   python nowcasting_pipeline.py")
print("\n2. Apply same changes to: UK, Japan, Germany, France, Italy, USA")
print("\n✅ Then ready for country-specific optimization!")