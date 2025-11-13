"""
CANADA DATA EXTENSION - Add Bloomberg Financial Features
=========================================================
V3: Using Bloomberg Corporate Bond Index + Government Bonds + Historical Data

Input:
- ../../../Data/canada_data.csv (OECD base data 2000-2025)
- ../../../Data/historical/canada_historical_1980-2000.csv (Historical 1980-2000)
- ../../../Data/bloomberg/Canada Corporate.xlsx
- ../../../Data/bloomberg/Canada Gov 2Y.xlsx
- ../../../Data/bloomberg/Canada Gov 5Y.xlsx
- ../../../Data/bloomberg/Canada Gov 10Y.xlsx

Output:
- ../../../Data/canada_extended_with_bloomberg.csv (Full 1980-2025)
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FRED_API_KEY = '3477091a9ab79fb20b9ac8aca531d2dd'
START_DATE = '1980-01-01'
END_DATE = '2024-12-31'

# FRED series (keep existing ones that work)
FRED_SERIES = {
    'consumer_sentiment': 'CSCICP03CAM665S',
    'employment_level': 'LFEMTTTTCAM647S',
    'building_permits': 'CANODCNPI03GPSAM',
}

# File paths
DATA_DIR = Path('../../../Data')
INPUT_CURRENT = DATA_DIR / 'canada_data.csv'
INPUT_HISTORICAL = DATA_DIR / 'historical' / 'canada_historical_1980-2000.csv'
BLOOMBERG_DIR = DATA_DIR / 'bloomberg'
OUTPUT_PATH = DATA_DIR / 'canada_extended_with_bloomberg.csv'

# Bloomberg files
BLOOMBERG_FILES = {
    'corporate': BLOOMBERG_DIR / 'Canada Corporate.xlsx',
    'gov_2y': BLOOMBERG_DIR / 'Canada Gov 2Y.xlsx',
    'gov_5y': BLOOMBERG_DIR / 'Canada Gov 5Y.xlsx',
    'gov_10y': BLOOMBERG_DIR / 'Canada Gov 10Y.xlsx',
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_and_merge_base_data():
    """Load and merge historical + current Canada data"""
    print("="*80)
    print("STEP 1: Loading & Merging Base Data")
    print("="*80 + "\n")
    
    # Load current data
    print(f"Loading current: {INPUT_CURRENT}")
    if not INPUT_CURRENT.exists():
        print(f"❌ File not found: {INPUT_CURRENT}")
        return pd.DataFrame()
    
    df_current = pd.read_csv(INPUT_CURRENT, index_col=0, parse_dates=True)
    print(f"  ✓ Shape: {df_current.shape}")
    print(f"    Range: {df_current.index[0].date()} to {df_current.index[-1].date()}")
    
    # Load historical data
    print(f"\nLoading historical: {INPUT_HISTORICAL}")
    if not INPUT_HISTORICAL.exists():
        print(f"⚠️  File not found: {INPUT_HISTORICAL}")
        print(f"  Continuing with only current data...")
        return df_current
    
    df_historical = pd.read_csv(INPUT_HISTORICAL, index_col=0, parse_dates=True)
    print(f"  ✓ Shape: {df_historical.shape}")
    print(f"    Range: {df_historical.index[0].date()} to {df_historical.index[-1].date()}")
    
    # Merge (historical first, then current - current overwrites on duplicates)
    print("\nMerging historical + current...")
    merged = pd.concat([df_historical, df_current], axis=0)
    merged = merged[~merged.index.duplicated(keep='last')].sort_index()
    
    print(f"  ✓ Merged shape: {merged.shape}")
    print(f"    Range: {merged.index[0].date()} to {merged.index[-1].date()}")
    print(f"    Columns: {merged.shape[1]}")
    
    return merged

def load_bloomberg_file(filepath, series_name):
    """Load a Bloomberg Excel export and return clean Series"""
    print(f"  Loading {filepath.name}...", end=' ')
    
    if not filepath.exists():
        print(f"✗ File not found!")
        return None
    
    try:
        # Read with header at row 4 (Bloomberg format)
        df = pd.read_excel(filepath, header=4)
        
        # Clean data
        df = df[df.iloc[:, 0] != 'Date']  # Remove header duplicates
        df = df.dropna(subset=[df.columns[0]])
        df.columns = ['Date', 'Value']
        
        # Convert types
        df['Date'] = pd.to_datetime(df['Date'])
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # Create series
        series = pd.Series(df['Value'].values, index=df['Date'], name=series_name)
        series = series.sort_index()
        series = series[~series.index.duplicated(keep='last')]
        
        print(f"✓ {len(series)} obs ({series.index.min().year}-{series.index.max().year})")
        return series
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def load_all_bloomberg():
    """Load all Bloomberg bond data"""
    print("\n" + "="*80)
    print("STEP 2: Loading Bloomberg Bond Data")
    print("="*80 + "\n")
    
    bloomberg_data = {}
    
    for key, filepath in BLOOMBERG_FILES.items():
        series = load_bloomberg_file(filepath, key)
        if series is not None:
            bloomberg_data[key] = series
    
    if not bloomberg_data:
        print("\n❌ No Bloomberg data loaded!")
        return pd.DataFrame()
    
    # Combine into DataFrame
    df = pd.DataFrame(bloomberg_data)
    print(f"\n✓ Loaded {len(bloomberg_data)} Bloomberg series")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df

def download_fred():
    """Download supplementary FRED data"""
    print("\n" + "="*80)
    print("STEP 3: Loading FRED Supplementary Data")
    print("="*80 + "\n")
    
    fred = Fred(api_key=FRED_API_KEY)
    fred_data = {}
    
    for name, code in FRED_SERIES.items():
        try:
            print(f"  {name} ({code})...", end=' ')
            data = fred.get_series(code, observation_start=START_DATE)
            if len(data) > 0:
                print(f"✓ {len(data)} obs")
                fred_data[name] = data
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ {str(e)[:50]}")
    
    if fred_data:
        df = pd.DataFrame(fred_data)
        print(f"\n✓ Downloaded {len(fred_data)} FRED series")
        return df
    else:
        print("\n⚠️  No FRED data downloaded")
        return pd.DataFrame()

def calculate_financial_indicators(bloomberg_df):
    """Calculate credit spread and yield curve features"""
    print("\n" + "="*80)
    print("STEP 4: Calculating Financial Indicators")
    print("="*80 + "\n")
    
    # FIRST: Resample Bloomberg data to quarterly
    print("Resampling Bloomberg data to quarterly...")
    bloomberg_quarterly = bloomberg_df.resample('Q').last()
    bloomberg_quarterly.index = bloomberg_quarterly.index.to_period('Q').to_timestamp(how='end')
    print(f"  ✓ {len(bloomberg_df)} monthly → {len(bloomberg_quarterly)} quarterly")
    
    indicators = pd.DataFrame(index=bloomberg_quarterly.index)
    calculated = []
    
    # Credit Spread (Corporate - Government 10Y)
    if 'corporate' in bloomberg_quarterly.columns and 'gov_10y' in bloomberg_quarterly.columns:
        indicators['credit_spread'] = bloomberg_quarterly['corporate'] - bloomberg_quarterly['gov_10y']
        n_valid = indicators['credit_spread'].notna().sum()
        calculated.append(f"credit_spread (corporate - 10Y): {n_valid} obs")
    
    # Yield Curve Slope (10Y - 2Y)
    if 'gov_10y' in bloomberg_quarterly.columns and 'gov_2y' in bloomberg_quarterly.columns:
        indicators['yield_curve_slope'] = bloomberg_quarterly['gov_10y'] - bloomberg_quarterly['gov_2y']
        n_valid = indicators['yield_curve_slope'].notna().sum()
        calculated.append(f"yield_curve_slope (10Y - 2Y): {n_valid} obs")
    
    # Yield Curve Curvature (10Y - 2*5Y + 2Y)
    if all(col in bloomberg_quarterly.columns for col in ['gov_10y', 'gov_5y', 'gov_2y']):
        indicators['yield_curve_curvature'] = (
            bloomberg_quarterly['gov_10y'] - 
            2 * bloomberg_quarterly['gov_5y'] + 
            bloomberg_quarterly['gov_2y']
        )
        n_valid = indicators['yield_curve_curvature'].notna().sum()
        calculated.append(f"yield_curve_curvature (10Y - 2*5Y + 2Y): {n_valid} obs")
    
    # Credit Spread Change
    if 'credit_spread' in indicators.columns:
        indicators['credit_spread_change'] = indicators['credit_spread'].diff()
        n_valid = indicators['credit_spread_change'].notna().sum()
        calculated.append(f"credit_spread_change (diff): {n_valid} obs")
    
    print(f"\n✓ Calculated {len(calculated)} indicators:")
    for calc in calculated:
        print(f"  - {calc}")
    
    # Also keep raw bond yields
    for col in ['gov_2y', 'gov_5y', 'gov_10y', 'corporate']:
        if col in bloomberg_quarterly.columns:
            indicators[f'bond_{col}'] = bloomberg_quarterly[col]
    
    return indicators

def merge_all_data(base, bloomberg_indicators, fred):
    """Merge all data sources"""
    print("\n" + "="*80)
    print("STEP 5: Merging All Data")
    print("="*80 + "\n")
    
    print(f"Base data (historical + current): {base.shape}")
    print(f"Bloomberg indicators: {bloomberg_indicators.shape}")
    print(f"FRED data: {fred.shape if not fred.empty else 'Empty'}")
    
    # Start with base
    combined = base.copy()
    
    # Add Bloomberg financial indicators
    if not bloomberg_indicators.empty:
        for col in bloomberg_indicators.columns:
            combined[col] = bloomberg_indicators[col]
            n_valid = bloomberg_indicators[col].notna().sum()
            date_range = f"{bloomberg_indicators[col].first_valid_index().year if bloomberg_indicators[col].notna().any() else 'N/A'}-{bloomberg_indicators[col].last_valid_index().year if bloomberg_indicators[col].notna().any() else 'N/A'}"
            print(f"  + {col}: {n_valid} values ({date_range})")
    
    # Add FRED data
    if not fred.empty:
        for col in fred.columns:
            if col not in combined.columns:
                combined[col] = fred[col]
                n_valid = fred[col].notna().sum()
                print(f"  + {col}: {n_valid} values")
            else:
                # Fill gaps in existing columns
                before = combined[col].notna().sum()
                combined[col] = combined[col].fillna(fred[col])
                after = combined[col].notna().sum()
                if after > before:
                    print(f"  + {col}: filled {after - before} gaps")
    
    print(f"\n✓ Combined shape: {combined.shape}")
    print(f"  Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    
    return combined

def resample_to_quarterly(df):
    """Resample base data to quarterly frequency"""
    print("\n" + "="*80)
    print("STEP 6: Resampling Base Data to Quarterly")
    print("="*80 + "\n")
    
    # Resample to quarter-end
    quarterly = df.resample('Q').last()
    quarterly.index = quarterly.index.to_period('Q').to_timestamp(how='end')
    
    print(f"✓ Resampled: {len(df)} rows → {len(quarterly)} quarters")
    start_q = quarterly.index.to_period('Q')[0]
    end_q = quarterly.index.to_period('Q')[-1]
    print(f"  Range: {start_q} to {end_q}")
    
    return quarterly

def save_output(df):
    """Save final dataset"""
    print("\n" + "="*80)
    print("STEP 7: Saving Output")
    print("="*80 + "\n")
    
    # Sort columns alphabetically
    df = df[sorted(df.columns)]
    
    # Save
    df.to_csv(OUTPUT_PATH)
    
    print(f"✓ SAVED: {OUTPUT_PATH}")
    print(f"  Shape: {df.shape}")
    print(f"  Quarters: {len(df)}")
    print(f"  Features: {df.shape[1]}")
    
    # Show data quality
    print("\n📊 Data Quality Summary:")
    print(f"\nDate range: {df.index[0].date()} to {df.index[-1].date()}")
    
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    print("\nColumns with most missing data (top 10):")
    for col, pct in missing_pct.head(10).items():
        print(f"  {col}: {pct:.1f}% missing")
    
    print("\n✅ Key financial indicators coverage:")
    for col in ['credit_spread', 'yield_curve_slope', 'yield_curve_curvature', 'credit_spread_change']:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct_valid = n_valid / len(df) * 100
            first_valid = df[col].first_valid_index()
            last_valid = df[col].last_valid_index()
            print(f"  {col}:")
            print(f"    {n_valid}/{len(df)} quarters ({pct_valid:.1f}%)")
            if first_valid and last_valid:
                print(f"    Range: {first_valid.date()} to {last_valid.date()}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print(" CANADA DATA EXTENSION - BLOOMBERG FINANCIAL INDICATORS")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nMerging THREE data sources:")
    print("  1. Historical data (1980-2000)")
    print("  2. Current OECD data (2000-2025)")
    print("  3. Bloomberg bond data (1989-2025)")
    print("\nFinancial indicators from Bloomberg:")
    print("  ✓ Canada Corporate Bond Index (I05510CA)")
    print("  ✓ Canada Government Bonds (2Y, 5Y, 10Y)")
    print("="*80 + "\n")
    
    # Load and merge base data (historical + current)
    base = load_and_merge_base_data()
    if base.empty:
        print("\n❌ ERROR: No base data loaded! Cannot proceed.")
        return
    
    # ⚡ RESAMPLE BASE TO QUARTERLY FIRST!
    base_quarterly = resample_to_quarterly(base)
    
    # Load Bloomberg data
    bloomberg_df = load_all_bloomberg()
    if bloomberg_df.empty:
        print("\n❌ ERROR: No Bloomberg data loaded! Cannot proceed.")
        print("\nCheck that these files exist:")
        for filepath in BLOOMBERG_FILES.values():
            exists = "✓" if filepath.exists() else "✗"
            print(f"  {exists} {filepath}")
        return
    
    # Download FRED supplements
    fred = download_fred()
    if not fred.empty:
        # Resample FRED to quarterly too
        print("\n" + "="*80)
        print("Resampling FRED data to quarterly...")
        print("="*80 + "\n")
        fred_quarterly = fred.resample('Q').last()
        fred_quarterly.index = fred_quarterly.index.to_period('Q').to_timestamp(how='end')
        print(f"  ✓ {len(fred)} → {len(fred_quarterly)} quarters")
    else:
        fred_quarterly = pd.DataFrame()
    
    # Calculate financial indicators (already resamples Bloomberg to quarterly)
    bloomberg_indicators = calculate_financial_indicators(bloomberg_df)
    
    # Merge everything (NOW all are quarterly!)
    combined = merge_all_data(base_quarterly, bloomberg_indicators, fred_quarterly)
    
    # Save (no more resampling needed!)
    save_output(combined)
    
    print("\n" + "="*80)
    print(" ✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput: {OUTPUT_PATH}")
    print("\n📊 Next Steps:")
    print("  1. Review output: Open CSV and check coverage")
    print("  2. Run preprocessing: cd ../../ && python preprocessing_pipeline.py")
    print("  3. Test model: cd ../../models/nowcasting_v7 && python nowcasting_pipeline.py")
    print("\n💡 Expected data coverage:")
    print("  1980-1989: Historical data only (no Bloomberg corporate yet)")
    print("  1989-2002: Historical + Gov bonds (corporate starts 2002)")
    print("  2002-2025: Full coverage (all indicators)")
    print("\n🎯 Model will use complete periods for training/testing")
    print("="*80)
    
if __name__ == '__main__':
    main()

