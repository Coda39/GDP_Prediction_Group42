"""
Merge Historical Data with Existing Data
=========================================
This script merges the downloaded 1980-2000 historical data with 
the existing 2001-2025 data to create extended datasets.

Usage:
    python merge_historical_data.py

Output:
    - raw_data/{country}_raw_extended.csv (1980-2025 combined)
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths adjusted for running from data_preprocessing/data_merging/
DATA_DIR = Path(__file__).parent.parent.parent / 'Data'
HISTORICAL_DIR = DATA_DIR / 'historical'
OUTPUT_SUFFIX = '_extended'

# Country file mappings (how they're named in Data/)
# Focus on USA first - add others after verifying this works
COUNTRY_FILES = {
    'usa': 'united_states_fred_data.csv',
    # Add others after USA works:
    # 'canada': 'canada_data.csv',
    # 'japan': 'japan_data.csv',
    # 'uk': 'united_kingdom_data.csv',
}

# ============================================================================
# MERGE FUNCTIONS
# ============================================================================

def load_and_check_data(file_path):
    """Load CSV and display basic info"""
    if not file_path.exists():
        print(f"    ✗ File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"    ✓ Loaded: {len(df)} rows, {df.shape[1]} columns")
    print(f"      Date range: {df.index[0]} to {df.index[-1]}")
    return df

def merge_historical_and_current(historical_df, current_df, country):
    """
    Merge historical (1980-2000) and current (2001-2025) data
    
    Parameters:
    - historical_df: DataFrame with 1980-2000 data
    - current_df: DataFrame with 2001-2025 data
    - country: Country name for logging
    
    Returns: Merged DataFrame
    """
    print(f"  Merging data...")
    
    # Check for overlapping columns
    common_cols = set(historical_df.columns) & set(current_df.columns)
    print(f"    Common columns: {len(common_cols)}")
    
    # Check for date overlap (should be none)
    date_overlap = set(historical_df.index) & set(current_df.index)
    if date_overlap:
        print(f"    ⚠ Warning: {len(date_overlap)} overlapping dates found!")
        print(f"      Will keep current data for overlapping dates")
    
    # Combine
    # Use outer join to keep all dates, current data takes precedence for overlaps
    merged = pd.concat([historical_df, current_df], axis=0)
    
    # Remove duplicates (keep current data)
    merged = merged[~merged.index.duplicated(keep='last')]
    
    # Sort by date
    merged = merged.sort_index()
    
    print(f"    ✓ Merged shape: {merged.shape}")
    print(f"    ✓ Date range: {merged.index[0]} to {merged.index[-1]}")
    print(f"    ✓ Total quarters: {len(merged)}")
    
    # Check for gaps
    expected_quarters = pd.date_range(start=merged.index[0], end=merged.index[-1], freq='Q')
    missing_quarters = set(expected_quarters) - set(merged.index)
    if missing_quarters:
        print(f"    ⚠ Warning: {len(missing_quarters)} missing quarters in date range")
    else:
        print(f"    ✓ No gaps in quarterly data")
    
    return merged

def check_data_quality(df, country):
    """Check merged data quality"""
    print(f"  Checking data quality...")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"    ⚠ Missing values detected:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"      - {col}: {count} ({pct:.1f}%)")
    else:
        print(f"    ✓ No missing values")
    
    # Data types
    print(f"    Column types:")
    print(f"      - Numeric: {df.select_dtypes(include=['float64', 'int64']).shape[1]}")
    print(f"      - Object: {df.select_dtypes(include=['object']).shape[1]}")
    
    return True

# ============================================================================
# MAIN MERGE PROCESS
# ============================================================================

def merge_country_data(country):
    """Merge historical and current data for one country"""
    print(f"\n{'='*70}")
    print(f"Merging {country.upper()} Data")
    print(f"{'='*70}")
    
    # Load historical data (1980-2000)
    print(f"  Loading historical data (1980-2000)...")
    historical_file = HISTORICAL_DIR / f'{country}_historical_1980-2000.csv'
    historical_df = load_and_check_data(historical_file)
    
    if historical_df is None:
        print(f"  ✗ Skipping {country} - no historical data found")
        return None
    
    # Load current data (2001-2025) using correct filename
    print(f"  Loading current data (2001-2025)...")
    current_filename = COUNTRY_FILES.get(country)
    if current_filename is None:
        print(f"  ✗ Unknown country: {country}")
        return None
        
    current_file = DATA_DIR / current_filename
    current_df = load_and_check_data(current_file)
    
    if current_df is None:
        print(f"  ✗ Skipping {country} - no current data found")
        return None
    
    # Merge
    merged_df = merge_historical_and_current(historical_df, current_df, country)
    
    # Quality check
    check_data_quality(merged_df, country)
    
    # Save with _extended suffix but keep original filename structure
    base_name = current_filename.replace('.csv', '')
    output_file = DATA_DIR / f'{base_name}{OUTPUT_SUFFIX}.csv'
    merged_df.to_csv(output_file, index=True)
    print(f"  ✓ Saved to: {output_file}")
    
    return merged_df

def main():
    """Main merge process"""
    print("="*70)
    print("Merge Historical Data with Current Data")
    print("="*70)
    print(f"\nThis will create extended datasets (1980-2025) by combining:")
    print(f"  - Historical: Data/historical/*_historical_1980-2000.csv")
    print(f"  - Current:    Data/*.csv")
    print(f"  - Output:     Data/*{OUTPUT_SUFFIX}.csv")
    
    # Check directories exist
    if not HISTORICAL_DIR.exists():
        print(f"\n✗ Historical data directory not found: {HISTORICAL_DIR}")
        print(f"  Please run download_historical_data.py first!")
        return
    
    if not DATA_DIR.exists():
        print(f"\n✗ Data directory not found: {DATA_DIR}")
        return
    
    # Merge each country
    results = {}
    for country in COUNTRY_FILES.keys():
        try:
            merged_df = merge_country_data(country)
            if merged_df is not None:
                results[country] = merged_df
        except Exception as e:
            print(f"\n✗ Error merging {country}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("MERGE SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully merged: {len(results)}/{len(COUNTRY_FILES)} countries\n")
    
    for country, df in results.items():
        quarters_1980_2000 = len(df[df.index.year <= 2000])
        quarters_2001_2025 = len(df[df.index.year >= 2001])
        
        print(f"{country.upper():10s}:")
        print(f"  Total quarters: {len(df)}")
        print(f"  - 1980-2000: {quarters_1980_2000} quarters")
        print(f"  - 2001-2025: {quarters_2001_2025} quarters")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print()
    
    print("✓ Extended datasets saved to Data/")
    print("\nNext steps:")
    print("1. Review the merged data")
    print("2. Update preprocessing_pipeline.py to use *_extended.csv")
    print("3. Re-run preprocessing to create new processed files")
    print("4. Re-train models with expanded training data")

if __name__ == '__main__':
    main()