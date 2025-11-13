"""
FRANCE DATA EXTENSION - Historical FRED + Bloomberg
=====================================================
1. Load base France data (2000-2025)
2. Extend with FRED historical data (1980-2000)
3. Add Bloomberg financial indicators (1999-2025)

Output: france_extended_with_bloomberg.csv (1980-2025)
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
START_DATE = '1991-01-01'
END_DATE = '2024-12-31'

# FRED series for France - ALL economic indicators
FRED_SERIES = {
    # GDP
    'gdp_real': 'CLVMNACSCAB1GQFR',  # Real GDP (billions of 2015 euros, quarterly)
    'gdp_nominal': 'CPMNACSCAB1GQFR',  # Nominal GDP (quarterly)
    
    # Labor market
    'unemployment_rate': 'LRHUTTTTFRQ156S',  # Unemployment rate
    'employment_level': 'LMEMTTTTFRQ659S',  # Employment level (thousands)
    
    # Prices
    'cpi_all_items': 'FRACPIALLMINMEI',  # CPI all items (monthly → quarterly)
    
    # Trade
    'exports_volume': 'XTEXVA01FRQ188S',  # Exports volume index
    'imports_volume': 'XTIMVA01FRQ188S',  # Imports volume index
    
    # Production
    'industrial_production_index': 'FRAPROINDMISMEI',  # Industrial production
    
    # Financial
    'interest_rate_short_term': 'IRSTCI01FRM156N',  # Short-term interest rate
    'interest_rate_long_term': 'IRLTLT01FRM156N',  # Long-term interest rate
    'stock_market_index': 'FCHI',  # CAC 40 index
    
    # Other
    'consumer_confidence': 'CSCICP03FRM665S',  # Consumer confidence
}
# File paths
DATA_DIR = Path('../../../Data')
INPUT_CURRENT = DATA_DIR / 'france_data_no_money_supply.csv'
BLOOMBERG_DIR = DATA_DIR / 'bloomberg'
OUTPUT_PATH = DATA_DIR / 'france_extended_with_bloomberg.csv'

# Bloomberg files
BLOOMBERG_FILES = {
    'corporate': BLOOMBERG_DIR / 'Europe Corporate.xlsx',
    'gov_2y': BLOOMBERG_DIR / 'France Gov 2Y.xlsx',
    'gov_5y': BLOOMBERG_DIR / 'France Gov 5Y.xlsx',
    'gov_10y': BLOOMBERG_DIR / 'France Gov 10Y.xlsx',
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def download_fred_historical():
    """Download ALL available FRED data for France (1980-2025)"""
    print("="*80)
    print("STEP 1: Downloading FRED Historical Data (1980-2025)")
    print("="*80 + "\n")
    
    fred = Fred(api_key=FRED_API_KEY)
    fred_data = {}
    
    for name, code in FRED_SERIES.items():
        try:
            print(f"  {name} ({code})...", end=' ')
            data = fred.get_series(code, observation_start=START_DATE, observation_end=END_DATE)
            if len(data) > 0:
                print(f"✓ {len(data)} obs ({data.index.min().year}-{data.index.max().year})")
                fred_data[name] = data
            else:
                print("✗ No data")
        except Exception as e:
            error_msg = str(e)[:60]
            print(f"✗ {error_msg}")
    
    if not fred_data:
        print("\n⚠️  No FRED data downloaded!")
        return pd.DataFrame()
    
    df = pd.DataFrame(fred_data)
    print(f"\n✓ Downloaded {len(fred_data)} FRED series")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df

def load_base_data():
    """Load current France data (2000-2025)"""
    print("\n" + "="*80)
    print("STEP 2: Loading Current OECD Data (2000-2025)")
    print("="*80 + "\n")
    
    print(f"Loading: {INPUT_CURRENT}")
    if not INPUT_CURRENT.exists():
        print(f"❌ File not found: {INPUT_CURRENT}")
        return pd.DataFrame()
    
    df = pd.read_csv(INPUT_CURRENT, index_col=0, parse_dates=True)
    print(f"  ✓ Shape: {df.shape}")
    print(f"    Range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"    Features: {df.shape[1]}")
    
    return df

def merge_fred_and_base(fred_df, base_df):
    """Merge FRED historical with OECD current data"""
    print("\n" + "="*80)
    print("STEP 3: Merging FRED Historical + OECD Current")
    print("="*80 + "\n")
    
    if fred_df.empty:
        print("⚠️  No FRED data to merge, using only base data")
        return base_df
    
    print("Merging strategy:")
    print("  - FRED data: 1980-2000 (historical)")
    print("  - Base data: 2000-2025 (current, priority for overlaps)")
    
    # Simple concat and deduplicate (base wins on conflicts)
    combined = pd.concat([fred_df, base_df], axis=0, sort=True)
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    
    print(f"\n✓ Combined shape: {combined.shape}")
    print(f"  Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    
    # Show coverage for key columns
    print("\nData coverage by period:")
    for col in ['gdp_real', 'unemployment_rate', 'cpi_all_items']:
        if col in combined.columns:
            pre_2000 = combined[combined.index < '2000-01-01'][col].notna().sum()
            post_2000 = combined[combined.index >= '2000-01-01'][col].notna().sum()
            print(f"  {col}: {pre_2000} quarters (1980-2000), {post_2000} quarters (2000-2025)")
    
    return combined

def load_bloomberg_file(filepath, series_name):
    """Load a Bloomberg Excel export"""
    print(f"  Loading {filepath.name}...", end=' ')
    
    if not filepath.exists():
        print(f"✗ File not found!")
        return None
    
    try:
        df = pd.read_excel(filepath, header=4)
        df = df[df.iloc[:, 0] != 'Date']
        df = df.dropna(subset=[df.columns[0]])
        df.columns = ['Date', 'Value']
        df['Date'] = pd.to_datetime(df['Date'])
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        series = pd.Series(df['Value'].values, index=df['Date'], name=series_name)
        series = series.sort_index()
        series = series[~series.index.duplicated(keep='last')]
        
        print(f"✓ {len(series)} obs ({series.index.min().year}-{series.index.max().year})")
        return series
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def load_all_bloomberg():
    """Load Bloomberg bond data"""
    print("\n" + "="*80)
    print("STEP 4: Loading Bloomberg Bond Data")
    print("="*80 + "\n")
    print("NOTE: Euro Corporate index shared across Germany/France/Italy\n")
    
    bloomberg_data = {}
    
    for key, filepath in BLOOMBERG_FILES.items():
        series = load_bloomberg_file(filepath, key)
        if series is not None:
            bloomberg_data[key] = series
    
    if not bloomberg_data:
        print("\n❌ No Bloomberg data loaded!")
        return pd.DataFrame()
    
    df = pd.DataFrame(bloomberg_data)
    print(f"\n✓ Loaded {len(bloomberg_data)} Bloomberg series")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df

def calculate_financial_indicators(bloomberg_df):
    """Calculate credit spread and yield curve features"""
    print("\n" + "="*80)
    print("STEP 5: Calculating Financial Indicators")
    print("="*80 + "\n")
    
    print("Resampling Bloomberg data to quarterly...")
    bloomberg_quarterly = bloomberg_df.resample('Q').last()
    bloomberg_quarterly.index = bloomberg_quarterly.index.to_period('Q').to_timestamp(how='end')
    print(f"  ✓ {len(bloomberg_df)} monthly → {len(bloomberg_quarterly)} quarterly")
    
    indicators = pd.DataFrame(index=bloomberg_quarterly.index)
    calculated = []
    
    # Credit Spread
    if 'corporate' in bloomberg_quarterly.columns and 'gov_10y' in bloomberg_quarterly.columns:
        indicators['credit_spread'] = bloomberg_quarterly['corporate'] - bloomberg_quarterly['gov_10y']
        n_valid = indicators['credit_spread'].notna().sum()
        calculated.append(f"credit_spread: {n_valid} obs")
    
    # Yield Curve Slope
    if 'gov_10y' in bloomberg_quarterly.columns and 'gov_2y' in bloomberg_quarterly.columns:
        indicators['yield_curve_slope'] = bloomberg_quarterly['gov_10y'] - bloomberg_quarterly['gov_2y']
        n_valid = indicators['yield_curve_slope'].notna().sum()
        calculated.append(f"yield_curve_slope: {n_valid} obs")
    
    # Yield Curve Curvature
    if all(col in bloomberg_quarterly.columns for col in ['gov_10y', 'gov_5y', 'gov_2y']):
        indicators['yield_curve_curvature'] = (
            bloomberg_quarterly['gov_10y'] - 
            2 * bloomberg_quarterly['gov_5y'] + 
            bloomberg_quarterly['gov_2y']
        )
        n_valid = indicators['yield_curve_curvature'].notna().sum()
        calculated.append(f"yield_curve_curvature: {n_valid} obs")
    
    # Credit Spread Change
    if 'credit_spread' in indicators.columns:
        indicators['credit_spread_change'] = indicators['credit_spread'].diff()
        n_valid = indicators['credit_spread_change'].notna().sum()
        calculated.append(f"credit_spread_change: {n_valid} obs")
    
    print(f"\n✓ Calculated {len(calculated)} indicators:")
    for calc in calculated:
        print(f"  - {calc}")
    
    # Keep raw yields
    for col in ['gov_2y', 'gov_5y', 'gov_10y', 'corporate']:
        if col in bloomberg_quarterly.columns:
            indicators[f'bond_{col}'] = bloomberg_quarterly[col]
    
    return indicators

def resample_to_quarterly(df):
    """Resample to quarterly"""
    print("\n" + "="*80)
    print("STEP 6: Resampling to Quarterly")
    print("="*80 + "\n")
    
    quarterly = df.resample('Q').last()
    quarterly.index = quarterly.index.to_period('Q').to_timestamp(how='end')
    
    print(f"✓ Resampled: {len(df)} rows → {len(quarterly)} quarters")
    print(f"  Range: {quarterly.index.to_period('Q')[0]} to {quarterly.index.to_period('Q')[-1]}")
    
    return quarterly

def merge_all_data(base, bloomberg_indicators):
    """Merge base + Bloomberg"""
    print("\n" + "="*80)
    print("STEP 7: Adding Bloomberg Indicators")
    print("="*80 + "\n")
    
    combined = base.copy()
    
    if not bloomberg_indicators.empty:
        for col in bloomberg_indicators.columns:
            combined[col] = bloomberg_indicators[col]
            n_valid = bloomberg_indicators[col].notna().sum()
            print(f"  + {col}: {n_valid} values")
    
    print(f"\n✓ Final shape: {combined.shape}")
    return combined

def save_output(df):
    """Save final dataset"""
    print("\n" + "="*80)
    print("STEP 8: Saving Output")
    print("="*80 + "\n")
    
    df = df[sorted(df.columns)]
    df.to_csv(OUTPUT_PATH)
    
    print(f"✓ SAVED: {OUTPUT_PATH}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    print("\n📊 Financial indicators coverage:")
    for col in ['credit_spread', 'yield_curve_slope', 'yield_curve_curvature']:
        if col in df.columns:
            coverage = df[col].notna().sum() / len(df) * 100
            print(f"  {col}: {coverage:.1f}%")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print(" FRANCE DATA EXTENSION - FRED HISTORICAL + BLOOMBERG")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Download FRED historical
    fred_df = download_fred_historical()
    
    # Load current data
    base_df = load_base_data()
    if base_df.empty:
        print("\n❌ ERROR: No base data!")
        return
    
    # Merge FRED + Base
    combined = merge_fred_and_base(fred_df, base_df)
    
    # Resample to quarterly
    combined_quarterly = resample_to_quarterly(combined)
    
    # Load Bloomberg
    bloomberg_df = load_all_bloomberg()
    if not bloomberg_df.empty:
        bloomberg_indicators = calculate_financial_indicators(bloomberg_df)
        final = merge_all_data(combined_quarterly, bloomberg_indicators)
    else:
        final = combined_quarterly
    
    # Save
    save_output(final)
    
    print("\n" + "="*80)
    print(" ✓ FRANCE COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()