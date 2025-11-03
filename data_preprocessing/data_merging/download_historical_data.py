"""
Download Historical Economic Data (1980-2024) from FRED
========================================================
This script downloads economic indicators from FRED API to extend
our training data back to 1980, adding high-inflation regime examples.

**V6 UPDATE**: Added financial market indicators (yield curve, credit spreads)
These are NOT GDP components - they're market prices that predict future GDP.

**V6.1 FIX**: Fixed S&P 500 resampling issue - use .last() for indices, not .mean()

Requirements:
    pip install fredapi pandas

Usage:
    1. Get FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html
    2. Set your API key in this script (line 30)
    3. Run: python download_historical_data.py
    4. Historical data will be saved to: raw_data/historical/
"""

import pandas as pd
from fredapi import Fred
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# FRED API Key
FRED_API_KEY = '3477091a9ab79fb20b9ac8aca531d2dd'

# Date range for historical data
START_DATE = '1980-01-01'
END_DATE = '2024-12-31'

# Output directory - save to Data/historical/ (go up 2 levels from data_merging)
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'Data' / 'historical'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FRED SERIES IDs FOR EACH COUNTRY
# ============================================================================

# USA FRED Series IDs
USA_SERIES = {
    # ========================================================================
    # GDP & National Accounts
    # ========================================================================
    'gdp_real': 'GDPC1',                    # Real GDP (Billions of Chained 2017 Dollars)
    'gdp_nominal': 'GDP',                   # Nominal GDP (Billions of Dollars)
    'household_consumption': 'PCEC',        # Personal Consumption Expenditures
    'capital_formation': 'GPDI',            # Gross Private Domestic Investment
    'government_spending': 'GCE',           # Government Consumption Expenditures
    'exports_volume': 'EXPGS',              # Exports of Goods and Services
    'imports_volume': 'IMPGS',              # Imports of Goods and Services
    
    # ========================================================================
    # Labor Market
    # ========================================================================
    'unemployment_rate': 'UNRATE',          # Unemployment Rate (%)
    'employment_level': 'CE16OV',           # Employment Level (Thousands)
    
    # ========================================================================
    # Prices & Inflation
    # ========================================================================
    'cpi_all_items': 'CPIAUCSL',           # Consumer Price Index
    
    # ========================================================================
    # Interest Rates (EXISTING - NOT GDP components, market prices)
    # ========================================================================
    'interest_rate_short_term': 'DFF',      # Federal Funds Rate (%)
    'interest_rate_long_term': 'GS10',      # 10-Year Treasury Rate (%)
    
    # ========================================================================
    # **NEW V6**: FINANCIAL MARKET INDICATORS
    # These are market prices, NOT GDP components!
    # They predict GDP because markets are forward-looking.
    # ========================================================================
    
    # Yield Curve (Treasury bonds - market prices)
    'interest_rate_10y': 'GS10',            # 10-Year Treasury Constant Maturity Rate
    'interest_rate_2y': 'GS2',              # 2-Year Treasury Constant Maturity Rate
    'interest_rate_5y': 'GS5',              # 5-Year Treasury Constant Maturity Rate
    'interest_rate_3m': 'TB3MS',            # 3-Month Treasury Bill Rate
    'yield_curve_spread': 'T10Y2Y',         # 10Y-2Y Treasury Spread (pre-calculated by FRED)
    
    # Credit Market (Corporate bonds - market prices, NOT GDP)
    'corporate_bond_baa': 'BAA',            # Moody's Seasoned Baa Corporate Bond Yield
    'corporate_bond_aaa': 'AAA',            # Moody's Seasoned Aaa Corporate Bond Yield
    
    # Banking & Financial Stress (NOT GDP)
    'ted_spread': 'TEDRATE',                # TED Spread (3M LIBOR - 3M Treasury)
    # Note: TEDRATE starts 1986, will have NaNs before that
    
    # ========================================================================
    # **NEW V6**: REAL ACTIVITY LEADING INDICATORS
    # These LEAD GDP changes, they are NOT components of GDP
    # ========================================================================
    
    # Consumer Confidence (Survey - predicts future consumption)
    'consumer_sentiment': 'UMCSENT',        # U of Michigan Consumer Sentiment Index
    
    # Housing Market (Permits LEAD construction spending by 6+ months)
    'building_permits': 'PERMIT',           # New Private Housing Units Authorized (Thousands)
    'housing_starts': 'HOUST',              # Housing Starts (Thousands of Units)
    
    # Industrial Capacity (Efficiency measure - NOT output)
    'capacity_utilization': 'TCU',          # Capacity Utilization: Total Industry (%)
    
    # ========================================================================
    # Production & Trade (EXISTING)
    # ========================================================================
    'industrial_production_index': 'INDPRO', # Industrial Production Index
    'trade_balance': 'BOPGSTB',             # Trade Balance (Millions of Dollars)
    
    # ========================================================================
    # Money & Markets (EXISTING)
    # ========================================================================
    'money_supply_broad': 'M2SL',           # M2 Money Supply (Billions)
    'stock_market_index': 'SP500',          # S&P 500 Index
    
    # ========================================================================
    # Population (for per capita calculations)
    # ========================================================================
    'population_total': 'POPTOTUSA647NWDB', # Total Population
    'population_working_age': 'LFWA64TTUSM647S', # Working Age Population
}

# Define which series should use which resampling method
# This is critical for avoiding the S&P 500 missing data issue!
RESAMPLE_METHODS = {
    # Use LAST value for indices and levels (point-in-time measurements)
    'last': [
        'stock_market_index',           # S&P 500 - use closing value
        'population_total',              # Population - point estimate
        'population_working_age',        # Population - point estimate
        'cpi_all_items',                # CPI - use end of quarter value
        'industrial_production_index',   # Index - use end of quarter value
    ],
    
    # Use MEAN for rates and percentages (averages over the quarter)
    'mean': [
        'unemployment_rate',
        'interest_rate_short_term',
        'interest_rate_long_term',
        'interest_rate_10y',
        'interest_rate_2y',
        'interest_rate_5y',
        'interest_rate_3m',
        'yield_curve_spread',
        'corporate_bond_baa',
        'corporate_bond_aaa',
        'ted_spread',
        'consumer_sentiment',
        'capacity_utilization',
    ],
    
    # Use SUM for flows (aggregate over the quarter)
    'sum': [
        'gdp_real',
        'gdp_nominal',
        'household_consumption',
        'capital_formation',
        'government_spending',
        'exports_volume',
        'imports_volume',
        'employment_level',
        'building_permits',
        'housing_starts',
        'trade_balance',
        'money_supply_broad',
    ]
}

# Canada FRED Series IDs
CANADA_SERIES = {
    'gdp_real': 'NAEXKP01CAQ189S',          # Real GDP
    'gdp_nominal': 'MKTGDPCAA646NWDB',      # Nominal GDP
    'unemployment_rate': 'LRUNTTTTCAQ156S',  # Unemployment Rate
    'cpi_all_items': 'CANCPIALLMINMEI',     # CPI
    'interest_rate_short_term': 'IR3TCD01CAM156N', # 3-Month Rate
    'industrial_production_index': 'CANPROINDMISMEI', # Industrial Production
    'exports_volume': 'XTEXVA01CAQ188S',    # Exports
    'imports_volume': 'XTIMVA01CAQ188S',    # Imports
    'stock_market_index': 'SPTSXCTTM',      # TSX Composite
    'population_total': 'POPTOTCAA647NWDB', # Population
}

# Japan FRED Series IDs
JAPAN_SERIES = {
    'gdp_real': 'JPNRGDPEXP',               # Real GDP
    'gdp_nominal': 'MKTGDPJPA646NWDB',      # Nominal GDP
    'unemployment_rate': 'LRUNTTTTJPM156S',  # Unemployment Rate
    'cpi_all_items': 'JPNCPIALLMINMEI',     # CPI
    'interest_rate_short_term': 'IR3TIB01JPM156N', # 3-Month Rate
    'industrial_production_index': 'JPNPROINDMISMEI', # Industrial Production
    'exports_volume': 'XTEXVA01JPQ188S',    # Exports
    'imports_volume': 'XTIMVA01JPQ188S',    # Imports
    'stock_market_index': 'NIKKEI225',      # Nikkei 225
    'population_total': 'POPTOTJPA647NWDB', # Population
}

# UK FRED Series IDs
UK_SERIES = {
    'gdp_real': 'NAEXKP01GBQ189S',          # Real GDP
    'gdp_nominal': 'MKTGDPGBA646NWDB',      # Nominal GDP
    'unemployment_rate': 'LRUNTTTTGBM156S',  # Unemployment Rate
    'cpi_all_items': 'GBRCPIALLMINMEI',     # CPI
    'interest_rate_short_term': 'IR3TIB01GBM156N', # 3-Month Rate
    'industrial_production_index': 'GBRPROINDMISMEI', # Industrial Production
    'exports_volume': 'XTEXVA01GBQ188S',    # Exports
    'imports_volume': 'XTIMVA01GBQ188S',    # Imports
    'stock_market_index': 'FTSE100',        # FTSE 100
    'population_total': 'POPTOTGBA647NWDB', # Population
}

COUNTRY_SERIES = {
    'usa': USA_SERIES,
    'canada': CANADA_SERIES,
    'japan': JAPAN_SERIES,
    'uk': UK_SERIES,
}

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_fred_data(fred, series_dict, start_date, end_date):
    """
    Download multiple series from FRED
    
    Parameters:
    - fred: Fred API object
    - series_dict: Dictionary of {column_name: series_id}
    - start_date: Start date for data
    - end_date: End date for data
    
    Returns: DataFrame with all series
    """
    data = {}
    missing_series = []
    partial_series = []
    
    print(f"  Downloading {len(series_dict)} series...")
    
    for col_name, series_id in series_dict.items():
        try:
            series = fred.get_series(series_id, start_date, end_date)
            # Ensure series has a datetime index
            if len(series) > 0:
                series.index = pd.to_datetime(series.index)
                data[col_name] = series
                
                # Check data quality
                missing_pct = series.isna().sum() / len(series) * 100
                
                if missing_pct > 50:
                    print(f"    ⚠ {col_name:30s} ({series_id}): {len(series)} obs, {missing_pct:.1f}% MISSING")
                    partial_series.append((col_name, series_id, missing_pct))
                elif missing_pct > 0:
                    print(f"    ✓ {col_name:30s} ({series_id}): {len(series)} obs, {missing_pct:.1f}% missing")
                else:
                    print(f"    ✓ {col_name:30s} ({series_id}): {len(series)} obs, complete")
            else:
                print(f"    ✗ {col_name:30s} ({series_id}): No data returned")
                missing_series.append((col_name, series_id))
        except Exception as e:
            print(f"    ✗ {col_name:30s} ({series_id}): FAILED - {str(e)}")
            missing_series.append((col_name, series_id))
    
    # Report issues
    if missing_series:
        print(f"\n  ⚠ WARNING: {len(missing_series)} series completely failed:")
        for col_name, series_id in missing_series:
            print(f"    - {col_name} ({series_id})")
    
    if partial_series:
        print(f"\n  ⚠ WARNING: {len(partial_series)} series have >50% missing data:")
        for col_name, series_id, missing_pct in partial_series:
            print(f"    - {col_name} ({series_id}): {missing_pct:.1f}% missing")
        print(f"    → These will be handled by preprocessing (forward-fill or drop)")
    
    # Combine into DataFrame - use outer join to handle different frequencies
    if data:
        df = pd.DataFrame(data)
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        return df
    else:
        return pd.DataFrame()

def resample_to_quarterly_smart(df):
    """
    Resample to quarterly using appropriate method for each column
    
    **V6.2 FIX**: For indices, forward-fill within quarter then take last value
    This ensures we get the last AVAILABLE value, not just the last date
    
    Parameters:
    - df: DataFrame with datetime index
    
    Returns: Quarterly DataFrame
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Separate columns by resampling method
    result_dfs = []
    
    # Method 1: LAST (for indices, levels, point-in-time)
    # CRITICAL: Forward-fill first to handle weekend/holiday gaps!
    last_cols = [col for col in df.columns if col in RESAMPLE_METHODS['last']]
    if last_cols:
        # Forward-fill BEFORE resampling to handle gaps
        df_last = df[last_cols].fillna(method='ffill').resample('QE').last()
        result_dfs.append(df_last)
        print(f"    → {len(last_cols)} series using .ffill().last() (indices/levels)")
    
    # Method 2: MEAN (for rates, percentages)
    mean_cols = [col for col in df.columns if col in RESAMPLE_METHODS['mean']]
    if mean_cols:
        df_mean = df[mean_cols].resample('QE').mean()
        result_dfs.append(df_mean)
        print(f"    → {len(mean_cols)} series using .mean() (rates/percentages)")
    
    # Method 3: SUM (for flows - but use last since FRED gives quarterly already)
    sum_cols = [col for col in df.columns if col in RESAMPLE_METHODS['sum']]
    if sum_cols:
        # For quarterly data, use .last() since they're already quarterly aggregates
        df_sum = df[sum_cols].resample('QE').last()
        result_dfs.append(df_sum)
        print(f"    → {len(sum_cols)} series using .last() (flows/aggregates)")
    
    # Catch any columns not in RESAMPLE_METHODS (default to mean)
    all_defined = set(RESAMPLE_METHODS['last'] + RESAMPLE_METHODS['mean'] + RESAMPLE_METHODS['sum'])
    undefined_cols = [col for col in df.columns if col not in all_defined]
    if undefined_cols:
        print(f"    ⚠ {len(undefined_cols)} undefined series, using .mean(): {undefined_cols}")
        df_undefined = df[undefined_cols].resample('QE').mean()
        result_dfs.append(df_undefined)
    
    # Combine all
    if result_dfs:
        return pd.concat(result_dfs, axis=1)
    else:
        return pd.DataFrame()

def check_data_quality(df, country):
    """
    Check data quality and report issues
    
    Parameters:
    - df: DataFrame to check
    - country: Country name
    """
    print(f"\n  {'='*66}")
    print(f"  DATA QUALITY REPORT: {country.upper()}")
    print(f"  {'='*66}")
    
    # Overall stats
    total_rows = len(df)
    total_cols = df.shape[1]
    
    print(f"  Total quarters: {total_rows}")
    print(f"  Total features: {total_cols}")
    
    # Missing data per column
    missing_stats = []
    for col in df.columns:
        if col == 'country':
            continue
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        if missing_count > 0:
            missing_stats.append((col, missing_count, missing_pct))
    
    if missing_stats:
        print(f"\n  Missing data by feature:")
        print(f"  {'Feature':<35} {'Missing':<10} {'Percent':<10}")
        print(f"  {'-'*60}")
        
        # Sort by percent missing (descending)
        missing_stats.sort(key=lambda x: x[2], reverse=True)
        
        for col, count, pct in missing_stats:
            status = "⚠ HIGH" if pct > 50 else "✓ OK" if pct < 10 else "⚠ MODERATE"
            print(f"  {col:<35} {count:<10} {pct:>6.1f}%  {status}")
    else:
        print(f"  ✓ No missing data!")
    
    print(f"  {'='*66}\n")
    
    return missing_stats

# ============================================================================
# MAIN DOWNLOAD PROCESS
# ============================================================================

def download_country_data(country, fred, start_date, end_date):
    """Download historical data for one country"""
    print(f"\n{'='*70}")
    print(f"Downloading {country.upper()} Historical Data ({start_date} to {end_date})")
    print(f"{'='*70}")
    
    series_dict = COUNTRY_SERIES[country]
    
    # Download data
    df = download_fred_data(fred, series_dict, start_date, end_date)
    
    if df.empty:
        print(f"  ✗ No data downloaded for {country}")
        return None
    
    # Resample to quarterly using smart method
    print(f"\n  Resampling to quarterly frequency (using appropriate methods)...")
    df_quarterly = resample_to_quarterly_smart(df)
    
    print(f"  ✓ Final shape: {df_quarterly.shape}")
    print(f"  ✓ Date range: {df_quarterly.index[0]} to {df_quarterly.index[-1]}")
    print(f"  ✓ Quarters: {len(df_quarterly)}")
    
    # Check data quality
    missing_stats = check_data_quality(df_quarterly, country)
    
    # Add country column
    df_quarterly['country'] = country
    
    # Save to CSV
    output_file = OUTPUT_DIR / f'{country}_extended_with_financial_1980-2024.csv'
    df_quarterly.to_csv(output_file, index=True)
    print(f"  ✓ Saved to: {output_file}")
    
    # Save data quality report
    if missing_stats:
        report_file = OUTPUT_DIR / f'{country}_data_quality_report_v6.txt'
        with open(report_file, 'w') as f:
            f.write(f"Data Quality Report: {country.upper()} V6 (with Financial Indicators)\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Features with missing data:\n\n")
            for col, count, pct in missing_stats:
                f.write(f"{col:<40} {count:>5} / {len(df_quarterly)} ({pct:>5.1f}%)\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"Note: Missing data will be handled by preprocessing pipeline\n")
            f.write(f"Strategies: forward-fill, interpolation, or feature dropping\n")
        print(f"  ✓ Data quality report saved to: {report_file}")
    
    return df_quarterly

def main():
    """Main download process"""
    print("="*70)
    print("FRED Historical Data Download (1980-2024)")
    print("V6.1: FIXED S&P 500 resampling issue!")
    print("="*70)
    
    # Check API key
    if FRED_API_KEY == 'YOUR_API_KEY_HERE':
        print("\n❌ ERROR: Please set your FRED API key!")
        print("\nTo get an API key:")
        print("1. Go to: https://fred.stlouisfed.org/")
        print("2. Create free account")
        print("3. Go to: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("4. Copy your API key")
        print("5. Paste it in this script (line 30)")
        return
    
    # Initialize FRED API
    print(f"\nInitializing FRED API...")
    try:
        fred = Fred(api_key=FRED_API_KEY)
        print("✓ Connected to FRED API")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    # Download data for each country
    # Focus on USA - best data quality back to 1980
    countries = ['usa']
    # Can add others later if USA works: ['canada', 'japan', 'uk']
    results = {}
    
    for country in countries:
        try:
            df = download_country_data(country, fred, START_DATE, END_DATE)
            if df is not None:
                results[country] = df
        except Exception as e:
            print(f"\n✗ Error downloading {country}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully downloaded: {len(results)}/{len(countries)} countries")
    for country, df in results.items():
        print(f"  ✓ {country.upper():10s}: {len(df)} quarters, {df.shape[1]} columns")
    
    print(f"\n✓ All files saved to: {OUTPUT_DIR}")
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Update preprocessing_pipeline.py with V6 features")
    print("2. Run: cd .. && python preprocessing_pipeline.py")
    print("3. Train V6 model and CHECK IF R² IMPROVES!")
    print("="*70)

if __name__ == '__main__':
    main()