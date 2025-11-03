"""
Data Audit Script - Check What You Actually Have
=================================================
Scans your existing CSV files and shows:
1. What features exist
2. Date ranges
3. Missing data percentages
4. What's missing for V7 model

Usage:
    python audit_existing_data.py
"""

import pandas as pd
from pathlib import Path
import sys

# V7 Required Features (from your USA model)
V7_REQUIRED_FEATURES = {
    'critical': [
        'credit_spread',              # 43% importance
        'yield_curve_slope',          # 4.1% importance  
        'yield_curve_curvature',      # 4.6% importance
        'credit_spread_change',       # 4.7% importance
    ],
    'important': [
        'unemployment_rate',          # 21.5% importance
        'stock_market_index',         # 10.7% importance
        'unemployment_rate_lag1',     # 9.9% importance
        'exports_volume',             # 7.9% importance
        'stock_volatility_4q',        # 6.0% importance
        'interest_rate_short_term',   # 5.1% importance
        'imports_volume',             # 4.8% importance
        'trade_balance',              # 4.3% importance
        'interest_rate_short_term_lag1', # 3.4% importance
        'employment_level',           # 2.7% importance
    ]
}

# Base features needed to DERIVE the above
BASE_FEATURES_NEEDED = {
    'critical': [
        'corporate_bond_baa',         # For credit_spread
        'corporate_bond_aaa',         # For credit_spread
        'interest_rate_10y',          # For yield_curve_slope
        'interest_rate_2y',           # For yield_curve_slope
        'interest_rate_5y',           # For yield_curve_curvature
    ],
    'important': [
        'unemployment_rate',
        'stock_market_index',
        'exports_volume',
        'imports_volume',
        'trade_balance',
        'interest_rate_short_term',
        'employment_level',
    ],
    'optional': [
        'consumer_sentiment',
        'housing_starts',
        'building_permits',
        'capacity_utilization',
    ]
}

def find_csv_files():
    """Find all relevant CSV files"""
    search_paths = [
        Path('Data'),
        Path('../Data'),
        Path('../../Data'),
        Path('.'),
    ]
    
    csv_files = {}
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        # Look for country CSV files
        for pattern in ['*_data.csv', '*_extended*.csv', '*_processed*.csv', '*_fred*.csv']:
            for csv_file in search_path.rglob(pattern):
                country = None
                fname = csv_file.stem.lower()
                
                if 'usa' in fname or 'united_states' in fname:
                    country = 'USA'
                elif 'canada' in fname:
                    country = 'Canada'
                elif 'uk' in fname or 'united_kingdom' in fname:
                    country = 'UK'
                elif 'japan' in fname:
                    country = 'Japan'
                
                if country:
                    if country not in csv_files:
                        csv_files[country] = []
                    csv_files[country].append(csv_file)
    
    return csv_files

def analyze_csv(file_path):
    """Analyze a single CSV file"""
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Get basic stats
        stats = {
            'file': file_path.name,
            'rows': len(df),
            'cols': len(df.columns),
            'start': df.index.min(),
            'end': df.index.max(),
            'features': list(df.columns),
            'missing_pct': {}
        }
        
        # Calculate missing data per column
        for col in df.columns:
            if col != 'country':
                missing = df[col].isna().sum()
                pct = (missing / len(df)) * 100
                stats['missing_pct'][col] = pct
        
        return df, stats
    
    except Exception as e:
        print(f"  ✗ Error reading {file_path.name}: {e}")
        return None, None

def check_feature_availability(stats, base_features):
    """Check which required features are available"""
    available = {'critical': [], 'important': [], 'optional': []}
    missing = {'critical': [], 'important': [], 'optional': []}
    
    for tier in ['critical', 'important', 'optional']:
        if tier not in base_features:
            continue
            
        for feature in base_features[tier]:
            if feature in stats['features']:
                # Check data quality
                missing_pct = stats['missing_pct'].get(feature, 100)
                available[tier].append((feature, missing_pct))
            else:
                missing[tier].append(feature)
    
    return available, missing

def print_country_report(country, files_stats):
    """Print report for one country"""
    print(f"\n{'='*80}")
    print(f"  {country}")
    print(f"{'='*80}")
    
    if not files_stats:
        print("  ✗ No CSV files found")
        return
    
    # Show all files found
    print(f"\n  Found {len(files_stats)} file(s):")
    for i, (file_path, stats) in enumerate(files_stats, 1):
        if stats:
            print(f"    [{i}] {stats['file']}")
            print(f"        {stats['rows']} rows, {stats['cols']} cols")
            print(f"        {stats['start']} to {stats['end']}")
    
    # Use the most complete file (most columns)
    best_stats = max(files_stats, key=lambda x: x[1]['cols'] if x[1] else 0)[1]
    
    if not best_stats:
        return
    
    print(f"\n  Using: {best_stats['file']} (most features)")
    
    # Check feature availability
    available, missing = check_feature_availability(best_stats, BASE_FEATURES_NEEDED)
    
    # Critical Features
    print(f"\n  CRITICAL FEATURES (for V7 model - 56% importance):")
    if available['critical']:
        print(f"    [+] Available:")
        for feat, miss_pct in available['critical']:
            status = "[!] HIGH MISSING" if miss_pct > 30 else "[OK]" if miss_pct < 10 else "[!]"
            print(f"       {status} {feat:<30} ({miss_pct:.1f}% missing)")
    else:
        print(f"    [X] None found")
    
    if missing['critical']:
        print(f"    [X] Missing:")
        for feat in missing['critical']:
            print(f"       [X] {feat}")
    
    # Important Features
    print(f"\n  IMPORTANT FEATURES (44% importance):")
    if available['important']:
        print(f"    [+] Available:")
        for feat, miss_pct in available['important']:
            status = "[!] HIGH MISSING" if miss_pct > 30 else "[OK]" if miss_pct < 10 else "[!]"
            print(f"       {status} {feat:<30} ({miss_pct:.1f}% missing)")
    else:
        print(f"    [X] None found")
    
    if missing['important']:
        print(f"    [X] Missing:")
        for feat in missing['important']:
            print(f"       [X] {feat}")
    
    # Optional Features
    if available['optional']:
        print(f"\n  OPTIONAL FEATURES (nice-to-have):")
        print(f"    [+] Available:")
        for feat, miss_pct in available['optional']:
            status = "[!] HIGH MISSING" if miss_pct > 30 else "[OK]" if miss_pct < 10 else "[!]"
            print(f"       {status} {feat:<30} ({miss_pct:.1f}% missing)")
    
    # Model viability assessment
    print(f"\n  MODEL VIABILITY:")
    critical_count = len(available['critical'])
    critical_total = len(BASE_FEATURES_NEEDED['critical'])
    important_count = len(available['important'])
    
    if critical_count == critical_total:
        print(f"    [+] EXCELLENT - All critical features available")
        print(f"       Expected R²: 0.15-0.25 (like USA)")
    elif critical_count >= 3:
        print(f"    [!] MODERATE - {critical_count}/{critical_total} critical features")
        print(f"       Expected R²: 0.10-0.18 (may need proxies)")
    else:
        print(f"    [X] POOR - Only {critical_count}/{critical_total} critical features")
        print(f"       Expected R²: 0.00-0.10 (like USA V5 baseline)")
        print(f"       Recommendation: Need more financial market data")

def generate_summary_table(all_country_stats):
    """Generate comparison table across countries"""
    print(f"\n{'='*80}")
    print(f"  CROSS-COUNTRY COMPARISON")
    print(f"{'='*80}\n")
    
    # Critical features comparison
    print(f"  Critical Features Availability:")
    print(f"  {'-'*76}")
    print(f"  {'Feature':<30} | {'USA':<8} | {'Canada':<8} | {'UK':<8} | {'Japan':<8}")
    print(f"  {'-'*76}")
    
    for feat in BASE_FEATURES_NEEDED['critical']:
        row = f"  {feat:<30} |"
        for country in ['USA', 'Canada', 'UK', 'Japan']:
            if country in all_country_stats:
                stats = all_country_stats[country]
                if feat in stats['features']:
                    miss_pct = stats['missing_pct'].get(feat, 100)
                    if miss_pct < 10:
                        status = " [+]    "
                    elif miss_pct < 30:
                        status = f" [!]{miss_pct:.0f}%  "
                    else:
                        status = f" [X]{miss_pct:.0f}%  "
                else:
                    status = " [X] NO "
            else:
                status = " ? NONE "
            row += f" {status} |"
        print(row)
    
    print(f"  {'-'*76}\n")
    
    # Date range comparison
    print(f"  Date Range Comparison:")
    print(f"  {'-'*76}")
    print(f"  {'Country':<15} | {'Start Date':<12} | {'End Date':<12} | {'Quarters':<10}")
    print(f"  {'-'*76}")
    
    for country in ['USA', 'Canada', 'UK', 'Japan']:
        if country in all_country_stats:
            stats = all_country_stats[country]
            start = stats['start'].strftime('%Y-%m-%d') if hasattr(stats['start'], 'strftime') else str(stats['start'])
            end = stats['end'].strftime('%Y-%m-%d') if hasattr(stats['end'], 'strftime') else str(stats['end'])
            quarters = stats['rows']
            print(f"  {country:<15} | {start:<12} | {end:<12} | {quarters:<10}")
        else:
            print(f"  {country:<15} | {'N/A':<12} | {'N/A':<12} | {'N/A':<10}")
    
    print(f"  {'-'*76}\n")

def main():
    """Main audit process"""
    print("="*80)
    print("  DATA AUDIT - Checking Existing CSV Files")
    print("="*80)
    
    # Find all CSV files
    csv_files = find_csv_files()
    
    if not csv_files:
        print("\n[X] No CSV files found in Data/ directories")
        print("\nSearched in:")
        print("  - ./Data/")
        print("  - ../Data/")
        print("  - ../../Data/")
        print("  - ./")
        print("\nMake sure you're running from the correct directory")
        return
    
    print(f"\nFound CSV files for: {', '.join(csv_files.keys())}")
    
    # Analyze each country
    all_country_stats = {}
    
    for country in ['USA', 'Canada', 'UK', 'Japan']:
        if country in csv_files:
            files_stats = []
            for file_path in csv_files[country]:
                df, stats = analyze_csv(file_path)
                if stats:
                    files_stats.append((file_path, stats))
                    # Save best stats for summary
                    if country not in all_country_stats or stats['cols'] > all_country_stats[country]['cols']:
                        all_country_stats[country] = stats
            
            print_country_report(country, files_stats)
    
    # Generate summary table
    if len(all_country_stats) > 1:
        generate_summary_table(all_country_stats)
    
    # Final recommendations
    print(f"\n{'='*80}")
    print(f"  RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    for country, stats in all_country_stats.items():
        available, missing = check_feature_availability(stats, BASE_FEATURES_NEEDED)
        critical_count = len(available['critical'])
        critical_total = len(BASE_FEATURES_NEEDED['critical'])
        
        if critical_count == critical_total:
            print(f"  [+] {country}: Ready for V7 model training")
        elif critical_count >= 3:
            print(f"  [!] {country}: Proceed with caution - missing {len(missing['critical'])} critical features")
            print(f"      Missing: {', '.join(missing['critical'])}")
            print(f"      -> Consider using proxies or alternative data sources")
        else:
            print(f"  [X] {country}: Not ready - need {critical_total - critical_count} more critical features")
            print(f"      Missing: {', '.join(missing['critical'])}")
            print(f"      -> Run fred_series_finder.py to find FRED codes")
    
    print("\n" + "="*80)
    print(f"  Next step: python fred_series_finder_all_countries.py")
    print("="*80)

if __name__ == '__main__':
    main()