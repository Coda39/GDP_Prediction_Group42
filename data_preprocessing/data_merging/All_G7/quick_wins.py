"""
Quick Wins - Download Easy Available Data
==========================================
These are CONFIRMED to exist on FRED. Let's grab them before manual search.

Run this first, then assess what's still missing.
"""

import pandas as pd
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

FRED_API_KEY = '3477091a9ab79fb20b9ac8aca531d2dd'
fred = Fred(api_key=FRED_API_KEY)

# ============================================================================
# EASY WINS - CONFIRMED TO EXIST
# ============================================================================

EASY_CANADA = {
    # We KNOW these exist from previous searches
    'interest_rate_10y': 'IRLTLT01CAM156N',      # CONFIRMED - 10Y bond
    'consumer_confidence': 'CSCICP03CAM665S',    # OECD consumer confidence
    'housing_starts': 'CANHOUST',                # Housing starts
    'employment_level': 'LFEMTTTTCAM647S',       # Employment found in search
    'building_permits': 'CANODCNPI03GPSAM',      # Building permits found
    
    # Try these - likely exist
    'prime_rate': 'CAPRIME',                     # Canada prime rate (worth trying)
}

EASY_UK = {
    # We KNOW these exist
    'interest_rate_10y': 'IRLTLT01GBM156N',      # CONFIRMED - 10Y gilt
    'bank_rate': 'BOERUKQ',                      # CONFIRMED - Bank of England rate
    'consumer_confidence': 'CSCICP03GBM665S',    # OECD consumer confidence
    'building_permits': 'GBRAPRAVALMQ',          # Building approvals
    'employment_level': 'SLEMPTOTLSPZSGBR',      # Employment found in search
}

EASY_JAPAN = {
    # Try these - appeared in search
    'consumer_confidence': 'CSCICP02JPM460S',    # Found in search
    'housing_starts': 'WSCNDW01JPQ659S',         # Found in search
    'employment_level': 'LREM64TTJPQ156N',       # Found in search
    
    # Try common codes
    'interest_rate_10y': 'IRLTLT01JPM156N',      # Standard pattern (worth trying)
}

# ============================================================================
# DOWNLOAD AND CHECK
# ============================================================================

def try_download(series_id, feature_name, start='1980-01-01'):
    """Try to download a series and report results"""
    try:
        data = fred.get_series(series_id, start_date=start)
        if len(data) > 0:
            first = data.index[0]
            last = data.index[-1]
            obs = len(data)
            missing = data.isna().sum()
            pct_miss = (missing / obs) * 100
            
            print(f"  [+] {feature_name:<25} {series_id:<20}")
            print(f"      {first.strftime('%Y-%m-%d')} to {last.strftime('%Y-%m-%d')} | {obs} obs | {pct_miss:.1f}% missing")
            
            if pct_miss > 30:
                print(f"      [!] WARNING: High missing data")
            
            return True, data
        else:
            print(f"  [X] {feature_name:<25} {series_id:<20} - No data returned")
            return False, None
            
    except Exception as e:
        print(f"  [X] {feature_name:<25} {series_id:<20} - FAILED: {str(e)[:50]}")
        return False, None

def check_country(country_name, series_dict):
    """Check all easy series for a country"""
    print(f"\n{'='*80}")
    print(f"  {country_name.upper()} - EASY WINS CHECK")
    print(f"{'='*80}")
    
    success = {}
    failed = []
    
    for feature, code in series_dict.items():
        worked, data = try_download(code, feature)
        if worked:
            success[feature] = code
        else:
            failed.append(feature)
    
    # Summary
    print(f"\n  SUMMARY:")
    print(f"    Success: {len(success)}/{len(series_dict)} features")
    if success:
        print(f"    Working codes:")
        for feat, code in success.items():
            print(f"      - {feat}: {code}")
    
    if failed:
        print(f"    Failed:")
        for feat in failed:
            print(f"      - {feat}")
    
    return success, failed

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("  QUICK WINS - Downloading Easy Available Data")
    print("  Let's grab confirmed features before manual search")
    print("="*80)
    
    all_results = {}
    
    # Check each country
    all_results['Canada'] = check_country('Canada', EASY_CANADA)
    all_results['UK'] = check_country('UK', EASY_UK)
    all_results['Japan'] = check_country('Japan', EASY_JAPAN)
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for country, (success, failed) in all_results.items():
        print(f"\n  {country}:")
        print(f"    Found: {len(success)} features")
        if success:
            for feat, code in success.items():
                print(f"      [+] {feat}")
        if failed:
            print(f"    Still need: {len(failed)} features")
            for feat in failed:
                print(f"      [X] {feat}")
    
    # What's still missing (critical features)
    print(f"\n{'='*80}")
    print(f"  WHAT'S STILL MISSING (CRITICAL)")
    print(f"{'='*80}")
    
    critical_missing = {
        'Canada': ['interest_rate_2y', 'interest_rate_5y', 'corporate_bond_aaa', 'corporate_bond_baa'],
        'UK': ['interest_rate_2y', 'interest_rate_5y', 'corporate_bond_aaa', 'corporate_bond_baa'],
        'Japan': ['interest_rate_2y', 'interest_rate_5y', 'interest_rate_10y', 'corporate_bond_aaa', 'corporate_bond_baa'],
    }
    
    for country, missing in critical_missing.items():
        success, _ = all_results[country]
        still_missing = [f for f in missing if f not in success]
        
        if still_missing:
            print(f"\n  {country} still needs:")
            for feat in still_missing:
                print(f"    [X] {feat}")
        else:
            print(f"\n  {country}: [+] ALL CRITICAL FEATURES FOUND!")
    
    print("\n" + "="*80)
    print("  NEXT STEPS")
    print("="*80)
    print("\n  If features found above:")
    print("    1. Update download_historical_data.py with working codes")
    print("    2. Re-run download script")
    print("\n  For still-missing critical features:")
    print("    3. Try manual FRED search (MANUAL_SEARCH_GUIDE.md)")
    print("    4. Or use proxy credit spread approach")
    print("="*80)

if __name__ == '__main__':
    main()