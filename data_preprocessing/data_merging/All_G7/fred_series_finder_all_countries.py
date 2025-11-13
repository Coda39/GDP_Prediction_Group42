"""
FRED Series Finder - All Countries (USA, Canada, UK, Japan)
=============================================================
Searches FRED for critical financial indicators for all countries.

Usage:
    python fred_series_finder_all_countries.py > fred_results.txt
"""

import pandas as pd
from fredapi import Fred
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FRED_API_KEY = '3477091a9ab79fb20b9ac8aca531d2dd'
fred = Fred(api_key=FRED_API_KEY)

# What we need to find (in order of importance)
SEARCHES = {
    'Canada': {
        'credit': ['Canada corporate bond AAA', 'Canada corporate bond BAA', 'Canada credit spread'],
        'yield_curve': ['Canada 2 year bond', 'Canada 5 year bond', 'Canada government bond 2y'],
        'real_activity': ['Canada consumer confidence', 'Canada housing starts', 'Canada building permits'],
        'employment': ['Canada employment level', 'Canada total employment'],
    },
    'UK': {
        'credit': ['UK corporate bond AAA', 'UK corporate bond BAA', 'Sterling corporate bond', 'UK credit spread'],
        'yield_curve': ['UK 2 year gilt', 'UK 5 year gilt', 'United Kingdom bond yield 2 year'],
        'real_activity': ['UK consumer confidence', 'UK building approvals', 'UK housing'],
        'employment': ['UK employment level', 'United Kingdom employment'],
    },
    'Japan': {
        'credit': ['Japan corporate bond', 'Japan credit spread'],
        'yield_curve': ['Japan 2 year bond', 'Japan 5 year bond', 'Japan government bond 2 year'],
        'real_activity': ['Japan consumer confidence', 'Japan housing starts'],
        'employment': ['Japan employment level', 'Japan total employment'],
    }
}

def search_and_check(search_term, limit=5):
    """Search FRED and check data quality"""
    try:
        results = fred.search(search_term, limit=limit)
        if len(results) == 0:
            return []
        
        series_list = []
        for idx, row in results.iterrows():
            series_id = row.get('id', 'N/A')
            
            # Check data quality
            try:
                data = fred.get_series(series_id, start_date='1980-01-01')
                if len(data) > 0:
                    first_date = data.index[0]
                    last_date = data.index[-1]
                    n_obs = len(data)
                    missing = data.isna().sum()
                    pct_missing = (missing / n_obs) * 100
                    years = last_date.year - first_date.year + 1
                    
                    # Only include if decent quality
                    if years >= 15 and pct_missing < 50 and last_date.year >= 2020:
                        series_list.append({
                            'id': series_id,
                            'title': row.get('title', 'N/A'),
                            'first': first_date.strftime('%Y-%m-%d'),
                            'last': last_date.strftime('%Y-%m-%d'),
                            'years': years,
                            'obs': n_obs,
                            'missing_pct': pct_missing,
                            'freq': row.get('frequency', 'N/A'),
                        })
            except:
                pass
        
        return series_list
    except:
        return []

def search_country(country, searches):
    """Search all categories for a country"""
    results = {}
    
    print(f"\n{'='*80}")
    print(f"  {country.upper()}")
    print(f"{'='*80}")
    
    for category, search_terms in searches.items():
        print(f"\n  [{category.upper()}]")
        category_results = []
        
        for term in search_terms:
            found = search_and_check(term, limit=3)
            if found:
                for series in found:
                    # Avoid duplicates
                    if series['id'] not in [r['id'] for r in category_results]:
                        category_results.append(series)
        
        if category_results:
            # Sort by years available (descending)
            category_results.sort(key=lambda x: x['years'], reverse=True)
            results[category] = category_results
            
            # Print top 3
            for i, series in enumerate(category_results[:3], 1):
                print(f"    {i}. {series['id']:<20} | {series['years']}y | {series['first']} to {series['last']}")
                print(f"       {series['title'][:70]}")
                if series['missing_pct'] > 0:
                    print(f"       [!] {series['missing_pct']:.1f}% missing")
        else:
            print(f"    [X] No series found")
            results[category] = []
    
    return results

def generate_code_template(all_results):
    """Generate Python dictionary template with found codes"""
    print(f"\n\n{'='*80}")
    print(f"  COPY-PASTE TEMPLATE FOR download_historical_data.py")
    print(f"{'='*80}\n")
    
    for country, categories in all_results.items():
        print(f"# {country.upper()} SERIES")
        print(f"{country.upper()}_SERIES = {{")
        
        # Existing features (keep)
        print(f"    # Existing features - KEEP THESE")
        if country == 'Canada':
            print(f"    'gdp_real': 'NAEXKP01CAQ189S',")
            print(f"    'gdp_nominal': 'MKTGDPCAA646NWDB',")
            print(f"    'unemployment_rate': 'LRUNTTTTCAQ156S',")
            print(f"    'cpi_all_items': 'CANCPIALLMINMEI',")
            print(f"    'interest_rate_short_term': 'IR3TCD01CAM156N',")
            print(f"    'industrial_production_index': 'CANPROINDMISMEI',")
            print(f"    'exports_volume': 'XTEXVA01CAQ188S',")
            print(f"    'imports_volume': 'XTIMVA01CAQ188S',")
            print(f"    'stock_market_index': 'SPTSXCTTM',")
            print(f"    'population_total': 'POPTOTCAA647NWDB',")
            print(f"    'interest_rate_10y': 'IRLTLT01CAM156N',  # CONFIRMED")
        elif country == 'UK':
            print(f"    'gdp_real': 'NAEXKP01GBQ189S',")
            print(f"    'gdp_nominal': 'MKTGDPGBA646NWDB',")
            print(f"    'unemployment_rate': 'LRUNTTTTGBM156S',")
            print(f"    'cpi_all_items': 'GBRCPIALLMINMEI',")
            print(f"    'interest_rate_short_term': 'IR3TIB01GBM156N',")
            print(f"    'industrial_production_index': 'GBRPROINDMISMEI',")
            print(f"    'exports_volume': 'XTEXVA01GBQ188S',")
            print(f"    'imports_volume': 'XTIMVA01GBQ188S',")
            print(f"    'stock_market_index': 'FTSE100',")
            print(f"    'population_total': 'POPTOTGBA647NWDB',")
            print(f"    'interest_rate_10y': 'IRLTLT01GBM156N',  # CONFIRMED")
        elif country == 'Japan':
            print(f"    'gdp_real': 'JPNRGDPEXP',")
            print(f"    'gdp_nominal': 'MKTGDPJPA646NWDB',")
            print(f"    'unemployment_rate': 'LRUNTTTTJPM156S',")
            print(f"    'cpi_all_items': 'JPNCPIALLMINMEI',")
            print(f"    'interest_rate_short_term': 'IR3TIB01JPM156N',")
            print(f"    'industrial_production_index': 'JPNPROINDMISMEI',")
            print(f"    'exports_volume': 'XTEXVA01JPQ188S',")
            print(f"    'imports_volume': 'XTIMVA01JPQ188S',")
            print(f"    'stock_market_index': 'NIKKEI225',")
            print(f"    'population_total': 'POPTOTJPA647NWDB',")
        
        print()
        print(f"    # NEW V6 FEATURES - ADD THESE")
        
        # Add found features
        if 'yield_curve' in categories and categories['yield_curve']:
            print(f"    # Yield Curve")
            for i, series in enumerate(categories['yield_curve'][:2]):
                var_name = f"interest_rate_{['2y', '5y'][i] if i < 2 else 'extra'}"
                print(f"    '{var_name}': '{series['id']}',  # {series['years']}y, {series['first']} to {series['last']}")
        else:
            print(f"    # 'interest_rate_2y': 'FIND_THIS',  # NEEDED")
            print(f"    # 'interest_rate_5y': 'FIND_THIS',  # NEEDED")
        
        print()
        if 'credit' in categories and categories['credit']:
            print(f"    # Corporate Bonds / Credit Spread")
            for i, series in enumerate(categories['credit'][:2]):
                var_name = f"corporate_bond_{['aaa', 'baa'][i] if i < 2 else 'other'}"
                print(f"    '{var_name}': '{series['id']}',  # {series['years']}y, {series['first']} to {series['last']}")
        else:
            print(f"    # 'corporate_bond_aaa': 'FIND_THIS',  # CRITICAL (43% importance)")
            print(f"    # 'corporate_bond_baa': 'FIND_THIS',  # CRITICAL (43% importance)")
        
        print()
        if 'real_activity' in categories and categories['real_activity']:
            print(f"    # Real Activity Indicators")
            for series in categories['real_activity'][:3]:
                if 'consumer' in series['title'].lower():
                    var_name = 'consumer_sentiment'
                elif 'housing' in series['title'].lower():
                    var_name = 'housing_starts'
                elif 'building' in series['title'].lower() or 'permit' in series['title'].lower():
                    var_name = 'building_permits'
                else:
                    continue
                print(f"    '{var_name}': '{series['id']}',  # {series['years']}y")
        
        if 'employment' in categories and categories['employment']:
            print()
            print(f"    # Employment Level")
            for series in categories['employment'][:1]:
                print(f"    'employment_level': '{series['id']}',  # {series['years']}y")
        
        print(f"}}")
        print()
    
    print("="*80)

def main():
    """Main search process"""
    print("="*80)
    print("  FRED SERIES FINDER - All Countries")
    print("  Searching for critical financial indicators...")
    print("="*80)
    
    all_results = {}
    
    for country, searches in SEARCHES.items():
        results = search_country(country, searches)
        all_results[country] = results
    
    # Generate template
    generate_code_template(all_results)
    
    print("\n\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    for country, categories in all_results.items():
        credit = len(categories.get('credit', []))
        yield_curve = len(categories.get('yield_curve', []))
        real = len(categories.get('real_activity', []))
        
        print(f"\n  {country}:")
        print(f"    Credit market: {credit} series found {'[+]' if credit > 0 else '[X]'}")
        print(f"    Yield curve: {yield_curve} series found {'[+]' if yield_curve > 0 else '[X]'}")
        print(f"    Real activity: {real} series found {'[+]' if real > 0 else '[!]'}")
        
        if credit == 0:
            print(f"    [!] WARNING: No credit spread data - model will struggle (43% importance)")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()