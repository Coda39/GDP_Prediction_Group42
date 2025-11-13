"""
Quick Reference: Known FRED Codes vs Missing Data
===================================================
What we KNOW exists vs what we NEED to find

Run this after fred_series_finder to see what's still missing.
"""

# ============================================================================
# WHAT WE KNOW EXISTS (From your download_historical_data.py)
# ============================================================================

KNOWN_EXISTING = {
    'USA': {
        'gdp_real': 'GDPC1',
        'unemployment_rate': 'UNRATE',
        'cpi_all_items': 'CPIAUCSL',
        'exports_volume': 'EXPGS',
        'imports_volume': 'IMPGS',
        'stock_market_index': 'SP500',
        'interest_rate_short_term': 'DFF',
        'interest_rate_10y': 'GS10',
        'interest_rate_2y': 'GS2',
        'interest_rate_5y': 'GS5',
        'corporate_bond_baa': 'BAA',
        'corporate_bond_aaa': 'AAA',
        'consumer_sentiment': 'UMCSENT',
        'housing_starts': 'HOUST',
        'employment_level': 'CE16OV',
    },
    'Canada': {
        'gdp_real': 'NAEXKP01CAQ189S',
        'unemployment_rate': 'LRUNTTTTCAQ156S',
        'cpi_all_items': 'CANCPIALLMINMEI',
        'exports_volume': 'XTEXVA01CAQ188S',
        'imports_volume': 'XTIMVA01CAQ188S',
        'stock_market_index': 'SPTSXCTTM',
        'interest_rate_short_term': 'IR3TCD01CAM156N',
        'interest_rate_10y': 'IRLTLT01CAM156N',  # CONFIRMED
        # Missing: 2y, 5y bonds, corporate bonds, consumer sentiment, housing, employment level
    },
    'UK': {
        'gdp_real': 'NAEXKP01GBQ189S',
        'unemployment_rate': 'LRUNTTTTGBM156S',
        'cpi_all_items': 'GBRCPIALLMINMEI',
        'exports_volume': 'XTEXVA01GBQ188S',
        'imports_volume': 'XTIMVA01GBQ188S',
        'stock_market_index': 'FTSE100',
        'interest_rate_short_term': 'IR3TIB01GBM156N',
        'interest_rate_10y': 'IRLTLT01GBM156N',  # CONFIRMED
        # Missing: 2y, 5y gilts, corporate bonds, consumer sentiment, housing, employment level
    },
    'Japan': {
        'gdp_real': 'JPNRGDPEXP',
        'unemployment_rate': 'LRUNTTTTJPM156S',
        'cpi_all_items': 'JPNCPIALLMINMEI',
        'exports_volume': 'XTEXVA01JPQ188S',
        'imports_volume': 'XTIMVA01JPQ188S',
        'stock_market_index': 'NIKKEI225',
        'interest_rate_short_term': 'IR3TIB01JPM156N',
        # Missing: ALL yield curve, corporate bonds, consumer sentiment, housing, employment level
    }
}

# ============================================================================
# WHAT WE NEED TO FIND (Priority order)
# ============================================================================

NEED_TO_FIND = {
    'Canada': {
        'CRITICAL (56% model importance)': [
            'interest_rate_2y',      # For yield_curve_slope
            'interest_rate_5y',      # For yield_curve_curvature
            'corporate_bond_aaa',    # For credit_spread (43% importance!)
            'corporate_bond_baa',    # For credit_spread (43% importance!)
        ],
        'IMPORTANT (additional features)': [
            'consumer_sentiment',    # Real activity indicator
            'housing_starts',        # Real activity indicator (CANHOUST exists?)
            'employment_level',      # Absolute employment (not just rate)
        ],
    },
    'UK': {
        'CRITICAL (56% model importance)': [
            'interest_rate_2y',      # 2-year gilt yield
            'interest_rate_5y',      # 5-year gilt yield
            'corporate_bond_aaa',    # High-grade Sterling corporate bonds
            'corporate_bond_baa',    # Medium-grade Sterling corporate bonds
        ],
        'IMPORTANT (additional features)': [
            'consumer_sentiment',    # Consumer confidence
            'building_permits',      # GBRAPRAVALMQ exists?
            'employment_level',      # Absolute employment
        ],
    },
    'Japan': {
        'CRITICAL (56% model importance)': [
            'interest_rate_10y',     # 10-year JGB
            'interest_rate_2y',      # 2-year JGB
            'interest_rate_5y',      # 5-year JGB
            'corporate_bond_aaa',    # Japanese corporate bonds
            'corporate_bond_baa',    # Japanese corporate bonds
        ],
        'IMPORTANT (additional features)': [
            'consumer_sentiment',    # Consumer confidence
            'housing_starts',        # Housing construction
            'employment_level',      # Absolute employment
        ],
    }
}

# ============================================================================
# PROXY OPTIONS (if corporate bonds not found)
# ============================================================================

PROXY_OPTIONS = {
    'Canada': {
        'credit_spread_proxy': 'Prime Rate - 10Y Bond',
        'prime_rate_code': 'FIND: Canada prime rate',
    },
    'UK': {
        'credit_spread_proxy': 'Bank Rate - 10Y Gilt',
        'bank_rate_code': 'BOERUKQ',  # Bank of England Official Bank Rate
    },
    'Japan': {
        'credit_spread_proxy': 'Corporate lending rate - 10Y JGB',
        'lending_rate_code': 'FIND: Japan corporate lending rate',
    }
}

# ============================================================================
# CHECKLIST
# ============================================================================

def print_checklist():
    """Print what to search for"""
    print("="*80)
    print("  CHECKLIST: What to Search For")
    print("="*80)
    
    for country in ['Canada', 'UK', 'Japan']:
        print(f"\n{country.upper()}")
        print("-" * 40)
        
        print("\n  CRITICAL (Must-have for good R²):")
        for feature in NEED_TO_FIND[country]['CRITICAL (56% model importance)']:
            status = "[+]" if feature in KNOWN_EXISTING[country] else "[X]"
            print(f"    {status} {feature}")
        
        print("\n  IMPORTANT (Nice-to-have):")
        for feature in NEED_TO_FIND[country]['IMPORTANT (additional features)']:
            print(f"    [!] {feature}")
        
        if country in PROXY_OPTIONS:
            print(f"\n  IF NO CORPORATE BONDS FOUND:")
            print(f"    -> Use proxy: {PROXY_OPTIONS[country]['credit_spread_proxy']}")
            if 'FIND' not in PROXY_OPTIONS[country].get('prime_rate_code', ''):
                print(f"    -> Code: {PROXY_OPTIONS[country].get('bank_rate_code', 'TBD')}")
    
    print("\n" + "="*80)
    print("  SEARCH STRATEGY")
    print("="*80)
    print("\n  1. Run: python fred_series_finder_all_countries.py")
    print("  2. Check FRED manually:")
    print("     - https://fred.stlouisfed.org/")
    print("     - Search for missing codes above")
    print("  3. Check central banks:")
    print("     - Bank of Canada: https://www.bankofcanada.ca/rates/")
    print("     - Bank of England: https://www.bankofengland.co.uk/statistics")
    print("     - Bank of Japan: https://www.boj.or.jp/en/statistics/")
    print("  4. Fill in template in download_historical_data.py")
    print("="*80)

# ============================================================================
# VIABILITY MATRIX
# ============================================================================

def print_viability():
    """Expected model performance by scenario"""
    print("\n" + "="*80)
    print("  MODEL VIABILITY BY SCENARIO")
    print("="*80)
    
    print("\n  Scenario A: Full Features (Credit Spreads + Yield Curve)")
    print("    Expected Test R²: 0.15-0.25")
    print("    Status: [+] Best case")
    
    print("\n  Scenario B: Yield Curve Only (No Credit Spreads)")
    print("    Expected Test R²: 0.05-0.12")
    print("    Status: [!] Moderate - missing 43% importance")
    
    print("\n  Scenario C: Proxy Credit Spread")
    print("    Expected Test R²: 0.10-0.18")
    print("    Status: [!] Acceptable compromise")
    
    print("\n  Scenario D: Basic Features Only")
    print("    Expected Test R²: 0.00-0.10")
    print("    Status: [X] Poor (like USA V5)")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    print_checklist()
    print_viability()
    
    print("\n\nNEXT STEPS:")
    print("-" * 40)
    print("1. Run: python audit_existing_data.py")
    print("   → See what you already have")
    print()
    print("2. Run: python fred_series_finder_all_countries.py > fred_results.txt")
    print("   → Search FRED for missing codes")
    print()
    print("3. Update: download_historical_data.py")
    print("   → Add found FRED codes")
    print()
    print("4. Run: python download_historical_data.py")
    print("   → Download extended data")
    print()
    print("5. Check results and decide: proceed or use proxies")
    print("="*80)