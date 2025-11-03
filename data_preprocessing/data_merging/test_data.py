"""
Quick Test Script - Verify Phase 3 Data
========================================
Run this after downloading and merging to verify everything worked.

Usage: python test_phase3_data.py
"""

import pandas as pd
from pathlib import Path

# Path adjusted for running from data_preprocessing/data_merging/
DATA_DIR = Path(__file__).parent.parent.parent / 'Data'

def test_extended_file(country, filename):
    """Test if extended file exists and has correct structure"""
    print(f"\n{'='*60}")
    print(f"Testing: {country.upper()}")
    print(f"{'='*60}")
    
    file_path = DATA_DIR / filename
    
    # Check if file exists
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"✓ File exists: {filename}")
    
    # Load and check
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        print(f"\n📊 Data Summary:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Total quarters: {len(df)}")
        
        # Check date ranges
        df_1980s = df[df.index.year < 1990]
        df_1990s = df[(df.index.year >= 1990) & (df.index.year < 2000)]
        df_2000s = df[(df.index.year >= 2000) & (df.index.year < 2010)]
        df_2010s = df[(df.index.year >= 2010) & (df.index.year < 2020)]
        df_2020s = df[df.index.year >= 2020]
        
        print(f"\n📅 Data by Decade:")
        print(f"  1980s: {len(df_1980s)} quarters")
        print(f"  1990s: {len(df_1990s)} quarters")
        print(f"  2000s: {len(df_2000s)} quarters")
        print(f"  2010s: {len(df_2010s)} quarters")
        print(f"  2020s: {len(df_2020s)} quarters")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        missing_pct = (missing / (len(df) * len(df.columns))) * 100
        
        print(f"\n🔍 Data Quality:")
        print(f"  Missing values: {missing} ({missing_pct:.1f}%)")
        
        # Verify historical data
        if df.index.min().year <= 1980:
            print(f"\n✅ SUCCESS: Historical data present (starts in {df.index.min().year})")
            
            # Check we have enough quarters
            if len(df) >= 160:  # Should have ~183
                print(f"✅ Good data length: {len(df)} quarters")
            else:
                print(f"⚠️  Short data: {len(df)} quarters (expected ~183)")
            
            return True
        else:
            print(f"\n❌ WARNING: No historical data (starts in {df.index.min().year}, expected 1980)")
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    """Test all extended files"""
    print("="*60)
    print("PHASE 3 DATA VERIFICATION TEST")
    print("="*60)
    print("\nThis script checks if historical data was successfully")
    print("downloaded and merged into your Data/ files.")
    
    # Files to test (USA only for now)
    files_to_test = {
        'USA': 'united_states_fred_data_extended.csv',
        # Add others after USA works:
        # 'Canada': 'canada_data_extended.csv',
        # 'Japan': 'japan_data_extended.csv',
        # 'UK': 'united_kingdom_data_extended.csv',
    }
    
    results = {}
    for country, filename in files_to_test.items():
        results[country] = test_extended_file(country, filename)
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n✅ Successfully verified: {success_count}/{total_count} countries\n")
    
    for country, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {country:10s}: {status}")
    
    if success_count == total_count:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"Your historical data is ready to use.")
        print(f"\nNext steps:")
        print(f"1. Update preprocessing_pipeline.py")
        print(f"2. Re-run preprocessing")
        print(f"3. Re-train models")
    elif success_count > 0:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"Some files are ready, others may need attention.")
    else:
        print(f"\n❌ NO FILES VERIFIED")
        print(f"Please check that you ran:")
        print(f"1. python download_historical_data.py")
        print(f"2. python merge_historical_data.py")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()