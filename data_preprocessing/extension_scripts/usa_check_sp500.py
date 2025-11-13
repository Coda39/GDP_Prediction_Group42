"""
Check S&P 500 data quality in USA extended dataset
"""

import pandas as pd
from pathlib import Path

# File path
DATA_DIR = Path('../../Data')
INPUT_FILE = DATA_DIR / 'extended' / 'usa_extended_with_financial_1980-2024.csv'

print("="*80)
print("S&P 500 DATA QUALITY CHECK")
print("="*80)

print(f"\nLoading: {INPUT_FILE}")

if not INPUT_FILE.exists():
    print(f"❌ File not found: {INPUT_FILE}")
    exit(1)

df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

print('\n📊 S&P 500 Summary:')
print(f'  Total quarters: {len(df)}')
print(f'  Non-null S&P 500: {df["stock_market_index"].notna().sum()}')
print(f'  Missing S&P 500: {df["stock_market_index"].isna().sum()}')
print(f'  Coverage: {df["stock_market_index"].notna().sum() / len(df) * 100:.1f}%')

if df["stock_market_index"].notna().any():
    print(f'  First non-null: {df["stock_market_index"].first_valid_index()}')
    print(f'  Last non-null: {df["stock_market_index"].last_valid_index()}')
    print(f'  Min value: {df["stock_market_index"].min():.2f}')
    print(f'  Max value: {df["stock_market_index"].max():.2f}')

print(f'\n📅 First 20 quarters:')
print(df[["stock_market_index"]].head(20))

print(f'\n📅 Last 20 quarters:')
print(df[["stock_market_index"]].tail(20))

if df["stock_market_index"].notna().any():
    print(f'\n✅ Sample of non-null values:')
    print(df[df["stock_market_index"].notna()][["stock_market_index"]].head(10))

print("\n" + "="*80)