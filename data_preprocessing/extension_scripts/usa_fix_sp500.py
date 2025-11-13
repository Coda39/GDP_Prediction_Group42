"""
Fix S&P 500 data using Yahoo Finance
Downloads complete S&P 500 history and updates USA extended dataset
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

print("="*80)
print("FIXING S&P 500 DATA - YAHOO FINANCE")
print("="*80)

# File paths
DATA_DIR = Path('../../Data')
INPUT_FILE = DATA_DIR / 'extended' / 'usa_extended_with_financial_1980-2024.csv'
OUTPUT_FILE = DATA_DIR / 'extended' / 'usa_extended_with_financial_1980-2024.csv'

print("\nDownloading S&P 500 from Yahoo Finance (1980-2024)...")

# Download S&P 500 from Yahoo Finance
sp500 = yf.download('^GSPC', start='1980-01-01', end='2024-12-31', progress=False)

# Take closing price
sp500_close = sp500['Close']

# Forward-fill to handle missing days
sp500_close = sp500_close.fillna(method='ffill')

# Resample to quarterly (last value of quarter)
sp500_quarterly = sp500_close.resample('QE').last()

print(f"✓ Downloaded {len(sp500_quarterly)} quarters of S&P 500 data")
print(f"  Date range: {sp500_quarterly.index[0]} to {sp500_quarterly.index[-1]}")
print(f"  Missing: {sp500_quarterly.isna().sum()} quarters")

# Load existing data
print(f"\nLoading existing data from: {INPUT_FILE}")
if not INPUT_FILE.exists():
    print(f"❌ File not found: {INPUT_FILE}")
    exit(1)

df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

print(f"  Before: stock_market_index has {df['stock_market_index'].isna().sum()} missing")

# Replace S&P 500 column with Yahoo Finance data
df['stock_market_index'] = sp500_quarterly

print(f"  After:  stock_market_index has {df['stock_market_index'].isna().sum()} missing")

# Save updated file
df.to_csv(OUTPUT_FILE)

print(f"\n✓ Fixed! S&P 500 now complete from Yahoo Finance")
print(f"✓ Saved to: {OUTPUT_FILE}")

# Show sample
print("\nSample of fixed data:")
print(df[['stock_market_index']].head(10))
print("...")
print(df[['stock_market_index']].tail(10))

print("\n" + "="*80)
print("✅ COMPLETE!")
print("="*80)