import pandas as pd

df = pd.read_csv('../../Data/historical/usa_extended_with_financial_1980-2024.csv', index_col=0, parse_dates=True)

print('S&P 500 info:')
print(f'  Total quarters: {len(df)}')
print(f'  Non-null S&P 500: {df["stock_market_index"].notna().sum()}')
print(f'  Missing S&P 500: {df["stock_market_index"].isna().sum()}')
print(f'  First non-null: {df["stock_market_index"].first_valid_index()}')
print(f'  Last non-null: {df["stock_market_index"].last_valid_index()}')

print(f'\nFirst 20 quarters:')
print(df[["stock_market_index"]].head(20))

print(f'\nLast 20 quarters:')
print(df[["stock_market_index"]].tail(20))

print(f'\nSample of non-null values:')
print(df[df["stock_market_index"].notna()][["stock_market_index"]].head(10))