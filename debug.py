import pandas as pd

df = pd.read_csv('data_preprocessing/resampled_data/germany_processed_unnormalized.csv', index_col=0, parse_dates=True)

print("="*60)
print("GERMANY GDP GROWTH CHECK")
print("="*60)

print("\nGDP Growth YoY stats:")
print(df['gdp_growth_yoy'].describe())

print("\nExtreme values:")
print(f"  Quarters with |growth| > 10%: {(abs(df['gdp_growth_yoy']) > 10).sum()}")
print(f"  Quarters with growth = 0: {(df['gdp_growth_yoy'] == 0).sum()}")

# Check around German reunification (1990-1991)
print("\n" + "="*60)
print("GERMAN REUNIFICATION PERIOD (1989-1993)")
print("="*60)
print(df.loc['1989-01-01':'1993-01-01', ['gdp_real', 'gdp_growth_yoy']])

# Financial features
print("\n" + "="*60)
print("FINANCIAL FEATURE COVERAGE")
print("="*60)
for feat in ['credit_spread', 'yield_curve_slope', 'yield_curve_curvature']:
    if feat in df.columns:
        coverage = df[feat].notna().sum() / len(df) * 100
        print(f"  {feat}: {coverage:.1f}%")