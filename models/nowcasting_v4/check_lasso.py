import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Load Canada data
print("=" * 60)
print("CANADA LASSO FEATURE ANALYSIS")
print("=" * 60)

# Load processed data
data_path = '../../data_preprocessing/resampled_data/canada_processed_normalized.csv'
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index)

# Define features (same as pipeline)
features = [
    'industrial_production_index', 'stock_market_index', 'interest_rate_short_term',
    'capital_formation', 'employment_level', 'unemployment_rate', 'cpi_annual_growth',
    'exports_volume', 'imports_volume', 'trade_balance',
    'gdp_volatility_4q', 'stock_volatility_4q', 'gdp_momentum', 'inflation_momentum',
    'high_inflation_regime', 'high_volatility_regime', 'inflation_x_volatility',
    'gdp_growth_yoy_lag4'
]

target = 'gdp_growth_yoy'

# Split data (2001-2018 train)
train_data = df[df.index.year <= 2018]
X_train = train_data[features].dropna()
y_train = train_data.loc[X_train.index, target]

# Train LASSO with alpha=0.01 (from your output)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# Show coefficients
coefs = pd.DataFrame({
    'Feature': features,
    'Coefficient': lasso.coef_
})
coefs = coefs[np.abs(coefs['Coefficient']) > 0.001].sort_values('Coefficient', key=abs, ascending=False)

print(f"\nUsed {len(coefs)} out of {len(features)} features:\n")
for _, row in coefs.iterrows():
    print(f"{row['Feature']:40s} {row['Coefficient']:8.4f}")

print("\n")

# UK
print("=" * 60)
print("UK LASSO FEATURE ANALYSIS")
print("=" * 60)

data_path = '../../data_preprocessing/resampled_data/uk_processed_normalized.csv'
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index)

train_data = df[df.index.year <= 2018]
X_train = train_data[features].dropna()
y_train = train_data.loc[X_train.index, target]

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

coefs = pd.DataFrame({
    'Feature': features,
    'Coefficient': lasso.coef_
})
coefs = coefs[np.abs(coefs['Coefficient']) > 0.001].sort_values('Coefficient', key=abs, ascending=False)

print(f"\nUsed {len(coefs)} out of {len(features)} features:\n")
for _, row in coefs.iterrows():
    print(f"{row['Feature']:40s} {row['Coefficient']:8.4f}")