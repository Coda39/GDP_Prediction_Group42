"""
Exploratory Data Visualization for GDP Nowcasting Project
=========================================================
This script performs comprehensive exploratory analysis on G7 and BRICS economic data.

Phase 1: Data Understanding & Visualization
- GDP time series across all countries
- Correlation analysis between indicators
- Missing data patterns
- Cross-country comparisons
- Indicator distributions and outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data directory
DATA_DIR = Path(__file__).parent.parent / 'Data'
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Country lists
G7_COUNTRIES = {
    'united_states': 'USA',
    'canada': 'Canada',
    'france': 'France',
    'germany': 'Germany',
    'italy': 'Italy',
    'japan': 'Japan',
    'united_kingdom': 'UK'
}

BRICS_COUNTRIES = {
    'brazil': 'Brazil',
    'russia': 'Russia',
    'india': 'India',
    'china': 'China',
    'south_africa': 'South Africa'
}

def load_g7_data():
    """Load G7 quarterly data"""
    g7_data = {}

    for country_file, country_name in G7_COUNTRIES.items():
        # Use appropriate file path
        if country_file == 'united_states':
            file_path = DATA_DIR / 'united_states_fred_data.csv'
        elif country_file in ['germany', 'france', 'italy']:
            file_path = DATA_DIR / f'{country_file}_data_no_money_supply.csv'
        else:
            file_path = DATA_DIR / f'{country_file}_data.csv'

        if file_path.exists():
            df = pd.read_csv(file_path)
            # Convert first column to datetime
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: 'date'})
            df = df.set_index('date')

            # For USA daily data, resample to quarterly
            if country_file == 'united_states':
                df = df.resample('QE').last()

            df['country'] = country_name
            g7_data[country_name] = df
            print(f"Loaded {country_name}: {df.shape[0]} observations, {df.shape[1]} variables")

    return g7_data

def load_brics_data():
    """Load BRICS annual data"""
    brics_data = {}

    for country_file, country_name in BRICS_COUNTRIES.items():
        file_path = DATA_DIR / f'{country_file}_worldbank_data.csv'

        if file_path.exists():
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['country'] = country_name
            brics_data[country_name] = df
            print(f"Loaded {country_name}: {df.shape[0]} observations, {df.shape[1]} variables")

    return brics_data

def plot_gdp_timeseries(g7_data, brics_data):
    """Plot 1: GDP time series for all countries"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # G7 GDP (Real GDP)
    ax1 = axes[0]
    for country, df in g7_data.items():
        if 'gdp_real' in df.columns:
            ax1.plot(df.index, df['gdp_real'], label=country, linewidth=2, marker='o', markersize=3, alpha=0.7)

    ax1.set_title('G7 Countries: Real GDP Over Time (2000-2025)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Real GDP (Billions, Local Currency)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.5, label='2008 Financial Crisis')
    ax1.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.5, label='COVID-19 Pandemic')

    # BRICS GDP (Constant)
    ax2 = axes[1]
    for country, df in brics_data.items():
        if 'gdp_constant' in df.columns:
            ax2.plot(df.index, df['gdp_constant'], label=country, linewidth=2, marker='s', markersize=6, alpha=0.7)

    ax2.set_title('BRICS Countries: Real GDP Over Time (2000-2024)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('GDP Constant (Local Currency)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.5)
    ax2.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_gdp_timeseries_all_countries.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: GDP time series plot")
    plt.close()

def plot_gdp_growth_rates(g7_data, brics_data):
    """Plot 2: GDP growth rates (YoY %)"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # G7 Growth Rates
    ax1 = axes[0]
    for country, df in g7_data.items():
        if 'gdp_real' in df.columns:
            growth = df['gdp_real'].pct_change(periods=4) * 100  # YoY quarterly
            ax1.plot(growth.index, growth, label=country, linewidth=2, alpha=0.7)

    ax1.set_title('G7 Countries: Real GDP Growth Rate (YoY %)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('GDP Growth Rate (%)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.5)
    ax1.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.5)

    # BRICS Growth Rates
    ax2 = axes[1]
    for country, df in brics_data.items():
        if 'gdp_constant' in df.columns:
            growth = df['gdp_constant'].pct_change() * 100  # YoY annual
            ax2.plot(growth.index, growth, label=country, linewidth=2, marker='s', markersize=6, alpha=0.7)

    ax2.set_title('BRICS Countries: Real GDP Growth Rate (YoY %)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('GDP Growth Rate (%)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.5)
    ax2.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_gdp_growth_rates.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: GDP growth rates plot")
    plt.close()

def plot_correlation_heatmaps(g7_data):
    """Plot 3: Correlation heatmaps for key G7 countries"""
    # Select representative countries
    countries_to_plot = ['USA', 'Canada', 'Japan']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, country in enumerate(countries_to_plot):
        if country in g7_data:
            df = g7_data[country].copy()

            # Select key indicators (exclude country column)
            key_cols = ['gdp_real', 'unemployment_rate', 'cpi_annual_growth',
                       'interest_rate_short_term', 'industrial_production_index',
                       'exports_volume', 'imports_volume', 'household_consumption',
                       'capital_formation', 'employment_level']

            # Filter to existing columns
            available_cols = [col for col in key_cols if col in df.columns]
            corr_df = df[available_cols].corr()

            # Plot heatmap
            sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, vmin=-1, vmax=1, ax=axes[idx],
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(f'{country} - Indicator Correlations', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Correlation heatmaps")
    plt.close()

def plot_missing_data_patterns(g7_data, brics_data):
    """Plot 4: Missing data visualization"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # G7 Missing Data
    ax1 = axes[0]
    g7_missing = {}
    for country, df in g7_data.items():
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        g7_missing[country] = missing_pct

    # Combine into DataFrame
    g7_missing_df = pd.DataFrame(g7_missing).T
    # Select top 15 most missing indicators
    top_missing = g7_missing_df.max(axis=0).sort_values(ascending=False).head(15).index

    sns.heatmap(g7_missing_df[top_missing], annot=True, fmt='.1f', cmap='Reds',
               ax=ax1, cbar_kws={'label': 'Missing %'})
    ax1.set_title('G7 Countries: Missing Data Patterns (Top 15 Indicators)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Indicators', fontsize=11)
    ax1.set_ylabel('Country', fontsize=11)

    # BRICS Missing Data
    ax2 = axes[1]
    brics_missing = {}
    for country, df in brics_data.items():
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        brics_missing[country] = missing_pct

    brics_missing_df = pd.DataFrame(brics_missing).T

    sns.heatmap(brics_missing_df, annot=True, fmt='.1f', cmap='Reds',
               ax=ax2, cbar_kws={'label': 'Missing %'})
    ax2.set_title('BRICS Countries: Missing Data Patterns (All Indicators)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Indicators', fontsize=11)
    ax2.set_ylabel('Country', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_missing_data_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Missing data patterns")
    plt.close()

def plot_indicator_distributions(g7_data):
    """Plot 5: Distribution of key indicators (boxplots)"""
    # Combine all G7 data
    all_g7 = pd.concat([df for df in g7_data.values()], axis=0)

    # Select key indicators for analysis
    indicators = ['unemployment_rate', 'cpi_annual_growth', 'interest_rate_short_term',
                 'industrial_production_index']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, indicator in enumerate(indicators):
        if indicator in all_g7.columns:
            # Prepare data for boxplot by country
            data_for_plot = []
            labels = []
            for country, df in g7_data.items():
                if indicator in df.columns:
                    data_for_plot.append(df[indicator].dropna())
                    labels.append(country)

            axes[idx].boxplot(data_for_plot, labels=labels, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
            axes[idx].set_title(f'Distribution: {indicator.replace("_", " ").title()}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value', fontsize=10)
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_indicator_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Indicator distributions")
    plt.close()

def plot_leading_indicators(g7_data):
    """Plot 6: Leading indicators vs GDP for USA"""
    if 'USA' not in g7_data:
        print("⚠ USA data not available, skipping leading indicators plot")
        return

    usa_df = g7_data['USA'].copy()

    # Calculate GDP growth
    usa_df['gdp_growth'] = usa_df['gdp_real'].pct_change(periods=4) * 100

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    axes = axes.flatten()

    # Indicators to compare
    leading_indicators = [
        ('industrial_production_index', 'Industrial Production'),
        ('employment_level', 'Employment Level'),
        ('capital_formation', 'Capital Formation'),
        ('stock_market_index', 'Stock Market Index'),
        ('interest_rate_short_term', 'Short-term Interest Rate'),
        ('household_consumption', 'Household Consumption')
    ]

    for idx, (indicator, title) in enumerate(leading_indicators):
        if indicator in usa_df.columns:
            ax = axes[idx]

            # Normalize both series to 0-100 scale for comparison
            gdp_norm = (usa_df['gdp_growth'] - usa_df['gdp_growth'].min()) / (usa_df['gdp_growth'].max() - usa_df['gdp_growth'].min()) * 100
            indicator_series = usa_df[indicator].dropna()
            ind_norm = (indicator_series - indicator_series.min()) / (indicator_series.max() - indicator_series.min()) * 100

            ax.plot(gdp_norm.index, gdp_norm, label='GDP Growth (Normalized)', linewidth=2, color='blue', alpha=0.7)
            ax.plot(ind_norm.index, ind_norm, label=f'{title} (Normalized)', linewidth=2, color='orange', alpha=0.7)

            ax.set_title(f'{title} vs GDP Growth', fontsize=12, fontweight='bold')
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('Normalized Value (0-100)', fontsize=10)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.3)
            ax.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_leading_indicators_usa.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Leading indicators plot")
    plt.close()

def plot_cross_country_synchronization(g7_data):
    """Plot 7: Economic cycle synchronization across G7"""
    fig, ax = plt.subplots(figsize=(16, 8))

    for country, df in g7_data.items():
        if 'gdp_real' in df.columns:
            # Calculate rolling 4-quarter growth rate
            growth = df['gdp_real'].pct_change(periods=4).rolling(window=4).mean() * 100
            ax.plot(growth.index, growth, label=country, linewidth=2, alpha=0.7)

    ax.set_title('G7 Economic Cycle Synchronization (4Q Rolling Average Growth)',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('GDP Growth Rate (%, 4Q MA)', fontsize=12)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.5, linewidth=2, label='2008 Crisis')
    ax.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.5, linewidth=2, label='COVID-19')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_cross_country_synchronization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Cross-country synchronization plot")
    plt.close()

def plot_trade_balance_analysis(g7_data):
    """Plot 8: Trade balance trends"""
    fig, ax = plt.subplots(figsize=(16, 8))

    for country, df in g7_data.items():
        if 'trade_balance' in df.columns:
            ax.plot(df.index, df['trade_balance'], label=country, linewidth=2, alpha=0.7)

    ax.set_title('G7 Trade Balance Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Trade Balance (Billions)', fontsize=12)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.5)
    ax.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_trade_balance_trends.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Trade balance plot")
    plt.close()

def generate_summary_statistics(g7_data, brics_data):
    """Generate and save summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # G7 Summary
    print("\n--- G7 COUNTRIES ---")
    for country, df in g7_data.items():
        print(f"\n{country}:")
        print(f"  Period: {df.index.min()} to {df.index.max()}")
        print(f"  Observations: {len(df)}")
        print(f"  Variables: {df.shape[1]}")
        if 'gdp_real' in df.columns:
            growth = df['gdp_real'].pct_change(periods=4) * 100
            print(f"  Avg GDP Growth: {growth.mean():.2f}%")
            print(f"  GDP Volatility (Std): {growth.std():.2f}%")
        missing_pct = (df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100
        print(f"  Overall Missing Data: {missing_pct:.1f}%")

    # BRICS Summary
    print("\n--- BRICS COUNTRIES ---")
    for country, df in brics_data.items():
        print(f"\n{country}:")
        print(f"  Period: {df.index.min()} to {df.index.max()}")
        print(f"  Observations: {len(df)}")
        print(f"  Variables: {df.shape[1]}")
        if 'gdp_constant' in df.columns:
            growth = df['gdp_constant'].pct_change() * 100
            print(f"  Avg GDP Growth: {growth.mean():.2f}%")
            print(f"  GDP Volatility (Std): {growth.std():.2f}%")
        missing_pct = (df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100
        print(f"  Overall Missing Data: {missing_pct:.1f}%")

def main():
    """Main execution function"""
    print("="*80)
    print("EXPLORATORY DATA VISUALIZATION - GDP NOWCASTING PROJECT")
    print("="*80)

    # Load data
    print("\n[1/9] Loading G7 data...")
    g7_data = load_g7_data()

    print("\n[2/9] Loading BRICS data...")
    brics_data = load_brics_data()

    # Generate visualizations
    print("\n[3/9] Plotting GDP time series...")
    plot_gdp_timeseries(g7_data, brics_data)

    print("\n[4/9] Plotting GDP growth rates...")
    plot_gdp_growth_rates(g7_data, brics_data)

    print("\n[5/9] Creating correlation heatmaps...")
    plot_correlation_heatmaps(g7_data)

    print("\n[6/9] Analyzing missing data patterns...")
    plot_missing_data_patterns(g7_data, brics_data)

    print("\n[7/9] Plotting indicator distributions...")
    plot_indicator_distributions(g7_data)

    print("\n[8/9] Analyzing leading indicators...")
    plot_leading_indicators(g7_data)

    print("\n[9/9] Additional analyses...")
    plot_cross_country_synchronization(g7_data)
    plot_trade_balance_analysis(g7_data)

    # Generate summary statistics
    generate_summary_statistics(g7_data, brics_data)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
