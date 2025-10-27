"""
Data Preprocessing Pipeline for GDP Nowcasting Project
=======================================================
This script implements comprehensive preprocessing for G7 and BRICS economic data.

Preprocessing Steps:
1. Frequency harmonization (resample to quarterly)
2. Missing data imputation (forward-fill + interpolation)
3. Feature engineering (growth rates, lags, ratios)
4. Scaling and normalization
5. Stationarity analysis
6. Save processed datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import interpolate
from scipy.stats import jarque_bera
import warnings
warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directories
DATA_DIR = Path(__file__).parent.parent / 'Data'
OUTPUT_DIR = Path(__file__).parent / 'resampled_data'
VIZ_DIR = Path(__file__).parent / 'preprocessing_figures'
VIZ_DIR.mkdir(exist_ok=True)

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

# Indicator categorization for imputation strategy
POLICY_INDICATORS = ['interest_rate_short_term', 'interest_rate_long_term']
STOCK_INDICATORS = ['stock_market_index']
ECONOMIC_INDICATORS = ['gdp_real', 'gdp_nominal', 'industrial_production_index',
                       'exports_volume', 'imports_volume', 'household_consumption',
                       'capital_formation', 'employment_level', 'money_supply_broad',
                       'government_spending', 'population_total', 'population_working_age',
                       'gdp_constant', 'gdp_current', 'gross_capital_formation',
                       'government_expenditure', 'population', 'urban_population',
                       'gni_per_capita', 'fdi_inflows', 'external_debt',
                       'manufacturing_value_added', 'services_value_added']
RATE_INDICATORS = ['unemployment_rate', 'cpi_all_items', 'cpi_annual_growth',
                   'exchange_rate_usd', 'unemployment', 'inflation', 'interest_rate',
                   'exchange_rate']
TRADE_INDICATORS = ['trade_balance', 'exports', 'imports']

class PreprocessingPipeline:
    """Complete preprocessing pipeline for economic data"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.preprocessing_log = []

    def log(self, message):
        """Log preprocessing steps"""
        if self.verbose:
            print(message)
        self.preprocessing_log.append(message)

    def load_g7_data(self):
        """Load G7 data from CSV files"""
        g7_data = {}

        for country_file, country_name in G7_COUNTRIES.items():
            # Determine file path
            if country_file == 'united_states':
                file_path = DATA_DIR / 'united_states_fred_data.csv'
            elif country_file in ['germany', 'france', 'italy']:
                file_path = DATA_DIR / f'{country_file}_data_no_money_supply.csv'
            else:
                file_path = DATA_DIR / f'{country_file}_data.csv'

            if file_path.exists():
                df = pd.read_csv(file_path)
                date_col = df.columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.rename(columns={date_col: 'date'})
                df = df.set_index('date')
                df['country'] = country_name
                g7_data[country_name] = df
                self.log(f"✓ Loaded {country_name}: {df.shape}")

        return g7_data

    def load_brics_data(self):
        """Load BRICS data from CSV files"""
        brics_data = {}

        for country_file, country_name in BRICS_COUNTRIES.items():
            file_path = DATA_DIR / f'{country_file}_worldbank_data.csv'

            if file_path.exists():
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df['country'] = country_name
                brics_data[country_name] = df
                self.log(f"✓ Loaded {country_name}: {df.shape}")

        return brics_data

    def resample_to_quarterly(self, df, method='last', country_name=''):
        """
        Resample data to quarterly frequency

        Parameters:
        - df: DataFrame with datetime index
        - method: 'last' for end-of-quarter, 'mean' for quarterly average
        - country_name: for logging
        """
        original_shape = df.shape

        # Separate country column if present
        country_col = None
        if 'country' in df.columns:
            country_col = df['country'].iloc[0]
            df = df.drop('country', axis=1)

        # Determine current frequency
        freq = pd.infer_freq(df.index)

        if freq and ('D' in freq or 'B' in freq):  # Daily or business daily
            # Resample daily to quarterly (end of quarter)
            df_quarterly = df.resample('QE').last()
            self.log(f"  → Resampled {country_name} from daily to quarterly: {original_shape} → {df_quarterly.shape}")

        elif freq and 'M' in freq:  # Monthly
            # Resample monthly to quarterly (end of quarter)
            df_quarterly = df.resample('QE').last()
            self.log(f"  → Resampled {country_name} from monthly to quarterly: {original_shape} → {df_quarterly.shape}")

        elif freq and ('Y' in freq or 'A' in freq):  # Annual
            # Upsample annual to quarterly using interpolation
            # First, create quarterly index spanning the data range
            start_date = df.index.min()
            end_date = df.index.max()
            quarterly_index = pd.date_range(start=start_date, end=end_date, freq='QE')

            # Reindex to quarterly
            df_quarterly = df.reindex(quarterly_index)

            # Interpolate using cubic spline for smooth transition
            for col in df_quarterly.columns:
                if df_quarterly[col].notna().sum() > 3:  # Need at least 4 points for cubic
                    df_quarterly[col] = df_quarterly[col].interpolate(method='cubic', limit_direction='both')
                else:
                    df_quarterly[col] = df_quarterly[col].interpolate(method='linear', limit_direction='both')

            self.log(f"  → Resampled {country_name} from annual to quarterly (interpolated): {original_shape} → {df_quarterly.shape}")

        else:
            # Already quarterly or irregular frequency - just ensure quarter-end alignment
            df_quarterly = df.resample('QE').last()
            self.log(f"  → Aligned {country_name} to quarter-end dates: {original_shape} → {df_quarterly.shape}")

        # Add country column back
        if country_col:
            df_quarterly['country'] = country_col

        return df_quarterly

    def impute_missing_data(self, df, country_name=''):
        """
        Impute missing data using intelligent strategies based on indicator type

        Strategy:
        - Policy rates: Forward-fill (rates persist until changed)
        - Stock indices: Forward-fill (carry last observation)
        - Economic volumes: Interpolation (linear/cubic)
        - Rates/percentages: Interpolation
        """
        original_missing = df.isnull().sum().sum()
        df_imputed = df.copy()

        for col in df_imputed.columns:
            if col == 'country':
                continue

            missing_count = df_imputed[col].isnull().sum()
            if missing_count == 0:
                continue

            # Determine imputation strategy
            if col in POLICY_INDICATORS or col in STOCK_INDICATORS:
                # Forward-fill for policy rates and stock indices
                df_imputed[col] = df_imputed[col].fillna(method='ffill')
                # If still missing at start, backfill
                df_imputed[col] = df_imputed[col].fillna(method='bfill')
                method_used = 'forward-fill'

            elif col in ECONOMIC_INDICATORS or col in RATE_INDICATORS or col in TRADE_INDICATORS:
                # Interpolate for economic indicators
                if df_imputed[col].notna().sum() > 3:
                    df_imputed[col] = df_imputed[col].interpolate(method='cubic', limit_direction='both')
                else:
                    df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')

                # Fill any remaining with forward/backward fill
                df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')
                method_used = 'interpolation'

            else:
                # Default: linear interpolation + forward-fill
                df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
                df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')
                method_used = 'default'

            remaining_missing = df_imputed[col].isnull().sum()
            if missing_count > 0 and self.verbose and remaining_missing < missing_count:
                self.log(f"    {col}: {missing_count} missing → {remaining_missing} after {method_used}")

        final_missing = df_imputed.isnull().sum().sum()
        self.log(f"  → {country_name} imputation: {original_missing} missing → {final_missing} remaining")

        return df_imputed

    def create_features(self, df, country_name=''):
        """
        Create engineered features:
        - Growth rates (YoY %)
        - Lagged values (t-1, t-2, t-4)
        - Ratios (trade/GDP, etc.)
        - Moving averages
        """
        df_featured = df.copy()
        feature_count = 0

        # 1. GDP Growth Rates
        if 'gdp_real' in df.columns:
            df_featured['gdp_growth_qoq'] = df['gdp_real'].pct_change() * 100
            df_featured['gdp_growth_yoy'] = df['gdp_real'].pct_change(periods=4) * 100
            feature_count += 2

        if 'gdp_constant' in df.columns:
            df_featured['gdp_growth_yoy'] = df['gdp_constant'].pct_change() * 100
            feature_count += 1

        # 2. Growth rates for key indicators
        growth_indicators = {
            'industrial_production_index': 'ip_growth',
            'employment_level': 'employment_growth',
            'household_consumption': 'consumption_growth',
            'capital_formation': 'investment_growth',
            'exports_volume': 'exports_growth',
            'imports_volume': 'imports_growth',
            'money_supply_broad': 'm2_growth'
        }

        for original, new_name in growth_indicators.items():
            if original in df.columns:
                df_featured[new_name] = df[original].pct_change(periods=4) * 100
                feature_count += 1

        # 3. Lagged features (t-1, t-2, t-4)
        lag_indicators = ['gdp_real', 'gdp_constant', 'unemployment_rate', 'unemployment',
                         'cpi_annual_growth', 'inflation', 'industrial_production_index',
                         'interest_rate_short_term', 'interest_rate']

        for indicator in lag_indicators:
            if indicator in df.columns:
                for lag in [1, 2, 4]:
                    df_featured[f'{indicator}_lag{lag}'] = df[indicator].shift(lag)
                    feature_count += 1

        # 4. Ratios
        if 'trade_balance' in df.columns and 'gdp_real' in df.columns:
            df_featured['trade_gdp_ratio'] = (df['trade_balance'] / df['gdp_real']) * 100
            feature_count += 1

        if 'trade_balance' in df.columns and 'gdp_constant' in df.columns:
            df_featured['trade_gdp_ratio'] = (df['trade_balance'] / df['gdp_constant']) * 100
            feature_count += 1

        if 'government_spending' in df.columns and 'gdp_real' in df.columns:
            df_featured['gov_gdp_ratio'] = (df['government_spending'] / df['gdp_real']) * 100
            feature_count += 1

        if 'government_expenditure' in df.columns and 'gdp_constant' in df.columns:
            df_featured['gov_gdp_ratio'] = (df['government_expenditure'] / df['gdp_constant']) * 100
            feature_count += 1

        # 5. Moving averages (4-quarter MA for smoothing)
        ma_indicators = ['gdp_growth_yoy', 'unemployment_rate', 'unemployment',
                        'cpi_annual_growth', 'inflation']

        for indicator in ma_indicators:
            if indicator in df_featured.columns:
                df_featured[f'{indicator}_ma4'] = df_featured[indicator].rolling(window=4).mean()
                feature_count += 1

        # 6. Differences (for non-stationary series)
        diff_indicators = ['gdp_real', 'gdp_constant', 'employment_level',
                          'industrial_production_index']

        for indicator in diff_indicators:
            if indicator in df.columns:
                df_featured[f'{indicator}_diff'] = df[indicator].diff()
                feature_count += 1

        self.log(f"  → Created {feature_count} engineered features for {country_name}")

        return df_featured

    def normalize_data(self, df, country_name='', exclude_cols=None):
        """
        Normalize data using z-score standardization (per country)

        Parameters:
        - df: DataFrame to normalize
        - country_name: for logging
        - exclude_cols: list of columns to exclude from normalization (e.g., 'country')
        """
        if exclude_cols is None:
            exclude_cols = ['country']

        df_normalized = df.copy()

        # Columns to normalize
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]

        # Calculate statistics for logging
        normalization_stats = {}

        for col in cols_to_normalize:
            if df[col].notna().sum() > 0:  # Only if data exists
                mean_val = df[col].mean()
                std_val = df[col].std()

                if std_val > 0:  # Avoid division by zero
                    df_normalized[col] = (df[col] - mean_val) / std_val
                    normalization_stats[col] = {'mean': mean_val, 'std': std_val}

        self.log(f"  → Normalized {len(normalization_stats)} columns for {country_name}")

        # Save normalization statistics for potential inverse transform
        stats_df = pd.DataFrame(normalization_stats).T
        stats_df.to_csv(OUTPUT_DIR / f'{country_name.lower().replace(" ", "_")}_normalization_stats.csv')

        return df_normalized

    def analyze_stationarity(self, df, country_name=''):
        """
        Visual stationarity analysis using rolling statistics

        For each key indicator:
        - Plot original series
        - Plot rolling mean (4-quarter window)
        - Plot rolling std (4-quarter window)
        """
        indicators_to_check = ['gdp_real', 'gdp_constant', 'gdp_growth_yoy',
                              'unemployment_rate', 'unemployment',
                              'cpi_annual_growth', 'inflation',
                              'industrial_production_index']

        available_indicators = [ind for ind in indicators_to_check if ind in df.columns]

        if not available_indicators:
            self.log(f"  ⚠ No key indicators found for stationarity analysis in {country_name}")
            return

        # Create subplots
        n_indicators = len(available_indicators)
        n_cols = 2
        n_rows = (n_indicators + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_indicators == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, indicator in enumerate(available_indicators):
            ax = axes[idx]

            # Original series
            ax.plot(df.index, df[indicator], label='Original', linewidth=1.5, alpha=0.7)

            # Rolling mean (4-quarter)
            rolling_mean = df[indicator].rolling(window=4).mean()
            ax.plot(df.index, rolling_mean, label='Rolling Mean (4Q)', linewidth=2, color='red')

            # Rolling std (4-quarter)
            rolling_std = df[indicator].rolling(window=4).std()
            ax.fill_between(df.index,
                           rolling_mean - rolling_std,
                           rolling_mean + rolling_std,
                           alpha=0.2, color='red', label='±1 Std Dev')

            ax.set_title(f'{indicator.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_indicators, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'{country_name} - Stationarity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(VIZ_DIR / f'stationarity_{country_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"  ✓ Saved stationarity plot for {country_name}")

    def process_country(self, df, country_name, is_brics=False):
        """
        Complete preprocessing pipeline for a single country

        Steps:
        1. Resample to quarterly
        2. Impute missing data
        3. Create features
        4. Normalize data
        5. Analyze stationarity
        6. Save processed data
        """
        self.log(f"\n{'='*60}")
        self.log(f"Processing: {country_name}")
        self.log(f"{'='*60}")

        # Step 1: Resample to quarterly
        self.log("[1/6] Resampling to quarterly frequency...")
        df_quarterly = self.resample_to_quarterly(df, country_name=country_name)

        # Step 2: Impute missing data
        self.log("[2/6] Imputing missing data...")
        df_imputed = self.impute_missing_data(df_quarterly, country_name=country_name)

        # Step 3: Create features (before normalization)
        self.log("[3/6] Creating engineered features...")
        df_featured = self.create_features(df_imputed, country_name=country_name)

        # Step 4: Save un-normalized version (for interpretability)
        self.log("[4/6] Saving un-normalized processed data...")
        output_file = OUTPUT_DIR / f'{country_name.lower().replace(" ", "_")}_processed_unnormalized.csv'
        df_featured.to_csv(output_file)
        self.log(f"  ✓ Saved to {output_file}")

        # Step 5: Normalize data
        self.log("[5/6] Normalizing data...")
        df_normalized = self.normalize_data(df_featured, country_name=country_name)

        # Step 6: Analyze stationarity (use un-normalized for visualization)
        self.log("[6/6] Analyzing stationarity...")
        self.analyze_stationarity(df_featured, country_name=country_name)

        # Save final normalized version
        output_file_norm = OUTPUT_DIR / f'{country_name.lower().replace(" ", "_")}_processed_normalized.csv'
        df_normalized.to_csv(output_file_norm)
        self.log(f"  ✓ Saved normalized data to {output_file_norm}")

        # Summary
        self.log(f"\nSummary for {country_name}:")
        self.log(f"  Original shape: {df.shape}")
        self.log(f"  Final shape: {df_normalized.shape}")
        self.log(f"  Date range: {df_normalized.index.min()} to {df_normalized.index.max()}")
        self.log(f"  Remaining missing values: {df_normalized.isnull().sum().sum()}")

        return df_normalized

    def create_combined_datasets(self, processed_data):
        """
        Create combined datasets for G7 and BRICS
        """
        self.log(f"\n{'='*60}")
        self.log("Creating combined datasets...")
        self.log(f"{'='*60}")

        # Separate G7 and BRICS
        g7_countries = ['USA', 'Canada', 'France', 'Germany', 'Italy', 'Japan', 'UK']
        brics_countries = ['Brazil', 'Russia', 'India', 'China', 'South Africa']

        g7_data = {k: v for k, v in processed_data.items() if k in g7_countries}
        brics_data = {k: v for k, v in processed_data.items() if k in brics_countries}

        # Combine G7
        if g7_data:
            g7_combined = pd.concat([df.assign(country=country) for country, df in g7_data.items()])
            g7_combined.to_csv(OUTPUT_DIR / 'g7_combined_processed.csv')
            self.log(f"  ✓ Saved G7 combined dataset: {g7_combined.shape}")

        # Combine BRICS
        if brics_data:
            brics_combined = pd.concat([df.assign(country=country) for country, df in brics_data.items()])
            brics_combined.to_csv(OUTPUT_DIR / 'brics_combined_processed.csv')
            self.log(f"  ✓ Saved BRICS combined dataset: {brics_combined.shape}")

        # Combine ALL
        all_combined = pd.concat([df.assign(country=country) for country, df in processed_data.items()])
        all_combined.to_csv(OUTPUT_DIR / 'all_countries_combined_processed.csv')
        self.log(f"  ✓ Saved ALL countries combined dataset: {all_combined.shape}")

    def generate_preprocessing_report(self):
        """
        Generate summary statistics and visualizations
        """
        self.log(f"\n{'='*60}")
        self.log("Generating preprocessing report...")
        self.log(f"{'='*60}")

        # Load all processed files
        processed_files = list(OUTPUT_DIR.glob('*_processed_unnormalized.csv'))

        summary_stats = []

        for file in processed_files:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            country = file.stem.replace('_processed_unnormalized', '').replace('_', ' ').title()

            stats = {
                'Country': country,
                'Observations': len(df),
                'Features': len(df.columns),
                'Start Date': df.index.min(),
                'End Date': df.index.max(),
                'Missing Values': df.isnull().sum().sum(),
                'Missing %': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            }
            summary_stats.append(stats)

        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(OUTPUT_DIR / 'preprocessing_summary.csv', index=False)
        self.log(f"  ✓ Saved preprocessing summary")

        return summary_df

def main():
    """Main execution function"""
    print("="*80)
    print("DATA PREPROCESSING PIPELINE - GDP NOWCASTING PROJECT")
    print("="*80)

    # Initialize pipeline
    pipeline = PreprocessingPipeline(verbose=True)

    # Load data
    print("\n[STEP 1] Loading raw data...")
    g7_data = pipeline.load_g7_data()
    brics_data = pipeline.load_brics_data()

    # Process each country
    print("\n[STEP 2] Processing G7 countries...")
    processed_data = {}

    for country, df in g7_data.items():
        processed_df = pipeline.process_country(df, country, is_brics=False)
        processed_data[country] = processed_df

    print("\n[STEP 3] Processing BRICS countries...")
    for country, df in brics_data.items():
        processed_df = pipeline.process_country(df, country, is_brics=True)
        processed_data[country] = processed_df

    # Create combined datasets
    print("\n[STEP 4] Creating combined datasets...")
    pipeline.create_combined_datasets(processed_data)

    # Generate report
    print("\n[STEP 5] Generating preprocessing report...")
    summary_df = pipeline.generate_preprocessing_report()

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print(f"Processed datasets saved to: {OUTPUT_DIR}")
    print(f"Stationarity plots saved to: {VIZ_DIR}")
    print("="*80)

    print("\nPreprocessing Summary:")
    print(summary_df.to_string(index=False))

    # Save log
    log_file = OUTPUT_DIR / 'preprocessing_log.txt'
    with open(log_file, 'w') as f:
        f.write('\n'.join(pipeline.preprocessing_log))
    print(f"\nDetailed log saved to: {log_file}")

if __name__ == "__main__":
    main()
