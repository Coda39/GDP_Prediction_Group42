"""
Data Preprocessing Pipeline for GDP Nowcasting Project
=======================================================
This script implements comprehensive preprocessing for G7 economic data.

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
OUTPUT_DIR.mkdir(exist_ok=True)

# G7 Countries configuration
G7_COUNTRIES = {
    'USA': {
        'file': 'historical/usa_extended_with_financial_1980-2024.csv',
        'active': True
    },
    'Canada': {
        'file': 'canada_extended_with_bloomberg.csv',  # Will be created
        'active': True  # Set to True when ready
    },
    'UK': {
        'file': 'uk_extended_with_bloomberg.csv',
        'active': True
    },
    'Japan': {
        'file': 'japan_extended_with_bloomberg.csv',
        'active': True
    },
    'Germany': {
        'file': 'germany_extended_with_bloomberg.csv',
        'active': True
    },
    'France': {
        'file': 'france_extended_with_bloomberg.csv',
        'active': True
    },
    'Italy': {
        'file': 'italy_extended_with_bloomberg.csv',
        'active': True
    }
}

# Select which countries to process
# CRITICAL: Process countries SEPARATELY to avoid cross-contamination
# Set this to ['Canada'] when processing Canada, ['USA'] for USA, etc.
COUNTRIES_TO_PROCESS = ['Italy']  # ← CHANGE THIS FOR EACH COUNTRY

# Indicator categorization for imputation strategy
POLICY_INDICATORS = ['interest_rate_short_term', 'interest_rate_long_term', 
                     'overnight_rate', 'federal_funds_rate']
STOCK_INDICATORS = ['stock_market_index', 'sp500_index']
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

    def load_country_data(self, countries_to_process):
        """
        Load data for specified countries
        
        Parameters:
        - countries_to_process: list of country names to process
        """
        country_data = {}
        
        for country in countries_to_process:
            if country not in G7_COUNTRIES:
                self.log(f"⚠ {country} not in G7_COUNTRIES configuration")
                continue
                
            config = G7_COUNTRIES[country]
            if not config['active']:
                self.log(f"⚠ {country} is not active - skipping")
                continue
                
            file_path = DATA_DIR / config['file']
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Handle date column
                date_col = df.columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.rename(columns={date_col: 'date'})
                df = df.set_index('date')
                df['country'] = country
                
                country_data[country] = df
                self.log(f"✓ Loaded {country}: {df.shape} from {config['file']}")
            else:
                self.log(f"✗ File not found for {country}: {file_path}")
                
        return country_data

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
            df_quarterly = df.resample('QE').last()
            self.log(f"  → Resampled {country_name} from daily to quarterly: {original_shape} → {df_quarterly.shape}")

        elif freq and 'M' in freq:  # Monthly
            df_quarterly = df.resample('QE').last()
            self.log(f"  → Resampled {country_name} from monthly to quarterly: {original_shape} → {df_quarterly.shape}")

        elif freq and ('Y' in freq or 'A' in freq):  # Annual
            start_date = df.index.min()
            end_date = df.index.max()
            quarterly_index = pd.date_range(start=start_date, end=end_date, freq='QE')
            df_quarterly = df.reindex(quarterly_index)

            for col in df_quarterly.columns:
                if df_quarterly[col].notna().sum() > 3:
                    df_quarterly[col] = df_quarterly[col].interpolate(method='cubic', limit_direction='both')
                else:
                    df_quarterly[col] = df_quarterly[col].interpolate(method='linear', limit_direction='both')

            self.log(f"  → Resampled {country_name} from annual to quarterly (interpolated): {original_shape} → {df_quarterly.shape}")

        else:
            df_quarterly = df.resample('QE').last()
            self.log(f"  → Aligned {country_name} to quarter-end dates: {original_shape} → {df_quarterly.shape}")

        # Add country column back
        if country_col:
            df_quarterly['country'] = country_col

        return df_quarterly

    def impute_missing_data(self, df, country_name=''):
        """
        Impute missing data using intelligent strategies based on indicator type
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
                df_imputed[col] = df_imputed[col].fillna(method='ffill')
                df_imputed[col] = df_imputed[col].fillna(method='bfill')
                method_used = 'forward-fill'

            elif col in ECONOMIC_INDICATORS or col in RATE_INDICATORS or col in TRADE_INDICATORS:
                if df_imputed[col].notna().sum() > 3:
                    df_imputed[col] = df_imputed[col].interpolate(method='cubic', limit_direction='both')
                else:
                    df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')

                df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')
                method_used = 'interpolation'

            else:
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
        
        # 3.5 Lagged GDP Growth Rates (Added for V4)
        # These are lags of the GROWTH RATE, not the raw GDP level
        if 'gdp_growth_yoy' in df_featured.columns:
            df_featured['gdp_growth_yoy_lag1'] = df_featured['gdp_growth_yoy'].shift(1)
            df_featured['gdp_growth_yoy_lag4'] = df_featured['gdp_growth_yoy'].shift(4)
            feature_count += 2
            self.log(f"  → Created lagged GDP growth features for {country_name}")

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
        
        # 7. Regime Detection Features (Added for V3)
        # Helps detect economic regime changes (stable vs volatile, low vs high inflation)
        # Critical for handling 2022-2025 test period which differs from 2001-2018 training
        
        if 'gdp_growth_yoy' in df_featured.columns:
            # Volatility Indicators (detect unstable periods)
            df_featured['gdp_volatility_4q'] = df_featured['gdp_growth_yoy'].rolling(window=4, min_periods=2).std()
            feature_count += 1
            
        if 'stock_market_index' in df.columns:
            df_featured['stock_volatility_4q'] = df['stock_market_index'].pct_change().rolling(window=4, min_periods=2).std()
            feature_count += 1
        
        if 'gdp_growth_yoy' in df_featured.columns:
            # Momentum Indicators (detect acceleration/deceleration)
            df_featured['gdp_momentum'] = df_featured['gdp_growth_yoy'] - df_featured['gdp_growth_yoy'].shift(4)
            feature_count += 1
            
        if 'cpi_annual_growth' in df.columns:
            df_featured['inflation_momentum'] = df['cpi_annual_growth'] - df['cpi_annual_growth'].shift(4)
            feature_count += 1
        
        if 'cpi_annual_growth' in df.columns:
            # Regime Flags (binary indicators for different economic states)
            df_featured['high_inflation_regime'] = (df['cpi_annual_growth'] > 2.5).astype(int)
            feature_count += 1
        
        if 'gdp_volatility_4q' in df_featured.columns:
            df_featured['high_volatility_regime'] = (df_featured['gdp_volatility_4q'] > df_featured['gdp_volatility_4q'].quantile(0.75)).astype(int)
            feature_count += 1
        
        if 'cpi_annual_growth' in df.columns and 'gdp_volatility_4q' in df_featured.columns:
            # Interaction Terms (capture regime-specific dynamics)
            df_featured['inflation_x_volatility'] = df['cpi_annual_growth'] * df_featured['gdp_volatility_4q']
            feature_count += 1

        self.log(f"  → Created {feature_count} engineered features for {country_name}")

        # ====================================================================
        # **V6 FINANCIAL FEATURES** (Extended Data with Financial Indicators)
        # These are NOT GDP components - they're market prices that predict GDP
        # ====================================================================
        
        self.log(f"  → Creating V6 financial features for {country_name}...")
        v6_count = 0
        
        # 1. YIELD CURVE INDICATORS (Treasury market prices)
        if 'interest_rate_10y' in df.columns and 'interest_rate_2y' in df.columns:
            df_featured['yield_curve_slope'] = df['interest_rate_10y'] - df['interest_rate_2y']
            df_featured['yield_curve_inverted'] = (df_featured['yield_curve_slope'] < 0).astype(int)
            v6_count += 2
            self.log(f"    ✓ Yield curve slope & inversion")
        
        if all(col in df.columns for col in ['interest_rate_10y', 'interest_rate_5y', 'interest_rate_2y']):
            df_featured['yield_curve_curvature'] = (
                df['interest_rate_10y'] - 2*df['interest_rate_5y'] + df['interest_rate_2y']
            )
            v6_count += 1
            self.log(f"    ✓ Yield curve curvature")
        
        # 2. CREDIT MARKET INDICATORS (Corporate bond market prices)
        if 'corporate_bond_baa' in df.columns and 'corporate_bond_aaa' in df.columns:
            df_featured['credit_spread'] = df['corporate_bond_baa'] - df['corporate_bond_aaa']
            df_featured['credit_spread_change'] = df_featured['credit_spread'].diff()
            v6_count += 2
            self.log(f"    ✓ Credit spread & change")

        elif 'credit_spread' in df_featured.columns and 'credit_spread_change' not in df_featured.columns:
            df_featured['credit_spread_change'] = df_featured['credit_spread'].diff()
            v6_count += 1
            self.log(f"    ✓ Credit spread change (from existing credit_spread)")
        
        # 3. FINANCIAL STRESS INDICATORS (Banking stress)
        if 'ted_spread' in df.columns:
            ted_mean = df['ted_spread'].rolling(window=20, min_periods=1).mean()
            df_featured['high_financial_stress'] = (df['ted_spread'] > ted_mean).astype(int)
            v6_count += 1
            self.log(f"    ✓ Financial stress indicator")
        
        # 4. INTERACTION TERMS (Regime-specific dynamics)
        if 'unemployment_rate' in df.columns and 'cpi_annual_growth' in df.columns:
            df_featured['unemployment_x_inflation'] = df['unemployment_rate'] * df['cpi_annual_growth']
            v6_count += 1
            self.log(f"    ✓ Unemployment × Inflation")
        
        # 5. STOCK MARKET FEATURES (Now fixed with Yahoo Finance data!)
        if 'stock_market_index' in df.columns:
            df_featured['stock_returns_1q'] = df['stock_market_index'].pct_change(periods=1) * 100
            df_featured['stock_volatility_4q_v6'] = df_featured['stock_returns_1q'].rolling(window=4, min_periods=1).std()
            v6_count += 2
            
            if 'stock_volatility_4q_v6' in df_featured.columns:
                df_featured['risk_adjusted_returns'] = df_featured['stock_returns_1q'] / (df_featured['stock_volatility_4q_v6'] + 0.01)
                v6_count += 1
            self.log(f"    ✓ Stock returns, volatility, risk-adjusted")
        
        # 6. SECOND-ORDER FEATURES (Momentum indicators - NO GDP to avoid circularity!)
        if 'unemployment_rate' in df.columns:
            df_featured['unemployment_momentum_v6'] = df['unemployment_rate'].diff()
            df_featured['unemployment_vs_5y_avg'] = (
                df['unemployment_rate'] - df['unemployment_rate'].rolling(window=20, min_periods=1).mean()
            )
            v6_count += 2
            self.log(f"    ✓ Unemployment momentum & deviation")
        
        if 'cpi_annual_growth' in df.columns:
            df_featured['inflation_acceleration_v6'] = df['cpi_annual_growth'].diff()
            v6_count += 1
            self.log(f"    ✓ Inflation acceleration")
        
        # 7. Z-SCORE INDICATORS (Normalized measures - NO GDP to avoid circularity!)
        if 'cpi_annual_growth' in df.columns:
            inflation_mean = df['cpi_annual_growth'].rolling(window=20, min_periods=5).mean()
            inflation_std = df['cpi_annual_growth'].rolling(window=20, min_periods=5).std()
            df_featured['inflation_z_score'] = (df['cpi_annual_growth'] - inflation_mean) / (inflation_std + 0.01)
            v6_count += 1
            self.log(f"    ✓ Inflation Z-score")
        
        # 8. COMPOSITE INDICES (Combine multiple indicators)
        financial_indicators = []
        if 'yield_curve_slope' in df_featured.columns:
            financial_indicators.append('yield_curve_slope')
        if 'credit_spread' in df_featured.columns:
            financial_indicators.append('credit_spread')
        if 'stock_returns_1q' in df_featured.columns:
            financial_indicators.append('stock_returns_1q')
        
        if len(financial_indicators) >= 2:
            temp_df = df_featured[financial_indicators].copy()
            for col in temp_df.columns:
                col_mean = temp_df[col].mean()
                col_std = temp_df[col].std()
                if col_std > 0:
                    temp_df[col] = (temp_df[col] - col_mean) / col_std
            df_featured['financial_conditions_index'] = temp_df.mean(axis=1)
            v6_count += 1
            self.log(f"    ✓ Financial conditions index ({len(financial_indicators)} components)")
        
        real_indicators = []
        if 'building_permits' in df.columns:
            real_indicators.append('building_permits')
        if 'consumer_sentiment' in df.columns:
            real_indicators.append('consumer_sentiment')
        if 'capacity_utilization' in df.columns:
            real_indicators.append('capacity_utilization')
        
        if len(real_indicators) >= 2:
            temp_df = df_featured[real_indicators].copy()
            for col in temp_df.columns:
                col_mean = temp_df[col].mean()
                col_std = temp_df[col].std()
                if col_std > 0:
                    temp_df[col] = (temp_df[col] - col_mean) / col_std
            df_featured['real_activity_index'] = temp_df.mean(axis=1)
            v6_count += 1
            self.log(f"    ✓ Real activity index ({len(real_indicators)} components)")
        
        self.log(f"  → Created {v6_count} V6 features for {country_name} (NO GDP CIRCULARITY)")

        return df_featured

    def normalize_data(self, df, country_name='', exclude_cols=None):
        """
        Normalize data using z-score standardization (PER COUNTRY - CRITICAL!)
        """
        if exclude_cols is None:
            exclude_cols = ['country']

        df_normalized = df.copy()
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
        normalization_stats = {}

        for col in cols_to_normalize:
            if df[col].notna().sum() > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()

                if std_val > 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
                    normalization_stats[col] = {'mean': mean_val, 'std': std_val}

        self.log(f"  → Normalized {len(normalization_stats)} columns for {country_name}")

        # Save normalization statistics
        stats_df = pd.DataFrame(normalization_stats).T
        stats_df.to_csv(OUTPUT_DIR / f'{country_name.lower()}_normalization_stats.csv')

        return df_normalized

    def process_country(self, df, country_name):
        """
        Complete preprocessing pipeline for a single country
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

        # Step 3: Create features
        self.log("[3/6] Creating engineered features...")
        df_featured = self.create_features(df_imputed, country_name=country_name)

        # Step 4: Save un-normalized version
        self.log("[4/6] Saving un-normalized processed data...")
        output_file = OUTPUT_DIR / f'{country_name.lower()}_processed_unnormalized.csv'
        df_featured.to_csv(output_file)
        self.log(f"  ✓ Saved to {output_file}")

        # Step 5: Normalize data
        self.log("[5/6] Normalizing data...")
        df_normalized = self.normalize_data(df_featured, country_name=country_name)

        # Step 6: Save normalized version
        output_file_norm = OUTPUT_DIR / f'{country_name.lower()}_processed_normalized.csv'
        df_normalized.to_csv(output_file_norm)
        self.log(f"  ✓ Saved normalized data to {output_file_norm}")

        # Summary
        self.log(f"\nSummary for {country_name}:")
        self.log(f"  Original shape: {df.shape}")
        self.log(f"  Final shape: {df_normalized.shape}")
        self.log(f"  Date range: {df_normalized.index.min()} to {df_normalized.index.max()}")
        self.log(f"  Remaining missing values: {df_normalized.isnull().sum().sum()}")

        return df_normalized

def main():
    """Main execution function"""
    print("="*80)
    print("DATA PREPROCESSING PIPELINE - GDP NOWCASTING PROJECT")
    print("="*80)
    print(f"\n⚠️  PROCESSING COUNTRIES: {COUNTRIES_TO_PROCESS}")
    print("   (Change COUNTRIES_TO_PROCESS variable to process different countries)")
    print("="*80)

    # Initialize pipeline
    pipeline = PreprocessingPipeline(verbose=True)

    # Load data for specified countries
    print(f"\n[STEP 1] Loading data for {COUNTRIES_TO_PROCESS}...")
    country_data = pipeline.load_country_data(COUNTRIES_TO_PROCESS)

    if not country_data:
        print("\n❌ No data loaded. Check that:")
        print("   1. Country is set to 'active': True in G7_COUNTRIES")
        print("   2. Data file exists at specified path")
        return

    # Process each country
    print(f"\n[STEP 2] Processing {len(country_data)} countries...")
    processed_data = {}

    for country, df in country_data.items():
        processed_df = pipeline.process_country(df, country)
        processed_data[country] = processed_df

    # Generate summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print(f"Processed datasets saved to: {OUTPUT_DIR}")
    print("="*80)

    # Save log
    log_file = OUTPUT_DIR / f'preprocessing_log_{"-".join(COUNTRIES_TO_PROCESS)}.txt'
    with open(log_file, 'w') as f:
        f.write('\n'.join(pipeline.preprocessing_log))
    print(f"\nDetailed log saved to: {log_file}")

if __name__ == "__main__":
    main()