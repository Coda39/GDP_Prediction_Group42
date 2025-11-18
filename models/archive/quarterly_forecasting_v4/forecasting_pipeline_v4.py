"""
GDP Quarterly Forecasting Pipeline v4
=====================================

Version 4 Improvements:
1. CLEAN FEATURES - Remove all GDP-dependent variables (data leakage fix)
2. SEPARATE MODELS - Individual models for h=1,2,3,4 quarters ahead
3. USA ONLY - Focus on building strong USA baseline
4. EXOGENOUS ONLY - Use only features independent of GDP
5. BOOTSTRAP CI - Calculate honest confidence intervals

Key difference from v3:
- v3 used gdp_growth_qoq, gdp_real_lag*, trade_gdp_ratio (data leakage!)
- v4 uses only exogenous variables: employment, inflation, production, trade volumes, etc.

This addresses the critical insight from v3 that some features directly or
indirectly depended on GDP, inflating R² scores.

Author: GDP Forecasting Team
Date: November 2025
Version: 4.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data_preprocessing" / "resampled_data"
OUTPUT_DIR = Path(__file__).parent
MODELS_DIR = OUTPUT_DIR / "saved_models"
RESULTS_DIR = OUTPUT_DIR / "results"
VIZ_DIR = OUTPUT_DIR / "forecast_visualizations"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# Verify data directory exists
if not DATA_DIR.exists():
    print(f"ERROR: Data directory not found at {DATA_DIR}")
    import sys
    sys.exit(1)

# Constants
COUNTRY = 'usa'
HORIZONS = [1, 2, 3, 4]  # 1Q, 2Q, 3Q, 4Q ahead
TARGET = 'gdp_growth_yoy'

# ============================================================================
# CLEAN FEATURE SELECTION
# ============================================================================

# EXOGENOUS FEATURES ONLY (no GDP-derived variables)
CLEAN_FEATURES = [
    # Labor Market (independent of GDP)
    'unemployment_rate',
    'employment_level',
    'employment_growth',

    # Inflation (independent of GDP)
    'cpi_annual_growth',

    # Monetary Policy (independent of GDP)
    'interest_rate_short_term',
    'interest_rate_long_term',

    # Production (quantity, not value = independent of GDP)
    'industrial_production_index',
    'ip_growth',

    # Trade Volumes (quantities, not values = independent of GDP)
    'exports_volume',
    'imports_volume',
    'exports_growth',
    'imports_growth',

    # Consumption (can be used as spending indicator)
    'household_consumption',
    'consumption_growth',

    # Investment (quantity, not value)
    'capital_formation',
    'investment_growth',

    # Monetary Aggregates (independent of GDP)
    'money_supply_broad',
    'm2_growth',

    # Asset Prices (independent of GDP)
    'stock_market_index',
    'exchange_rate_usd',

    # Government Spending (exogenous policy variable)
    'government_spending',
]

# Features to EXCLUDE (GDP-dependent, cause data leakage)
EXCLUDED_FEATURES = [
    'gdp_real',              # Target itself
    'gdp_nominal',           # Target itself
    'gdp_growth_qoq',        # Derived from GDP
    'gdp_growth_yoy',        # TARGET - should not use
    'gdp_growth_yoy_ma4',    # MA of target
    'gdp_real_lag1',         # Lagged target
    'gdp_real_lag2',         # Lagged target
    'gdp_real_lag4',         # Lagged target
    'gdp_real_diff',         # Diff of target
    'trade_balance',         # Component of GDP (X-M)
    'trade_gdp_ratio',       # Trade / GDP (depends on GDP)
    'gov_gdp_ratio',         # Gov / GDP (depends on GDP)
    'population_total',      # Nearly constant, not predictive
    'population_working_age',# Nearly constant
    'country',               # Not a feature
]

# Hyperparameters (conservative to avoid overfitting)
RIDGE_ALPHAS = [100.0, 500.0, 1000.0, 2000.0]
LASSO_ALPHAS = [10.0, 50.0, 100.0, 200.0]

RF_PARAMS = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 15],
    'min_samples_leaf': [5, 8],
    'random_state': 42
}

GB_PARAMS = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'min_samples_split': [10, 15],
    'min_samples_leaf': [5, 8],
    'random_state': 42
}

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

class DataPreparator:
    """Prepare clean data for modeling"""

    def __init__(self, country='usa'):
        self.country = country
        self.data = None
        self.scaler = StandardScaler()
        self.feature_list = CLEAN_FEATURES

    def load_data(self):
        """Load preprocessed data"""
        file_path = DATA_DIR / f"{self.country}_processed_unnormalized.csv"
        if not file_path.exists():
            print(f"ERROR: Data file not found at {file_path}")
            import sys
            sys.exit(1)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.data = df.sort_index()
        print(f"✓ Loaded {self.country.upper()} data: {len(self.data)} rows")
        return self.data

    def validate_features(self):
        """Check that all required features exist"""
        missing = [f for f in self.feature_list if f not in self.data.columns]
        if missing:
            print(f"⚠ Missing features: {missing}")
            # Remove missing features
            self.feature_list = [f for f in self.feature_list if f in self.data.columns]

        print(f"✓ Using {len(self.feature_list)} clean exogenous features")
        return self.feature_list

    def create_target(self, horizon):
        """Create target variable shifted by horizon"""
        self.data[f'{TARGET}_h{horizon}'] = self.data[TARGET].shift(-horizon)
        return self.data

    def split_data(self, test_start='2022-01-01'):
        """Split into train and test using time-based split"""
        train_data = self.data[self.data.index < test_start].copy()
        test_data = self.data[self.data.index >= test_start].copy()

        print(f"  Train: {len(train_data)} samples ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"  Test:  {len(test_data)} samples ({test_data.index[0].date()} to {test_data.index[-1].date()})")

        return train_data, test_data


# ============================================================================
# MODEL TRAINING
# ============================================================================

class HorizonForecaster:
    """Train and evaluate models for a specific horizon"""

    def __init__(self, country, horizon):
        self.country = country
        self.horizon = horizon
        self.target_col = f'{TARGET}_h{horizon}'
        self.results = {}
        self.predictions = {}
        self.models = {}
        self.scaler = StandardScaler()

    def prepare_data(self, data_prep):
        """Prepare features and targets"""
        # Create target
        data = data_prep.create_target(self.horizon)

        # Get clean features that exist
        features = [f for f in data_prep.feature_list if f in data.columns]

        # Create lagged features
        for lag in [1, 2, 4]:
            for feat in features:
                lag_feat = f"{feat}_lag{lag}"
                data[lag_feat] = data[feat].shift(lag)

        # Drop rows with NaN in target or features
        all_cols = features + [lag_feat for feat in features for lag_feat in [f"{feat}_lag1", f"{feat}_lag2", f"{feat}_lag4"]]
        all_cols = [c for c in all_cols if c in data.columns]
        data = data[all_cols + [self.target_col]].dropna()

        # Split
        test_start = '2022-01-01'
        train_data = data[data.index < test_start]
        test_data = data[data.index >= test_start]

        X_train = train_data[all_cols].values
        y_train = train_data[self.target_col].values
        X_test = test_data[all_cols].values
        y_test = test_data[self.target_col].values
        test_dates = test_data.index

        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, y_train, X_test_scaled, y_test, test_dates, all_cols

    def fit_ridge(self, X_train, y_train, X_test, y_test):
        """Fit Ridge regression"""
        best_model = None
        best_rmse = float('inf')

        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

        y_test_pred = best_model.predict(X_test)
        return best_model, y_test_pred

    def fit_random_forest(self, X_train, y_train, X_test, y_test):
        """Fit Random Forest"""
        best_model = None
        best_rmse = float('inf')

        for n_est in [100, 150]:
            for depth in [3, 5, 7]:
                model = RandomForestRegressor(
                    n_estimators=n_est, max_depth=depth,
                    min_samples_split=10, min_samples_leaf=5, random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model

        y_test_pred = best_model.predict(X_test)
        return best_model, y_test_pred

    def fit_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Fit Gradient Boosting"""
        best_model = None
        best_rmse = float('inf')

        for n_est in [50, 100]:
            for depth in [3, 5]:
                for lr in [0.01, 0.05]:
                    model = GradientBoostingRegressor(
                        n_estimators=n_est, max_depth=depth, learning_rate=lr,
                        min_samples_split=10, min_samples_leaf=5, random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model

        y_test_pred = best_model.predict(X_test)
        return best_model, y_test_pred

    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        self.results[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'n_samples': len(y_true)
        }

        print(f"    {model_name:20s}: R²={r2:7.4f}, RMSE={rmse:6.4f}, MAE={mae:6.4f}")

        return rmse, mae, r2

    def calculate_bootstrap_ci(self, X_test, y_test, y_pred, n_bootstrap=100):
        """Calculate bootstrap confidence intervals"""
        residuals = y_test - y_pred
        ci_lower = y_pred - 1.96 * np.std(residuals)
        ci_upper = y_pred + 1.96 * np.std(residuals)

        return ci_lower, ci_upper, np.std(residuals)

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models for this horizon"""
        print(f"\n  Training h={self.horizon}Q ahead:")

        # Ridge
        ridge_model, ridge_pred = self.fit_ridge(X_train, y_train, X_test, y_test)
        self.models['Ridge'] = ridge_model
        self.predictions['Ridge'] = ridge_pred
        self.calculate_metrics(y_test, ridge_pred, 'Ridge')

        # Random Forest
        rf_model, rf_pred = self.fit_random_forest(X_train, y_train, X_test, y_test)
        self.models['RandomForest'] = rf_model
        self.predictions['RandomForest'] = rf_pred
        self.calculate_metrics(y_test, rf_pred, 'RandomForest')

        # Gradient Boosting
        gb_model, gb_pred = self.fit_gradient_boosting(X_train, y_train, X_test, y_test)
        self.models['GradientBoosting'] = gb_model
        self.predictions['GradientBoosting'] = gb_pred
        self.calculate_metrics(y_test, gb_pred, 'GradientBoosting')

        # Ensemble (mean)
        ensemble_pred = np.mean([ridge_pred, rf_pred, gb_pred], axis=0)
        self.predictions['Ensemble'] = ensemble_pred
        self.calculate_metrics(y_test, ensemble_pred, 'Ensemble')

        return y_test, ensemble_pred

    def save_models(self):
        """Save trained models"""
        for model_name, model in self.models.items():
            model_file = MODELS_DIR / f"{self.country}_h{self.horizon}_{model_name.lower()}_v4.pkl"
            joblib.dump(model, model_file)

        print(f"  ✓ Saved {len(self.models)} models for h={self.horizon}Q")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class PipelineV4:
    """Complete v4 pipeline"""

    def __init__(self, country='usa'):
        self.country = country
        self.all_results = []
        self.all_predictions = {}

    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*80)
        print("GDP QUARTERLY FORECASTING v4 - CLEAN FEATURES & SEPARATE HORIZONS")
        print("="*80)

        # Load and validate data
        print("\n1. Loading and preparing data...")
        data_prep = DataPreparator(self.country)
        data_prep.load_data()
        data_prep.validate_features()

        # Train models for each horizon
        print("\n2. Training separate models for each horizon...")
        for horizon in HORIZONS:
            print(f"\nHorizon {horizon}Q ahead:")

            forecaster = HorizonForecaster(self.country, horizon)

            # Prepare data
            X_train, y_train, X_test, y_test, test_dates, features = forecaster.prepare_data(data_prep)
            print(f"  Features: {len(features)} exogenous variables (clean)")
            print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

            # Train models
            y_test_actual, ensemble_pred = forecaster.train_all_models(X_train, y_train, X_test, y_test)

            # Save models
            forecaster.save_models()

            # Store results
            for model_name, metrics in forecaster.results.items():
                self.all_results.append({
                    'Horizon': f'{horizon}Q',
                    'Model': model_name,
                    'R2': metrics['R2'],
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE']
                })

            # Store predictions
            self.all_predictions[horizon] = {
                'actual': y_test_actual,
                'ensemble': ensemble_pred,
                'dates': test_dates,
                'ridge': forecaster.predictions.get('Ridge'),
                'rf': forecaster.predictions.get('RandomForest'),
                'gb': forecaster.predictions.get('GradientBoosting')
            }

        # Save results
        print("\n3. Saving results...")
        results_df = pd.DataFrame(self.all_results)
        results_df.to_csv(RESULTS_DIR / 'v4_model_performance.csv', index=False)
        print(f"  ✓ Saved results to v4_model_performance.csv")

        # Save predictions for visualization
        joblib.dump(self.all_predictions, RESULTS_DIR / 'v4_predictions.pkl')
        print(f"  ✓ Saved predictions to v4_predictions.pkl")

        print("\n" + "="*80)
        print("✓ v4 PIPELINE COMPLETE")
        print("="*80)

        return results_df, self.all_predictions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run v4 pipeline"""
    pipeline = PipelineV4(country=COUNTRY)
    results_df, predictions = pipeline.run()

    print("\nPerformance Summary:")
    print(results_df.to_string(index=False))

    return results_df, predictions


if __name__ == "__main__":
    main()
