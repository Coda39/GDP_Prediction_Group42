"""
GDP Nowcasting Pipeline - V7 (Financial Indicators)
====================================================
This script trains and evaluates multiple models for nowcasting current quarter GDP growth.

VERSION 6 CHANGES (October 31, 2025):
- Added financial market indicators (yield curve, credit spreads, TED spread)
- Added real activity indicators (consumer sentiment, building permits, capacity utilization)
- Added interaction terms (unemployment × inflation) and composite indices
- Extended training data from 72 quarters (2000-2018) to 180 quarters (1980-2024)
- These are NOT GDP components - they're forward-looking market prices

NEW V7 FEATURES (~18-20 features):
Financial Indicators:
  - Yield curve: slope, inversion, curvature
  - Credit spreads: BAA-AAA spread and changes
  - Financial stress: TED spread indicator
  - Stock market: returns, volatility, risk-adjusted returns
  
Real Activity Indicators:
  - Consumer sentiment (U of Michigan)
  - Building permits (leading housing indicator)
  - Capacity utilization
  


Expected Improvement: 
- V5 Test R²: -0.091 (72 quarters, 2000-2018)
- V7 Test R²: +0.10 to +0.20 (180 quarters, 1980-2024, with financial indicators)

Key Hypothesis: Financial indicators (yield curve, credit spreads) are proven GDP predictors
that work across multiple economic regimes (stagflation, dot-com, financial crisis, COVID)

Models:
1. Linear Regression (Baseline)
2. Ridge Regression (L2 Regularization)
3. LASSO Regression (L1 Regularization, Feature Selection)
4. Random Forest Regressor
5. XGBoost Regressor
6. Gradient Boosting Regressor

Target: Current quarter GDP Growth (YoY %)
Training Data: 1980-2024 (180 quarters)
Test Period: 2022-2025
Countries: USA (focus)

Reference: See technical analysis document for full V7 rationale
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
from datetime import datetime

# ML libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directories
DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / 'data_preprocessing' / 'outputs' / 'processed_data'
OUTPUT_DIR = Path(__file__).parent / 'results'
MODEL_DIR = Path(__file__).parent / 'saved_models'
FIG_DIR = Path(__file__).parent / 'figures'

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Focus countries (A+ quality from preprocessing)
FOCUS_COUNTRIES = ['france'] 

# Features to drop
DROP_FEATURES = ['money_supply_broad', 'm2_growth', 'country']

# Target variable
TARGET = 'gdp_growth_yoy'

# ============================================================================
# V7 FEATURE GROUPS
# ============================================================================

# Original leading/coincident indicators (V1-V5)
LEADING_INDICATORS = [
    #'industrial_production_index', Not helping
    'stock_market_index',
    'interest_rate_short_term',
    #'capital_formation', Not helping
]

COINCIDENT_INDICATORS = [
    'employment_level',
    'unemployment_rate',
    #'cpi_annual_growth', Not helping
    'exports_volume',
    'imports_volume',
    'trade_balance'
]

# Original regime features (V3-V5)
REGIME = [
    'stock_volatility_4q',       # Stock-based, not GDP
    #'inflation_momentum',         # Inflation-based, not GDP. Not helping
    #'high_inflation_regime',      # Inflation-based, not GDP. Not helping
]

# Financial Market Indicators (NOT GDP components!)
FINANCIAL_INDICATORS = [
    # Yield curve (proven recession predictor)
    'yield_curve_slope',          # 10Y-2Y Treasury spread
    'yield_curve_curvature',      # Curvature measure
    
    # Credit markets (proven GDP predictor)
    'credit_spread',              # BAA-AAA corporate bond spread
    'credit_spread_change',       # Change in credit spread
]

#  Real Activity Leading Indicators use these later if you want more R^2 but it can be an overbearing feature
REAL_ACTIVITY_INDICATORS = [
    'housing_starts',             # Housing activity
]

# Complete nowcasting feature set (V1-V5 + V7)
NOWCAST_FEATURES = LEADING_INDICATORS + COINCIDENT_INDICATORS + REGIME + FINANCIAL_INDICATORS

class GDPNowcastingPipeline:
    """Complete nowcasting pipeline for current quarter GDP growth"""

    def __init__(self, verbose=True):
        """
        Initialize nowcasting pipeline

        Parameters:
        - verbose: print progress
        """
        self.verbose = verbose
        self.models = {}
        self.results = {}
        self.feature_importance = {}

    def log(self, message):
        """Print if verbose"""
        if self.verbose:
            print(message)

    def load_and_prepare_data(self, country):
        """
        Load preprocessed data and prepare for nowcasting

        V7 UPDATE: Now loads extended data (1980-2024) with financial indicators

        Key difference from forecasting:
        - No future shifting of target (we predict current quarter)
        - Use only leading/coincident indicators (no GDP lags)
        - Focus on features available before GDP release

        Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load unnormalized data for interpretability
        file_path = DATA_DIR / f'{country.lower()}_processed_unnormalized.csv'

        if not file_path.exists():
            self.log(f"⚠ File not found: {file_path}")
            return None

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Drop M2 and country columns
        df = df.drop(columns=[col for col in DROP_FEATURES if col in df.columns], errors='ignore')

        # Drop rows with missing target
        df = df.dropna(subset=[TARGET])

        self.log(f"  Data shape after preparation: {df.shape}")
        self.log(f"  Date range: {df.index.min()} to {df.index.max()}")

        # Define train/val/test splits (temporal)
        train_end = '2018-12-31'
        val_end = '2021-12-31'

        train_df = df.loc[:train_end]
        val_df = df.loc[train_end:val_end].iloc[1:]
        test_df = df.loc[val_end:].iloc[1:]

        self.log(f"  Train: {len(train_df)} quarters ({train_df.index.min()} to {train_df.index.max()})")
        self.log(f"  Val:   {len(val_df)} quarters ({val_df.index.min() if len(val_df) > 0 else 'N/A'} to {val_df.index.max() if len(val_df) > 0 else 'N/A'})")
        self.log(f"  Test:  {len(test_df)} quarters ({test_df.index.min() if len(test_df) > 0 else 'N/A'} to {test_df.index.max() if len(test_df) > 0 else 'N/A'})")

        # Select nowcasting features (available features from predefined list)
        available_features = [col for col in NOWCAST_FEATURES if col in df.columns]

        # Add lagged indicators (1 quarter lag is acceptable for nowcasting - previous quarter data)
        lag_features = []
        for feature in available_features:
            lag_col = f'{feature}_lag1'
            if lag_col in df.columns:
                lag_features.append(lag_col)

        feature_cols = available_features + lag_features

        self.log(f"  Nowcasting features selected: {len(feature_cols)}")
        self.log(f"    - Core features (V1-V5): {len(LEADING_INDICATORS) + len(COINCIDENT_INDICATORS) + len(REGIME)}")
        self.log(f"    - financial features: {len([f for f in available_features if f in FINANCIAL_INDICATORS])}")
        self.log(f"    - Lagged features (t-1): {len(lag_features)}")

        # Handle remaining missing values
        missing_pct = (train_df[feature_cols].isnull().sum() / len(train_df)) * 100
        high_missing = missing_pct[missing_pct > 30].index.tolist()

        if high_missing:
            self.log(f"  Dropping {len(high_missing)} features with >30% missing data")
            feature_cols = [col for col in feature_cols if col not in high_missing]

        # Forward-fill remaining missing values
        train_df[feature_cols] = train_df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        val_df[feature_cols] = val_df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        test_df[feature_cols] = test_df[feature_cols].fillna(method='ffill').fillna(method='bfill')

        # Prepare X and y (no shifting - current quarter target)
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET]

        X_val = val_df[feature_cols] if len(val_df) > 0 else pd.DataFrame()
        y_val = val_df[TARGET] if len(val_df) > 0 else pd.Series()

        X_test = test_df[feature_cols] if len(test_df) > 0 else pd.DataFrame()
        y_test = test_df[TARGET] if len(test_df) > 0 else pd.Series()

        self.log(f"  Final feature count: {len(feature_cols)}")

        # ===== ADD THIS DEBUG BLOCK =====
        self.log("\n" + "="*70)
        self.log("V7 FEATURE DEBUG")
        self.log("="*70)
        self.log(f"Features defined in NOWCAST_FEATURES: {len(NOWCAST_FEATURES)}")
        self.log(f"Features actually in data: {len(feature_cols)}")
        
        # Check financial features
        Financial_in_data = [f for f in FINANCIAL_INDICATORS if f in feature_cols]
        Financial_missing = [f for f in FINANCIAL_INDICATORS if f not in feature_cols]
        self.log(f"\nV7 features present: {len(Financial_in_data)}/{len(FINANCIAL_INDICATORS)}")
        if Financial_in_data:
            self.log(f"  Present: {Financial_in_data}")
        if Financial_missing:
            self.log(f"  MISSING: {Financial_missing}")
        
        # NaN check
        self.log(f"\nNaN counts:")
        nan_counts = X_train.isnull().sum().sum()
        self.log(f"  Total NaNs in X_train: {nan_counts}")
        high_nan_features = (X_train.isnull().sum() / len(X_train) * 100)
        high_nan_features = high_nan_features[high_nan_features > 20]
        if len(high_nan_features) > 0:
            self.log(f"  Features with >20% NaNs:")
            for feat, pct in high_nan_features.items():
                self.log(f"    {feat}: {pct:.1f}%")
        
        self.log(f"\nActual feature columns being used:")
        self.log(f"  {feature_cols[:10]}... (showing first 10)")
        self.log("="*70)
        # ===== END DEBUG BLOCK =====


        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def train_linear_regression(self, X_train, y_train):
        """Train baseline linear regression"""
        self.log("  [1/6] Training Linear Regression (Baseline)...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_ridge(self, X_train, y_train):
        """Train Ridge regression with CV"""
        self.log("  [2/6] Training Ridge Regression (L2)...")
        # Use stronger regularization based on forecasting lessons
        param_grid = {'alpha': [1.0, 10.0, 100.0, 500.0, 1000.0]}

        tscv = TimeSeriesSplit(n_splits=3)
        ridge = Ridge()

        grid_search = GridSearchCV(ridge, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"    Best alpha: {grid_search.best_params_['alpha']}")
        return grid_search.best_estimator_

    def train_lasso(self, X_train, y_train):
        """Train LASSO regression with CV"""
        self.log("  [3/6] Training LASSO Regression (L1)...")
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

        tscv = TimeSeriesSplit(n_splits=3)
        lasso = Lasso(max_iter=10000)

        grid_search = GridSearchCV(lasso, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"    Best alpha: {grid_search.best_params_['alpha']}")

        # Count non-zero coefficients (feature selection)
        n_features = np.sum(grid_search.best_estimator_.coef_ != 0)
        self.log(f"    Features selected: {n_features}/{len(X_train.columns)}")

        return grid_search.best_estimator_

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest with conservative hyperparameters"""
        self.log("  [4/6] Training Random Forest...")
        # More conservative to reduce overfitting
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],  # Shallower trees
            'min_samples_split': [10, 20],  # Larger split requirement
            'min_samples_leaf': [5, 10]  # Larger leaf requirement
        }

        tscv = TimeSeriesSplit(n_splits=3)
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        self.log(f"    Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost with conservative hyperparameters"""
        self.log("  [5/6] Training XGBoost...")
        param_grid = {
            'n_estimators': [50, 100],  # Fewer trees
            'max_depth': [3, 5],  # Shallower trees
            'learning_rate': [0.01, 0.05],  # Slower learning
            'subsample': [0.8],
            'colsample_bytree': [0.8],  # Feature sampling
            'reg_alpha': [0.1, 1.0],  # L1 regularization
            'reg_lambda': [1.0, 10.0]  # L2 regularization
        }

        tscv = TimeSeriesSplit(n_splits=3)
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(xgb_model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        self.log(f"    Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting with conservative hyperparameters"""
        self.log("  [6/6] Training Gradient Boosting...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10]
        }

        tscv = TimeSeriesSplit(n_splits=3)
        gb = GradientBoostingRegressor(random_state=42)

        grid_search = GridSearchCV(gb, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        self.log(f"    Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def evaluate_model(self, model, X, y, dataset_name):
        """Evaluate model performance"""
        if len(X) == 0:
            return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan}

        predictions = model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y - predictions) / np.where(y == 0, 1, y))) * 100

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'predictions': predictions,
            'actuals': y.values
        }

    def train_all_models(self, country):
        """Train all models for a given country"""
        self.log(f"\n{'='*70}")
        self.log(f"Training Nowcasting Models: {country}")
        self.log(f"{'='*70}")

        # Load data
        self.log("[Step 1/3] Loading and preparing data (with financial indicators)...")
        data = self.load_and_prepare_data(country)

        if data is None:
            self.log(f"⚠ Skipping {country} - data not available")
            return None

        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = data

        # Train models
        self.log("\n[Step 2/3] Training models...")
        models = {
            'Linear Regression': self.train_linear_regression(X_train, y_train),
            'Ridge': self.train_ridge(X_train, y_train),
            'LASSO': self.train_lasso(X_train, y_train),
            'Random Forest': self.train_random_forest(X_train, y_train),
            'XGBoost': self.train_xgboost(X_train, y_train),
            'Gradient Boosting': self.train_gradient_boosting(X_train, y_train)
        }


        # ===== ADD THIS BLOCK HERE =====
        # Train stacking ensemble
        self.log("\n[BONUS] Training Stacking Ensemble...")
        from sklearn.ensemble import StackingRegressor
        
        estimators = [
            ('ridge', models['Ridge']),
            ('lasso', models['LASSO']),
            ('rf', models['Random Forest']),
            ('xgb', models['XGBoost']),
            ('gb', models['Gradient Boosting'])
        ]
        
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1
        )
        
        stacking.fit(X_train, y_train)
        models['Stacking Ensemble'] = stacking
        self.log("  ✓ Stacking Ensemble trained")
        # ===== END ADD =====

        # Evaluate models
        self.log("\n[Step 3/3] Evaluating models...")
        results = {}

        for model_name, model in models.items():
            results[model_name] = {
                'train': self.evaluate_model(model, X_train, y_train, 'Train'),
                'val': self.evaluate_model(model, X_val, y_val, 'Val'),
                'test': self.evaluate_model(model, X_test, y_test, 'Test')
            }

            # Save model
            model_file = MODEL_DIR / f'{country.lower()}_nowcast_v7_{model_name.lower().replace(" ", "_")}.pkl'
            joblib.dump(model, model_file)

        # Store results
        self.models[country] = models
        self.results[country] = results

        # Extract feature importance for tree-based models
        self.feature_importance[country] = {}
        for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
            if hasattr(models[model_name], 'feature_importances_'):
                self.feature_importance[country][model_name] = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': models[model_name].feature_importances_
                }).sort_values('importance', ascending=False)

                self.log(f"\n{model_name} - Top 10 Features:")
                self.log(self.feature_importance[country][model_name].head(10).to_string(index=False))

        # Print summary
        self.print_results_summary(country)

        return results

    def print_results_summary(self, country):
        """Print results summary table"""
        self.log(f"\n{'='*70}")
        self.log(f" Nowcasting Results Summary: {country}")
        self.log(f"{'='*70}")

        results = self.results[country]

        # Create summary table
        summary_data = []
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Train RMSE': f"{metrics['train']['RMSE']:.3f}",
                'Val RMSE': f"{metrics['val']['RMSE']:.3f}" if not np.isnan(metrics['val']['RMSE']) else 'N/A',
                'Test RMSE': f"{metrics['test']['RMSE']:.3f}" if not np.isnan(metrics['test']['RMSE']) else 'N/A',
                'Train R²': f"{metrics['train']['R2']:.3f}",
                'Val R²': f"{metrics['val']['R2']:.3f}" if not np.isnan(metrics['val']['R2']) else 'N/A',
                'Test R²': f"{metrics['test']['R2']:.3f}" if not np.isnan(metrics['test']['R2']) else 'N/A'
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        self.log("\n" + summary_df.to_string(index=False))

        # Save to CSV
        summary_df.to_csv(OUTPUT_DIR / f'{country.lower()}_nowcast_V7_results.csv', index=False)

    def plot_predictions(self, country):
        results = self.results[country]
        
        n_models = len(results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten()

        for idx, (model_name, metrics) in enumerate(results.items()):
            ax = axes[idx]

            # Test set predictions
            if not np.isnan(metrics['test']['RMSE']):
                actuals = metrics['test']['actuals']
                predictions = metrics['test']['predictions']

                ax.scatter(actuals, predictions, alpha=0.6, s=50)
                ax.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()],
                       'r--', linewidth=2, label='Perfect Prediction')

                ax.set_xlabel('Actual GDP Growth (%)', fontsize=10)
                ax.set_ylabel('Predicted GDP Growth (%)', fontsize=10)
                ax.set_title(f'{model_name}\nTest R² = {metrics["test"]["R2"]:.3f}, RMSE = {metrics["test"]["RMSE"]:.3f}',
                           fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'{country} - V7 GDP Nowcasting Results (with Financial Indicators)\nPredictions vs Actuals',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIG_DIR / f'{country.lower()}_nowcast_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"  ✓ Saved predictions plot: {country}_nowcast_predictions.png")

    def plot_feature_importance(self, country):
        """Plot feature importance for tree-based models"""
        if country not in self.feature_importance or not self.feature_importance[country]:
            return

        n_models = len(self.feature_importance[country])
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, importance_df) in enumerate(self.feature_importance[country].items()):
            ax = axes[idx]

            # Top 15 features
            top_features = importance_df.head(15)

            ax.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=9)
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_title(f'{model_name}\nTop 15 Features', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'{country} - Nowcasting Feature Importance',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIG_DIR / f'{country.lower()}_nowcast_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"  ✓ Saved feature importance plot: {country}_nowcast_feature_importance.png")

    def plot_time_series_predictions(self, country):
        """Plot time series of predictions vs actuals"""
        if country not in self.results:
            return

        results = self.results[country]

        # Get best model (highest test R²)
        best_model_name = max(results.keys(),
                             key=lambda k: results[k]['test']['R2'] if not np.isnan(results[k]['test']['R2']) else -np.inf)

        best_metrics = results[best_model_name]

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot actuals and predictions over time
        if not np.isnan(best_metrics['test']['RMSE']):
            test_actuals = best_metrics['test']['actuals']
            test_predictions = best_metrics['test']['predictions']

            # Get test dates
            file_path = DATA_DIR / f'{country.lower()}_processed_unnormalized.csv'
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            val_end = '2021-12-31'
            test_dates = df.loc[val_end:].iloc[1:].index[:len(test_actuals)]

            ax.plot(test_dates, test_actuals, 'o-', label='Actual GDP Growth', linewidth=2, markersize=6)
            ax.plot(test_dates, test_predictions, 's--', label=f'{best_model_name} Predictions',
                   linewidth=2, markersize=6, alpha=0.7)

            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('Quarter', fontsize=12)
            ax.set_ylabel('GDP Growth (YoY %)', fontsize=12)
            ax.set_title(f'{country} - Nowcasting Time Series (with Financial Indicators)\nBest Model: {best_model_name} (Test R² = {best_metrics["test"]["R2"]:.3f})',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(FIG_DIR / f'{country.lower()}_nowcast_timeseries.png', dpi=300, bbox_inches='tight')
            plt.close()

            self.log(f"  ✓ Saved time series plot: {country}_nowcast_timeseries.png")

def main():
    """Main execution"""
    print("="*80)
    print("GDP NOWCASTING PIPELINE V7 - WITH FINANCIAL INDICATORS")
    print("="*80)
    print("Extended Training Data: 1980-2024 (180 quarters)")
    print("New Features: Yield Curve, Credit Spreads, Consumer Sentiment, etc.")
    print("V5 Baseline: Test R² = -0.091 (72 quarters, no financial indicators)")
    print("V7 Target:   Test R² > 0.10 (180 quarters, with financial indicators)")
    print("="*80)

    pipeline = GDPNowcastingPipeline(verbose=True)

    # Train models for focus countries
    for country in FOCUS_COUNTRIES:
        results = pipeline.train_all_models(country)

        if results:
            # Generate visualizations
            print(f"\nGenerating V7 visualizations for {country}...")
            pipeline.plot_predictions(country)
            pipeline.plot_feature_importance(country)
            pipeline.plot_time_series_predictions(country)

    # Combine results across countries
    print(f"\n{'='*80}")
    print("V7 Cross-Country Nowcasting Summary")
    print(f"{'='*80}")

    all_results = []
    for country in FOCUS_COUNTRIES:
        if country in pipeline.results:
            for model_name, metrics in pipeline.results[country].items():
                all_results.append({
                    'Country': country,
                    'Model': model_name,
                    'Test_RMSE': metrics['test']['RMSE'],
                    'Test_MAE': metrics['test']['MAE'],
                    'Test_R2': metrics['test']['R2'],
                    'Test_MAPE': metrics['test']['MAPE']
                })

    combined_df = pd.DataFrame(all_results)
    combined_df.to_csv(OUTPUT_DIR / 'all_countries_nowcast_V7_results.csv', index=False)

    # Print best models
    if len(combined_df) > 0:
        print("\nBest V7 Model by Country (Test R²):")
        best_models = combined_df.loc[combined_df.groupby('Country')['Test_R2'].idxmax()]
        print(best_models[['Country', 'Model', 'Test_R2', 'Test_RMSE']].to_string(index=False))

    print("\n" + "="*80)
    print("V7 NOWCASTING PIPELINE COMPLETE!")
    print("="*80)
    print(f"Key Changes from V5:")
    print(f"  - Training data: 72 quarters (2000-2018) → 180 quarters (1980-2024)")
    print(f"  - Features: ~13 baseline → ~30 with {len(FINANCIAL_INDICATORS)} financial features")
    print(f"  - S&P 500: Fixed (now complete from Yahoo Finance)")
    print(f"  - Added: Yield curve, credit spreads, financial stress indicators")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*80)
    print("\n🎯 Check Test R² values above to see if financial indicators improved predictions!")

if __name__ == "__main__":
    main()