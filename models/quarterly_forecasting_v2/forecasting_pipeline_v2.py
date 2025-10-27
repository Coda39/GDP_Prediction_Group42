"""
GDP Quarterly Forecasting Pipeline v2
======================================

Improvements over v1:
1. Feature selection: Reduce from 49 to 10-15 features using LASSO
2. Stronger regularization: Ridge alpha 500-1000, LASSO alpha 10-100
3. Conservative hyperparameters (shallow trees, larger min_samples)
4. Walk-forward validation for robust evaluation
5. Ensemble models weighted by validation performance
6. Comparison with v1 baseline results

Author: GDP Forecasting Team
Date: October 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data_preprocessing" / "resampled_data"
OUTPUT_DIR = Path(__file__).parent
MODELS_DIR = OUTPUT_DIR / "saved_models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
COMPARISON_DIR = OUTPUT_DIR / "comparison_plots"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
COMPARISON_DIR.mkdir(exist_ok=True)

# Constants
COUNTRIES = ['usa', 'canada', 'japan', 'uk']
HORIZONS = [1, 4]  # 1Q and 4Q ahead
TARGET = 'gdp_growth_yoy'

# Hyperparameters - Much more conservative than v1
RIDGE_ALPHAS = [100.0, 500.0, 1000.0, 2000.0]  # v1: [0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [10.0, 50.0, 100.0, 200.0]  # v1: [0.001, 0.01, 0.1, 1.0, 10.0]

RF_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],  # v1: [5, 10, 15] - shallower
    'min_samples_split': [10, 20],  # v1: [5, 10] - larger
    'min_samples_leaf': [5, 10]  # v1: default - added constraint
}

XGB_PARAMS = {
    'n_estimators': [50, 100],  # v1: [100, 200] - fewer
    'max_depth': [3, 5],  # v1: [3, 5, 7] - shallower
    'learning_rate': [0.01, 0.05],  # v1: [0.01, 0.1] - slower
    'subsample': [0.8],  # v1: [0.8, 1.0]
    'reg_alpha': [0.1, 1.0],  # L1 regularization - NEW
    'reg_lambda': [1.0, 10.0]  # L2 regularization - NEW
}

GB_PARAMS = {
    'n_estimators': [50, 100],  # v1: [100, 200]
    'max_depth': [3, 5],  # v1: [3, 5] - kept shallow
    'learning_rate': [0.01, 0.05],  # v1: [0.01, 0.1]
    'min_samples_split': [10, 20],  # v1: default
    'min_samples_leaf': [5, 10]  # v1: default
}


class GDPForecastingPipelineV2:
    """Improved GDP Forecasting Pipeline with feature selection and stronger regularization"""

    def __init__(self, horizon=1, n_features=15, verbose=True):
        """
        Initialize forecasting pipeline

        Parameters:
        -----------
        horizon : int
            Forecasting horizon in quarters (1 or 4)
        n_features : int
            Number of features to select (10-15 recommended)
        verbose : bool
            Print progress messages
        """
        self.horizon = horizon
        self.n_features = n_features
        self.verbose = verbose
        self.results = []
        self.selected_features = {}  # Store selected features per country

    def log(self, message):
        """Print message if verbose"""
        if self.verbose:
            print(message)

    def load_and_prepare_data(self, country):
        """
        Load preprocessed data and create train/val/test splits

        Returns:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
        """
        # Load normalized data
        file_path = DATA_DIR / f"{country}_processed_normalized.csv"
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Drop rows with missing target
        df = df.dropna(subset=[TARGET])

        # Create future target (shift backward by horizon)
        df[f'{TARGET}_future'] = df[TARGET].shift(-self.horizon)

        # Drop rows with missing future target
        df = df.dropna(subset=[f'{TARGET}_future'])

        # Features: all columns except target, future target, and country
        feature_cols = [col for col in df.columns if col not in [TARGET, f'{TARGET}_future', 'country']]

        # Drop rows with any missing values in features
        df = df.dropna(subset=feature_cols)

        X = df[feature_cols]
        y = df[f'{TARGET}_future']

        # Temporal split (same as v1 for fair comparison)
        train_end = '2018-12-31'
        val_end = '2021-12-31'

        X_train = X.loc[:train_end]
        X_val = X.loc[train_end:val_end].iloc[1:]
        X_test = X.loc[val_end:].iloc[1:]

        y_train = y.loc[:train_end]
        y_val = y.loc[train_end:val_end].iloc[1:]
        y_test = y.loc[val_end:].iloc[1:]

        self.log(f"    Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        self.log(f"    Features: {len(feature_cols)} (before selection)")

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def select_features_lasso(self, X_train, y_train, X_val, y_val, X_test, feature_names):
        """
        Use LASSO with high alpha to select top N features

        Returns:
        --------
        X_train_selected, X_val_selected, X_test_selected, selected_features
        """
        self.log(f"    Performing feature selection (target: {self.n_features} features)...")

        # Train LASSO with very high alpha for aggressive feature selection
        lasso = Lasso(alpha=10.0, max_iter=10000, random_state=42)
        lasso.fit(X_train, y_train)

        # Get feature importance (absolute coefficients)
        importance = np.abs(lasso.coef_)

        # Select top N features
        top_indices = np.argsort(importance)[-self.n_features:]
        selected_features = [feature_names[i] for i in top_indices]

        # Filter datasets
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        X_test_selected = X_test[selected_features]

        self.log(f"    Selected {len(selected_features)} features")
        self.log(f"    Top 5: {selected_features[-5:]}")

        return X_train_selected, X_val_selected, X_test_selected, selected_features

    def train_ridge(self, X_train, y_train):
        """Train Ridge regression with strong regularization"""
        self.log("      Training Ridge...")

        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {'alpha': RIDGE_ALPHAS}

        ridge = Ridge(max_iter=10000, random_state=42)
        grid_search = GridSearchCV(ridge, param_grid, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"        Best alpha: {grid_search.best_params_['alpha']}")
        return grid_search.best_estimator_

    def train_lasso(self, X_train, y_train):
        """Train LASSO with strong regularization"""
        self.log("      Training LASSO...")

        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {'alpha': LASSO_ALPHAS}

        lasso = Lasso(max_iter=10000, random_state=42)
        grid_search = GridSearchCV(lasso, param_grid, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"        Best alpha: {grid_search.best_params_['alpha']}")
        return grid_search.best_estimator_

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest with conservative hyperparameters"""
        self.log("      Training Random Forest...")

        tscv = TimeSeriesSplit(n_splits=3)

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, RF_PARAMS, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"        Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost with regularization"""
        self.log("      Training XGBoost...")

        tscv = TimeSeriesSplit(n_splits=3)

        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(xgb_model, XGB_PARAMS, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"        Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting with conservative hyperparameters"""
        self.log("      Training Gradient Boosting...")

        tscv = TimeSeriesSplit(n_splits=3)

        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, GB_PARAMS, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"        Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def create_ensemble(self, models, model_names, X_val, y_val):
        """
        Create weighted ensemble based on validation R²
        Only include models with positive validation R²
        """
        self.log("      Creating ensemble...")

        val_r2_scores = []
        for model in models:
            y_val_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_val_pred)
            val_r2_scores.append(max(0, r2))  # Clip negative to 0

        # If all models have negative R², use equal weights
        if sum(val_r2_scores) == 0:
            weights = np.ones(len(models)) / len(models)
            self.log(f"        All models negative R² - using equal weights")
        else:
            # Normalize to sum to 1
            weights = np.array(val_r2_scores) / sum(val_r2_scores)
            self.log(f"        Ensemble weights: {dict(zip(model_names, weights))}")

        return weights

    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test, model_name):
        """Evaluate model on train/val/test sets"""

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'test_r2': r2_score(y_test, y_test_pred),
        }

        # Calculate MAPE (handle division by zero)
        def safe_mape(y_true, y_pred):
            mask = y_true != 0
            if mask.sum() == 0:
                return np.nan
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        metrics['train_mape'] = safe_mape(y_train, y_train_pred)
        metrics['val_mape'] = safe_mape(y_val, y_val_pred)
        metrics['test_mape'] = safe_mape(y_test, y_test_pred)

        return metrics, y_test_pred

    def run_country(self, country):
        """Run forecasting pipeline for one country"""
        self.log(f"\n{'='*80}")
        self.log(f"Country: {country.upper()}, Horizon: {self.horizon}Q Ahead")
        self.log(f"{'='*80}")

        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
            self.load_and_prepare_data(country)

        # Feature selection
        X_train_sel, X_val_sel, X_test_sel, selected_features = \
            self.select_features_lasso(X_train, y_train, X_val, y_val, X_test, feature_names)

        # Store selected features
        self.selected_features[country] = selected_features

        # Train models
        self.log("\n    Training models...")
        models = {}

        models['Ridge'] = self.train_ridge(X_train_sel, y_train)
        models['LASSO'] = self.train_lasso(X_train_sel, y_train)
        models['Random Forest'] = self.train_random_forest(X_train_sel, y_train)
        models['XGBoost'] = self.train_xgboost(X_train_sel, y_train)
        models['Gradient Boosting'] = self.train_gradient_boosting(X_train_sel, y_train)

        # Create ensemble
        model_names = list(models.keys())
        model_list = list(models.values())
        ensemble_weights = self.create_ensemble(model_list, model_names, X_val_sel, y_val)

        # Ensemble predictions
        y_test_pred_ensemble = np.zeros(len(y_test))
        for model, weight in zip(model_list, ensemble_weights):
            y_test_pred_ensemble += weight * model.predict(X_test_sel)

        # Evaluate all models
        self.log("\n    Evaluating models...")
        for model_name, model in models.items():
            metrics, y_test_pred = self.evaluate_model(
                model, X_train_sel, X_val_sel, X_test_sel,
                y_train, y_val, y_test, model_name
            )

            result = {
                'country': country,
                'horizon': self.horizon,
                'model': model_name,
                'version': 'v2',
                'n_features': len(selected_features),
                **metrics
            }
            self.results.append(result)

            self.log(f"      {model_name:20s} | Test R²: {metrics['test_r2']:7.3f} | Test RMSE: {metrics['test_rmse']:.3f}%")

            # Save model
            model_filename = f"{country}_h{self.horizon}_{model_name.lower().replace(' ', '_')}_v2.pkl"
            joblib.dump(model, MODELS_DIR / model_filename)

        # Evaluate ensemble
        metrics_ensemble = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_ensemble)),
            'test_mae': mean_absolute_error(y_test, y_test_pred_ensemble),
            'test_r2': r2_score(y_test, y_test_pred_ensemble),
        }

        result_ensemble = {
            'country': country,
            'horizon': self.horizon,
            'model': 'Ensemble',
            'version': 'v2',
            'n_features': len(selected_features),
            'test_rmse': metrics_ensemble['test_rmse'],
            'test_mae': metrics_ensemble['test_mae'],
            'test_r2': metrics_ensemble['test_r2'],
        }
        self.results.append(result_ensemble)

        self.log(f"      {'Ensemble':20s} | Test R²: {metrics_ensemble['test_r2']:7.3f} | Test RMSE: {metrics_ensemble['test_rmse']:.3f}%")

        # Save ensemble weights
        ensemble_info = {
            'weights': dict(zip(model_names, ensemble_weights)),
            'models': model_names
        }
        joblib.dump(ensemble_info, MODELS_DIR / f"{country}_h{self.horizon}_ensemble_v2.pkl")

        return models, selected_features

    def run_all_countries(self):
        """Run forecasting pipeline for all countries"""
        self.log("\n" + "="*80)
        self.log(f"GDP FORECASTING PIPELINE V2 - {self.horizon}Q Ahead")
        self.log(f"Feature Selection: Top {self.n_features} features")
        self.log(f"Regularization: STRONG (Ridge α=100-2000, LASSO α=10-200)")
        self.log("="*80)

        for country in COUNTRIES:
            self.run_country(country)

        # Save results
        self.save_results()
        self.log("\n" + "="*80)
        self.log("PIPELINE V2 COMPLETE")
        self.log("="*80)

    def save_results(self):
        """Save results to CSV"""
        df_results = pd.DataFrame(self.results)

        # Save per-horizon results
        output_file = RESULTS_DIR / f"all_countries_h{self.horizon}_v2_results.csv"
        df_results.to_csv(output_file, index=False)
        self.log(f"\nResults saved to: {output_file}")

        # Save selected features
        features_file = RESULTS_DIR / f"selected_features_h{self.horizon}_v2.csv"
        features_data = []
        for country, features in self.selected_features.items():
            for i, feat in enumerate(features, 1):
                features_data.append({'country': country, 'rank': i, 'feature': feat})
        pd.DataFrame(features_data).to_csv(features_file, index=False)
        self.log(f"Selected features saved to: {features_file}")


def load_v1_results(horizon):
    """Load v1 baseline results for comparison"""
    v1_file = BASE_DIR / "models" / "quarterly_forecasting" / "results" / f"all_countries_h{horizon}_results.csv"
    if v1_file.exists():
        df = pd.read_csv(v1_file)
        df['version'] = 'v1'
        return df
    return None


def create_comparison_plots(horizon):
    """Create comparison plots between v1 and v2"""
    print(f"\nCreating comparison plots for {horizon}Q horizon...")

    # Load v1 and v2 results
    v1_results = load_v1_results(horizon)
    v2_file = RESULTS_DIR / f"all_countries_h{horizon}_v2_results.csv"
    v2_results = pd.read_csv(v2_file)

    if v1_results is None:
        print("  Warning: v1 results not found, skipping comparison")
        return

    # Combine results
    combined = pd.concat([v1_results, v2_results], ignore_index=True)

    # Plot 1: Test R² comparison by country
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Performance Comparison: v1 vs v2 ({horizon}Q Ahead)', fontsize=16, fontweight='bold')

    for idx, country in enumerate(COUNTRIES):
        ax = axes[idx // 2, idx % 2]

        df_country = combined[combined['country'] == country]

        # Group by model and version
        pivot = df_country.pivot_table(index='model', columns='version', values='test_r2', aggfunc='first')

        # Plot
        x = np.arange(len(pivot))
        width = 0.35

        if 'v1' in pivot.columns:
            ax.bar(x - width/2, pivot['v1'], width, label='v1 (baseline)', alpha=0.8)
        if 'v2' in pivot.columns:
            ax.bar(x + width/2, pivot['v2'], width, label='v2 (improved)', alpha=0.8)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Mean baseline')
        ax.set_xlabel('Model')
        ax.set_ylabel('Test R²')
        ax.set_title(f'{country.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / f'test_r2_comparison_h{horizon}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: test_r2_comparison_h{horizon}.png")
    plt.close()

    # Plot 2: Test RMSE comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Test RMSE Comparison: v1 vs v2 ({horizon}Q Ahead)', fontsize=16, fontweight='bold')

    for idx, country in enumerate(COUNTRIES):
        ax = axes[idx // 2, idx % 2]

        df_country = combined[combined['country'] == country]
        pivot = df_country.pivot_table(index='model', columns='version', values='test_rmse', aggfunc='first')

        x = np.arange(len(pivot))
        width = 0.35

        if 'v1' in pivot.columns:
            ax.bar(x - width/2, pivot['v1'], width, label='v1 (baseline)', alpha=0.8)
        if 'v2' in pivot.columns:
            ax.bar(x + width/2, pivot['v2'], width, label='v2 (improved)', alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Test RMSE (%)')
        ax.set_title(f'{country.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / f'test_rmse_comparison_h{horizon}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: test_rmse_comparison_h{horizon}.png")
    plt.close()

    # Plot 3: Overall improvement summary
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate average test R² per version per country
    summary = combined.groupby(['country', 'version'])['test_r2'].mean().reset_index()
    pivot_summary = summary.pivot(index='country', columns='version', values='test_r2')

    x = np.arange(len(pivot_summary))
    width = 0.35

    if 'v1' in pivot_summary.columns:
        bars1 = ax.bar(x - width/2, pivot_summary['v1'], width, label='v1 (baseline)', alpha=0.8)
    if 'v2' in pivot_summary.columns:
        bars2 = ax.bar(x + width/2, pivot_summary['v2'], width, label='v2 (improved)', alpha=0.8)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Mean baseline')
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Average Test R²', fontsize=12)
    ax.set_title(f'Overall Model Performance: v1 vs v2 ({horizon}Q Ahead)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in pivot_summary.index], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / f'overall_improvement_h{horizon}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: overall_improvement_h{horizon}.png")
    plt.close()


def main():
    """Run improved forecasting pipeline for both horizons"""
    print("\n" + "="*80)
    print("GDP QUARTERLY FORECASTING PIPELINE V2")
    print("Improvements: Feature Selection + Strong Regularization + Conservative Hyperparameters")
    print("="*80)

    for horizon in HORIZONS:
        # Run v2 pipeline
        pipeline = GDPForecastingPipelineV2(horizon=horizon, n_features=15, verbose=True)
        pipeline.run_all_countries()

        # Create comparison plots
        create_comparison_plots(horizon)

    print("\n" + "="*80)
    print("ALL PIPELINES COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Comparison plots saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
