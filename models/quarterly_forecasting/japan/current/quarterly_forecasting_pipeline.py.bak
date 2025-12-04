"""
GDP Quarterly Forecasting Pipeline - V1
========================================
This script trains and evaluates multiple models for forecasting GDP growth at 1Q, 2Q, 3Q, and 4Q horizons.

Key Differences from Nowcasting:
- Target is shifted forward by h quarters (h=1,2,3,4)
- Uses ONLY exogenous features (no GDP-dependent variables to avoid data leakage)
- Each horizon has separate models
- Bootstrap confidence intervals (80% and 95%) for uncertainty quantification
- Weighted ensemble with model disagreement incorporated into uncertainty

Models (5 per horizon):
1. Ridge Regression - L2 regularization
2. LASSO Regression - L1 regularization + feature selection
3. Random Forest - Tree ensemble
4. XGBoost - Gradient boosting
5. Gradient Boosting - Alternative ensemble

Total: 20 models per country (4 horizons × 5 models)

Target: GDP Growth h quarters ahead (YoY %)
Training Data: 1980-2024
Horizons: h=1, h=2, h=3, h=4 quarters
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

warnings.filterwarnings("ignore")

# Styling
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Directories
DATA_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "data_preprocessing"
    / "outputs"
    / "processed_data"
)
OUTPUT_DIR = Path(__file__).parent / "results"
MODEL_DIR = Path(__file__).parent / "saved_models"
FIG_DIR = Path(__file__).parent / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Country configuration
COUNTRY = "japan"

# Forecast horizons
HORIZONS = [1, 2, 3, 4]

# Target variable
TARGET = "gdp_growth_yoy"

# Bootstrap parameters
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# ============================================================================
# EXOGENOUS FEATURES (NO GDP-DEPENDENT VARIABLES)
# ============================================================================
# These features do not directly depend on GDP to avoid data leakage

FORECAST_FEATURES = [
    # Labor market
    'unemployment_rate',
    'employment_level',
    'employment_growth',

    # Inflation
    'cpi_annual_growth',

    # Monetary policy
    'interest_rate_short_term',
    'interest_rate_long_term',

    # Production
    'industrial_production_index',
    'ip_growth',

    # Trade volumes (not ratios with GDP)
    'exports_volume',
    'imports_volume',
    'exports_growth',
    'imports_growth',

    # Consumption
    'household_consumption',
    'consumption_growth',

    # Investment
    'capital_formation',
    'investment_growth',

    # Government
    'government_spending',

    # Financial markets
    'stock_market_index',
    'yield_curve_slope',
    'yield_curve_curvature',
    'credit_spread',
    'stock_volatility_4q',
]


class QuarterlyForecastingPipeline:
    """Complete forecasting pipeline for multi-horizon GDP growth prediction"""

    def __init__(self, country, verbose=True):
        """
        Initialize forecasting pipeline

        Parameters:
        - country: Country name (e.g., 'usa', 'canada')
        - verbose: print progress
        """
        self.country = country
        self.verbose = verbose
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.ensemble_results = {}

    def log(self, message):
        """Print if verbose"""
        if self.verbose:
            print(message)

    def load_and_prepare_data(self, horizon):
        """
        Load preprocessed data and prepare for h-step ahead forecasting

        Key difference from nowcasting:
        - Target is shifted FORWARD by h quarters
        - Only use exogenous features (no GDP-dependent variables)
        - Lose h quarters at end of dataset due to forward shifting

        Parameters:
        - horizon: Forecast horizon (1, 2, 3, or 4 quarters ahead)

        Returns: (X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates, feature_cols)
        """
        # Load unnormalized data for interpretability
        file_path = DATA_DIR / f"{self.country.lower()}_processed_unnormalized.csv"

        if not file_path.exists():
            self.log(f"⚠ File not found: {file_path}")
            return None

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Select only available exogenous features
        available_features = [col for col in FORECAST_FEATURES if col in df.columns]

        self.log(f"  Available features: {len(available_features)}/{len(FORECAST_FEATURES)}")

        # Create h-step ahead target by shifting FORWARD
        # Example: If h=1, y[t] = gdp_growth[t+1]
        df['target'] = df[TARGET].shift(-horizon)

        # Drop rows with missing target (last h quarters have no future data)
        df_clean = df.dropna(subset=['target'])

        self.log(f"  Data shape after h={horizon} shift: {df_clean.shape}")
        self.log(f"  Date range: {df_clean.index.min()} to {df_clean.index.max()}")
        self.log(f"  Lost {len(df) - len(df_clean)} quarters at end due to forward shift")

        # Define train/val/test splits (temporal)
        train_end = "2018-12-31"
        val_end = "2021-12-31"

        train_df = df_clean.loc[:train_end]
        val_df = df_clean.loc[train_end:val_end].iloc[1:]
        test_df = df_clean.loc[val_end:].iloc[1:]

        self.log(
            f"  Train: {len(train_df)} quarters ({train_df.index.min()} to {train_df.index.max()})"
        )
        self.log(
            f"  Val:   {len(val_df)} quarters ({val_df.index.min() if len(val_df) > 0 else 'N/A'} to {val_df.index.max() if len(val_df) > 0 else 'N/A'})"
        )
        self.log(
            f"  Test:  {len(test_df)} quarters ({test_df.index.min() if len(test_df) > 0 else 'N/A'} to {test_df.index.max() if len(test_df) > 0 else 'N/A'})"
        )

        # Handle missing values in features
        missing_pct = (train_df[available_features].isnull().sum() / len(train_df)) * 100
        high_missing = missing_pct[missing_pct > 30].index.tolist()

        if high_missing:
            self.log(f"  Dropping {len(high_missing)} features with >30% missing data: {high_missing}")
            available_features = [col for col in available_features if col not in high_missing]

        # Forward-fill remaining missing values
        train_df[available_features] = (
            train_df[available_features].fillna(method="ffill").fillna(method="bfill")
        )
        val_df[available_features] = (
            val_df[available_features].fillna(method="ffill").fillna(method="bfill")
        )
        test_df[available_features] = (
            test_df[available_features].fillna(method="ffill").fillna(method="bfill")
        )

        # Prepare X and y
        X_train = train_df[available_features]
        y_train = train_df['target']
        train_dates = train_df.index

        X_val = val_df[available_features] if len(val_df) > 0 else pd.DataFrame()
        y_val = val_df['target'] if len(val_df) > 0 else pd.Series()
        val_dates = val_df.index if len(val_df) > 0 else pd.DatetimeIndex([])

        X_test = test_df[available_features] if len(test_df) > 0 else pd.DataFrame()
        y_test = test_df['target'] if len(test_df) > 0 else pd.Series()
        test_dates = test_df.index if len(test_df) > 0 else pd.DatetimeIndex([])

        self.log(f"  Final feature count: {len(available_features)}")

        return X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates, available_features

    def bootstrap_confidence_intervals(self, model, X_train, y_train, X_test, y_test, n_bootstrap=1000):
        """
        Calculate bootstrap confidence intervals by resampling residuals

        Returns:
        - predictions: Point predictions
        - lower_80: 80% lower bound
        - upper_80: 80% upper bound
        - lower_95: 95% lower bound
        - upper_95: 95% upper bound
        """
        # Get training residuals
        train_predictions = model.predict(X_train)
        residuals = y_train.values - train_predictions

        # Point predictions for test set
        predictions = model.predict(X_test)

        # Bootstrap by resampling residuals
        bootstrap_predictions = np.zeros((n_bootstrap, len(predictions)))

        np.random.seed(RANDOM_SEED)
        for i in range(n_bootstrap):
            # Resample residuals with replacement
            bootstrap_residuals = np.random.choice(residuals, size=len(predictions), replace=True)
            # Add resampled residuals to point predictions
            bootstrap_predictions[i, :] = predictions + bootstrap_residuals

        # Calculate percentiles
        lower_80 = np.percentile(bootstrap_predictions, 10, axis=0)
        upper_80 = np.percentile(bootstrap_predictions, 90, axis=0)
        lower_95 = np.percentile(bootstrap_predictions, 2.5, axis=0)
        upper_95 = np.percentile(bootstrap_predictions, 97.5, axis=0)

        return predictions, lower_80, upper_80, lower_95, upper_95

    def calculate_ensemble(self, all_predictions, all_lower_80, all_upper_80, all_lower_95, all_upper_95, weights):
        """
        Calculate weighted ensemble predictions and uncertainty bands

        Ensemble variance accounts for:
        1. Individual model uncertainty (variance within each model)
        2. Model disagreement (variance across models)

        Formula:
        - Mean: Σ(w_i × pred_i)
        - Variance: Σ(w_i² × var_i) + Σ(w_i × (pred_i - mean)²)
        """
        # Weighted mean prediction
        ensemble_pred = np.sum([w * pred for w, pred in zip(weights, all_predictions)], axis=0)

        # For each test point, calculate ensemble variance
        n_points = len(all_predictions[0])
        ensemble_lower_80 = np.zeros(n_points)
        ensemble_upper_80 = np.zeros(n_points)
        ensemble_lower_95 = np.zeros(n_points)
        ensemble_upper_95 = np.zeros(n_points)

        for i in range(n_points):
            # Individual model variances (from their confidence intervals)
            # 80% CI: ~1.28 std, 95% CI: ~1.96 std
            variances_80 = [((upper[i] - lower[i]) / (2 * 1.28)) ** 2
                           for lower, upper in zip(all_lower_80, all_upper_80)]
            variances_95 = [((upper[i] - lower[i]) / (2 * 1.96)) ** 2
                           for lower, upper in zip(all_lower_95, all_upper_95)]

            predictions_at_i = [pred[i] for pred in all_predictions]

            # Model disagreement variance
            disagreement_var = np.sum([w * (pred - ensemble_pred[i]) ** 2
                                      for w, pred in zip(weights, predictions_at_i)])

            # Total variance = weighted individual variances + disagreement
            total_var_80 = np.sum([w**2 * var for w, var in zip(weights, variances_80)]) + disagreement_var
            total_var_95 = np.sum([w**2 * var for w, var in zip(weights, variances_95)]) + disagreement_var

            # Confidence intervals
            ensemble_lower_80[i] = ensemble_pred[i] - 1.28 * np.sqrt(total_var_80)
            ensemble_upper_80[i] = ensemble_pred[i] + 1.28 * np.sqrt(total_var_80)
            ensemble_lower_95[i] = ensemble_pred[i] - 1.96 * np.sqrt(total_var_95)
            ensemble_upper_95[i] = ensemble_pred[i] + 1.96 * np.sqrt(total_var_95)

        return ensemble_pred, ensemble_lower_80, ensemble_upper_80, ensemble_lower_95, ensemble_upper_95

    def train_ridge(self, X_train, y_train):
        """Train Ridge regression with CV"""
        self.log("    [1/5] Training Ridge Regression...")
        param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}

        tscv = TimeSeriesSplit(n_splits=3)
        ridge = Ridge()

        grid_search = GridSearchCV(
            ridge, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.log(f"      Best alpha: {grid_search.best_params_['alpha']}")
        return grid_search.best_estimator_

    def train_lasso(self, X_train, y_train):
        """Train LASSO regression with CV"""
        self.log("    [2/5] Training LASSO Regression...")
        param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

        tscv = TimeSeriesSplit(n_splits=3)
        lasso = Lasso(max_iter=10000)

        grid_search = GridSearchCV(
            lasso, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.log(f"      Best alpha: {grid_search.best_params_['alpha']}")

        # Count non-zero coefficients
        n_features = np.sum(grid_search.best_estimator_.coef_ != 0)
        self.log(f"      Features selected: {n_features}/{len(X_train.columns)}")

        return grid_search.best_estimator_

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest with conservative hyperparameters"""
        self.log("    [3/5] Training Random Forest...")
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "min_samples_split": [10, 20],
            "min_samples_leaf": [5, 10],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        rf = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)

        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost with conservative hyperparameters"""
        self.log("    [4/5] Training XGBoost...")
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_alpha": [0.1, 1.0],
            "reg_lambda": [1.0, 10.0],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1)

        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_

    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting with conservative hyperparameters"""
        self.log("    [5/5] Training Gradient Boosting...")
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.05],
            "subsample": [0.8],
            "min_samples_split": [10, 20],
            "min_samples_leaf": [5, 10],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        gb = GradientBoostingRegressor(random_state=RANDOM_SEED)

        grid_search = GridSearchCV(
            gb,
            param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_

    def evaluate_model_with_ci(self, model, X_train, y_train, X, y, dataset_name):
        """Evaluate model with confidence intervals"""
        if len(X) == 0:
            return None

        # Get predictions with confidence intervals
        predictions, lower_80, upper_80, lower_95, upper_95 = self.bootstrap_confidence_intervals(
            model, X_train, y_train, X, y, n_bootstrap=N_BOOTSTRAP
        )

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / np.where(y == 0, 1, y))) * 100

        # Calculate coverage (what % of actuals fall within CI)
        coverage_80 = np.mean((y.values >= lower_80) & (y.values <= upper_80)) * 100
        coverage_95 = np.mean((y.values >= lower_95) & (y.values <= upper_95)) * 100

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape,
            "coverage_80": coverage_80,
            "coverage_95": coverage_95,
            "predictions": predictions,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "actuals": y.values,
        }

    def train_all_models_for_horizon(self, horizon):
        """Train all models for a specific forecast horizon"""
        self.log(f"\n{'=' * 70}")
        self.log(f"Training Forecasting Models: {self.country.upper()} - Horizon h={horizon}Q")
        self.log(f"{'=' * 70}")

        # Load data
        self.log(f"[Step 1/4] Loading and preparing h={horizon} data...")
        data = self.load_and_prepare_data(horizon)

        if data is None:
            self.log(f"⚠ Skipping h={horizon} - data not available")
            return None

        X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates, feature_cols = data

        # Train models
        self.log(f"\n[Step 2/4] Training 5 models for h={horizon}...")
        models = {
            "Ridge": self.train_ridge(X_train, y_train),
            "LASSO": self.train_lasso(X_train, y_train),
            "Random Forest": self.train_random_forest(X_train, y_train),
            "XGBoost": self.train_xgboost(X_train, y_train),
            "Gradient Boosting": self.train_gradient_boosting(X_train, y_train),
        }

        # Evaluate models with confidence intervals
        self.log(f"\n[Step 3/4] Evaluating models with bootstrap CI...")
        results = {}

        for model_name, model in models.items():
            results[model_name] = {
                "train": self.evaluate_model_with_ci(model, X_train, y_train, X_train, y_train, "Train"),
                "val": self.evaluate_model_with_ci(model, X_train, y_train, X_val, y_val, "Val"),
                "test": self.evaluate_model_with_ci(model, X_train, y_train, X_test, y_test, "Test"),
            }

            # Save model
            model_file = MODEL_DIR / f"{self.country}_h{horizon}_{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, model_file)

        # Calculate ensemble
        self.log(f"\n[Step 4/4] Calculating weighted ensemble...")

        # Use validation R² to determine weights (higher R² = higher weight)
        val_r2_scores = {name: res["val"]["R2"] if res["val"] is not None else -np.inf
                        for name, res in results.items()}

        # Convert R² to weights (use softmax on positive R² values)
        r2_values = np.array(list(val_r2_scores.values()))
        # Shift to make all positive, then normalize
        r2_shifted = r2_values - r2_values.min() + 0.1
        weights_array = r2_shifted / r2_shifted.sum()
        weights = {name: w for name, w in zip(val_r2_scores.keys(), weights_array)}

        self.log(f"  Ensemble weights (based on Val R²):")
        for name, w in weights.items():
            self.log(f"    {name}: {w:.3f}")

        # Calculate ensemble predictions for test set
        if len(X_test) > 0:
            all_predictions = [results[name]["test"]["predictions"] for name in models.keys()]
            all_lower_80 = [results[name]["test"]["lower_80"] for name in models.keys()]
            all_upper_80 = [results[name]["test"]["upper_80"] for name in models.keys()]
            all_lower_95 = [results[name]["test"]["lower_95"] for name in models.keys()]
            all_upper_95 = [results[name]["test"]["upper_95"] for name in models.keys()]
            weights_list = [weights[name] for name in models.keys()]

            ensemble_pred, ensemble_lower_80, ensemble_upper_80, ensemble_lower_95, ensemble_upper_95 = \
                self.calculate_ensemble(all_predictions, all_lower_80, all_upper_80,
                                       all_lower_95, all_upper_95, weights_list)

            # Evaluate ensemble
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / np.where(y_test == 0, 1, y_test))) * 100

            coverage_80 = np.mean((y_test.values >= ensemble_lower_80) & (y_test.values <= ensemble_upper_80)) * 100
            coverage_95 = np.mean((y_test.values >= ensemble_lower_95) & (y_test.values <= ensemble_upper_95)) * 100

            self.log(f"\n  Ensemble Test Performance:")
            self.log(f"    RMSE: {ensemble_rmse:.3f}")
            self.log(f"    R²: {ensemble_r2:.3f}")
            self.log(f"    80% CI Coverage: {coverage_80:.1f}%")
            self.log(f"    95% CI Coverage: {coverage_95:.1f}%")

            # Store ensemble results
            self.ensemble_results[horizon] = {
                "predictions": ensemble_pred,
                "lower_80": ensemble_lower_80,
                "upper_80": ensemble_upper_80,
                "lower_95": ensemble_lower_95,
                "upper_95": ensemble_upper_95,
                "actuals": y_test.values,
                "dates": test_dates,
                "weights": weights,
                "RMSE": ensemble_rmse,
                "MAE": ensemble_mae,
                "R2": ensemble_r2,
                "MAPE": ensemble_mape,
                "coverage_80": coverage_80,
                "coverage_95": coverage_95,
            }

        # Store results
        self.models[horizon] = models
        self.results[horizon] = results

        # Extract feature importance
        self.feature_importance[horizon] = {}
        for model_name in ["Random Forest", "XGBoost", "Gradient Boosting"]:
            if hasattr(models[model_name], "feature_importances_"):
                self.feature_importance[horizon][model_name] = pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "importance": models[model_name].feature_importances_,
                    }
                ).sort_values("importance", ascending=False)

        # Save results
        self.save_results(horizon, test_dates)

        return results

    def save_results(self, horizon, test_dates):
        """Save results to CSV with ensemble predictions"""
        results = self.results[horizon]

        # Create summary table
        summary_data = []
        for model_name, metrics in results.items():
            row = {
                "Country": self.country.upper(),
                "Horizon": f"h{horizon}",
                "Model": model_name,
                "Train_RMSE": f"{metrics['train']['RMSE']:.3f}",
                "Val_RMSE": f"{metrics['val']['RMSE']:.3f}" if metrics['val'] is not None else "N/A",
                "Test_RMSE": f"{metrics['test']['RMSE']:.3f}" if metrics['test'] is not None else "N/A",
                "Train_R2": f"{metrics['train']['R2']:.3f}",
                "Val_R2": f"{metrics['val']['R2']:.3f}" if metrics['val'] is not None else "N/A",
                "Test_R2": f"{metrics['test']['R2']:.3f}" if metrics['test'] is not None else "N/A",
                "Test_Coverage_80": f"{metrics['test']['coverage_80']:.1f}%" if metrics['test'] is not None else "N/A",
                "Test_Coverage_95": f"{metrics['test']['coverage_95']:.1f}%" if metrics['test'] is not None else "N/A",
            }
            summary_data.append(row)

        # Add ensemble row
        if horizon in self.ensemble_results:
            ens = self.ensemble_results[horizon]
            ensemble_row = {
                "Country": self.country.upper(),
                "Horizon": f"h{horizon}",
                "Model": "Ensemble",
                "Train_RMSE": "N/A",
                "Val_RMSE": "N/A",
                "Test_RMSE": f"{ens['RMSE']:.3f}",
                "Train_R2": "N/A",
                "Val_R2": "N/A",
                "Test_R2": f"{ens['R2']:.3f}",
                "Test_Coverage_80": f"{ens['coverage_80']:.1f}%",
                "Test_Coverage_95": f"{ens['coverage_95']:.1f}%",
            }
            summary_data.append(ensemble_row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(OUTPUT_DIR / f"{self.country}_h{horizon}_results.csv", index=False)

        self.log(f"\n  ✓ Results saved: {self.country}_h{horizon}_results.csv")

    def plot_predictions(self, horizon):
        """Plot predictions vs actuals for all models"""
        results = self.results[horizon]

        n_models = len(results) + 1  # +1 for ensemble
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten()

        idx = 0
        for model_name, metrics in results.items():
            ax = axes[idx]

            if metrics["test"] is not None:
                actuals = metrics["test"]["actuals"]
                predictions = metrics["test"]["predictions"]

                ax.scatter(actuals, predictions, alpha=0.6, s=50)
                ax.plot(
                    [actuals.min(), actuals.max()],
                    [actuals.min(), actuals.max()],
                    "r--",
                    linewidth=2,
                    label="Perfect Prediction",
                )

                ax.set_xlabel("Actual GDP Growth (%)", fontsize=10)
                ax.set_ylabel("Predicted GDP Growth (%)", fontsize=10)
                ax.set_title(
                    f"{model_name}\nTest R² = {metrics['test']['R2']:.3f}, RMSE = {metrics['test']['RMSE']:.3f}",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            idx += 1

        # Plot ensemble
        if horizon in self.ensemble_results:
            ax = axes[idx]
            ens = self.ensemble_results[horizon]

            ax.scatter(ens["actuals"], ens["predictions"], alpha=0.6, s=50, color='purple')
            ax.plot(
                [ens["actuals"].min(), ens["actuals"].max()],
                [ens["actuals"].min(), ens["actuals"].max()],
                "r--",
                linewidth=2,
                label="Perfect Prediction",
            )

            ax.set_xlabel("Actual GDP Growth (%)", fontsize=10)
            ax.set_ylabel("Predicted GDP Growth (%)", fontsize=10)
            ax.set_title(
                f"Weighted Ensemble\nTest R² = {ens['R2']:.3f}, RMSE = {ens['RMSE']:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for i in range(idx + 1, len(axes)):
            axes[i].axis('off')

        plt.suptitle(
            f"{self.country.upper()} - {horizon}Q Ahead Forecasting\nPredictions vs Actuals",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / f"{self.country}_h{horizon}_predictions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        self.log(f"  ✓ Saved predictions plot: {self.country}_h{horizon}_predictions.png")

    def plot_feature_importance(self, horizon):
        """Plot feature importance for tree-based models"""
        if horizon not in self.feature_importance or not self.feature_importance[horizon]:
            return

        n_models = len(self.feature_importance[horizon])
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, importance_df) in enumerate(self.feature_importance[horizon].items()):
            ax = axes[idx]

            # Top 15 features
            top_features = importance_df.head(15)

            ax.barh(
                range(len(top_features)),
                top_features["importance"],
                color="steelblue",
                alpha=0.7,
            )
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features["feature"], fontsize=9)
            ax.set_xlabel("Importance", fontsize=10)
            ax.set_title(
                f"{model_name}\nTop 15 Features", fontsize=11, fontweight="bold"
            )
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis="x")

        plt.suptitle(
            f"{self.country.upper()} - {horizon}Q Ahead Feature Importance",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / f"{self.country}_h{horizon}_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        self.log(f"  ✓ Saved feature importance plot: {self.country}_h{horizon}_feature_importance.png")


def main():
    """Main execution"""
    print("=" * 80)
    print(f"GDP QUARTERLY FORECASTING PIPELINE - {COUNTRY.upper()}")
    print("=" * 80)
    print(f"Horizons: {HORIZONS} quarters ahead")
    print(f"Models per horizon: 5 (Ridge, LASSO, RF, XGBoost, GB)")
    print(f"Total models: {len(HORIZONS) * 5} = 20")
    print(f"Ensemble: Weighted by validation R² with uncertainty quantification")
    print(f"Confidence Intervals: Bootstrap (80% and 95%)")
    print("=" * 80)

    pipeline = QuarterlyForecastingPipeline(country=COUNTRY, verbose=True)

    # Train models for all horizons
    for horizon in HORIZONS:
        results = pipeline.train_all_models_for_horizon(horizon)

        if results:
            # Generate visualizations
            print(f"\nGenerating visualizations for h={horizon}...")
            pipeline.plot_predictions(horizon)
            pipeline.plot_feature_importance(horizon)

    print("\n" + "=" * 80)
    print("QUARTERLY FORECASTING PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Figures saved to: {FIG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
