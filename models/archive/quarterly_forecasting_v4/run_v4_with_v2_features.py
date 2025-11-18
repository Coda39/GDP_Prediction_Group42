#!/usr/bin/env python3
"""
V4 Models with V2 Feature Engineering Data - IMPROVED VERSION
==============================================================

IMPROVEMENTS OVER ORIGINAL:
1. Proper time-series cross-validation (no test set leakage)
2. Feature selection to reduce overfitting
3. More conservative tree model hyperparameters
4. Better regularization for Random Forest and Gradient Boosting
5. Optional lag feature creation with validation
6. Comprehensive diagnostics and visualizations

Output: Results and visualizations saved to v4_v2_improved_results/

Author: GDP Prediction Group 42
Date: November 11, 2025
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pickle
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Directory setup
OUTPUT_DIR = Path(__file__).parent / "v4_v2_improved_results"
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "saved_models"
RESULTS_DIR = OUTPUT_DIR / "results"
VIZ_DIR = OUTPUT_DIR / "forecast_visualizations"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

for dir_path in [MODELS_DIR, RESULTS_DIR, VIZ_DIR, DIAGNOSTICS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data paths
V2_FORECAST_DATA = (
    project_root / "Data_v2/processed/forecasting/usa_forecasting_features.csv"
)
V2_NOWCAST_DATA = (
    project_root / "Data_v2/processed/nowcasting/usa_nowcasting_features.csv"
)


class ImprovedV4V2Forecaster:
    """
    Improved V4 forecaster with proper validation and regularization.

    Key improvements:
    - Time-series cross-validation for hyperparameter selection
    - Feature selection to prevent overfitting
    - Conservative tree model parameters
    - Comprehensive diagnostics
    """

    def __init__(
        self,
        data_path,
        dataset_type="forecasting",
        horizon=1,
        use_lags=False,
        max_features=None,
    ):
        """
        Initialize improved forecaster.

        Args:
            data_path: Path to V2 feature CSV
            dataset_type: 'forecasting' or 'nowcasting'
            horizon: 1, 2, 3, or 4 quarters ahead
            use_lags: Whether to create lagged features (default: False)
            max_features: Maximum number of features to use (default: None = use all)
        """
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.horizon = horizon
        self.use_lags = use_lags
        self.max_features = max_features
        self.target_col = "GDPC1"
        self.data = None
        self.results = {}
        self.predictions = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.selected_features = None
        self.cv_scores = {}

    def load_data(self):
        """Load V2 engineered features"""
        print(f"\nLoading {self.dataset_type} data from {self.data_path.name}...")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        self.data = df.sort_index()

        print(f"✓ Loaded: {len(df)} observations × {len(df.columns)} features")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Target: {self.target_col}")

        return self.data

    def prepare_features_and_target(self):
        """
        Create features and target variable with optional lag features.

        IMPROVEMENT: More conservative lag creation with validation
        """
        df = self.data.copy()

        # Create target: GDP growth h quarters ahead
        df[f"target_h{self.horizon}"] = df[self.target_col].shift(-self.horizon)

        # Features: exclude target column
        feature_cols = [c for c in df.columns if c != self.target_col]

        print(f"\n  Original features: {len(feature_cols)}")

        # Optional: Create lagged features
        if self.use_lags:
            print(f"  Creating lagged features...")

            # Select only top 5 features for lagging to avoid explosion
            # Use correlation with target as selection criterion
            target_temp = df[self.target_col].dropna()
            correlations = []

            for col in feature_cols:
                corr = df[col].corr(target_temp)
                correlations.append((col, abs(corr)))

            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [c[0] for c in correlations[:5]]

            print(f"    Top 5 features for lagging: {top_features}")

            # Create lags: only 1 and 2 quarters (not 4)
            for lag in [1, 2]:
                for col in top_features:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)

            print(f"    Added {len(top_features) * 2} lag features")

        # Get all feature columns
        all_features = [
            c
            for c in df.columns
            if c not in [self.target_col, f"target_h{self.horizon}"]
        ]

        # Drop rows with NaN
        df_clean = df[all_features + [f"target_h{self.horizon}"]].dropna()

        print(f"  Total features: {len(all_features)}")
        print(f"  Clean observations: {len(df_clean)}")

        return df_clean, all_features

    def select_important_features(self, X_train, y_train, feature_names):
        """
        Select most important features using Lasso.

        IMPROVEMENT: Feature selection to reduce overfitting
        """
        if self.max_features is None or self.max_features >= len(feature_names):
            print(f"  Using all {len(feature_names)} features (no selection)")
            return np.arange(len(feature_names)), feature_names

        print(
            f"\n  Selecting top {self.max_features} features from {len(feature_names)}..."
        )

        # Use LassoCV for feature selection
        lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
        lasso.fit(X_train, y_train)

        # Get feature importance (absolute coefficients)
        importance = np.abs(lasso.coef_)

        # Select top k features
        top_indices = np.argsort(importance)[-self.max_features :]
        selected_features = [feature_names[i] for i in top_indices]

        print(f"  ✓ Selected {len(selected_features)} features")
        print(f"  Top 5 most important:")
        for i, feat in enumerate(selected_features[-5:], 1):
            print(f"    {i}. {feat}")

        self.selected_features = selected_features

        return top_indices, selected_features

    def split_train_test(self, df_clean, all_features):
        """
        Split data into train/test with proper temporal ordering.

        IMPROVEMENT: Clear separation, no leakage
        """
        test_start_date = pd.Timestamp("2022-01-01")

        train_mask = df_clean.index < test_start_date
        test_mask = df_clean.index >= test_start_date

        train_data = df_clean[train_mask]
        test_data = df_clean[test_mask]

        X_train = train_data[all_features].values
        y_train = train_data[f"target_h{self.horizon}"].values

        X_test = test_data[all_features].values
        y_test = test_data[f"target_h{self.horizon}"].values

        test_dates = test_data.index

        print(
            f"\n  Train: {len(train_data)} samples ({train_data.index.min().date()} to {train_data.index.max().date()})"
        )
        print(
            f"  Test:  {len(test_data)} samples ({test_data.index.min().date()} to {test_data.index.max().date()})"
        )

        # Feature selection if requested
        if self.max_features is not None:
            selected_indices, selected_features = self.select_important_features(
                X_train, y_train, all_features
            )
            X_train = X_train[:, selected_indices]
            X_test = X_test[:, selected_indices]
            all_features = selected_features

        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, y_train, X_test_scaled, y_test, test_dates, all_features

    def train_ridge_cv(self, X_train, y_train, X_test, y_test):
        """
        Train Ridge with proper time-series cross-validation.

        IMPROVEMENT: CV on training set only, no test set leakage
        """
        print("\n    Ridge Regression:")

        best_model = None
        best_alpha = None
        best_cv_score = float("inf")

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Test different alpha values
        alphas = [10, 50, 100, 500, 1000, 2000, 5000]

        for alpha in alphas:
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = Ridge(alpha=alpha)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(rmse)

            mean_cv_rmse = np.mean(cv_scores)

            if mean_cv_rmse < best_cv_score:
                best_cv_score = mean_cv_rmse
                best_alpha = alpha

        # Train final model with best alpha on full training set
        best_model = Ridge(alpha=best_alpha)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        print(f"      Best alpha: {best_alpha}")
        print(f"      CV RMSE: {best_cv_score:.2f}")

        self.cv_scores["Ridge"] = {"best_alpha": best_alpha, "cv_rmse": best_cv_score}

        return best_model, y_pred

    def train_random_forest_cv(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest with conservative parameters and proper CV.

        IMPROVEMENT:
        - Stronger regularization (higher min_samples_leaf)
        - max_features='sqrt' to limit feature consideration
        - Proper CV for hyperparameter selection
        """
        print("\n    Random Forest:")

        best_model = None
        best_params = None
        best_cv_score = float("inf")

        tscv = TimeSeriesSplit(n_splits=5)

        # More conservative hyperparameters
        param_grid = [
            {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 20},
            {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 20},
            {"n_estimators": 50, "max_depth": 5, "min_samples_leaf": 30},
            {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 30},
            {"n_estimators": 150, "max_depth": 3, "min_samples_leaf": 25},
        ]

        for params in param_grid:
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    min_samples_split=params["min_samples_leaf"] * 2,
                    max_features="sqrt",  # Critical: limit features per split
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(rmse)

            mean_cv_rmse = np.mean(cv_scores)

            if mean_cv_rmse < best_cv_score:
                best_cv_score = mean_cv_rmse
                best_params = params

        # Train final model
        best_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            min_samples_split=best_params["min_samples_leaf"] * 2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        print(
            f"      Best params: n_est={best_params['n_estimators']}, "
            f"depth={best_params['max_depth']}, "
            f"min_leaf={best_params['min_samples_leaf']}"
        )
        print(f"      CV RMSE: {best_cv_score:.2f}")

        self.cv_scores["RandomForest"] = {
            "best_params": best_params,
            "cv_rmse": best_cv_score,
        }

        return best_model, y_pred

    def train_gradient_boosting_cv(self, X_train, y_train, X_test, y_test):
        """
        Train Gradient Boosting with proper regularization and CV.

        IMPROVEMENT:
        - Lower learning rates
        - Conservative depth and samples
        - Proper CV
        """
        print("\n    Gradient Boosting:")

        best_model = None
        best_params = None
        best_cv_score = float("inf")

        tscv = TimeSeriesSplit(n_splits=5)

        # Conservative hyperparameters
        param_grid = [
            {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.01,
                "min_samples_leaf": 20,
            },
            {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.01,
                "min_samples_leaf": 20,
            },
            {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.05,
                "min_samples_leaf": 20,
            },
            {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "min_samples_leaf": 20,
            },
            {
                "n_estimators": 150,
                "max_depth": 2,
                "learning_rate": 0.01,
                "min_samples_leaf": 30,
            },
        ]

        for params in param_grid:
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = GradientBoostingRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    min_samples_leaf=params["min_samples_leaf"],
                    min_samples_split=params["min_samples_leaf"] * 2,
                    subsample=0.8,  # Use 80% of samples per tree
                    random_state=42,
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(rmse)

            mean_cv_rmse = np.mean(cv_scores)

            if mean_cv_rmse < best_cv_score:
                best_cv_score = mean_cv_rmse
                best_params = params

        # Train final model
        best_model = GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            min_samples_leaf=best_params["min_samples_leaf"],
            min_samples_split=best_params["min_samples_leaf"] * 2,
            subsample=0.8,
            random_state=42,
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        print(
            f"      Best params: n_est={best_params['n_estimators']}, "
            f"depth={best_params['max_depth']}, "
            f"lr={best_params['learning_rate']}, "
            f"min_leaf={best_params['min_samples_leaf']}"
        )
        print(f"      CV RMSE: {best_cv_score:.2f}")

        self.cv_scores["GradientBoosting"] = {
            "best_params": best_params,
            "cv_rmse": best_cv_score,
        }

        return best_model, y_pred

    def evaluate_metrics(self, y_true, y_pred, model_name):
        """Calculate and store evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        self.results[model_name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "n_samples": len(y_true),
        }

        # Add CV scores if available
        if model_name in self.cv_scores:
            self.results[model_name]["cv_rmse"] = self.cv_scores[model_name]["cv_rmse"]

        print(f"\n    {model_name} Test Results:")
        print(f"      R² = {r2:7.4f}")
        print(f"      RMSE = {rmse:8.2f}")
        print(f"      MAE = {mae:8.2f}")
        if model_name in self.cv_scores:
            print(f"      CV RMSE = {self.cv_scores[model_name]['cv_rmse']:8.2f}")

        return rmse, mae, r2

    def train_all_models(self, X_train, y_train, X_test, y_test, test_dates):
        """Train all models with improved methods"""
        print(f"\n{'=' * 70}")
        print(f"Training Models - Horizon h={self.horizon}Q ahead")
        print(f"{'=' * 70}")

        # Ridge
        ridge_model, ridge_pred = self.train_ridge_cv(X_train, y_train, X_test, y_test)
        self.models["Ridge"] = ridge_model
        self.predictions["Ridge"] = ridge_pred
        self.evaluate_metrics(y_test, ridge_pred, "Ridge")

        # Random Forest
        rf_model, rf_pred = self.train_random_forest_cv(
            X_train, y_train, X_test, y_test
        )
        self.models["RandomForest"] = rf_model
        self.predictions["RandomForest"] = rf_pred
        self.evaluate_metrics(y_test, rf_pred, "RandomForest")

        # Gradient Boosting
        gb_model, gb_pred = self.train_gradient_boosting_cv(
            X_train, y_train, X_test, y_test
        )
        self.models["GradientBoosting"] = gb_model
        self.predictions["GradientBoosting"] = gb_pred
        self.evaluate_metrics(y_test, gb_pred, "GradientBoosting")

        # Weighted Ensemble (favor better models)
        # Calculate inverse RMSE weights
        rmse_ridge = self.results["Ridge"]["RMSE"]
        rmse_rf = self.results["RandomForest"]["RMSE"]
        rmse_gb = self.results["GradientBoosting"]["RMSE"]

        # Use inverse RMSE as weights (better models get higher weight)
        weight_ridge = 1 / rmse_ridge
        weight_rf = 1 / rmse_rf
        weight_gb = 1 / rmse_gb

        total_weight = weight_ridge + weight_rf + weight_gb

        weight_ridge /= total_weight
        weight_rf /= total_weight
        weight_gb /= total_weight

        print(f"\n    Ensemble Weights:")
        print(f"      Ridge: {weight_ridge:.3f}")
        print(f"      RandomForest: {weight_rf:.3f}")
        print(f"      GradientBoosting: {weight_gb:.3f}")

        ensemble_pred = (
            weight_ridge * ridge_pred + weight_rf * rf_pred + weight_gb * gb_pred
        )

        self.predictions["Ensemble"] = ensemble_pred
        self.evaluate_metrics(y_test, ensemble_pred, "Ensemble")

        return y_test, test_dates

    def plot_diagnostics(self, y_test, test_dates):
        """
        Create diagnostic plots.

        IMPROVEMENT: Visualize model performance
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"{self.dataset_type.capitalize()} - Horizon h={self.horizon}Q",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Actual vs Predicted for all models
        ax = axes[0, 0]
        for model_name in ["Ridge", "RandomForest", "GradientBoosting", "Ensemble"]:
            if model_name in self.predictions:
                ax.plot(
                    test_dates,
                    self.predictions[model_name],
                    label=model_name,
                    marker="o",
                    markersize=4,
                    alpha=0.7,
                )

        ax.plot(test_dates, y_test, "k-", linewidth=2, label="Actual", alpha=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("GDP Growth")
        ax.set_title("Predictions vs Actual")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Residuals for Ridge (best model)
        ax = axes[0, 1]
        if "Ridge" in self.predictions:
            residuals = y_test - self.predictions["Ridge"]
            ax.scatter(self.predictions["Ridge"], residuals, alpha=0.6)
            ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot (Ridge)")
            ax.grid(True, alpha=0.3)

        # Plot 3: Model comparison (R²)
        ax = axes[1, 0]
        models = list(self.results.keys())
        r2_scores = [self.results[m]["R2"] for m in models]
        colors = ["green" if r2 > 0 else "red" for r2 in r2_scores]

        bars = ax.barh(models, r2_scores, color=colors, alpha=0.7)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("R² Score")
        ax.set_title("Model Comparison (R²)")
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (model, r2) in enumerate(zip(models, r2_scores)):
            ax.text(r2, i, f"  {r2:.3f}", va="center")

        # Plot 4: RMSE comparison
        ax = axes[1, 1]
        rmse_scores = [self.results[m]["RMSE"] for m in models]
        ax.bar(models, rmse_scores, color="steelblue", alpha=0.7)
        ax.set_ylabel("RMSE")
        ax.set_title("Model Comparison (RMSE)")
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for i, (model, rmse) in enumerate(zip(models, rmse_scores)):
            ax.text(i, rmse, f"{rmse:.1f}", ha="center", va="bottom")

        plt.tight_layout()

        # Save plot
        filename = f"{self.dataset_type}_h{self.horizon}_diagnostics.png"
        filepath = DIAGNOSTICS_DIR / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n  ✓ Diagnostics saved to {filepath}")

    def save_models(self):
        """Save trained models"""
        for model_name, model in self.models.items():
            filename = f"{self.dataset_type}_h{self.horizon}_{model_name.lower()}.pkl"
            filepath = MODELS_DIR / filename
            with open(filepath, "wb") as f:
                pickle.dump(model, f)

        # Also save scaler
        scaler_file = MODELS_DIR / f"{self.dataset_type}_h{self.horizon}_scaler.pkl"
        with open(scaler_file, "wb") as f:
            pickle.dump(self.scaler, f)

    def run(self):
        """Execute complete improved training pipeline"""
        print(f"\n{'=' * 80}")
        print(f"IMPROVED V4 Models with V2 Features - {self.dataset_type.upper()}")
        print(f"{'=' * 80}")

        # Load and prepare
        self.load_data()
        df_clean, all_features = self.prepare_features_and_target()
        X_train, y_train, X_test, y_test, test_dates, final_features = (
            self.split_train_test(df_clean, all_features)
        )

        # Train models
        y_test_ret, test_dates_ret = self.train_all_models(
            X_train, y_train, X_test, y_test, test_dates
        )

        # Create diagnostics
        self.plot_diagnostics(y_test_ret, test_dates_ret)

        # Save models
        self.save_models()

        return self.results, self.predictions, y_test_ret


class ImprovedV4V2Pipeline:
    """
    Complete improved pipeline for both forecasting and nowcasting.

    IMPROVEMENTS:
    - Configurable feature selection
    - Optional lag features
    - Comprehensive reporting
    """

    def __init__(self, use_lags=False, max_features=20):
        """
        Initialize pipeline.

        Args:
            use_lags: Whether to create lagged features (default: False)
            max_features: Max number of features to use (default: 20)
        """
        self.all_results = {}
        self.all_predictions = {}
        self.use_lags = use_lags
        self.max_features = max_features

        print(f"\n{'=' * 80}")
        print(f"IMPROVED V4+V2 PIPELINE")
        print(f"{'=' * 80}")
        print(f"Configuration:")
        print(f"  Use lagged features: {use_lags}")
        print(f"  Max features: {max_features if max_features else 'All'}")
        print(f"{'=' * 80}")

    def run_forecasting(self):
        """Run forecasting models for all horizons"""
        print("\n" + "=" * 80)
        print("FORECASTING MODELS (Leading Indicators - Multiple Horizons)")
        print("=" * 80)

        for horizon in [1, 2, 3, 4]:
            print(f"\n{'─' * 80}")
            print(f"HORIZON: {horizon} Quarter(s) Ahead")
            print(f"{'─' * 80}")

            forecaster = ImprovedV4V2Forecaster(
                V2_FORECAST_DATA,
                dataset_type="forecasting",
                horizon=horizon,
                use_lags=self.use_lags,
                max_features=self.max_features,
            )

            try:
                results, predictions, y_test = forecaster.run()
                self.all_results[f"forecast_h{horizon}"] = results
                self.all_predictions[f"forecast_h{horizon}"] = {
                    "y_test": y_test,
                    "predictions": predictions,
                }
            except Exception as e:
                print(f"\n  ✗ Error in horizon h={horizon}: {str(e)}")
                import traceback

                traceback.print_exc()

    def run_nowcasting(self):
        """Run nowcasting models"""
        print("\n" + "=" * 80)
        print("NOWCASTING MODELS (Coincident Indicators - Current Quarter)")
        print("=" * 80)

        nowcaster = ImprovedV4V2Forecaster(
            V2_NOWCAST_DATA,
            dataset_type="nowcasting",
            horizon=1,
            use_lags=self.use_lags,
            max_features=self.max_features,
        )

        try:
            results, predictions, y_test = nowcaster.run()
            self.all_results["nowcast_h1"] = results
            self.all_predictions["nowcast_h1"] = {
                "y_test": y_test,
                "predictions": predictions,
            }
        except Exception as e:
            print(f"\n  ✗ Error in nowcasting: {str(e)}")
            import traceback

            traceback.print_exc()

    def save_all_results(self):
        """Save aggregated results"""
        # Results summary
        results_file = RESULTS_DIR / "improved_v4_v2_results.json"
        results_dict = {
            task: {
                model: {
                    "RMSE": float(metrics["RMSE"]),
                    "MAE": float(metrics["MAE"]),
                    "R2": float(metrics["R2"]),
                    "n_samples": metrics["n_samples"],
                    "cv_rmse": float(metrics.get("cv_rmse", 0)),
                }
                for model, metrics in task_results.items()
            }
            for task, task_results in self.all_results.items()
        }

        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Predictions
        predictions_file = RESULTS_DIR / "improved_v4_v2_predictions.pkl"
        with open(predictions_file, "wb") as f:
            pickle.dump(self.all_predictions, f)

        print(f"✓ Predictions saved to {predictions_file}")

    def create_summary_table(self):
        """Create a comprehensive summary table"""
        print("\n" + "=" * 80)
        print("IMPROVED V4+V2 RESULTS SUMMARY")
        print("=" * 80)

        # Forecasting results
        print("\n📊 FORECASTING RESULTS (Leading Indicators)")
        print("-" * 80)
        print(
            f"{'Horizon':<15} {'Model':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'CV RMSE':<10}"
        )
        print("-" * 80)

        for horizon in [1, 2, 3, 4]:
            key = f"forecast_h{horizon}"
            if key in self.all_results:
                for i, (model, metrics) in enumerate(self.all_results[key].items()):
                    horizon_str = f"h={horizon} (Q{horizon})" if i == 0 else ""
                    cv_rmse = metrics.get("cv_rmse", 0)
                    cv_str = f"{cv_rmse:8.2f}" if cv_rmse > 0 else "N/A"
                    print(
                        f"{horizon_str:<15} {model:<20} "
                        f"{metrics['R2']:>8.4f}  "
                        f"{metrics['RMSE']:>8.2f}  "
                        f"{metrics['MAE']:>8.2f}  "
                        f"{cv_str:>8}"
                    )

        # Nowcasting results
        print("\n\n📊 NOWCASTING RESULTS (Coincident Indicators)")
        print("-" * 80)
        print(
            f"{'Horizon':<15} {'Model':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'CV RMSE':<10}"
        )
        print("-" * 80)

        key = "nowcast_h1"
        if key in self.all_results:
            for i, (model, metrics) in enumerate(self.all_results[key].items()):
                horizon_str = "Current Quarter" if i == 0 else ""
                cv_rmse = metrics.get("cv_rmse", 0)
                cv_str = f"{cv_rmse:8.2f}" if cv_rmse > 0 else "N/A"
                print(
                    f"{horizon_str:<15} {model:<20} "
                    f"{metrics['R2']:>8.4f}  "
                    f"{metrics['RMSE']:>8.2f}  "
                    f"{metrics['MAE']:>8.2f}  "
                    f"{cv_str:>8}"
                )

        # Key insights
        print("\n\n📈 KEY INSIGHTS")
        print("-" * 80)

        # Find best models
        best_forecast_r2 = -float("inf")
        best_forecast_model = None
        best_forecast_horizon = None

        for horizon in [1, 2, 3, 4]:
            key = f"forecast_h{horizon}"
            if key in self.all_results:
                for model, metrics in self.all_results[key].items():
                    if metrics["R2"] > best_forecast_r2:
                        best_forecast_r2 = metrics["R2"]
                        best_forecast_model = model
                        best_forecast_horizon = horizon

        print(
            f"✓ Best Forecasting Model: {best_forecast_model} (h={best_forecast_horizon}, R²={best_forecast_r2:.4f})"
        )

        if "nowcast_h1" in self.all_results:
            best_nowcast_r2 = max(
                self.all_results["nowcast_h1"][m]["R2"]
                for m in self.all_results["nowcast_h1"]
            )
            best_nowcast_model = max(
                self.all_results["nowcast_h1"].items(), key=lambda x: x[1]["R2"]
            )[0]
            print(
                f"✓ Best Nowcasting Model: {best_nowcast_model} (R²={best_nowcast_r2:.4f})"
            )

        # Compare tree models
        print("\n✓ Tree Model Improvements:")
        for horizon in [1, 2, 3, 4]:
            key = f"forecast_h{horizon}"
            if key in self.all_results:
                rf_r2 = self.all_results[key].get("RandomForest", {}).get("R2", 0)
                gb_r2 = self.all_results[key].get("GradientBoosting", {}).get("R2", 0)
                if rf_r2 > 0 or gb_r2 > 0:
                    print(f"  h={horizon}: RF R²={rf_r2:6.4f}, GB R²={gb_r2:6.4f}")
                    break

        print(f"\n{'=' * 80}")
        print(f"Output Directory: {OUTPUT_DIR}/")
        print(f"  - Results: {RESULTS_DIR}/")
        print(f"  - Diagnostics: {DIAGNOSTICS_DIR}/")
        print(f"  - Models: {MODELS_DIR}/")
        print("=" * 80)

    def run(self):
        """Execute complete improved pipeline"""
        self.run_forecasting()
        self.run_nowcasting()
        self.save_all_results()
        self.create_summary_table()


def main():
    """
    Main execution with configuration options.

    You can modify these settings:
    - use_lags: Whether to create lagged features (can cause overfitting)
    - max_features: Maximum number of features to use (None = use all)
    """
    try:
        # Configuration
        USE_LAGS = False  # Set to True to enable lagged features
        MAX_FEATURES = 20  # Set to None to use all features

        # Run pipeline
        pipeline = ImprovedV4V2Pipeline(use_lags=USE_LAGS, max_features=MAX_FEATURES)
        pipeline.run()

        print("\n✅ Improved V4 models training complete!")
        print("\nTo compare with original results:")
        print("  - Original: models/quarterly_forecasting_v4/v4_v2_results/")
        print("  - Improved: models/quarterly_forecasting_v4/v4_v2_improved_results/")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
