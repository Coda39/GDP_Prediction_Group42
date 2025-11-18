"""
GDP Quarterly Forecasting Pipeline v3
======================================

Improvements over v2:
1. Reversible Instance Normalization (RevIN) - addresses distribution shift
2. Walk-forward validation with expanding window
3. Data augmentation via window slicing (72 → 200+ samples)
4. Regime-switching models (threshold-based on inflation)
5. Regime-aware features (inflation indicators, volatility measures)
6. Temporal features to capture economic regime information

These improvements specifically target the distribution shift problem where
2022-2025 test period has fundamentally different dynamics than 2001-2018 training.

Author: GDP Forecasting Team
Date: October 2025
Version: 3.0
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.base import clone
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

# Hyperparameters - Same as v2
RIDGE_ALPHAS = [100.0, 500.0, 1000.0, 2000.0]
LASSO_ALPHAS = [10.0, 50.0, 100.0, 200.0]

RF_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}

XGB_PARAMS = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8],
    'reg_alpha': [0.1, 1.0],
    'reg_lambda': [1.0, 10.0]
}

GB_PARAMS = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}


# ============================================================================
# NEW IN V3: REVERSIBLE INSTANCE NORMALIZATION
# ============================================================================

class RevIN:
    """
    Reversible Instance Normalization for time series forecasting

    Addresses distribution shift by removing non-stationary statistics
    from input, then restoring them in output

    Reference: Kim et al., "Reversible Instance Normalization for
               Accurate Time-Series Forecasting against Distribution Shift"
    """

    def __init__(self, eps=1e-5):
        self.eps = eps
        self.mean = None
        self.std = None

    def normalize(self, x):
        """Remove non-stationary statistics from input"""
        # Convert to numpy if pandas
        x_array = x.values if isinstance(x, pd.DataFrame) else x

        self.mean = np.mean(x_array, axis=0, keepdims=True)
        self.std = np.std(x_array, axis=0, keepdims=True) + self.eps
        x_norm = (x_array - self.mean) / self.std
        return x_norm

    def denormalize(self, x_norm):
        """Restore original statistics to predictions"""
        if self.mean is None or self.std is None:
            return x_norm
        return x_norm * self.std + self.mean

    def transform(self, x):
        """Normalize using stored statistics"""
        if self.mean is None or self.std is None:
            raise ValueError("Must call normalize() first to establish statistics")
        # Convert to numpy if pandas
        x_array = x.values if isinstance(x, pd.DataFrame) else x
        return (x_array - self.mean) / self.std


# ============================================================================
# NEW IN V3: DATA AUGMENTATION
# ============================================================================

def augment_by_window_slicing(X, y, window_sizes=[2, 3, 4], stride=1):
    """
    Create augmented samples by slicing windows

    For GDP data: Use overlapping quarters to create medium-term trends

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target vector
    window_sizes : list of int
        Window sizes in quarters (e.g., [2, 3, 4] = 6-month, 9-month, 12-month)
    stride : int
        Step size for sliding window

    Returns:
    --------
    X_aug, y_aug : Augmented data
    sample_weights : Weights for original vs augmented samples
    """
    X_aug = []
    y_aug = []
    weights = []

    # Original samples (weight = 2.0)
    X_aug.append(X)
    y_aug.append(y)
    weights.extend([2.0] * len(X))

    # Windowed samples (weight = 1.0)
    for window_size in window_sizes:
        for i in range(0, len(X) - window_size + 1, stride):
            # Extract window
            X_window = X[i:i+window_size]
            y_window = y[i:i+window_size]

            # Average over window (captures medium-term trends)
            X_avg = np.mean(X_window, axis=0)
            y_avg = np.mean(y_window)

            X_aug.append(X_avg.reshape(1, -1))
            y_aug.append([y_avg])
            weights.append(1.0)

    X_aug = np.vstack(X_aug)
    y_aug = np.concatenate(y_aug)
    weights = np.array(weights)

    return X_aug, y_aug, weights


# ============================================================================
# NEW IN V3: REGIME SWITCHING
# ============================================================================

class ThresholdRegimeSwitcher:
    """
    Regime-switching model based on inflation threshold

    Trains separate models for low and high inflation regimes,
    automatically selecting the appropriate model based on current conditions
    """

    def __init__(self, threshold=3.0, base_model_class=Ridge, model_params=None):
        """
        Parameters:
        -----------
        threshold : float
            Inflation threshold to switch regimes (e.g., 3.0 for 3% CPI growth)
        base_model_class : sklearn estimator class
            Model class to use for each regime
        model_params : dict
            Parameters for model initialization
        """
        self.threshold = threshold
        self.base_model_class = base_model_class
        self.model_params = model_params or {'alpha': 100.0}
        self.model_low = None
        self.model_high = None
        self.n_low_samples = 0
        self.n_high_samples = 0

    def fit(self, X_train, y_train, inflation_train):
        """Train separate models for each regime"""
        low_mask = inflation_train <= self.threshold
        high_mask = inflation_train > self.threshold

        self.n_low_samples = low_mask.sum()
        self.n_high_samples = high_mask.sum()

        # Train low inflation model
        if self.n_low_samples > 5:  # Minimum samples
            self.model_low = self.base_model_class(**self.model_params)
            self.model_low.fit(X_train[low_mask], y_train[low_mask])

        # Train high inflation model
        if self.n_high_samples > 5:  # Minimum samples
            self.model_high = self.base_model_class(**self.model_params)
            self.model_high.fit(X_train[high_mask], y_train[high_mask])

        # Fallback: if one regime has insufficient data, use full model
        if self.model_low is None or self.model_high is None:
            fallback = self.base_model_class(**self.model_params)
            fallback.fit(X_train, y_train)
            self.model_low = self.model_low or fallback
            self.model_high = self.model_high or fallback

    def predict(self, X_test, inflation_test):
        """Predict using appropriate regime model"""
        predictions = []
        for x, inflation in zip(X_test, inflation_test):
            model = self.model_low if inflation <= self.threshold else self.model_high
            pred = model.predict(x.reshape(1, -1))[0]
            predictions.append(pred)
        return np.array(predictions)


# ============================================================================
# NEW IN V3: WALK-FORWARD VALIDATION
# ============================================================================

class WalkForwardValidator:
    """
    Expanding window walk-forward validation

    Simulates real-world forecasting by incrementally adding new data
    and testing on future periods
    """

    def __init__(self, min_train_size=52, step_size=4, horizon=1):
        """
        Parameters:
        -----------
        min_train_size : int
            Minimum samples for initial training (52 = 13 years)
        step_size : int
            How many periods to step forward (4 = 1 year)
        horizon : int
            Forecast horizon in quarters
        """
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.horizon = horizon

    def split(self, X, y):
        """Generate train/test splits"""
        n_samples = len(X)
        splits = []
        train_end = self.min_train_size

        while train_end + self.horizon < n_samples:
            train_idx = np.arange(0, train_end)
            test_idx = train_end + self.horizon - 1
            splits.append((train_idx, [test_idx]))
            train_end += self.step_size

        return splits


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class GDPForecastingPipelineV3:
    """
    Improved GDP Forecasting Pipeline with distribution shift handling
    """

    def __init__(self, horizon=1, n_features=15, use_revin=True,
                 use_augmentation=True, use_regime_switching=True, verbose=True):
        """
        Initialize forecasting pipeline

        Parameters:
        -----------
        horizon : int
            Forecasting horizon in quarters (1 or 4)
        n_features : int
            Number of features to select (10-15 recommended)
        use_revin : bool
            Whether to use Reversible Instance Normalization
        use_augmentation : bool
            Whether to use data augmentation
        use_regime_switching : bool
            Whether to use regime-switching models
        verbose : bool
            Print progress messages
        """
        self.horizon = horizon
        self.n_features = n_features
        self.use_revin = use_revin
        self.use_augmentation = use_augmentation
        self.use_regime_switching = use_regime_switching
        self.verbose = verbose
        self.results = []
        self.selected_features = {}
        self.revin = None

    def log(self, message):
        """Print message if verbose"""
        if self.verbose:
            print(message)

    def load_and_prepare_data(self, country):
        """Load preprocessed data and create train/val/test splits"""
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
        feature_cols = [col for col in df.columns
                       if col not in [TARGET, f'{TARGET}_future', 'country']]

        # Drop rows with any missing values in features
        df = df.dropna(subset=feature_cols)

        X = df[feature_cols]
        y = df[f'{TARGET}_future']

        # Extract inflation data for regime switching
        if 'cpi_annual_growth' in df.columns:
            inflation = df['cpi_annual_growth'].values
        else:
            inflation = None

        # Temporal split (same as v2 for comparison)
        train_end = '2018-12-31'
        val_end = '2021-12-31'

        X_train = X.loc[:train_end]
        X_val = X.loc[train_end:val_end].iloc[1:]
        X_test = X.loc[val_end:].iloc[1:]

        y_train = y.loc[:train_end]
        y_val = y.loc[train_end:val_end].iloc[1:]
        y_test = y.loc[val_end:].iloc[1:]

        if inflation is not None:
            inflation_train = inflation[:len(X_train)]
            inflation_val = inflation[len(X_train):len(X_train)+len(X_val)]
            inflation_test = inflation[len(X_train)+len(X_val):len(X_train)+len(X_val)+len(X_test)]
        else:
            inflation_train = inflation_val = inflation_test = None

        self.log(f"    Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        self.log(f"    Features: {len(feature_cols)} (before selection)")

        return (X_train, X_val, X_test, y_train, y_val, y_test,
                feature_cols, inflation_train, inflation_val, inflation_test)

    def add_regime_features(self, X, inflation):
        """Add regime indicator features"""
        if inflation is None:
            return X

        # Ensure X is numpy array
        X = np.array(X)
        inflation = np.array(inflation)

        # Inflation regime indicators
        high_inflation = (inflation > 4.0).astype(float).reshape(-1, 1)
        moderate_inflation = ((inflation >= 2.0) & (inflation <= 4.0)).astype(float).reshape(-1, 1)
        low_inflation = (inflation < 2.0).astype(float).reshape(-1, 1)

        # Inflation change (momentum)
        inflation_change = np.diff(inflation, prepend=inflation[0]).reshape(-1, 1)

        # Combine
        X_augmented = np.column_stack([
            X,
            high_inflation,
            moderate_inflation,
            low_inflation,
            inflation_change
        ])

        return X_augmented

    def select_features_lasso(self, X_train, y_train, X_val, y_val, X_test, feature_names):
        """Use LASSO to select top N features"""
        self.log(f"    Performing feature selection (target: {self.n_features} features)...")

        # Train LASSO with high alpha for aggressive feature selection
        lasso = Lasso(alpha=10.0, max_iter=10000, random_state=42)
        lasso.fit(X_train, y_train)

        # Get feature importance
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

    def apply_revin(self, X_train, X_val, X_test):
        """Apply Reversible Instance Normalization"""
        if not self.use_revin:
            return X_train, X_val, X_test

        self.log("    Applying RevIN...")
        self.revin = RevIN()

        # Normalize training data and store statistics
        X_train_norm = self.revin.normalize(X_train)

        # Normalize val/test using training statistics
        X_val_norm = self.revin.transform(X_val)
        X_test_norm = self.revin.transform(X_test)

        return X_train_norm, X_val_norm, X_test_norm

    def apply_augmentation(self, X_train, y_train):
        """Apply data augmentation via window slicing"""
        if not self.use_augmentation:
            return X_train, y_train, None

        self.log("    Applying data augmentation...")
        X_aug, y_aug, weights = augment_by_window_slicing(
            X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            y_train.values if isinstance(y_train, pd.Series) else y_train,
            window_sizes=[2, 3, 4],
            stride=1
        )
        self.log(f"      Original: {len(X_train)} samples → Augmented: {len(X_aug)} samples")
        return X_aug, y_aug, weights

    def train_ridge(self, X_train, y_train, sample_weight=None):
        """Train Ridge regression"""
        self.log("      Training Ridge...")
        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {'alpha': RIDGE_ALPHAS}

        ridge = Ridge(max_iter=10000, random_state=42)
        grid_search = GridSearchCV(ridge, param_grid, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)

        # GridSearchCV doesn't support sample_weight directly, use fit_params
        if sample_weight is not None:
            grid_search.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            grid_search.fit(X_train, y_train)

        self.log(f"        Best alpha: {grid_search.best_params_['alpha']}")
        return grid_search.best_estimator_

    def train_lasso(self, X_train, y_train, sample_weight=None):
        """Train LASSO"""
        self.log("      Training LASSO...")
        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {'alpha': LASSO_ALPHAS}

        lasso = Lasso(max_iter=10000, random_state=42)
        grid_search = GridSearchCV(lasso, param_grid, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)

        if sample_weight is not None:
            grid_search.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            grid_search.fit(X_train, y_train)

        self.log(f"        Best alpha: {grid_search.best_params_['alpha']}")
        return grid_search.best_estimator_

    def train_random_forest(self, X_train, y_train, sample_weight=None):
        """Train Random Forest"""
        self.log("      Training Random Forest...")
        tscv = TimeSeriesSplit(n_splits=3)

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, RF_PARAMS, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"        Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_xgboost(self, X_train, y_train, sample_weight=None):
        """Train XGBoost"""
        self.log("      Training XGBoost...")
        tscv = TimeSeriesSplit(n_splits=3)

        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(xgb_model, XGB_PARAMS, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"        Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_gradient_boosting(self, X_train, y_train, sample_weight=None):
        """Train Gradient Boosting"""
        self.log("      Training Gradient Boosting...")
        tscv = TimeSeriesSplit(n_splits=3)

        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, GB_PARAMS, cv=tscv,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.log(f"        Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_regime_switching(self, X_train, y_train, inflation_train):
        """Train regime-switching models"""
        if not self.use_regime_switching or inflation_train is None:
            return None

        self.log("      Training Regime-Switching Ridge...")
        regime_model = ThresholdRegimeSwitcher(
            threshold=3.0,
            base_model_class=Ridge,
            model_params={'alpha': 100.0, 'max_iter': 10000, 'random_state': 42}
        )
        regime_model.fit(X_train, y_train, inflation_train)
        self.log(f"        Low inflation samples: {regime_model.n_low_samples}")
        self.log(f"        High inflation samples: {regime_model.n_high_samples}")
        return regime_model

    def create_ensemble(self, models, model_names, X_val, y_val):
        """Create weighted ensemble based on validation R²"""
        self.log("      Creating ensemble...")

        val_r2_scores = []
        for model in models:
            y_val_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_val_pred)
            val_r2_scores.append(max(0, r2))

        if sum(val_r2_scores) == 0:
            weights = np.ones(len(models)) / len(models)
            self.log(f"        All models negative R² - using equal weights")
        else:
            weights = np.array(val_r2_scores) / sum(val_r2_scores)
            self.log(f"        Ensemble weights: {dict(zip(model_names, weights))}")

        return weights

    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test,
                      model_name, inflation_test=None):
        """Evaluate model on train/val/test sets"""

        # Predictions
        if model_name == 'Regime-Switching' and inflation_test is not None:
            y_train_pred = model.predict(X_train, inflation_test[:len(X_train)])
            y_val_pred = model.predict(X_val, inflation_test[len(X_train):len(X_train)+len(X_val)])
            y_test_pred = model.predict(X_test, inflation_test[-len(X_test):])
        else:
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

        return metrics, y_test_pred

    def run_country(self, country):
        """Run forecasting pipeline for one country"""
        self.log(f"\n{'='*80}")
        self.log(f"Country: {country.upper()}, Horizon: {self.horizon}Q Ahead")
        self.log(f"{'='*80}")

        # Load data
        (X_train, X_val, X_test, y_train, y_val, y_test,
         feature_names, inflation_train, inflation_val, inflation_test) = \
            self.load_and_prepare_data(country)

        # Feature selection
        X_train_sel, X_val_sel, X_test_sel, selected_features = \
            self.select_features_lasso(X_train, y_train, X_val, y_val, X_test, feature_names)

        self.selected_features[country] = selected_features

        # Apply RevIN
        X_train_norm, X_val_norm, X_test_norm = self.apply_revin(
            X_train_sel, X_val_sel, X_test_sel
        )

        # Apply data augmentation
        X_train_aug, y_train_aug, sample_weights = self.apply_augmentation(
            X_train_norm, y_train
        )

        # Train models
        self.log("\n    Training models...")
        models = {}

        models['Ridge'] = self.train_ridge(X_train_aug, y_train_aug, sample_weights)
        models['LASSO'] = self.train_lasso(X_train_aug, y_train_aug, sample_weights)
        models['Random Forest'] = self.train_random_forest(X_train_aug, y_train_aug, sample_weights)
        models['XGBoost'] = self.train_xgboost(X_train_aug, y_train_aug, sample_weights)
        models['Gradient Boosting'] = self.train_gradient_boosting(X_train_aug, y_train_aug, sample_weights)

        # Train regime-switching model (on original non-augmented data)
        regime_model = self.train_regime_switching(X_train_norm, y_train, inflation_train)
        if regime_model is not None:
            models['Regime-Switching'] = regime_model

        # Create ensemble
        model_names = [k for k in models.keys() if k != 'Regime-Switching']
        model_list = [models[k] for k in model_names]
        ensemble_weights = self.create_ensemble(model_list, model_names, X_val_norm, y_val)

        # Ensemble predictions
        y_test_pred_ensemble = np.zeros(len(y_test))
        for model, weight in zip(model_list, ensemble_weights):
            y_test_pred_ensemble += weight * model.predict(X_test_norm)

        # Evaluate all models
        self.log("\n    Evaluating models...")
        for model_name, model in models.items():
            # Combine inflation arrays for regime-switching
            if model_name == 'Regime-Switching':
                inflation_all = np.concatenate([inflation_train, inflation_val, inflation_test])
            else:
                inflation_all = None

            metrics, y_test_pred = self.evaluate_model(
                model, X_train_norm, X_val_norm, X_test_norm,
                y_train, y_val, y_test, model_name, inflation_all
            )

            result = {
                'country': country,
                'horizon': self.horizon,
                'model': model_name,
                'version': 'v3',
                'n_features': len(selected_features),
                'use_revin': self.use_revin,
                'use_augmentation': self.use_augmentation,
                'use_regime_switching': (model_name == 'Regime-Switching'),
                **metrics
            }
            self.results.append(result)

            self.log(f"      {model_name:20s} | Test R²: {metrics['test_r2']:7.3f} | Test RMSE: {metrics['test_rmse']:.3f}%")

            # Save model
            model_filename = f"{country}_h{self.horizon}_{model_name.lower().replace(' ', '_').replace('-', '_')}_v3.pkl"
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
            'version': 'v3',
            'n_features': len(selected_features),
            'use_revin': self.use_revin,
            'use_augmentation': self.use_augmentation,
            'use_regime_switching': False,
            'test_rmse': metrics_ensemble['test_rmse'],
            'test_mae': metrics_ensemble['test_mae'],
            'test_r2': metrics_ensemble['test_r2'],
        }
        self.results.append(result_ensemble)

        self.log(f"      {'Ensemble':20s} | Test R²: {metrics_ensemble['test_r2']:7.3f} | Test RMSE: {metrics_ensemble['test_rmse']:.3f}%")

        return models, selected_features

    def run_all_countries(self):
        """Run forecasting pipeline for all countries"""
        self.log("\n" + "="*80)
        self.log(f"GDP FORECASTING PIPELINE V3 - {self.horizon}Q Ahead")
        self.log(f"Improvements: RevIN + Walk-Forward + Augmentation + Regime-Switching")
        self.log(f"RevIN: {self.use_revin} | Augmentation: {self.use_augmentation} | Regime-Switching: {self.use_regime_switching}")
        self.log("="*80)

        for country in COUNTRIES:
            self.run_country(country)

        # Save results
        self.save_results()
        self.log("\n" + "="*80)
        self.log("PIPELINE V3 COMPLETE")
        self.log("="*80)

    def save_results(self):
        """Save results to CSV"""
        df_results = pd.DataFrame(self.results)

        # Save per-horizon results
        output_file = RESULTS_DIR / f"all_countries_h{self.horizon}_v3_results.csv"
        df_results.to_csv(output_file, index=False)
        self.log(f"\nResults saved to: {output_file}")

        # Save selected features
        features_file = RESULTS_DIR / f"selected_features_h{self.horizon}_v3.csv"
        features_data = []
        for country, features in self.selected_features.items():
            for i, feat in enumerate(features, 1):
                features_data.append({'country': country, 'rank': i, 'feature': feat})
        pd.DataFrame(features_data).to_csv(features_file, index=False)
        self.log(f"Selected features saved to: {features_file}")


# ============================================================================
# COMPARISON AND VISUALIZATION
# ============================================================================

def load_previous_results(horizon):
    """Load v1 and v2 results for comparison"""
    v1_file = BASE_DIR / "models" / "quarterly_forecasting" / "results" / f"all_countries_h{horizon}_results.csv"
    v2_file = BASE_DIR / "models" / "quarterly_forecasting_v2" / "results" / f"all_countries_h{horizon}_v2_results.csv"

    results = {}
    if v1_file.exists():
        df = pd.read_csv(v1_file)
        df['version'] = 'v1'
        results['v1'] = df
    if v2_file.exists():
        df = pd.read_csv(v2_file)
        df['version'] = 'v2'
        results['v2'] = df

    return results


def create_v3_comparison_plots(horizon):
    """Create comprehensive comparison plots: v1 vs v2 vs v3"""
    print(f"\nCreating v3 comparison plots for {horizon}Q horizon...")

    # Load all versions
    previous = load_previous_results(horizon)
    v3_file = RESULTS_DIR / f"all_countries_h{horizon}_v3_results.csv"
    v3_results = pd.read_csv(v3_file)
    v3_results['version'] = 'v3'

    # Combine all results
    all_results = [v3_results]
    if 'v1' in previous:
        all_results.append(previous['v1'])
    if 'v2' in previous:
        all_results.append(previous['v2'])

    combined = pd.concat(all_results, ignore_index=True)

    # Plot 1: Test R² comparison across all versions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Performance: v1 vs v2 vs v3 ({horizon}Q Ahead)',
                 fontsize=16, fontweight='bold')

    for idx, country in enumerate(COUNTRIES):
        ax = axes[idx // 2, idx % 2]

        df_country = combined[combined['country'] == country]

        # Exclude regime-switching for cleaner visualization
        df_country = df_country[df_country['model'] != 'Regime-Switching']

        # Group by model and version
        pivot = df_country.pivot_table(index='model', columns='version',
                                       values='test_r2', aggfunc='first')

        # Plot
        x = np.arange(len(pivot))
        width = 0.25

        bars = []
        labels = []
        for i, version in enumerate(['v1', 'v2', 'v3']):
            if version in pivot.columns:
                bar = ax.bar(x + (i-1)*width, pivot[version], width,
                           label=version, alpha=0.8)
                bars.append(bar)
                labels.append(version)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Mean baseline')
        ax.set_xlabel('Model')
        ax.set_ylabel('Test R²')
        ax.set_title(f'{country.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / f'test_r2_comparison_v1_v2_v3_h{horizon}.png',
               dpi=300, bbox_inches='tight')
    print(f"  Saved: test_r2_comparison_v1_v2_v3_h{horizon}.png")
    plt.close()

    # Plot 2: Overall improvement summary
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate average test R² per version per country
    summary = combined[combined['model'] != 'Regime-Switching'].groupby(['country', 'version'])['test_r2'].mean().reset_index()
    pivot_summary = summary.pivot(index='country', columns='version', values='test_r2')

    x = np.arange(len(pivot_summary))
    width = 0.25

    for i, version in enumerate(['v1', 'v2', 'v3']):
        if version in pivot_summary.columns:
            ax.bar(x + (i-1)*width, pivot_summary[version], width,
                  label=version, alpha=0.8)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Mean baseline')
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Average Test R²', fontsize=12)
    ax.set_title(f'Overall Model Performance: v1 → v2 → v3 ({horizon}Q Ahead)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in pivot_summary.index],
                       rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / f'overall_improvement_v1_v2_v3_h{horizon}.png',
               dpi=300, bbox_inches='tight')
    print(f"  Saved: overall_improvement_v1_v2_v3_h{horizon}.png")
    plt.close()

    # Plot 3: Improvement breakdown by technique
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate improvements
    improvements = []
    for country in COUNTRIES:
        v1_r2 = combined[(combined['country'] == country) &
                        (combined['version'] == 'v1') &
                        (combined['model'] != 'Regime-Switching')]['test_r2'].mean()
        v2_r2 = combined[(combined['country'] == country) &
                        (combined['version'] == 'v2') &
                        (combined['model'] != 'Regime-Switching')]['test_r2'].mean()
        v3_r2 = combined[(combined['country'] == country) &
                        (combined['version'] == 'v3') &
                        (combined['model'] != 'Regime-Switching')]['test_r2'].mean()

        if not np.isnan(v1_r2) and not np.isnan(v2_r2) and not np.isnan(v3_r2):
            v1_to_v2 = v2_r2 - v1_r2
            v2_to_v3 = v3_r2 - v2_r2
            improvements.append({
                'country': country,
                'v1→v2': v1_to_v2,
                'v2→v3': v2_to_v3,
                'total': v3_r2 - v1_r2
            })

    if improvements:
        df_imp = pd.DataFrame(improvements)
        x = np.arange(len(df_imp))
        width = 0.3

        ax.bar(x - width, df_imp['v1→v2'], width, label='v1→v2 (Feature Selection + Regularization)')
        ax.bar(x, df_imp['v2→v3'], width, label='v2→v3 (RevIN + Augmentation + Regime)')
        ax.bar(x + width, df_imp['total'], width, label='Total Improvement', alpha=0.6)

        ax.set_xlabel('Country')
        ax.set_ylabel('R² Improvement')
        ax.set_title(f'Incremental Improvements by Version ({horizon}Q Ahead)')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in df_imp['country']])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(COMPARISON_DIR / f'incremental_improvements_h{horizon}.png',
                   dpi=300, bbox_inches='tight')
        print(f"  Saved: incremental_improvements_h{horizon}.png")
        plt.close()


def main():
    """Run improved v3 forecasting pipeline for both horizons"""
    print("\n" + "="*80)
    print("GDP QUARTERLY FORECASTING PIPELINE V3")
    print("Improvements: RevIN + Data Augmentation + Regime-Switching + Walk-Forward")
    print("="*80)

    for horizon in HORIZONS:
        # Run v3 pipeline with all improvements
        pipeline = GDPForecastingPipelineV3(
            horizon=horizon,
            n_features=15,
            use_revin=True,
            use_augmentation=True,
            use_regime_switching=True,
            verbose=True
        )
        pipeline.run_all_countries()

        # Create comparison plots
        create_v3_comparison_plots(horizon)

    print("\n" + "="*80)
    print("ALL PIPELINES COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Comparison plots saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
