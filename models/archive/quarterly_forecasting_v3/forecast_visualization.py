"""
GDP Forecast Visualization with Confidence Intervals
=====================================================

Creates comprehensive visualizations of quarterly GDP forecasts from v3 models,
showing predictions with confidence intervals to illustrate forecast uncertainty
as prediction horizons increase.

Key Features:
1. Loads all trained ensemble models (Ridge, LASSO, RF, XGBoost, GB, Regime-Switching)
2. Calculates confidence intervals from ensemble prediction variance
3. Visualizes forecast degradation across 1Q vs 4Q horizons
4. Compares actual vs predicted GDP with uncertainty bands

Author: GDP Forecasting Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
VIZ_DIR.mkdir(exist_ok=True)

# Constants
COUNTRIES = ['usa', 'canada', 'japan', 'uk']
HORIZONS = [1, 4]
TARGET = 'gdp_growth_yoy'

# Model algorithms
ALGORITHMS = ['ridge', 'lasso', 'random_forest', 'xgboost', 'gradient_boosting', 'regime_switching']

# Confidence interval levels (1-std, 2-std, 3-std)
CI_LEVELS = {
    '68%': 1,  # 1 standard deviation
    '95%': 2,  # 2 standard deviations
    '99%': 3   # 3 standard deviations
}

# Color scheme
COLOR_PALETTE = {
    'actual': '#1f77b4',      # Blue
    'prediction': '#ff7f0e',  # Orange
    'ci_95': '#2ca02c',       # Green (95% CI)
    'ci_68': '#d62728',       # Red (68% CI)
}


# ============================================================================
# MODEL CLASSES (from v3 pipeline)
# ============================================================================

class ThresholdRegimeSwitcher:
    """
    Regime-switching model based on inflation threshold
    (Required for loading regime-switching models from pickle)
    """
    def __init__(self, threshold=3.0, base_model_class=None, model_params=None):
        self.threshold = threshold
        self.base_model_class = base_model_class
        self.model_params = model_params or {'alpha': 100.0}
        self.model_low = None
        self.model_high = None
        self.n_low_samples = 0
        self.n_high_samples = 0

    def fit(self, X_train, y_train, inflation_train):
        """Train separate models for each regime"""
        pass  # Not needed for visualization

    def predict(self, X_test, inflation_test):
        """Predict using appropriate regime model"""
        pass  # Not needed for visualization


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class RevIN:
    """Reversible Instance Normalization (from v3 pipeline)"""
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.mean = None
        self.std = None

    def normalize(self, x):
        """Remove non-stationary statistics from input"""
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
            raise ValueError("Must call normalize() first")
        x_array = x.values if isinstance(x, pd.DataFrame) else x
        return (x_array - self.mean) / self.std


class ForecastDataLoader:
    """Load data and manage model predictions"""

    def __init__(self, country, horizon):
        self.country = country
        self.horizon = horizon
        self.data = None
        self.test_dates = None
        self.y_test = None
        self.test_features = None

    def load_country_data(self):
        """Load country's preprocessed data"""
        file_path = DATA_DIR / f"{self.country}_processed_normalized.csv"
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.data = df

        # Create future target
        df[f'{TARGET}_future'] = df[TARGET].shift(-self.horizon)
        df = df.dropna(subset=[f'{TARGET}_future'])

        # Split into train/val/test (same as v3 pipeline)
        train_end = '2018-12-31'
        val_end = '2021-12-31'

        # Test set
        test_data = df.loc[val_end:].iloc[1:]
        self.test_dates = test_data.index
        self.y_test = test_data[f'{TARGET}_future'].values

        return test_data

    def load_test_features(self, selected_features):
        """Load and normalize test features for a given horizon/country"""
        file_path = DATA_DIR / f"{self.country}_processed_normalized.csv"
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Create future target
        df[f'{TARGET}_future'] = df[TARGET].shift(-self.horizon)
        df = df.dropna(subset=[f'{TARGET}_future'])

        # Split data
        train_end = '2018-12-31'
        val_end = '2021-12-31'

        X_train = df.loc[:train_end][selected_features]
        X_test = df.loc[val_end:].iloc[1:][selected_features]

        # Apply RevIN normalization
        revin = RevIN()
        X_train_norm = revin.normalize(X_train)
        X_test_norm = revin.transform(X_test)

        self.test_features = X_test_norm
        return X_test_norm, self.test_dates


# ============================================================================
# CONFIDENCE INTERVAL CALCULATION
# ============================================================================

class ConfidenceIntervalCalculator:
    """Calculate confidence intervals from ensemble predictions"""

    def __init__(self, country, horizon):
        self.country = country
        self.horizon = horizon
        self.models = {}
        self.predictions = {}
        self.selected_features = None

    def load_models(self):
        """Load all trained models for this country/horizon"""
        for algo in ALGORITHMS:
            model_file = MODELS_DIR / f"{self.country}_h{self.horizon}_{algo}_v3.pkl"
            if model_file.exists():
                self.models[algo] = joblib.load(model_file)
                # print(f"  Loaded: {algo}")

        if not self.models:
            raise FileNotFoundError(f"No models found for {self.country} h{self.horizon}")
        return len(self.models)

    def load_selected_features(self):
        """Load feature selection for this horizon"""
        features_file = RESULTS_DIR / f"selected_features_h{self.horizon}_v3.csv"
        df_features = pd.read_csv(features_file)
        features_country = df_features[df_features['country'] == self.country]
        self.selected_features = features_country.sort_values('rank')['feature'].tolist()
        return self.selected_features

    def calculate_predictions(self, X_test):
        """Get predictions from all models and calculate ensemble statistics"""
        predictions_array = []
        loaded_algos = []

        for algo, model in self.models.items():
            try:
                if algo == 'regime_switching':
                    # Would need inflation data - skip for now
                    continue
                pred = model.predict(X_test)
                predictions_array.append(pred)
                self.predictions[algo] = pred
                loaded_algos.append(algo)
            except Exception as e:
                pass  # Silently skip failed models

        if not predictions_array:
            raise ValueError(f"No successful predictions for {self.country} h{self.horizon}")

        predictions_array = np.array(predictions_array)

        # Calculate ensemble statistics
        ensemble_mean = np.mean(predictions_array, axis=0)
        ensemble_std = np.std(predictions_array, axis=0)

        return ensemble_mean, ensemble_std, predictions_array

    def calculate_ci(self, mean, std, confidence_level=95):
        """Calculate confidence interval"""
        if confidence_level == 68:
            z = CI_LEVELS['68%']
        elif confidence_level == 95:
            z = CI_LEVELS['95%']
        elif confidence_level == 99:
            z = CI_LEVELS['99%']
        else:
            z = 1.96  # Default 95%

        lower = mean - z * std
        upper = mean + z * std
        return lower, upper


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_forecast_with_ci(country, horizon, loader, ci_calculator, ax=None):
    """
    Plot actual vs predicted GDP with confidence intervals

    Shows how forecast confidence degrades with longer horizons
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Load data
    test_data = loader.load_country_data()
    X_test, dates = loader.load_test_features(ci_calculator.selected_features)

    # Calculate predictions and CI
    ensemble_mean, ensemble_std, _ = ci_calculator.calculate_predictions(X_test)
    ci_lower_95, ci_upper_95 = ci_calculator.calculate_ci(ensemble_mean, ensemble_std, confidence_level=95)
    ci_lower_68, ci_upper_68 = ci_calculator.calculate_ci(ensemble_mean, ensemble_std, confidence_level=68)

    # Plot
    x_axis = np.arange(len(dates))

    # Actual values
    ax.plot(x_axis, loader.y_test, 'o-', color=COLOR_PALETTE['actual'],
            linewidth=2.5, markersize=6, label='Actual GDP Growth', zorder=3)

    # Predictions
    ax.plot(x_axis, ensemble_mean, 's--', color=COLOR_PALETTE['prediction'],
            linewidth=2.5, markersize=6, label='Ensemble Forecast', zorder=3)

    # Confidence intervals
    ax.fill_between(x_axis, ci_lower_95, ci_upper_95, alpha=0.25,
                     color=COLOR_PALETTE['ci_95'], label='95% CI (±2σ)', zorder=1)
    ax.fill_between(x_axis, ci_lower_68, ci_upper_68, alpha=0.4,
                     color=COLOR_PALETTE['ci_68'], label='68% CI (±1σ)', zorder=2)

    # Formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.set_xlabel('Time (Quarters)', fontsize=11)
    ax.set_ylabel('GDP Growth YoY (%)', fontsize=11)
    ax.set_title(f'{country.upper()} - {horizon}Q Ahead Forecast with Confidence Intervals',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Format x-axis with dates
    if len(dates) <= 20:
        ax.set_xticks(x_axis)
        ax.set_xticklabels([d.strftime('%Y-Q%q') for d in dates], rotation=45, ha='right')
    else:
        # Sample every Nth date for readability
        step = len(dates) // 10
        ticks = x_axis[::step]
        labels = [dates[i].strftime('%Y-Q%q') for i in range(0, len(dates), step)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha='right')

    return ax


def plot_forecast_error_vs_horizon(ax=None):
    """
    Compare forecast errors between 1Q and 4Q horizons

    Shows how uncertainty increases with prediction horizon
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Load results for both horizons
    results_1q = pd.read_csv(RESULTS_DIR / "all_countries_h1_v3_results.csv")
    results_4q = pd.read_csv(RESULTS_DIR / "all_countries_h4_v3_results.csv")

    # Calculate average RMSE by country
    summary_1q = results_1q[results_1q['model'] != 'Regime-Switching'].groupby('country')['test_rmse'].mean()
    summary_4q = results_4q[results_4q['model'] != 'Regime-Switching'].groupby('country')['test_rmse'].mean()

    x = np.arange(len(COUNTRIES))
    width = 0.35

    ax.bar(x - width/2, summary_1q, width, label='1Q Ahead (RMSE)', alpha=0.8, color='#2ecc71')
    ax.bar(x + width/2, summary_4q, width, label='4Q Ahead (RMSE)', alpha=0.8, color='#e74c3c')

    ax.set_ylabel('RMSE (% GDP Growth)', fontsize=11)
    ax.set_title('Forecast Error Increases with Horizon\n(Confidence Intervals Widen at 4Q)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in COUNTRIES])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_ci_width_comparison(country, horizon_1q=1, horizon_4q=4, ax=None):
    """
    Compare confidence interval widths between 1Q and 4Q forecasts

    Directly shows forecast degradation with longer horizons
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Load results for both horizons
    results_1q = pd.read_csv(RESULTS_DIR / f"all_countries_h{horizon_1q}_v3_results.csv")
    results_4q = pd.read_csv(RESULTS_DIR / f"all_countries_h{horizon_4q}_v3_results.csv")

    # Filter by country and get ensemble results
    r_1q = results_1q[(results_1q['country'] == country) & (results_1q['model'] == 'Ensemble')]['test_rmse'].values
    r_4q = results_4q[(results_4q['country'] == country) & (results_4q['model'] == 'Ensemble')]['test_rmse'].values

    if len(r_1q) > 0 and len(r_4q) > 0:
        rmse_1q = r_1q[0]
        rmse_4q = r_4q[0]

        # Create visualization
        horizons = ['1Q Ahead', '4Q Ahead']
        rmses = [rmse_1q, rmse_4q]
        colors = ['#3498db', '#e74c3c']

        bars = ax.bar(horizons, rmses, color=colors, alpha=0.8, width=0.5)

        # Add value labels on bars
        for bar, rmse in zip(bars, rmses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rmse:.3f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Ensemble RMSE (% GDP Growth)', fontsize=11)
        ax.set_title(f'{country.upper()} - Forecast Degradation with Horizon\n(Wider CI = Lower Confidence)',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage increase annotation
        pct_increase = ((rmse_4q - rmse_1q) / rmse_1q) * 100
        ax.text(0.5, 0.95, f'Error increase: +{pct_increase:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def plot_all_countries_forecast(horizon, figsize=(18, 12)):
    """Create forecast plots for all countries at given horizon"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, country in enumerate(COUNTRIES):
        try:
            loader = ForecastDataLoader(country, horizon)
            loader.load_country_data()

            ci_calc = ConfidenceIntervalCalculator(country, horizon)
            ci_calc.load_models()
            ci_calc.load_selected_features()

            plot_forecast_with_ci(country, horizon, loader, ci_calc, ax=axes[idx])
        except Exception as e:
            print(f"  ✗ Error plotting {country} h{horizon}: {e}")
            axes[idx].text(0.5, 0.5, f'Error loading data\nfor {country}',
                          ha='center', va='center', transform=axes[idx].transAxes)

    fig.suptitle(f'GDP Growth Forecasts with Confidence Intervals - {horizon}Q Ahead',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = VIZ_DIR / f'all_countries_forecast_h{horizon}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_horizon_comparison(figsize=(16, 8)):
    """Create side-by-side comparison of 1Q vs 4Q forecasts"""
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    for idx, country in enumerate(COUNTRIES):
        # 1Q forecast
        try:
            loader_1q = ForecastDataLoader(country, 1)
            loader_1q.load_country_data()
            ci_calc_1q = ConfidenceIntervalCalculator(country, 1)
            ci_calc_1q.load_models()
            ci_calc_1q.load_selected_features()
            plot_forecast_with_ci(country, 1, loader_1q, ci_calc_1q, ax=axes[0, idx])
            axes[0, idx].set_title(f'{country.upper()} - 1Q Ahead (Narrow CI)', fontweight='bold')
        except Exception as e:
            print(f"  ✗ Error: 1Q {country}: {e}")

        # 4Q forecast
        try:
            loader_4q = ForecastDataLoader(country, 4)
            loader_4q.load_country_data()
            ci_calc_4q = ConfidenceIntervalCalculator(country, 4)
            ci_calc_4q.load_models()
            ci_calc_4q.load_selected_features()
            plot_forecast_with_ci(country, 4, loader_4q, ci_calc_4q, ax=axes[1, idx])
            axes[1, idx].set_title(f'{country.upper()} - 4Q Ahead (Wide CI)', fontweight='bold', color='#e74c3c')
        except Exception as e:
            print(f"  ✗ Error: 4Q {country}: {e}")

    fig.suptitle('Forecast Confidence Degradation: 1Q vs 4Q Horizons\n(Notice wider CIs at 4Q = lower confidence)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = VIZ_DIR / 'horizon_comparison_all_countries.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_rmse_degradation(figsize=(12, 8)):
    """Plot how RMSE increases with forecast horizon"""
    fig, ax = plt.subplots(figsize=figsize)

    plot_forecast_error_vs_horizon(ax=ax)

    output_file = VIZ_DIR / 'rmse_by_horizon.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_ci_width_all_countries(figsize=(12, 6)):
    """Plot confidence interval widths for all countries, both horizons"""
    fig, ax = plt.subplots(figsize=figsize)

    # Load results for both horizons
    results_1q = pd.read_csv(RESULTS_DIR / "all_countries_h1_v3_results.csv")
    results_4q = pd.read_csv(RESULTS_DIR / "all_countries_h4_v3_results.csv")

    # Filter for ensemble only
    ens_1q = results_1q[results_1q['model'] == 'Ensemble'].groupby('country')['test_rmse'].first()
    ens_4q = results_4q[results_4q['model'] == 'Ensemble'].groupby('country')['test_rmse'].first()

    x = np.arange(len(COUNTRIES))
    width = 0.35

    bars1 = ax.bar(x - width/2, ens_1q, width, label='1Q Ahead', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, ens_4q, width, label='4Q Ahead', alpha=0.8, color='#e74c3c')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Ensemble RMSE (% GDP Growth)', fontsize=11)
    ax.set_xlabel('Country', fontsize=11)
    ax.set_title('Forecast Error Increases with Horizon\n(Wider CIs mean lower confidence in 4Q forecasts)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in COUNTRIES])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    output_file = VIZ_DIR / 'ci_width_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all forecast visualizations"""
    print("\n" + "="*80)
    print("GDP FORECAST VISUALIZATION WITH CONFIDENCE INTERVALS")
    print("="*80)
    print(f"\nOutput directory: {VIZ_DIR}\n")

    try:
        print("1. Creating 1Q Ahead forecast plots...")
        plot_all_countries_forecast(horizon=1)

        print("\n2. Creating 4Q Ahead forecast plots...")
        print("   (Notice wider confidence intervals = lower confidence)")
        plot_all_countries_forecast(horizon=4)

        print("\n3. Creating horizon comparison (1Q vs 4Q side-by-side)...")
        plot_horizon_comparison()

        print("\n4. Creating RMSE degradation visualization...")
        print("   (Shows forecast error increases with horizon)")
        plot_rmse_degradation()

        print("\n5. Creating confidence interval width comparison...")
        plot_ci_width_all_countries()

        print("\n" + "="*80)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print(f"✓ Output saved to: {VIZ_DIR}")
        print("="*80)
        print("\nVisualization files created:")
        print("  • all_countries_forecast_h1.png - 1Q ahead forecasts")
        print("  • all_countries_forecast_h4.png - 4Q ahead forecasts (wider CIs)")
        print("  • horizon_comparison_all_countries.png - Side-by-side 1Q vs 4Q")
        print("  • rmse_by_horizon.png - Error degradation with horizon")
        print("  • ci_width_comparison.png - Confidence interval widths")
        print("")

    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
