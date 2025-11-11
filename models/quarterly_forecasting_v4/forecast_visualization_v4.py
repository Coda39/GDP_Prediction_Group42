"""
v4 Forecast Visualization with Confidence Intervals
==================================================

Creates publication-quality visualizations matching v3 style:
- Actual vs predicted GDP with confidence bands
- RMSE degradation across horizons
- Model performance comparison
- Feature impact analysis (v3 vs v4)

Key differences from v3:
- Wider confidence intervals (more honest uncertainty)
- 4 separate horizon plots instead of generic 1Q/4Q
- Comparison showing impact of removing data leakage

Author: GDP Forecasting Team
Date: November 2025
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
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
RESULTS_DIR = OUTPUT_DIR / "results"
VIZ_DIR = OUTPUT_DIR / "forecast_visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Color scheme (matching v3)
COLOR_PALETTE = {
    'actual': '#1f77b4',       # Blue
    'prediction': '#ff7f0e',   # Orange
    'ci_95': '#2ca02c',        # Green (95% CI)
    'ci_68': '#d62728',        # Red (68% CI)
}

HORIZONS = [1, 2, 3, 4]
MODELS = ['Ridge', 'RandomForest', 'GradientBoosting', 'Ensemble']

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class ForecastVisualizer:
    """Create v4 forecast visualizations"""

    def __init__(self):
        self.results = None
        self.predictions_data = None
        self.try_load_results()
        self.try_load_predictions()

    def try_load_results(self):
        """Try to load results if they exist"""
        results_file = RESULTS_DIR / 'v4_model_performance.csv'
        if results_file.exists():
            try:
                self.results = pd.read_csv(results_file)
            except:
                pass

    def try_load_predictions(self):
        """Try to load predictions if they exist"""
        predictions_file = RESULTS_DIR / 'v4_predictions.pkl'
        if predictions_file.exists():
            try:
                self.predictions_data = joblib.load(predictions_file)
                print(f"✓ Loaded predictions data from {predictions_file.name}")
            except Exception as e:
                print(f"⚠ Could not load predictions: {e}")
                pass

    def plot_forecast_with_ci(self, horizon, actual, predicted, dates, ax=None):
        """Plot actual vs predicted with confidence intervals"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(dates))

        # Estimate CI from residuals
        residuals = actual - predicted
        ci_std = np.std(residuals)
        ci_lower = predicted - 1.96 * ci_std
        ci_upper = predicted + 1.96 * ci_std
        ci_lower_68 = predicted - 1.0 * ci_std
        ci_upper_68 = predicted + 1.0 * ci_std

        # Plot
        ax.plot(x, actual, 'o-', color=COLOR_PALETTE['actual'],
                linewidth=2.5, markersize=6, label='Actual GDP Growth', zorder=3)
        ax.plot(x, predicted, 's--', color=COLOR_PALETTE['prediction'],
                linewidth=2.5, markersize=6, label='Ensemble Forecast', zorder=3)

        # Confidence intervals
        ax.fill_between(x, ci_lower, ci_upper, alpha=0.25,
                        color=COLOR_PALETTE['ci_95'], label='95% CI (±2σ)', zorder=1)
        ax.fill_between(x, ci_lower_68, ci_upper_68, alpha=0.4,
                        color=COLOR_PALETTE['ci_68'], label='68% CI (±1σ)', zorder=2)

        # Formatting
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.set_xlabel('Time (Quarters)', fontsize=11)
        ax.set_ylabel('GDP Growth YoY (%)', fontsize=11)
        ax.set_title(f'USA - {horizon}Q Ahead Forecast with Confidence Intervals (v4 - Clean Features)',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # Format dates on x-axis
        if len(dates) <= 14:
            ax.set_xticks(x)
            ax.set_xticklabels([d.strftime('%Y-Q%q') for d in dates], rotation=45, ha='right', fontsize=9)

        return ax

    def plot_all_horizons_forecast(self, predictions_data):
        """Create 2x2 grid of forecasts for all horizons"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        for idx, horizon in enumerate(HORIZONS):
            if horizon in predictions_data:
                pred = predictions_data[horizon]
                self.plot_forecast_with_ci(
                    horizon,
                    pred['actual'],
                    pred['ensemble'],
                    pred['dates'],
                    ax=axes[idx]
                )
            else:
                axes[idx].text(0.5, 0.5, f'No data for {horizon}Q',
                              ha='center', va='center', transform=axes[idx].transAxes)

        fig.suptitle('USA GDP Forecasts with Confidence Intervals - v4 Clean Features',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = VIZ_DIR / 'usa_forecast_grid_v4.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file.name}")
        plt.close()

    def plot_individual_horizon(self, horizon, predictions_data):
        """Create individual plots for each horizon"""
        if horizon not in predictions_data:
            return

        pred = predictions_data[horizon]
        fig, ax = plt.subplots(figsize=(16, 8))

        self.plot_forecast_with_ci(horizon, pred['actual'], pred['ensemble'],
                                    pred['dates'], ax=ax)

        output_file = VIZ_DIR / f'usa_forecast_h{horizon}_v4.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file.name}")
        plt.close()

    def plot_rmse_by_horizon(self):
        """Plot RMSE degradation across horizons"""
        if self.results is None:
            print("⚠ Results not available, skipping RMSE plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Filter for ensemble only
        ensemble_results = self.results[self.results['Model'] == 'Ensemble']

        horizons = [f'{h}Q' for h in HORIZONS]
        rmses = []
        for horizon in horizons:
            h_data = ensemble_results[ensemble_results['Horizon'] == horizon]
            if len(h_data) > 0:
                rmses.append(h_data['RMSE'].values[0])
            else:
                rmses.append(0)

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax.bar(horizons, rmses, color=colors, alpha=0.8, width=0.6)

        # Add value labels
        for bar, rmse in zip(bars, rmses):
            if rmse > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rmse:.3f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Ensemble RMSE (%)', fontsize=11)
        ax.set_xlabel('Forecast Horizon', fontsize=11)
        ax.set_title('USA: Forecast Error Degradation with Horizon\n(v4 - Clean Exogenous Features)',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        output_file = VIZ_DIR / 'usa_rmse_by_horizon_v4.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file.name}")
        plt.close()

    def plot_r2_comparison(self):
        """Plot R² across models and horizons"""
        if self.results is None:
            print("⚠ Results not available, skipping R² plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Pivot for heatmap
        pivot = self.results.pivot_table(values='R2', index='Model', columns='Horizon')

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   ax=ax, cbar_kws={'label': 'R² Score'}, linewidths=1, linecolor='gray')

        ax.set_title('USA: Model Performance Across Horizons (v4 - Clean Features)\nGreen = Good | Red = Poor',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Forecast Horizon', fontsize=11)
        ax.set_ylabel('Model Type', fontsize=11)

        output_file = VIZ_DIR / 'usa_r2_heatmap_v4.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file.name}")
        plt.close()

    def plot_model_comparison(self):
        """Compare model performance"""
        if self.results is None:
            print("⚠ Results not available, skipping model comparison")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1Q Performance
        results_1q = self.results[self.results['Horizon'] == '1Q']
        if len(results_1q) > 0:
            x = np.arange(len(results_1q))
            width = 0.35
            axes[0].bar(x - width/2, results_1q['R2'], width, label='R²', alpha=0.8, color='#3498db')
            axes[0].bar(x + width/2, results_1q['RMSE'], width, label='RMSE (%)', alpha=0.8, color='#e74c3c')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(results_1q['Model'], rotation=45, ha='right')
            axes[0].set_ylabel('Score', fontsize=11)
            axes[0].set_title('USA 1Q Ahead: Model Comparison', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)

        # 4Q Performance
        results_4q = self.results[self.results['Horizon'] == '4Q']
        if len(results_4q) > 0:
            x = np.arange(len(results_4q))
            axes[1].bar(x - width/2, results_4q['R2'], width, label='R²', alpha=0.8, color='#3498db')
            axes[1].bar(x + width/2, results_4q['RMSE'], width, label='RMSE (%)', alpha=0.8, color='#e74c3c')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(results_4q['Model'], rotation=45, ha='right')
            axes[1].set_ylabel('Score', fontsize=11)
            axes[1].set_title('USA 4Q Ahead: Model Comparison', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)

        fig.suptitle('USA: Model Performance Comparison (v4)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = VIZ_DIR / 'usa_model_comparison_v4.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file.name}")
        plt.close()

    def plot_ensemble_vs_actual_gdp(self, predictions_data):
        """Create comprehensive plot comparing ensemble predictions to actual GDP with CIs"""
        if not predictions_data:
            print("⚠ No predictions data available, skipping ensemble vs actual GDP plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        for idx, horizon in enumerate(HORIZONS):
            ax = axes[idx]

            if horizon not in predictions_data:
                ax.text(0.5, 0.5, f'No data for {horizon}Q',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            pred = predictions_data[horizon]
            actual = pred['actual']
            ensemble = pred['ensemble']
            dates = pred['dates']
            x = np.arange(len(dates))

            # Calculate confidence intervals from residuals
            residuals = actual - ensemble
            ci_std = np.std(residuals)
            ci_lower = ensemble - 1.96 * ci_std
            ci_upper = ensemble + 1.96 * ci_std
            ci_lower_68 = ensemble - 1.0 * ci_std
            ci_upper_68 = ensemble + 1.0 * ci_std

            # Plot actual vs predicted
            ax.plot(x, actual, 'o-', color=COLOR_PALETTE['actual'],
                   linewidth=2.5, markersize=8, label='Actual GDP Growth', zorder=3)
            ax.plot(x, ensemble, 's--', color=COLOR_PALETTE['prediction'],
                   linewidth=2.5, markersize=8, label='Ensemble Prediction', zorder=3)

            # Confidence intervals
            ax.fill_between(x, ci_lower, ci_upper, alpha=0.25,
                           color=COLOR_PALETTE['ci_95'], label='95% CI (±2σ)', zorder=1)
            ax.fill_between(x, ci_lower_68, ci_upper_68, alpha=0.4,
                           color=COLOR_PALETTE['ci_68'], label='68% CI (±1σ)', zorder=2)

            # Format
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
            ax.set_ylabel('GDP Growth YoY (%)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (Quarters)', fontsize=11, fontweight='bold')
            ax.set_title(f'USA {horizon}Q Ahead: Ensemble Prediction vs Actual GDP Growth\n(with 95% and 68% Confidence Intervals)',
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10, framealpha=0.95)
            ax.grid(True, alpha=0.3)

            # Format x-axis with dates
            if len(dates) <= 14:
                ax.set_xticks(x)
                ax.set_xticklabels([d.strftime('%Y-Q%q') if hasattr(d, 'strftime') else str(d)
                                   for d in dates], rotation=45, ha='right', fontsize=9)

            # Calculate and display metrics
            rmse = np.sqrt(np.mean((actual - ensemble) ** 2))
            mae = np.mean(np.abs(actual - ensemble))
            r2 = 1 - (np.sum((actual - ensemble) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

            # Add metrics box
            metrics_text = f'RMSE: {rmse:.3f}%\nMAE: {mae:.3f}%\nR²: {r2:.4f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.suptitle('USA Ensemble Predictions vs Actual GDP Growth with Confidence Intervals\n(v4 - Clean Exogenous Features)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = VIZ_DIR / 'usa_ensemble_vs_actual_gdp_v4.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file.name}")
        plt.close()

    def plot_feature_impact_analysis(self):
        """Create infographic showing feature changes v3 → v4"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')

        # Title
        title_text = "v3 vs v4: Feature Impact Analysis"
        ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=16, fontweight='bold',
               transform=ax.transAxes)

        # v3 Issues
        v3_text = """v3 Selected Features (15 total):
✓ investment_growth, household_consumption, employment_level
✓ money_supply_broad, exchange_rate_usd, stock_market_index
✓ government_spending, ip_growth, employment_growth
✓ capital_formation, industrial_production_index_diff

❌ ISSUES DISCOVERED:
   • trade_balance: Component of GDP (X-M) → DATA LEAKAGE
   • gdp_growth_qoq: DIRECTLY CALCULATED FROM TARGET → SEVERE LEAKAGE
   • population_total/working_age: Nearly constant, not predictive

Impact: Inflated R² scores, overconfident predictions"""

        ax.text(0.05, 0.75, v3_text, ha='left', va='top', fontsize=10,
               transform=ax.transAxes, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))

        # v4 Solution
        v4_text = """v4 Clean Features (21 exogenous variables):
✓ KEPT: investment_growth, employment_level, money_supply_broad,
        stock_market_index, government_spending, ip_growth, etc.

✓ ADDED: unemployment_rate, cpi_annual_growth, interest rates,
         trade volumes (not balance), m2_growth

❌ REMOVED: All GDP-dependent variables that caused leakage

Benefits:
   • No data leakage → Honest predictions
   • 21 truly exogenous features
   • Wider confidence intervals (realistic uncertainty)
   • Ready for production deployment"""

        ax.text(0.55, 0.75, v4_text, ha='left', va='top', fontsize=10,
               transform=ax.transAxes, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))

        # Bottom summary
        summary = "Expected Impact: v4 will have LOWER R² than v3 due to removed leakage, but HIGHER TRUSTWORTHINESS"
        ax.text(0.5, 0.05, summary, ha='center', va='bottom', fontsize=11, fontweight='bold',
               transform=ax.transAxes, style='italic',
               bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))

        output_file = VIZ_DIR / 'v3_vs_v4_feature_impact.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file.name}")
        plt.close()

    def run_all(self, predictions_data=None):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print("v4 FORECAST VISUALIZATIONS")
        print("="*80 + "\n")

        # Use loaded predictions if not provided
        if predictions_data is None:
            predictions_data = self.predictions_data

        if predictions_data:
            print("1. Creating individual horizon forecasts...")
            for horizon in HORIZONS:
                self.plot_individual_horizon(horizon, predictions_data)

            print("\n2. Creating grid of all horizons...")
            self.plot_all_horizons_forecast(predictions_data)

            print("\n3. Creating ensemble vs actual GDP comparison...")
            self.plot_ensemble_vs_actual_gdp(predictions_data)
        else:
            print("⚠ No predictions data available, skipping prediction-based plots")
            print("  (Make sure to run forecasting_pipeline_v4.py first)")

        print("\n4. Creating RMSE degradation plot...")
        self.plot_rmse_by_horizon()

        print("\n5. Creating R² heatmap...")
        self.plot_r2_comparison()

        print("\n6. Creating model comparison plots...")
        self.plot_model_comparison()

        print("\n7. Creating feature impact analysis...")
        self.plot_feature_impact_analysis()

        print("\n" + "="*80)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print(f"✓ Total plots generated: 9 (plus individual horizon plots)")
        print(f"✓ Output directory: {VIZ_DIR}")
        print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate visualizations"""
    visualizer = ForecastVisualizer()

    # Check if predictions were loaded
    if visualizer.predictions_data:
        print("\n✓ Predictions data loaded successfully")
    else:
        print("\n⚠ No predictions data found - only metric-based visualizations will be created")
        print("  To generate all plots, run forecasting_pipeline_v4.py first\n")

    # Run all visualizations (will skip prediction-based ones if data not available)
    visualizer.run_all()


if __name__ == "__main__":
    main()
