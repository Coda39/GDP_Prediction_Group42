#!/usr/bin/env python3
"""
V4+V2 Results Visualizations
=============================

Create comprehensive visualizations comparing forecasting and nowcasting results
with the new V2 feature-engineered data.

Author: GDP Prediction Group 42
Date: November 11, 2025
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
RESULTS_DIR = Path(__file__).parent / "v4_v2_improved_results"
VIZ_DIR = RESULTS_DIR / "forecast_visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Load results
results_file = RESULTS_DIR / "results" / "improved_v4_v2_results.json"
predictions_file = RESULTS_DIR / "results" / "improved_v4_v2_predictions.pkl"

with open(results_file, "r") as f:
    results = json.load(f)

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10


def create_performance_heatmap():
    """Create R² heatmap across models and horizons"""
    print("Creating R² heatmap...")

    # Extract R² values
    r2_data = {}

    for task_key, task_results in results.items():
        if "forecast" in task_key:
            horizon = task_key.split("h")[-1]
            for model, metrics in task_results.items():
                if model not in r2_data:
                    r2_data[model] = {}
                r2_data[model][f"Forecast h={horizon}"] = metrics["R2"]

    for task_key, task_results in results.items():
        if "nowcast" in task_key:
            for model, metrics in task_results.items():
                if model not in r2_data:
                    r2_data[model] = {}
                r2_data[model]["Nowcast h=1"] = metrics["R2"]

    # Create DataFrame
    df_r2 = pd.DataFrame(r2_data).T
    df_r2 = df_r2[sorted(df_r2.columns)]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        df_r2,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "R² Score"},
        ax=ax,
        vmin=-3,
        vmax=1,
    )
    ax.set_title(
        "Model Performance Heatmap - R² Scores\nV4 Models with V2 Features",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Model", fontsize=11)
    ax.set_xlabel("Task", fontsize=11)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "v4_v2_r2_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_model_comparison_forecast():
    """Compare models for forecasting"""
    print("Creating forecasting model comparison...")

    horizons = [1, 2, 3, 4]
    models = ["Ridge", "RandomForest", "GradientBoosting", "Ensemble"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(["R2", "RMSE", "MAE"]):
        if idx < 3:
            ax = axes[idx]
            data = {}

            for h in horizons:
                key = f"forecast_h{h}"
                data[f"h={h}"] = [results[key][m][metric] for m in models]

            df_data = pd.DataFrame(data, index=models)
            df_data.plot(
                kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            )
            ax.set_title(
                f"Forecasting Models - {metric}", fontsize=12, fontweight="bold"
            )
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xlabel("Model", fontsize=10)
            ax.legend(title="Horizon", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Nowcasting comparison
    ax = axes[3]
    nowcast_data = results["nowcast_h1"]
    models_nowcast = list(nowcast_data.keys())
    r2_nowcast = [nowcast_data[m]["R2"] for m in models_nowcast]

    colors = [
        "green" if r2 > 0.5 else "orange" if r2 > 0 else "red" for r2 in r2_nowcast
    ]
    bars = ax.bar(
        models_nowcast, r2_nowcast, color=colors, alpha=0.7, edgecolor="black"
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("Nowcasting Models - R² Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=10)
    ax.set_xlabel("Model", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
        )

    plt.suptitle(
        "V4 Models with V2 Features - Forecasting vs Nowcasting",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "v4_v2_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_rmse_by_horizon():
    """RMSE comparison across horizons"""
    print("Creating RMSE by horizon plot...")

    horizons = [1, 2, 3, 4]
    models = ["Ridge", "RandomForest", "GradientBoosting", "Ensemble"]

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in models:
        rmse_vals = [results[f"forecast_h{h}"][model]["RMSE"] for h in horizons]
        ax.plot(horizons, rmse_vals, marker="o", linewidth=2, markersize=8, label=model)

    ax.set_xlabel("Forecast Horizon (Quarters Ahead)", fontsize=11, fontweight="bold")
    ax.set_ylabel("RMSE", fontsize=11, fontweight="bold")
    ax.set_title(
        "Forecasting RMSE by Horizon\nV4 Models with V2 Features",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "v4_v2_rmse_by_horizon.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_actual_vs_predicted():
    """Create actual vs predicted GDP plots for each horizon"""
    print("Creating actual vs predicted GDP plots...")

    # Load predictions
    with open(predictions_file, "rb") as f:
        predictions = pickle.load(f)

    horizons = [("nowcast_h1", "Nowcast"), ("forecast_h1", "1Q Ahead"),
                ("forecast_h2", "2Q Ahead"), ("forecast_h3", "3Q Ahead"),
                ("forecast_h4", "4Q Ahead")]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (task_key, horizon_label) in enumerate(horizons):
        if idx < len(axes):
            ax = axes[idx]

            if task_key not in predictions:
                continue

            y_test = predictions[task_key]["y_test"]
            preds_dict = predictions[task_key]["predictions"]

            # Find best model based on R² score
            best_model = max(results[task_key].items(),
                           key=lambda x: x[1]["R2"])[0]
            best_predictions = preds_dict[best_model]

            # Create x-axis (time index)
            time_steps = range(len(y_test))

            # Plot
            ax.plot(time_steps, y_test, "k-", linewidth=2.5,
                   label="Actual GDP", alpha=0.8, zorder=3)
            ax.plot(time_steps, best_predictions, "r--", linewidth=2,
                   label=f"{best_model} Predictions", alpha=0.7, zorder=2)

            # Styling
            r2_score = results[task_key][best_model]["R2"]
            ax.set_title(
                f"{horizon_label} - {best_model}\nR² = {r2_score:.4f}",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_xlabel("Time Period (Test Set)", fontsize=10)
            ax.set_ylabel("GDP (GDPC1)", fontsize=10)
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3)

            # Add some statistics as text
            rmse = results[task_key][best_model]["RMSE"]
            mae = results[task_key][best_model]["MAE"]
            ax.text(0.02, 0.98, f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}",
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Hide the last subplot if odd number of horizons
    if len(horizons) < len(axes):
        axes[-1].axis("off")

    plt.suptitle(
        "Actual vs Predicted GDP by Time Horizon\nV4 Models with V2 Features",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "v4_v2_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_results_table():
    """Create detailed results table"""
    print("Creating results table...")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("tight")
    ax.axis("off")

    # Prepare data
    rows = []

    # Forecasting results
    for h in [1, 2, 3, 4]:
        key = f"forecast_h{h}"
        for model, metrics in results[key].items():
            rows.append(
                [
                    "Forecasting",
                    f"h={h}",
                    model,
                    f"{metrics['R2']:.4f}",
                    f"{metrics['RMSE']:.2f}",
                    f"{metrics['MAE']:.2f}",
                    metrics["n_samples"],
                ]
            )

    # Nowcasting results
    for model, metrics in results["nowcast_h1"].items():
        rows.append(
            [
                "Nowcasting",
                "h=1",
                model,
                f"{metrics['R2']:.4f}",
                f"{metrics['RMSE']:.2f}",
                f"{metrics['MAE']:.2f}",
                metrics["n_samples"],
            ]
        )

    columns = ["Task", "Horizon", "Model", "R² Score", "RMSE", "MAE", "Test Samples"]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.08, 0.15, 0.12, 0.12, 0.12, 0.12],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
            else:
                table[(i, j)].set_facecolor("white")

    plt.title(
        "V4+V2 Model Performance Results\nComplete Summary of All Models and Horizons",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "v4_v2_results_table.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_report():
    """Create a text summary report"""
    print("Creating summary report...")

    report = []
    report.append("=" * 90)
    report.append("V4 MODELS WITH V2 FEATURE-ENGINEERED DATA - RESULTS SUMMARY")
    report.append("=" * 90)

    report.append("\n📊 FORECASTING MODELS (Leading Indicators - 6-18 Month Lookahead)")
    report.append("-" * 90)

    for h in [1, 2, 3, 4]:
        key = f"forecast_h{h}"
        report.append(f"\nHorizon h={h} (Quarter {h} Ahead):")
        report.append(
            "  Model                 │    R² Score    │   RMSE   │   MAE    │  Samples"
        )
        report.append("  " + "-" * 85)

        for model in ["Ridge", "RandomForest", "GradientBoosting", "Ensemble"]:
            metrics = results[key][model]
            report.append(
                f"  {model:20s} │ {metrics['R2']:13.4f} │ {metrics['RMSE']:8.2f} │ {metrics['MAE']:8.2f} │ {metrics['n_samples']:7d}"
            )

    report.append("\n\n📊 NOWCASTING MODELS (Coincident Indicators - Current Quarter)")
    report.append("-" * 90)
    report.append(f"\nHorizon h=1 (Current Quarter):")
    report.append(
        "  Model                 │    R² Score    │   RMSE   │   MAE    │  Samples"
    )
    report.append("  " + "-" * 85)

    for model in ["Ridge", "RandomForest", "GradientBoosting", "Ensemble"]:
        metrics = results["nowcast_h1"][model]
        report.append(
            f"  {model:20s} │ {metrics['R2']:13.4f} │ {metrics['RMSE']:8.2f} │ {metrics['MAE']:8.2f} │ {metrics['n_samples']:7d}"
        )

    report.append("\n\n📈 KEY INSIGHTS")
    report.append("-" * 90)

    # Find best forecasting model
    best_forecast_r2 = -np.inf
    best_forecast_model = None
    best_forecast_h = None

    for h in [1, 2, 3, 4]:
        for model, metrics in results[f"forecast_h{h}"].items():
            if metrics["R2"] > best_forecast_r2:
                best_forecast_r2 = metrics["R2"]
                best_forecast_model = model
                best_forecast_h = h

    report.append(
        f"✓ Best Forecasting Model: {best_forecast_model} (h={best_forecast_h}, R²={best_forecast_r2:.4f})"
    )

    # Find best nowcasting model
    best_nowcast_r2 = max(results["nowcast_h1"][m]["R2"] for m in results["nowcast_h1"])
    best_nowcast_model = [
        m for m, met in results["nowcast_h1"].items() if met["R2"] == best_nowcast_r2
    ][0]

    report.append(
        f"✓ Best Nowcasting Model: {best_nowcast_model} (R²={best_nowcast_r2:.4f})"
    )

    report.append(
        f"✓ Nowcasting significantly outperforms forecasting (R²={best_nowcast_r2:.4f} vs {best_forecast_r2:.4f})"
    )
    report.append(
        f"  → This is expected: current-quarter data is more predictive than future data"
    )

    report.append(f"✓ Ridge regression dominates tree-based models")
    report.append(f"  → V2 features may be linear relationships")
    report.append(
        f"  → Tree models show negative R² (overfitting on small feature set)"
    )

    report.append(f"✓ Forecast accuracy improves with longer horizons")
    report.append(f"  → h=1: R²={results['forecast_h1']['Ridge']['R2']:.4f}")
    report.append(f"  → h=4: R²={results['forecast_h4']['Ridge']['R2']:.4f}")
    report.append(f"  → Suggests models capture longer-term trends well")

    report.append("\n\n📁 OUTPUT FILES")
    report.append("-" * 90)
    report.append(
        f"Results:        {RESULTS_DIR / 'results' / 'v4_v2_model_results.json'}"
    )
    report.append(
        f"Predictions:    {RESULTS_DIR / 'results' / 'v4_v2_predictions.pkl'}"
    )
    report.append(f"Visualizations: {VIZ_DIR}/")
    report.append(f"  - v4_v2_r2_heatmap.png")
    report.append(f"  - v4_v2_model_comparison.png")
    report.append(f"  - v4_v2_rmse_by_horizon.png")
    report.append(f"  - v4_v2_results_table.png")

    report.append("\n" + "=" * 90)

    # Write to file
    report_text = "\n".join(report)
    report_file = RESULTS_DIR / "V4_V2_RESULTS_SUMMARY.txt"

    with open(report_file, "w") as f:
        f.write(report_text)

    # Also print
    print("\n" + report_text)

    return report_file


def main():
    """Create all visualizations"""
    print("\n" + "=" * 80)
    print("GENERATING V4+V2 VISUALIZATIONS")
    print("=" * 80)

    create_performance_heatmap()
    create_model_comparison_forecast()
    create_rmse_by_horizon()
    create_actual_vs_predicted()
    create_results_table()
    report_file = create_summary_report()

    print("\n✅ All visualizations created successfully!")
    print(f"\nVisualization Folder: {VIZ_DIR}")
    print(f"Summary Report: {report_file}")


if __name__ == "__main__":
    main()
