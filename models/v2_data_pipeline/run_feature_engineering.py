#!/usr/bin/env python3
"""
Feature Engineering Execution Script
=====================================

This script executes the complete feature engineering pipeline:
1. Loads raw FRED data from Data_v2/raw/fred/
2. Creates features using all 7 feature engineering modules
3. Applies task-specific feature selection (forecasting vs nowcasting)
4. Normalizes data using regime-aware scaling
5. Saves processed outputs to Data_v2/processed/

Author: GDP Prediction Group 42
Date: November 11, 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import feature engineering modules
from models.v2_data_pipeline.feature_engineering.hard_data_features import HardDataFeatures
from models.v2_data_pipeline.feature_engineering.soft_data_features import SoftDataFeatures
from models.v2_data_pipeline.feature_engineering.financial_features import FinancialFeatures
from models.v2_data_pipeline.feature_engineering.alternative_features import AlternativeDataFeatures
from models.v2_data_pipeline.feature_engineering.interaction_features import InteractionFeatures
from models.v2_data_pipeline.feature_engineering.signal_processing import SignalProcessing
from models.v2_data_pipeline.feature_selection.leading_lagging_classifier import FeatureClassifier


class FeatureEngineeringPipeline:
    """Execute complete feature engineering pipeline."""

    def __init__(self, data_root: str = None):
        """Initialize pipeline."""
        if data_root is None:
            data_root = str(project_root / "Data_v2")

        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw" / "fred"
        self.processed_dir = self.data_root / "processed"

        # Create output directories
        (self.processed_dir / "forecasting").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "nowcasting").mkdir(parents=True, exist_ok=True)

        self.df = None
        self.df_forecast = None
        self.df_nowcast = None

    def load_raw_data(self) -> pd.DataFrame:
        """Load and combine all raw FRED data."""
        print("\n" + "="*80)
        print("STEP 1: Loading Raw FRED Data")
        print("="*80)

        # Load metadata to understand data organization
        metadata_path = self.raw_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"\nLoading {len(metadata)} FRED indicators from {self.raw_dir}")

        all_data = {}
        loaded_count = 0

        # Load data organized by frequency
        freq_dirs = {'d': 'd', 'w': 'w', 'm': 'm', 'q': 'q'}

        for indicator, meta in metadata.items():
            freq = meta.get('frequency', 'm')
            freq_dir = self.raw_dir / freq_dirs.get(freq, 'm')

            csv_path = freq_dir / f"{indicator}.csv"

            if csv_path.exists():
                try:
                    data = pd.read_csv(csv_path)

                    # Convert date column to datetime
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date'])
                        data.set_index('date', inplace=True)

                    # Extract value column and convert to numeric
                    if 'value' in data.columns:
                        values = pd.to_numeric(data['value'], errors='coerce')
                    elif indicator in data.columns:
                        values = pd.to_numeric(data[indicator], errors='coerce')
                    else:
                        # Use the last numeric column
                        values = pd.to_numeric(data.iloc[:, -1], errors='coerce')

                    values.name = indicator
                    all_data[indicator] = values
                    loaded_count += 1
                    print(f"  ✓ {indicator:15} ({freq.upper():6}) - {len(data):5} obs")
                except Exception as e:
                    print(f"  ✗ {indicator:15} - ERROR: {str(e)[:50]}")
            else:
                print(f"  ✗ {indicator:15} - File not found: {csv_path}")

        # Combine all data into single DataFrame
        print(f"\nCombining {loaded_count} indicators into single DataFrame...")
        df = pd.DataFrame(all_data)

        # Forward fill for daily data, then align to quarterly frequency
        # This handles mixed-frequency data appropriately
        df = df.fillna(method='ffill')

        # Drop rows with all NaN
        df = df.dropna(how='all')

        print(f"Combined DataFrame shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Missing values: {df.isnull().sum().sum()}")

        self.df = df
        return df

    def create_features(self) -> pd.DataFrame:
        """Execute feature engineering pipeline."""
        print("\n" + "="*80)
        print("STEP 2: Creating Features (All 7 Modules)")
        print("="*80)

        if self.df is None:
            raise ValueError("Must load raw data first")

        df = self.df.copy()
        initial_cols = len(df.columns)

        # 1. Hard Data Features
        print("\n[1/7] Creating Hard Data Features...")
        df = HardDataFeatures.create_all_hard_features(df)
        hard_cols = len(df.columns) - initial_cols
        print(f"      Added {hard_cols} hard data features")

        # 2. Soft Data Features
        print("[2/7] Creating Soft Data Features...")
        cols_before = len(df.columns)
        df = SoftDataFeatures.create_all_soft_features(df)
        soft_cols = len(df.columns) - cols_before
        print(f"      Added {soft_cols} soft data features")

        # 3. Financial Features
        print("[3/7] Creating Financial Features...")
        cols_before = len(df.columns)
        df = FinancialFeatures.create_all_financial_features(df)
        fin_cols = len(df.columns) - cols_before
        print(f"      Added {fin_cols} financial features")

        # 4. Alternative Data Features
        print("[4/7] Creating Alternative Data Features...")
        cols_before = len(df.columns)
        df = AlternativeDataFeatures.create_all_alternative_features(df)
        alt_cols = len(df.columns) - cols_before
        print(f"      Added {alt_cols} alternative data features")

        # 5. Interaction Features
        print("[5/7] Creating Interaction Features...")
        cols_before = len(df.columns)
        df = InteractionFeatures.create_all_interaction_features(df)
        int_cols = len(df.columns) - cols_before
        print(f"      Added {int_cols} interaction features")

        # 6. Signal Processing Features
        print("[6/7] Creating Signal Processing Features...")
        cols_before = len(df.columns)
        df = SignalProcessing.create_all_signal_processing_features(df)
        sig_cols = len(df.columns) - cols_before
        print(f"      Added {sig_cols} signal processing features")

        # Drop any remaining NaN rows
        df = df.dropna(how='any')

        total_features = len(df.columns)
        print(f"\n✓ Total features created: {total_features}")
        print(f"  Final DataFrame shape: {df.shape}")

        self.df = df
        return df

    def select_task_specific_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select forecasting vs nowcasting features."""
        print("\n" + "="*80)
        print("STEP 3: Task-Specific Feature Selection")
        print("="*80)

        if self.df is None:
            raise ValueError("Must create features first")

        # Get forecasting features (leading indicators only)
        print("\nExtracting forecasting features (leading indicators)...")
        forecast_cols, forecast_info = FeatureClassifier.get_features_for_task(
            'forecasting', self.df.columns
        )

        print(f"  ✓ Forecasting features: {len(forecast_cols)}")

        # Get nowcasting features (coincident indicators only)
        print("Extracting nowcasting features (coincident indicators)...")
        nowcast_cols, nowcast_info = FeatureClassifier.get_features_for_task(
            'nowcasting', self.df.columns
        )

        print(f"  ✓ Nowcasting features: {len(nowcast_cols)}")

        # Create task-specific DataFrames
        # Ensure target variable (GDPC1) is included in both
        if 'GDPC1' not in forecast_cols:
            forecast_cols = ['GDPC1'] + forecast_cols
        if 'GDPC1' not in nowcast_cols:
            nowcast_cols = ['GDPC1'] + nowcast_cols

        self.df_forecast = self.df[forecast_cols].dropna(how='any')
        self.df_nowcast = self.df[nowcast_cols].dropna(how='any')

        print(f"\nForecasting dataset shape: {self.df_forecast.shape}")
        print(f"Nowcasting dataset shape: {self.df_nowcast.shape}")

        return self.df_forecast, self.df_nowcast

    def normalize_data(self):
        """Normalize data using regime-aware Z-score normalization."""
        print("\n" + "="*80)
        print("STEP 4: Regime-Aware Data Normalization")
        print("="*80)

        # Simple regime detection based on volatility
        def detect_regime(series, window=60):
            """Detect economic regime based on volatility."""
            rolling_std = series.rolling(window=window).std()
            median_std = rolling_std.median()
            return (rolling_std > median_std * 1.5).astype(int)

        # Use GDPC1 to detect regimes (high volatility = crisis regime)
        if 'GDPC1' in self.df_forecast.columns:
            gdp_returns = self.df_forecast['GDPC1'].pct_change()
            regime = detect_regime(gdp_returns)
        else:
            regime = pd.Series(0, index=self.df_forecast.index)

        print("\nNormalizing forecasting dataset...")
        print(f"  Regime split: Normal={sum(regime==0)}, Crisis={sum(regime==1)}")

        # Normalize by regime
        df_forecast_norm = self.df_forecast.copy()
        df_nowcast_norm = self.df_nowcast.copy()

        for col in df_forecast_norm.columns:
            if col != 'GDPC1':
                # Normal regime
                mask_normal = regime == 0
                if mask_normal.sum() > 0:
                    mean_normal = df_forecast_norm.loc[mask_normal, col].mean()
                    std_normal = df_forecast_norm.loc[mask_normal, col].std()
                    if std_normal > 0:
                        df_forecast_norm.loc[mask_normal, col] = (
                            (df_forecast_norm.loc[mask_normal, col] - mean_normal) / std_normal
                        )

                # Crisis regime
                mask_crisis = regime == 1
                if mask_crisis.sum() > 0:
                    mean_crisis = df_forecast_norm.loc[mask_crisis, col].mean()
                    std_crisis = df_forecast_norm.loc[mask_crisis, col].std()
                    if std_crisis > 0:
                        df_forecast_norm.loc[mask_crisis, col] = (
                            (df_forecast_norm.loc[mask_crisis, col] - mean_crisis) / std_crisis
                        )

        # Same for nowcasting
        for col in df_nowcast_norm.columns:
            if col != 'GDPC1':
                mask_normal = regime == 0
                if mask_normal.sum() > 0:
                    mean_normal = df_nowcast_norm.loc[mask_normal, col].mean()
                    std_normal = df_nowcast_norm.loc[mask_normal, col].std()
                    if std_normal > 0:
                        df_nowcast_norm.loc[mask_normal, col] = (
                            (df_nowcast_norm.loc[mask_normal, col] - mean_normal) / std_normal
                        )

                mask_crisis = regime == 1
                if mask_crisis.sum() > 0:
                    mean_crisis = df_nowcast_norm.loc[mask_crisis, col].mean()
                    std_crisis = df_nowcast_norm.loc[mask_crisis, col].std()
                    if std_crisis > 0:
                        df_nowcast_norm.loc[mask_crisis, col] = (
                            (df_nowcast_norm.loc[mask_crisis, col] - mean_crisis) / std_crisis
                        )

        print("✓ Normalization complete")

        self.df_forecast = df_forecast_norm
        self.df_nowcast = df_nowcast_norm

    def save_outputs(self):
        """Save processed datasets to disk."""
        print("\n" + "="*80)
        print("STEP 5: Saving Processed Outputs")
        print("="*80)

        # Save forecasting dataset
        forecast_path = self.processed_dir / "forecasting" / "usa_forecasting_features.csv"
        self.df_forecast.to_csv(forecast_path)
        print(f"\n✓ Forecasting dataset saved: {forecast_path}")
        print(f"  Shape: {self.df_forecast.shape}")
        print(f"  Date range: {self.df_forecast.index.min()} to {self.df_forecast.index.max()}")

        # Save nowcasting dataset
        nowcast_path = self.processed_dir / "nowcasting" / "usa_nowcasting_features.csv"
        self.df_nowcast.to_csv(nowcast_path)
        print(f"\n✓ Nowcasting dataset saved: {nowcast_path}")
        print(f"  Shape: {self.df_nowcast.shape}")
        print(f"  Date range: {self.df_nowcast.index.min()} to {self.df_nowcast.index.max()}")

        # Save metadata
        metadata = {
            'forecasting': {
                'shape': self.df_forecast.shape,
                'features': len(self.df_forecast.columns),
                'observations': len(self.df_forecast),
                'date_range': [str(self.df_forecast.index.min()), str(self.df_forecast.index.max())],
                'feature_list': list(self.df_forecast.columns)
            },
            'nowcasting': {
                'shape': self.df_nowcast.shape,
                'features': len(self.df_nowcast.columns),
                'observations': len(self.df_nowcast),
                'date_range': [str(self.df_nowcast.index.min()), str(self.df_nowcast.index.max())],
                'feature_list': list(self.df_nowcast.columns)
            }
        }

        metadata_path = self.processed_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Metadata saved: {metadata_path}")

    def print_summary(self):
        """Print summary of feature engineering results."""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*80)

        print("\n📊 Forecasting Dataset (Leading Indicators)")
        print("-" * 80)
        print(f"  Features: {self.df_forecast.shape[1]}")
        print(f"  Observations: {self.df_forecast.shape[0]}")
        print(f"  Date Range: {self.df_forecast.index.min().date()} to {self.df_forecast.index.max().date()}")
        print(f"  Location: Data_v2/processed/forecasting/usa_forecasting_features.csv")

        print("\n📊 Nowcasting Dataset (Coincident Indicators)")
        print("-" * 80)
        print(f"  Features: {self.df_nowcast.shape[1]}")
        print(f"  Observations: {self.df_nowcast.shape[0]}")
        print(f"  Date Range: {self.df_nowcast.index.min().date()} to {self.df_nowcast.index.max().date()}")
        print(f"  Location: Data_v2/processed/nowcasting/usa_nowcasting_features.csv")

        print("\n✅ STATUS: Feature engineering pipeline complete!")
        print("-" * 80)
        print("\nYou can now:")
        print("  1. Train forecasting models on Data_v2/processed/forecasting/usa_forecasting_features.csv")
        print("  2. Train nowcasting models on Data_v2/processed/nowcasting/usa_nowcasting_features.csv")
        print("  3. Run model comparisons between forecasting and nowcasting approaches")
        print("  4. Proceed to short-term next steps (model training, performance evaluation)")


def main():
    """Execute the complete feature engineering pipeline."""
    try:
        # Initialize pipeline
        pipeline = FeatureEngineeringPipeline()

        # Execute steps
        pipeline.load_raw_data()
        pipeline.create_features()
        pipeline.select_task_specific_features()
        pipeline.normalize_data()
        pipeline.save_outputs()
        pipeline.print_summary()

        print("\n" + "="*80)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
