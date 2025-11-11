"""
Signal Processing Feature Engineering
Applies advanced signal processing techniques to extract patterns:
- Wavelet decomposition (trend + cyclical components)
- Spectral analysis for business cycle frequency
- Seasonal decomposition (where applicable)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SignalProcessing:
    """
    Applies signal processing techniques to economic time series
    Decomposes signals into trend, cyclical, and residual components
    """

    @staticmethod
    def simple_wavelet_decomposition(series: pd.Series, window_size: int = 12) -> Tuple[pd.Series, pd.Series]:
        """
        Simple wavelet-like decomposition using moving averages

        This is a simplified approach to wavelet decomposition suitable for economic data:
        - Trend: Long moving average
        - Cyclical: Series minus trend

        Args:
            series: Time series to decompose
            window_size: Window for trend extraction (12 = 1 year for monthly data)

        Returns:
            Tuple of (trend_component, cyclical_component)
        """
        # Extract trend using moving average
        trend = series.rolling(window=window_size, center=True, min_periods=1).mean()

        # Cyclical component is residual after removing trend
        cyclical = series - trend

        return trend, cyclical

    @staticmethod
    def create_wavelet_features(df: pd.DataFrame, target_cols: Optional[list] = None) -> pd.DataFrame:
        """
        Create wavelet decomposition features for key economic indicators

        Args:
            df: Input DataFrame
            target_cols: List of columns to decompose (default: key macro indicators)

        Returns:
            DataFrame with wavelet components added
        """
        features = df.copy()

        # Default columns to decompose if not specified
        if target_cols is None:
            target_cols = [
                'DGS10',  # 10Y Treasury
                'INDPRO',  # Industrial Production
                'MMNRNJ',  # Manufacturing PMI
                'UMCSENT',  # Consumer Sentiment
                'BAMLH0A0HYM2'  # Credit spread
            ]

        for col in target_cols:
            if col in df.columns:
                try:
                    trend, cyclical = SignalProcessing.simple_wavelet_decomposition(df[col])
                    features[f'{col}_trend'] = trend
                    features[f'{col}_cycle'] = cyclical

                    logger.info(f"Decomposed {col} into trend and cyclical components")
                except Exception as e:
                    logger.warning(f"Failed to decompose {col}: {str(e)}")

        return features

    @staticmethod
    def extract_business_cycle_frequency(series: pd.Series, freq: str = 'm') -> pd.Series:
        """
        Extract business cycle frequency components

        Business cycle frequencies (Burn-Mitchell):
        - Monthly data: 8-32 months (typical recession-recovery cycle)
        - Quarterly data: 2-8 quarters (6-24 months)

        Uses high-pass filter to isolate business cycle fluctuations

        Args:
            series: Time series
            freq: Data frequency ('d'=daily, 'w'=weekly, 'm'=monthly, 'q'=quarterly)

        Returns:
            Business cycle component
        """
        # Simple HP filter approximation using differencing
        # More sophisticated approach would use Hodrick-Prescott filter

        if freq == 'q':
            window = 8  # Quarterly: 2-8 quarter business cycle
        elif freq == 'm':
            window = 24  # Monthly: 8-32 month business cycle
        else:
            window = 12  # Default

        # Extract cyclical via moving average subtraction
        trend = series.rolling(window=window, center=True, min_periods=1).mean()
        cycle = series - trend

        return cycle

    @staticmethod
    def create_cyclical_features(df: pd.DataFrame, freq: str = 'd') -> pd.DataFrame:
        """
        Create business cycle component features

        Args:
            df: Input DataFrame
            freq: Data frequency (daily='d', weekly='w', monthly='m', quarterly='q')

        Returns:
            DataFrame with cyclical components
        """
        features = df.copy()

        # Key series to extract business cycle from
        cycle_cols = {
            'DGS10': 'yield_cycle',
            'INDPRO': 'production_cycle',
            'SP500': 'equity_cycle',
            'BAMLH0A0HYM2': 'credit_cycle'
        }

        for col, output_name in cycle_cols.items():
            if col in df.columns:
                try:
                    cycle = SignalProcessing.extract_business_cycle_frequency(df[col], freq)
                    features[output_name] = cycle
                    logger.info(f"Extracted {output_name} from {col}")
                except Exception as e:
                    logger.warning(f"Failed to extract cycle from {col}: {str(e)}")

        return features

    @staticmethod
    def create_velocity_acceleration_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create velocity and acceleration features

        These capture the rate of change of changes
        - Velocity: First derivative (momentum)
        - Acceleration: Second derivative (change in momentum)

        Economic interpretation:
        - Rising velocity + falling acceleration = peak/turning point
        - Negative velocity + rising acceleration = trough/recovery signal

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with velocity/acceleration features
        """
        features = df.copy()

        key_series = ['INDPRO', 'PAYEMS', 'RSXFS', 'DGS10', 'BAMLH0A0HYM2']

        for col in key_series:
            if col in df.columns:
                # Velocity (1st derivative / momentum)
                velocity = df[col].diff(1)
                features[f'{col}_velocity'] = velocity

                # Acceleration (2nd derivative)
                acceleration = velocity.diff(1)
                features[f'{col}_acceleration'] = acceleration

                # Combined momentum score
                velocity_norm = (velocity - velocity.mean()) / velocity.std()
                accel_norm = (acceleration - acceleration.mean()) / acceleration.std()
                features[f'{col}_momentum_score'] = velocity_norm + accel_norm

        logger.info("Created velocity and acceleration features")
        return features

    @staticmethod
    def create_rate_of_change_acceleration(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rate-of-change and acceleration features for key indicators

        These are particularly useful for identifying turning points in the economy

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with ROC and acceleration features
        """
        features = df.copy()

        # Create short and long-term ROC
        for col in ['INDPRO', 'PAYEMS', 'RSXFS', 'UMCSENT', 'MMNRNJ']:
            if col in df.columns:
                # Rate of change
                features[f'{col}_roc_3m'] = df[col].pct_change(3)
                features[f'{col}_roc_6m'] = df[col].pct_change(6)
                features[f'{col}_roc_12m'] = df[col].pct_change(12)

                # Acceleration in ROC
                features[f'{col}_roc_acceleration'] = features[f'{col}_roc_3m'].diff(1)

        logger.info("Created rate of change and acceleration features")
        return features

    @staticmethod
    def create_all_signal_processing_features(df: pd.DataFrame, freq: str = 'd',
                                            config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all signal processing features

        Args:
            df: Input DataFrame
            freq: Data frequency
            config: Optional configuration

        Returns:
            DataFrame with all signal processing features
        """
        features = df.copy()

        features = SignalProcessing.create_wavelet_features(features)
        features = SignalProcessing.create_cyclical_features(features, freq)
        features = SignalProcessing.create_velocity_acceleration_features(features)
        features = SignalProcessing.create_rate_of_change_acceleration(features)

        logger.info(f"Created all signal processing features")
        return features
