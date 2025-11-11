"""
Soft Data Feature Engineering
Creates features from surveys and expectations data:
- Purchasing Managers Index (ISM Manufacturing & Services)
- Consumer Sentiment and Confidence
- Business expectations and outlook
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SoftDataFeatures:
    """
    Engineers features from soft data (surveys and sentiment)
    Soft data advantages: High frequency, forward-looking, no revisions
    """

    @staticmethod
    def create_pmi_features(df: pd.DataFrame, mfg_pmi_col: str = 'MMNRNJ',
                           mfg_orders_col: str = 'NAPMNPI',
                           services_pmi_col: str = 'IMNRNJ') -> pd.DataFrame:
        """
        Create Purchasing Managers Index features (LEADING indicators)

        Args:
            df: Input DataFrame with PMI data
            mfg_pmi_col: Manufacturing PMI column
            mfg_orders_col: Manufacturing New Orders Index column
            services_pmi_col: Services PMI column

        Returns:
            DataFrame with PMI features added
        """
        features = df.copy()

        # Manufacturing PMI (LEADING indicator)
        if mfg_pmi_col in df.columns:
            features[f'{mfg_pmi_col}_level'] = df[mfg_pmi_col]
            # PMI > 50 = expansion, < 50 = contraction
            features[f'{mfg_pmi_col}_expansion'] = (df[mfg_pmi_col] > 50).astype(int)
            features[f'{mfg_pmi_col}_momentum'] = df[mfg_pmi_col].diff(1)
            features[f'{mfg_pmi_col}_ma3'] = df[mfg_pmi_col].rolling(3, min_periods=1).mean()
            # Distance from neutral (50)
            features[f'{mfg_pmi_col}_distance_neutral'] = df[mfg_pmi_col] - 50

        # Manufacturing New Orders (KEY LEADING indicator - directly predicts production demand)
        if mfg_orders_col in df.columns:
            features[f'{mfg_orders_col}_level'] = df[mfg_orders_col]
            features[f'{mfg_orders_col}_expansion'] = (df[mfg_orders_col] > 50).astype(int)
            features[f'{mfg_orders_col}_momentum'] = df[mfg_orders_col].diff(1)
            features[f'{mfg_orders_col}_ma3'] = df[mfg_orders_col].rolling(3, min_periods=1).mean()

        # Services PMI (LEADING indicator - captures services sector which is ~80% of US economy)
        if services_pmi_col in df.columns:
            features[f'{services_pmi_col}_level'] = df[services_pmi_col]
            features[f'{services_pmi_col}_expansion'] = (df[services_pmi_col] > 50).astype(int)
            features[f'{services_pmi_col}_momentum'] = df[services_pmi_col].diff(1)
            features[f'{services_pmi_col}_ma3'] = df[services_pmi_col].rolling(3, min_periods=1).mean()

        # Composite PMI (if both available)
        if mfg_pmi_col in df.columns and services_pmi_col in df.columns:
            # Weight: Manufacturing 30%, Services 70% (approximate US economy composition)
            features['composite_pmi'] = (0.3 * df[mfg_pmi_col] + 0.7 * df[services_pmi_col])
            features['composite_pmi_expansion'] = (features['composite_pmi'] > 50).astype(int)

        logger.info(f"Created PMI features")
        return features

    @staticmethod
    def create_sentiment_features(df: pd.DataFrame, umich_col: str = 'UMCSENT',
                                 tcb_col: str = 'CONFCSL') -> pd.DataFrame:
        """
        Create consumer sentiment and confidence features (LEADING indicators)

        Args:
            df: Input DataFrame with sentiment data
            umich_col: University of Michigan Consumer Sentiment Index column
            tcb_col: Conference Board Consumer Confidence Index column

        Returns:
            DataFrame with sentiment features added
        """
        features = df.copy()

        # University of Michigan Consumer Sentiment (long-running, closely watched)
        if umich_col in df.columns:
            features[f'{umich_col}_level'] = df[umich_col]
            features[f'{umich_col}_change_1m'] = df[umich_col].diff(1)
            features[f'{umich_col}_change_12m'] = df[umich_col].diff(12)
            features[f'{umich_col}_ma3'] = df[umich_col].rolling(3, min_periods=1).mean()
            features[f'{umich_col}_ma12'] = df[umich_col].rolling(12, min_periods=1).mean()

            # Sentiment deviation from trend (expansion/contraction signal)
            ma12 = df[umich_col].rolling(12, min_periods=1).mean()
            features[f'{umich_col}_above_trend'] = df[umich_col] - ma12

        # Conference Board Consumer Confidence
        if tcb_col in df.columns:
            features[f'{tcb_col}_level'] = df[tcb_col]
            features[f'{tcb_col}_change_1m'] = df[tcb_col].diff(1)
            features[f'{tcb_col}_change_12m'] = df[tcb_col].diff(12)
            features[f'{tcb_col}_ma3'] = df[tcb_col].rolling(3, min_periods=1).mean()
            features[f'{tcb_col}_ma12'] = df[tcb_col].rolling(12, min_periods=1).mean()

        # Average sentiment (if both available)
        if umich_col in df.columns and tcb_col in df.columns:
            # Normalize both to 0-100 scale if needed
            umich_norm = (df[umich_col] - df[umich_col].min()) / (df[umich_col].max() - df[umich_col].min()) * 100
            tcb_norm = (df[tcb_col] - df[tcb_col].min()) / (df[tcb_col].max() - df[tcb_col].min()) * 100
            features['avg_consumer_sentiment'] = (umich_norm + tcb_norm) / 2

        logger.info(f"Created consumer sentiment features")
        return features

    @staticmethod
    def create_expectation_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create forward-looking expectation features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with expectation features
        """
        features = df.copy()

        # Extract expectations components if available in sentiment data
        # Note: More sophisticated approach would separate current conditions from expectations
        # Most sentiment indices contain both components; ideally need raw survey data

        # Placeholder for expectations-specific features
        # In practice, you would extract expectations subcomponents from the raw survey

        logger.info(f"Created expectation features")
        return features

    @staticmethod
    def create_all_soft_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all soft data features

        Args:
            df: Input DataFrame with survey and sentiment data
            config: Optional configuration dictionary

        Returns:
            DataFrame with all soft data features
        """
        features = df.copy()

        # Create features for each category
        features = SoftDataFeatures.create_pmi_features(features)
        features = SoftDataFeatures.create_sentiment_features(features)
        features = SoftDataFeatures.create_expectation_features(features)

        logger.info(f"Created soft data features")
        return features
