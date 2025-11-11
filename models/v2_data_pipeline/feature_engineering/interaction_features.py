"""
Interaction Feature Engineering
Creates combined features that capture multi-dimensional economic relationships
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class InteractionFeatures:
    """
    Creates interaction features combining multiple economic signals
    These capture non-linear relationships and complex economic dynamics
    """

    @staticmethod
    def create_financial_stress_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features capturing financial stress signals

        Interactions:
        - Yield curve × Credit spreads: Policy tightening signal
        - Stock price × Credit spread: Financial stress in equities
        - VIX × Credit spread: Market uncertainty combined with credit stress
        """
        features = df.copy()

        # Yield curve × Credit spread (policy tightening)
        if 'yield_spread_10y_2y' in df.columns and 'BAMLH0A0HYM2_level' in df.columns:
            features['financial_stress_yc_cs'] = (
                (df['yield_spread_10y_2y'] < 0).astype(int) *  # Inverted curve
                (df['BAMLH0A0HYM2_level'] > df['BAMLH0A0HYM2_level'].rolling(60, min_periods=1).mean()).astype(int)
            )

        # Stock volatility × Credit spread
        if 'SP500_volatility_20d' in df.columns and 'BAMLH0A0HYM2_level' in df.columns:
            features['market_credit_stress'] = (
                df['SP500_volatility_20d'] * df['BAMLH0A0HYM2_level']
            ).fillna(0)

        # VIX elevation × Credit stress
        if 'VIXCLS_elevated' in df.columns and 'BAMLH0A0HYM2_above_ma60' in df.columns:
            features['fear_credit_interaction'] = (
                df['VIXCLS_elevated'] * df['BAMLH0A0HYM2_above_ma60']
            )

        logger.info("Created financial stress interaction features")
        return features

    @staticmethod
    def create_labor_sentiment_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features capturing labor market sentiment dynamics

        Interactions:
        - ISM Employment × Unemployment rate: Labor weakness signal
        - Jobless claims × Consumer sentiment: Confidence in employment
        """
        features = df.copy()

        # Labor market weakness: Claims rising while sentiment falling
        if 'ICSA_ma13w' in df.columns and 'UMCSENT_level' in df.columns:
            # Normalize both to 0-1 scale for comparison
            claims_norm = (df['ICSA_ma13w'] - df['ICSA_ma13w'].min()) / (df['ICSA_ma13w'].max() - df['ICSA_ma13w'].min())
            sentiment_norm = (df['UMCSENT_level'] - df['UMCSENT_level'].min()) / (df['UMCSENT_level'].max() - df['UMCSENT_level'].min())

            # Divergence: high claims (bad) + low sentiment (bad) = major weakness
            features['labor_weakness_signal'] = claims_norm * (1 - sentiment_norm)

        logger.info("Created labor sentiment interaction features")
        return features

    @staticmethod
    def create_demand_supply_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features capturing demand vs supply dynamics

        Interactions:
        - Retail sales × Capacity utilization: Demand absorption capacity
        - Income growth × Consumption growth: Spending sustainability
        - PMI New Orders × Industrial Production: Orders → Production flow
        """
        features = df.copy()

        # Consumption growth sustainability: Income backing spending
        if 'W875RX1_pct_change_1m' in df.columns and 'RSXFS_pct_change_1m' in df.columns:
            features['consumption_income_divergence'] = (
                df['RSXFS_pct_change_1m'] - df['W875RX1_pct_change_1m']
            )
            # Positive = consumption outpacing income (unsustainable)

        # Demand vs production capacity
        if 'RSXFS_pct_change_1m' in df.columns and 'CUMFSL_level' in df.columns:
            features['demand_capacity_pressure'] = (
                df['RSXFS_pct_change_1m'] * df['CUMFSL_level']
            ).fillna(0)

        # Orders leading to production
        if 'NAPMNPI_level' in df.columns and 'INDPRO_pct_change_1m' in df.columns:
            # Orders should lead production
            features['orders_production_lag'] = (
                df['NAPMNPI_level'].shift(1) - 50
            ) * df['INDPRO_pct_change_1m']

        logger.info("Created demand supply interaction features")
        return features

    @staticmethod
    def create_uncertainty_investment_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features capturing uncertainty's impact on investment

        Economic rationale: High uncertainty causes firms to delay capital expenditure

        Interactions:
        - Policy uncertainty × Investment growth: Precautionary behavior
        - Yield curve × VIX: Combined financial stress
        - EPU × Business confidence: Sentiment under uncertainty
        """
        features = df.copy()

        # Uncertainty depressing investment sentiment
        if 'USEPUINDXD_level' in df.columns and 'NAPMNPI_level' in df.columns:
            features['uncertainty_investment_drag'] = (
                df['USEPUINDXD_level'] * (100 - df['NAPMNPI_level'])
            ) / 100

        # Combined financial+policy uncertainty
        if 'USEPUINDXD_ma60' in df.columns and 'VIXCLS_level' in df.columns:
            epu_norm = df['USEPUINDXD_ma60'] / df['USEPUINDXD_ma60'].max()
            vix_norm = df['VIXCLS_level'] / df['VIXCLS_level'].max()
            features['combined_uncertainty'] = epu_norm + vix_norm

        logger.info("Created uncertainty investment interaction features")
        return features

    @staticmethod
    def create_all_interaction_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all interaction features

        Args:
            df: Input DataFrame with base features
            config: Optional configuration

        Returns:
            DataFrame with interaction features added
        """
        features = df.copy()

        features = InteractionFeatures.create_financial_stress_interactions(features)
        features = InteractionFeatures.create_labor_sentiment_interactions(features)
        features = InteractionFeatures.create_demand_supply_interactions(features)
        features = InteractionFeatures.create_uncertainty_investment_interactions(features)

        logger.info(f"Created all interaction features")
        return features
