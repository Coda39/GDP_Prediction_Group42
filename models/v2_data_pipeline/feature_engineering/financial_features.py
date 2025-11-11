"""
Financial Market Feature Engineering
Creates features from financial markets:
- Yield Curve and Treasury spreads (KEY leading indicator for recessions)
- Credit spreads (high-yield bond spreads)
- Equity market indices and returns
- Volatility indices (VIX)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FinancialFeatures:
    """
    Engineers features from financial market data
    Financial markets are forward-looking and aggregate market expectations
    """

    @staticmethod
    def create_yield_curve_features(df: pd.DataFrame, dgs10_col: str = 'DGS10',
                                    dgs2_col: str = 'DGS2', dgs3m_col: str = 'DGS3MO') -> pd.DataFrame:
        """
        Create yield curve features (KEY LEADING INDICATOR)

        Yield curve inversion has preceded nearly every US recession
        Economic rationale: Inversion signals market expects Fed will cut rates due to slowdown

        Args:
            df: Input DataFrame with Treasury rates
            dgs10_col: 10-Year Treasury rate column
            dgs2_col: 2-Year Treasury rate column
            dgs3m_col: 3-Month Treasury rate column

        Returns:
            DataFrame with yield curve features
        """
        features = df.copy()

        # 10Y-2Y Spread (classic recession predictor)
        if dgs10_col in df.columns and dgs2_col in df.columns:
            features['yield_spread_10y_2y'] = df[dgs10_col] - df[dgs2_col]
            features['yield_spread_10y_2y_inversion'] = (features['yield_spread_10y_2y'] < 0).astype(int)
            features['yield_spread_10y_2y_ma3m'] = features['yield_spread_10y_2y'].rolling(
                63, min_periods=1).mean()  # ~3 months of trading days
            features['yield_spread_10y_2y_change'] = features['yield_spread_10y_2y'].diff(1)

        # 10Y-3M Spread (alternative measure)
        if dgs10_col in df.columns and dgs3m_col in df.columns:
            features['yield_spread_10y_3m'] = df[dgs10_col] - df[dgs3m_col]
            features['yield_spread_10y_3m_inversion'] = (features['yield_spread_10y_3m'] < 0).astype(int)

        # 2Y-3M Spread (shorter-term slope)
        if dgs2_col in df.columns and dgs3m_col in df.columns:
            features['yield_spread_2y_3m'] = df[dgs2_col] - df[dgs3m_col]

        # Individual rate levels
        if dgs10_col in df.columns:
            features[f'{dgs10_col}_level'] = df[dgs10_col]
            features[f'{dgs10_col}_change'] = df[dgs10_col].diff(1)

        if dgs2_col in df.columns:
            features[f'{dgs2_col}_level'] = df[dgs2_col]

        if dgs3m_col in df.columns:
            features[f'{dgs3m_col}_level'] = df[dgs3m_col]

        # Rate volatility (as signal of market uncertainty)
        if dgs10_col in df.columns:
            features['rate_volatility_10y'] = df[dgs10_col].rolling(20, min_periods=1).std()

        logger.info(f"Created yield curve features")
        return features

    @staticmethod
    def create_credit_spread_features(df: pd.DataFrame, baa_spread_col: str = 'BAMLH0A0HYM2') -> pd.DataFrame:
        """
        Create credit spread features (LEADING indicator)

        High-yield (junk bond) spread measures risk premium
        Widening spread signals market fears recession (higher default probability)

        Args:
            df: Input DataFrame with credit spread data
            baa_spread_col: Baa corporate bond spread column

        Returns:
            DataFrame with credit spread features
        """
        features = df.copy()

        if baa_spread_col in df.columns:
            features[f'{baa_spread_col}_level'] = df[baa_spread_col]
            features[f'{baa_spread_col}_change'] = df[baa_spread_col].diff(1)
            features[f'{baa_spread_col}_ma20'] = df[baa_spread_col].rolling(20, min_periods=1).mean()
            features[f'{baa_spread_col}_ma60'] = df[baa_spread_col].rolling(60, min_periods=1).mean()

            # Credit stress signal (widening from recent lows)
            ma60 = df[baa_spread_col].rolling(60, min_periods=1).mean()
            features[f'{baa_spread_col}_above_ma60'] = df[baa_spread_col] - ma60

            # Rate of change (acceleration of credit stress)
            features[f'{baa_spread_col}_accel'] = features[f'{baa_spread_col}_change'].rolling(20, min_periods=1).mean()

        logger.info(f"Created credit spread features")
        return features

    @staticmethod
    def create_equity_features(df: pd.DataFrame, sp500_col: str = 'SP500') -> pd.DataFrame:
        """
        Create equity market features

        Stock prices are forward-looking (reflect discounted future earnings)
        However, note: S&P 500 has globalization disconnect (companies earn internationally)

        Args:
            df: Input DataFrame with equity index data
            sp500_col: S&P 500 index column

        Returns:
            DataFrame with equity features
        """
        features = df.copy()

        if sp500_col in df.columns:
            # Price levels and returns
            features[f'{sp500_col}_level'] = df[sp500_col]
            features[f'{sp500_col}_return_1d'] = df[sp500_col].pct_change(1)
            features[f'{sp500_col}_return_5d'] = df[sp500_col].pct_change(5)
            features[f'{sp500_col}_return_20d'] = df[sp500_col].pct_change(20)
            features[f'{sp500_col}_return_60d'] = df[sp500_col].pct_change(60)

            # Moving averages
            features[f'{sp500_col}_ma20'] = df[sp500_col].rolling(20, min_periods=1).mean()
            features[f'{sp500_col}_ma60'] = df[sp500_col].rolling(60, min_periods=1).mean()

            # Trend (above/below MA)
            features[f'{sp500_col}_above_ma60'] = (df[sp500_col] > features[f'{sp500_col}_ma60']).astype(int)
            features[f'{sp500_col}_above_ma20'] = (df[sp500_col] > features[f'{sp500_col}_ma20']).astype(int)

            # Volatility
            features[f'{sp500_col}_volatility_20d'] = df[sp500_col].pct_change(1).rolling(20, min_periods=1).std()

            # Momentum
            features[f'{sp500_col}_momentum'] = features[f'{sp500_col}_return_20d'].rolling(20, min_periods=1).mean()

        logger.info(f"Created equity features")
        return features

    @staticmethod
    def create_volatility_features(df: pd.DataFrame, vix_col: str = 'VIXCLS') -> pd.DataFrame:
        """
        Create volatility index features

        VIX (Fear Index) measures market implied volatility
        Sudden VIX spikes associated with economic surprises and market stress

        Args:
            df: Input DataFrame with VIX data
            vix_col: VIX index column

        Returns:
            DataFrame with volatility features
        """
        features = df.copy()

        if vix_col in df.columns:
            features[f'{vix_col}_level'] = df[vix_col]
            features[f'{vix_col}_change'] = df[vix_col].diff(1)
            features[f'{vix_col}_ma20'] = df[vix_col].rolling(20, min_periods=1).mean()

            # High volatility signals (market stress)
            vix_ma = df[vix_col].rolling(60, min_periods=1).mean()
            features[f'{vix_col}_elevated'] = (df[vix_col] > vix_ma).astype(int)

            # Volatility of volatility
            features[f'{vix_col}_volatility'] = df[vix_col].diff(1).rolling(20, min_periods=1).std()

            # Spikes in VIX
            features[f'{vix_col}_spike'] = (features[f'{vix_col}_change'] > features[f'{vix_col}_change'].std() * 2).astype(int)

        logger.info(f"Created volatility features")
        return features

    @staticmethod
    def create_term_structure_factors(df: pd.DataFrame, dgs10_col: str = 'DGS10',
                                     dgs2_col: str = 'DGS2', dgs3m_col: str = 'DGS3MO') -> pd.DataFrame:
        """
        Create principal component analysis (PCA) factors of yield curve

        Research shows 99% of yield curve movements explained by 3 factors:
        1. Level (average height of curve - long-run policy stance)
        2. Slope (10Y-2Y spread - classic recession predictor)
        3. Curvature (hump in middle - 5Y relative to 2Y and 10Y)

        Args:
            df: Input DataFrame with multiple rates
            dgs10_col: 10-Year rate
            dgs2_col: 2-Year rate
            dgs3m_col: 3-Month rate

        Returns:
            DataFrame with term structure factors
        """
        features = df.copy()

        if dgs10_col in df.columns and dgs2_col in df.columns and dgs3m_col in df.columns:
            # Level factor (average of all rates)
            features['term_level'] = (df[dgs10_col] + df[dgs2_col] + df[dgs3m_col]) / 3

            # Slope factor (10Y-3M spread)
            features['term_slope'] = df[dgs10_col] - df[dgs3m_col]

            # Curvature factor (2Y relative to 3M and 10Y - the "hump")
            features['term_curvature'] = (2 * df[dgs2_col]) - df[dgs10_col] - df[dgs3m_col]

            # Moving averages of factors
            features['term_level_ma'] = features['term_level'].rolling(60, min_periods=1).mean()
            features['term_slope_ma'] = features['term_slope'].rolling(60, min_periods=1).mean()

        logger.info(f"Created term structure factors")
        return features

    @staticmethod
    def create_all_financial_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all financial market features

        Args:
            df: Input DataFrame with financial data
            config: Optional configuration dictionary

        Returns:
            DataFrame with all financial features
        """
        features = df.copy()

        # Create features for each category
        features = FinancialFeatures.create_yield_curve_features(features)
        features = FinancialFeatures.create_credit_spread_features(features)
        features = FinancialFeatures.create_equity_features(features)
        features = FinancialFeatures.create_volatility_features(features)
        features = FinancialFeatures.create_term_structure_factors(features)

        logger.info(f"Created financial features")
        return features
