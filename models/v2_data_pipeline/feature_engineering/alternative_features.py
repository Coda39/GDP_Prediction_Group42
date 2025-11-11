"""
Alternative Data Feature Engineering
Creates features from non-traditional, cutting-edge data sources:
- Economic Policy Uncertainty (EPU) Index - text analysis of newspapers
- World Uncertainty Index (WUI) - from Economist Intelligence Unit reports
- Shipping/Maritime data - real-time measure of global trade
- Optional: Google Trends, AI-based sentiment analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AlternativeDataFeatures:
    """
    Engineers features from alternative data sources (big data, text analysis, etc.)
    These are high-frequency, forward-looking indicators not subject to large revisions
    """

    @staticmethod
    def create_policy_uncertainty_features(df: pd.DataFrame, epu_col: str = 'USEPUINDXD') -> pd.DataFrame:
        """
        Create Economic Policy Uncertainty (EPU) features

        EPU Index methodology: Text analysis of newspapers counting articles with
        keywords related to (1) Economy, (2) Uncertainty, (3) Policy/Regulation

        Economic Rationale: Uncertainty about taxes, regulations, trade policy causes
        firms and households to delay spending/investment decisions, directly suppressing GDP

        Source: www.policyuncertainty.com (Baker, Bloom, Davis)

        Args:
            df: Input DataFrame with EPU data
            epu_col: EPU index column

        Returns:
            DataFrame with EPU features
        """
        features = df.copy()

        if epu_col in df.columns:
            features[f'{epu_col}_level'] = df[epu_col]
            # Monthly moving average (smooth out daily noise)
            features[f'{epu_col}_ma20'] = df[epu_col].rolling(20, min_periods=1).mean()
            features[f'{epu_col}_ma60'] = df[epu_col].rolling(60, min_periods=1).mean()

            # Uncertainty shocks (spikes above recent average)
            features[f'{epu_col}_above_ma60'] = df[epu_col] - features[f'{epu_col}_ma60']
            features[f'{epu_col}_shock'] = (
                (df[epu_col] > features[f'{epu_col}_ma60'] + features[f'{epu_col}_ma60'].std()) \
                .astype(int)
            )

            # Trend in uncertainty
            features[f'{epu_col}_trend'] = features[f'{epu_col}_ma60'].diff(1)
            features[f'{epu_col}_increasing'] = (features[f'{epu_col}_trend'] > 0).astype(int)

            # Lagged features (uncertainty effects persist)
            features[f'{epu_col}_lag1'] = df[epu_col].shift(1)
            features[f'{epu_col}_lag4'] = df[epu_col].shift(4)

        logger.info(f"Created Economic Policy Uncertainty features")
        return features

    @staticmethod
    def create_world_uncertainty_features(df: pd.DataFrame, wui_col: str = 'WUIUS') -> pd.DataFrame:
        """
        Create World Uncertainty Index (WUI) features

        WUI methodology: Text mining of Economist Intelligence Unit reports
        for 143 countries, counts article keywords related to uncertainty

        Provides globally comparable uncertainty measure
        Research shows WUI "innovations foreshadow significant declines in output"

        Source: www.worlduncertaintyindex.com (Ahir, Bloom, Furceri)

        Args:
            df: Input DataFrame with WUI data
            wui_col: WUI index column

        Returns:
            DataFrame with WUI features
        """
        features = df.copy()

        if wui_col in df.columns:
            features[f'{wui_col}_level'] = df[wui_col]
            features[f'{wui_col}_change'] = df[wui_col].diff(1)
            features[f'{wui_col}_ma4q'] = df[wui_col].rolling(4, min_periods=1).mean()  # 4 quarters

            # WUI elevated signal
            ma_level = df[wui_col].rolling(12, min_periods=1).mean()
            features[f'{wui_col}_elevated'] = (df[wui_col] > ma_level).astype(int)

            # Lagged features
            features[f'{wui_col}_lag1'] = df[wui_col].shift(1)

        logger.info(f"Created World Uncertainty Index features")
        return features

    @staticmethod
    def create_shipping_data_features(df: pd.DataFrame, bdi_col: Optional[str] = None,
                                     ccfi_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create shipping/maritime data features

        Baltic Dry Index (BDI): Cost to ship raw materials (iron ore, coal, etc.)
        - Direct measure of raw material demand from manufacturing

        China Containerized Freight Index (CCFI): Cost to ship finished goods from China
        - Direct measure of global trade in manufactured goods

        Economic Rationale: These are real-time proxies for physical trade flows,
        not subject to revisions or reporting lags

        Args:
            df: Input DataFrame with shipping data
            bdi_col: Optional Baltic Dry Index column
            ccfi_col: Optional China Container Freight Index column

        Returns:
            DataFrame with shipping features
        """
        features = df.copy()

        # Baltic Dry Index
        if bdi_col and bdi_col in df.columns:
            features[f'{bdi_col}_level'] = df[bdi_col]
            features[f'{bdi_col}_pct_change'] = df[bdi_col].pct_change(1)
            features[f'{bdi_col}_ma20'] = df[bdi_col].rolling(20, min_periods=1).mean()

            # BDI trend signal
            features[f'{bdi_col}_above_ma'] = (df[bdi_col] > features[f'{bdi_col}_ma20']).astype(int)

        # China Container Freight Index
        if ccfi_col and ccfi_col in df.columns:
            features[f'{ccfi_col}_level'] = df[ccfi_col]
            features[f'{ccfi_col}_pct_change'] = df[ccfi_col].pct_change(1)
            features[f'{ccfi_col}_ma20'] = df[ccfi_col].rolling(20, min_periods=1).mean()

        logger.info(f"Created shipping data features")
        return features

    @staticmethod
    def create_google_trends_features(df: pd.DataFrame, search_cols: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Create Google Trends features

        Google Trends data is a cost-free proxy for public interest in economic topics
        Examples: Searches for "recession", "unemployment", "stimulus"
        Or search categories: "Food" (consumption), "Tourism" (spending)

        Advantage: Real-time, high-frequency, no revisions, free
        Caveat: Data is sampled and normalized - must download multiple times and average

        Args:
            df: Input DataFrame with Google Trends data
            search_cols: Optional dictionary of {search_term: column_name}

        Returns:
            DataFrame with Google Trends features
        """
        features = df.copy()

        if search_cols:
            for search_term, col in search_cols.items():
                if col in df.columns:
                    features[f'gt_{search_term}_level'] = df[col]
                    features[f'gt_{search_term}_ma4w'] = df[col].rolling(4, min_periods=1).mean()
                    features[f'gt_{search_term}_ma13w'] = df[col].rolling(13, min_periods=1).mean()
                    features[f'gt_{search_term}_trend'] = features[f'gt_{search_term}_ma13w'].diff(1)

        logger.info(f"Created Google Trends features")
        return features

    @staticmethod
    def create_ai_sentiment_features(df: pd.DataFrame, ai_score_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create AI-based sentiment features

        Cutting-edge approach: Use large language models (ChatGPT, Claude) to analyze
        corporate earnings call transcripts (120,000+ transcripts, 2006-present)

        Prompt model as "financial expert" to score manager optimism about US economy

        Advantage: Surveys consumer sentiment (reactive), but this surveys
        C-suite executives (decision-makers) on their actual economic expectations

        Research shows "AI Economy Score" robustly predicts GDP growth, production,
        employment up to 10 quarters into future

        Args:
            df: Input DataFrame with AI sentiment scores
            ai_score_col: Optional AI economy score column

        Returns:
            DataFrame with AI sentiment features
        """
        features = df.copy()

        if ai_score_col and ai_score_col in df.columns:
            features[f'{ai_score_col}_level'] = df[ai_score_col]
            features[f'{ai_score_col}_change'] = df[ai_score_col].diff(1)
            features[f'{ai_score_col}_ma4q'] = df[ai_score_col].rolling(4, min_periods=1).mean()
            features[f'{ai_score_col}_trend'] = features[f'{ai_score_col}_ma4q'].diff(1)

        logger.info(f"Created AI sentiment features")
        return features

    @staticmethod
    def create_credit_conditions_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite credit conditions features

        Combines multiple credit indicators:
        - Credit spreads (high-yield bond spreads)
        - Lending standards (from Fed surveys if available)
        - Credit volume growth
        - Delinquency rates

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with credit conditions features
        """
        features = df.copy()

        # Note: Implementation depends on available data columns
        # This is a placeholder for the feature engineering framework

        logger.info(f"Created credit conditions features")
        return features

    @staticmethod
    def create_all_alternative_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all alternative data features

        Args:
            df: Input DataFrame with alternative data
            config: Optional configuration with additional sources

        Returns:
            DataFrame with all alternative data features
        """
        features = df.copy()

        # Create features for each category
        features = AlternativeDataFeatures.create_policy_uncertainty_features(features)
        features = AlternativeDataFeatures.create_world_uncertainty_features(features)
        features = AlternativeDataFeatures.create_shipping_data_features(features)
        features = AlternativeDataFeatures.create_google_trends_features(features)
        features = AlternativeDataFeatures.create_ai_sentiment_features(features)
        features = AlternativeDataFeatures.create_credit_conditions_features(features)

        logger.info(f"Created alternative data features")
        return features
