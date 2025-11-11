"""
Hard Data Feature Engineering
Creates features from official government statistics:
- Industrial Production, Manufacturing Output
- Labor Market: Employment, Hours, Claims
- Consumption: Retail Sales, Personal Income
- Housing: Building Permits, Housing Starts
- Trade: Exports, Imports, Trade Balance
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HardDataFeatures:
    """
    Engineers features from hard economic data (official statistics)
    """

    @staticmethod
    def create_production_features(df: pd.DataFrame, ip_col: str = 'INDPRO',
                                   cu_col: str = 'CUMFSL') -> pd.DataFrame:
        """
        Create production and capacity utilization features

        Args:
            df: Input DataFrame with industrial production and capacity utilization
            ip_col: Column name for Industrial Production Index
            cu_col: Column name for Capacity Utilization

        Returns:
            DataFrame with production features added
        """
        features = df.copy()

        if ip_col in df.columns:
            # Industrial production growth rates
            features[f'{ip_col}_pct_change_1m'] = df[ip_col].pct_change(1)  # MoM
            features[f'{ip_col}_pct_change_12m'] = df[ip_col].pct_change(12)  # YoY
            features[f'{ip_col}_ma3'] = df[ip_col].rolling(3, min_periods=1).mean()
            features[f'{ip_col}_ma12'] = df[ip_col].rolling(12, min_periods=1).mean()

            # Trend and acceleration
            features[f'{ip_col}_trend'] = features[f'{ip_col}_ma12'].diff(1)
            features[f'{ip_col}_acceleration'] = features[f'{ip_col}_trend'].diff(1)

        if cu_col in df.columns:
            # Capacity utilization level and changes
            features[f'{cu_col}_level'] = df[cu_col]
            features[f'{cu_col}_change_1m'] = df[cu_col].diff(1)
            features[f'{cu_col}_change_12m'] = df[cu_col].diff(12)
            features[f'{cu_col}_ma3'] = df[cu_col].rolling(3, min_periods=1).mean()

        logger.info(f"Created {len([c for c in features.columns if 'INDPRO' in c or 'CUMFSL' in c])} production features")
        return features

    @staticmethod
    def create_labor_features(df: pd.DataFrame, payrolls_col: str = 'PAYEMS',
                             hours_col: str = 'AWHMBS', claims_col: str = 'ICSA',
                             unemployment_col: str = 'UNRATE') -> pd.DataFrame:
        """
        Create labor market features

        Args:
            df: Input DataFrame with labor data
            payrolls_col: Non-farm payrolls column
            hours_col: Average weekly hours column
            claims_col: Initial jobless claims column
            unemployment_col: Unemployment rate column

        Returns:
            DataFrame with labor features added
        """
        features = df.copy()

        # Payroll features
        if payrolls_col in df.columns:
            features[f'{payrolls_col}_pct_change_1m'] = df[payrolls_col].pct_change(1)
            features[f'{payrolls_col}_pct_change_12m'] = df[payrolls_col].pct_change(12)
            features[f'{payrolls_col}_ma3'] = df[payrolls_col].rolling(3, min_periods=1).mean()

        # Hours worked features (LEADING indicator - firms cut hours before hiring)
        if hours_col in df.columns:
            features[f'{hours_col}_level'] = df[hours_col]
            features[f'{hours_col}_change_1m'] = df[hours_col].diff(1)
            features[f'{hours_col}_change_3m'] = df[hours_col].diff(3)
            features[f'{hours_col}_ma3'] = df[hours_col].rolling(3, min_periods=1).mean()

        # Initial jobless claims features (LEADING indicator - weekly frequency aggregated)
        if claims_col in df.columns:
            features[f'{claims_col}_level'] = df[claims_col]
            features[f'{claims_col}_ma4w'] = df[claims_col].rolling(4, min_periods=1).mean()
            features[f'{claims_col}_ma13w'] = df[claims_col].rolling(13, min_periods=1).mean()
            features[f'{claims_col}_change_4w'] = df[claims_col].diff(4)

        # Unemployment rate features (LAGGING indicator)
        if unemployment_col in df.columns:
            features[f'{unemployment_col}_level'] = df[unemployment_col]
            features[f'{unemployment_col}_change_1m'] = df[unemployment_col].diff(1)
            features[f'{unemployment_col}_change_12m'] = df[unemployment_col].diff(12)

        logger.info(f"Created labor market features")
        return features

    @staticmethod
    def create_consumption_features(df: pd.DataFrame, retail_col: str = 'RSXFS',
                                   income_col: str = 'W875RX1', savings_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create consumption and income features

        Args:
            df: Input DataFrame with consumption data
            retail_col: Real retail sales column
            income_col: Real personal income column
            savings_col: Optional savings rate column

        Returns:
            DataFrame with consumption features added
        """
        features = df.copy()

        # Retail sales features (COINCIDENT indicator)
        if retail_col in df.columns:
            features[f'{retail_col}_pct_change_1m'] = df[retail_col].pct_change(1)
            features[f'{retail_col}_pct_change_12m'] = df[retail_col].pct_change(12)
            features[f'{retail_col}_ma3'] = df[retail_col].rolling(3, min_periods=1).mean()
            features[f'{retail_col}_ma12'] = df[retail_col].rolling(12, min_periods=1).mean()

        # Personal income features (COINCIDENT indicator)
        if income_col in df.columns:
            features[f'{income_col}_pct_change_1m'] = df[income_col].pct_change(1)
            features[f'{income_col}_pct_change_12m'] = df[income_col].pct_change(12)
            features[f'{income_col}_ma3'] = df[income_col].rolling(3, min_periods=1).mean()

            # Consumption-to-income ratio (if both available)
            if retail_col in df.columns:
                features['consumption_income_ratio'] = df[retail_col] / df[income_col]

        # Savings rate
        if savings_col and savings_col in df.columns:
            features[f'{savings_col}_level'] = df[savings_col]
            features[f'{savings_col}_change_1m'] = df[savings_col].diff(1)

        logger.info(f"Created consumption features")
        return features

    @staticmethod
    def create_housing_features(df: pd.DataFrame, permits_col: str = 'PERMIT',
                               starts_col: str = 'HOUST') -> pd.DataFrame:
        """
        Create housing sector features (LEADING indicators)

        Args:
            df: Input DataFrame with housing data
            permits_col: Building permits column
            starts_col: Housing starts column

        Returns:
            DataFrame with housing features added
        """
        features = df.copy()

        # Building permits (LEADING - precedes actual construction by 3-6 months)
        if permits_col in df.columns:
            features[f'{permits_col}_level'] = df[permits_col]
            features[f'{permits_col}_pct_change_1m'] = df[permits_col].pct_change(1)
            features[f'{permits_col}_pct_change_12m'] = df[permits_col].pct_change(12)
            features[f'{permits_col}_ma3'] = df[permits_col].rolling(3, min_periods=1).mean()

        # Housing starts (LEADING)
        if starts_col in df.columns:
            features[f'{starts_col}_level'] = df[starts_col]
            features[f'{starts_col}_pct_change_1m'] = df[starts_col].pct_change(1)
            features[f'{starts_col}_pct_change_12m'] = df[starts_col].pct_change(12)
            features[f'{starts_col}_ma3'] = df[starts_col].rolling(3, min_periods=1).mean()

            # Permit-to-starts ratio (if both available)
            if permits_col in df.columns:
                features['permits_starts_ratio'] = df[permits_col] / (df[starts_col] + 1)

        logger.info(f"Created housing features")
        return features

    @staticmethod
    def create_trade_features(df: pd.DataFrame, exports_col: str = 'EXPGS',
                             imports_col: str = 'IMPGS') -> pd.DataFrame:
        """
        Create international trade features

        Args:
            df: Input DataFrame with trade data
            exports_col: Exports column
            imports_col: Imports column

        Returns:
            DataFrame with trade features added
        """
        features = df.copy()

        # Export features (COINCIDENT indicator for external demand)
        if exports_col in df.columns:
            features[f'{exports_col}_pct_change_1m'] = df[exports_col].pct_change(1)
            features[f'{exports_col}_pct_change_12m'] = df[exports_col].pct_change(12)
            features[f'{exports_col}_ma3'] = df[exports_col].rolling(3, min_periods=1).mean()

        # Import features (COINCIDENT indicator for domestic demand)
        if imports_col in df.columns:
            features[f'{imports_col}_pct_change_1m'] = df[imports_col].pct_change(1)
            features[f'{imports_col}_pct_change_12m'] = df[imports_col].pct_change(12)
            features[f'{imports_col}_ma3'] = df[imports_col].rolling(3, min_periods=1).mean()

        # Trade balance features
        if exports_col in df.columns and imports_col in df.columns:
            features['trade_balance'] = df[exports_col] - df[imports_col]
            features['trade_balance_ratio'] = (df[exports_col] - df[imports_col]) / (df[exports_col] + df[imports_col] + 1)
            features['export_import_ratio'] = df[exports_col] / (df[imports_col] + 1)

        logger.info(f"Created trade features")
        return features

    @staticmethod
    def create_all_hard_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all hard data features

        Args:
            df: Input DataFrame with all economic data
            config: Optional configuration dictionary for column names

        Returns:
            DataFrame with all hard data features
        """
        features = df.copy()

        # Create features for each category
        features = HardDataFeatures.create_production_features(features)
        features = HardDataFeatures.create_labor_features(features)
        features = HardDataFeatures.create_consumption_features(features)
        features = HardDataFeatures.create_housing_features(features)
        features = HardDataFeatures.create_trade_features(features)

        logger.info(f"Created {len(features.columns)} total columns (including original data)")
        return features
