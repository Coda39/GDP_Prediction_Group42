"""
Leading/Lagging Feature Classifier
Classifies economic indicators by their timing relative to the business cycle
Ensures forecasting models use only leading indicators (no lookahead bias)
Ensures nowcasting models use only coincident indicators (no future information)
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class FeatureClassifier:
    """
    Classifies features as Leading, Coincident, or Lagging based on economic theory
    Supports task-specific feature selection for forecasting vs nowcasting
    """

    # Based on "A Quantitative Analyst's Guide to GDP Forecasting"
    INDICATOR_CLASSIFICATION = {
        # =====================================================================
        # LEADING INDICATORS - Change 6-18 months BEFORE GDP
        # =====================================================================
        'leading': {
            # Yield Curve (MOST important recession predictor)
            'T10Y2Y': 'yield_curve_spread',
            'T10Y3M': 'yield_curve_spread',
            'yield_spread_10y_2y': 'yield_curve_spread',
            'yield_spread_10y_3m': 'yield_curve_spread',

            # Housing (Most interest-rate sensitive sector)
            'PERMIT': 'housing',
            'HOUST': 'housing',

            # Labor (Hours lead hiring, claims lead layoffs)
            'AWHMBS': 'labor_hours',
            'ICSA': 'labor_claims',
            'AWHMBS_change_1m': 'labor_hours',
            'ICSA_ma13w': 'labor_claims',

            # Purchasing Managers Index (NEW ORDERS especially)
            'NAPMNPI': 'pmi_orders',
            'MMNRNJ': 'pmi_manufacturing',
            'IMNRNJ': 'pmi_services',
            'composite_pmi': 'pmi_composite',

            # Consumer Sentiment (expectations component especially)
            'UMCSENT': 'consumer_sentiment',
            'CONFCSL': 'consumer_confidence',

            # Credit Spreads (Risk premium for default)
            'BAMLH0A0HYM2': 'credit_spreads',

            # Stock Market
            'SP500': 'equity_market',
            'SP500_return_20d': 'equity_market',

            # Financial Volatility
            'VIXCLS': 'volatility',

            # Policy & Uncertainty
            'USEPUINDXD': 'policy_uncertainty',
            'USEPUINDXD_level': 'policy_uncertainty',
            'USEPUINDXD_ma60': 'policy_uncertainty',
            'WUIUS': 'world_uncertainty',

            # Term Structure Factors
            'term_level': 'term_structure',
            'term_slope': 'term_structure',
            'term_curvature': 'term_structure',

            # Building Permits
            'PERMIT_level': 'housing',
            'permits_starts_ratio': 'housing'
        },

        # =====================================================================
        # COINCIDENT INDICATORS - Move SIMULTANEOUSLY with GDP
        # =====================================================================
        'coincident': {
            # Labor (Actual employment is coincident)
            'PAYEMS': 'labor_payroll',
            'PAYEMS_pct_change_1m': 'labor_payroll',
            'UNRATE': 'unemployment',

            # Production
            'INDPRO': 'industrial_production',
            'INDPRO_pct_change_1m': 'industrial_production',
            'CUMFSL': 'capacity_utilization',

            # Income
            'W875RX1': 'personal_income',
            'W875RX1_pct_change_1m': 'personal_income',

            # Consumption
            'RSXFS': 'retail_sales',
            'RSXFS_pct_change_1m': 'retail_sales',

            # Trade (actual flows, not expectations)
            'EXPGS': 'exports',
            'IMPGS': 'imports',
            'EXPGS_pct_change_1m': 'exports',
            'IMPGS_pct_change_1m': 'imports',

            # Manufacturing Activity (if current month, not forecast)
            'MMNRNJ_level': 'pmi_manufacturing',
            'IMNRNJ_level': 'pmi_services'
        },

        # =====================================================================
        # LAGGING INDICATORS - Change AFTER GDP has already turned
        # =====================================================================
        'lagging': {
            'UNRATE': 'unemployment',
            'CPIAUCSL': 'inflation',
            'PSAVERT': 'savings_rate'
        }
    }

    @classmethod
    def classify_feature(cls, feature_name: str) -> str:
        """
        Classify a single feature

        Args:
            feature_name: Name of the feature

        Returns:
            Classification: 'leading', 'coincident', 'lagging', or 'unknown'
        """
        for category, features in cls.INDICATOR_CLASSIFICATION.items():
            if feature_name in features:
                return category

        # Try to infer from feature name
        if any(keyword in feature_name.lower() for keyword in ['yield', 'spread', 'permit', 'hours', 'claims', 'pmi', 'sentiment', 'confidence', 'credit', 'vix', 'uncertainty', 'stock', 'sp500']):
            return 'leading'
        elif any(keyword in feature_name.lower() for keyword in ['payroll', 'indpro', 'production', 'income', 'retail', 'sales', 'export', 'import']):
            return 'coincident'
        elif any(keyword in feature_name.lower() for keyword in ['unemployment', 'cpi', 'inflation']):
            return 'lagging'

        return 'unknown'

    @classmethod
    def get_features_for_task(cls, task: str, all_features: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Get appropriate feature set for a specific task

        Args:
            task: 'forecasting' or 'nowcasting'
            all_features: List of all available features

        Returns:
            Tuple of (filtered_features, feature_classifications)
        """
        if task == 'forecasting':
            # Use only LEADING indicators for 6-18 month forecasts
            # (No coincident/lagging to avoid lookahead bias)
            allowed_categories = ['leading']
            reason = "Leading indicators only (no future information)"

        elif task == 'nowcasting':
            # Use only COINCIDENT indicators for current quarter estimates
            # Leading indicators are for future predictions, not now
            # Lagging indicators are confirmed after-the-fact
            allowed_categories = ['coincident']
            reason = "Coincident indicators only (current economic activity)"

        else:
            raise ValueError(f"Unknown task: {task}. Choose 'forecasting' or 'nowcasting'")

        # Filter features
        filtered_features = []
        feature_classifications = {}

        for feature in all_features:
            classification = cls.classify_feature(feature)
            if classification in allowed_categories:
                filtered_features.append(feature)
                feature_classifications[feature] = classification

        logger.info(f"\n{task.upper()} Feature Selection:")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  Total features provided: {len(all_features)}")
        logger.info(f"  Allowed categories: {', '.join(allowed_categories)}")
        logger.info(f"  Features selected: {len(filtered_features)}")
        logger.info(f"  By category: {dict((cat, sum(1 for f in feature_classifications.values() if f == cat)) for cat in allowed_categories)}")

        return filtered_features, feature_classifications

    @classmethod
    def print_classification_summary(cls):
        """Print summary of all classified indicators"""
        print("\n" + "=" * 80)
        print("FEATURE CLASSIFICATION SUMMARY")
        print("=" * 80)

        for category in ['leading', 'coincident', 'lagging']:
            features = cls.INDICATOR_CLASSIFICATION.get(category, {})
            print(f"\n{category.upper()} INDICATORS ({len(features)} total):")
            print("-" * 80)

            # Group by subcategory
            subcategories = {}
            for feature, subcat in features.items():
                if subcat not in subcategories:
                    subcategories[subcat] = []
                subcategories[subcat].append(feature)

            for subcat in sorted(subcategories.keys()):
                print(f"  {subcat}:")
                for feature in sorted(subcategories[subcat]):
                    print(f"    - {feature}")


def create_feature_selection_report(forecasting_features: List[str], nowcasting_features: List[str],
                                   output_path: str = 'Data_v2/processed/feature_selection_report.txt'):
    """
    Create a detailed feature selection report

    Args:
        forecasting_features: Features selected for forecasting
        nowcasting_features: Features selected for nowcasting
        output_path: Path to save report
    """
    report = "=" * 80 + "\n"
    report += "FEATURE SELECTION REPORT\n"
    report += "V2 Data Pipeline - Research-Based Feature Framework\n"
    report += "=" * 80 + "\n\n"

    report += "FORECASTING FEATURES (Leading Indicators)\n"
    report += "-" * 80 + "\n"
    report += f"Total: {len(forecasting_features)} features\n"
    report += "Purpose: Predict GDP 6-18 months in advance\n"
    report += "Logic: Only use indicators that LEAD the business cycle\n\n"
    report += "Features:\n"
    for feat in sorted(forecasting_features):
        report += f"  - {feat}\n"

    report += "\n\nNOWCASTING FEATURES (Coincident Indicators)\n"
    report += "-" * 80 + "\n"
    report += f"Total: {len(nowcasting_features)} features\n"
    report += "Purpose: Estimate current quarter GDP\n"
    report += "Logic: Only use indicators that move with current economic activity\n\n"
    report += "Features:\n"
    for feat in sorted(nowcasting_features):
        report += f"  - {feat}\n"

    report += "\n\nKEY PRINCIPLE\n"
    report += "-" * 80 + "\n"
    report += "Forecasting and Nowcasting use DIFFERENT feature sets:\n"
    report += "- FORECASTING: Leading indicators (yield curve, sentiment, spreads)\n"
    report += "- NOWCASTING: Coincident indicators (payroll, production, income)\n"
    report += "This prevents lookahead bias and respects information timing\n"

    # Write report
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Feature selection report saved to {output_path}")


if __name__ == '__main__':
    # Print classification summary
    FeatureClassifier.print_classification_summary()

    # Example usage
    all_features = [
        'T10Y2Y',  # Leading
        'PAYEMS',  # Coincident
        'UNRATE',  # Lagging
        'PERMIT',  # Leading
        'INDPRO',  # Coincident
        'MMNRNJ',  # Leading (PMI new orders)
        'UMCSENT'  # Leading (sentiment)
    ]

    # Get forecasting features
    forecast_feats, forecast_class = FeatureClassifier.get_features_for_task('forecasting', all_features)
    print(f"\nForecasting features: {forecast_feats}")

    # Get nowcasting features
    nowcast_feats, nowcast_class = FeatureClassifier.get_features_for_task('nowcasting', all_features)
    print(f"Nowcasting features: {nowcast_feats}")
