"""
Data Collection Utilities
Provides retry logic, rate limiting, error handling, and validation for API data collection
"""

import time
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Data_v2/logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RetryHandler:
    """
    Implements exponential backoff retry logic for API calls
    """

    def __init__(self, max_retries: int = 3, initial_delay: float = 2, backoff_factor: float = 1.5):
        """
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            backoff_factor: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with exponential backoff retry logic

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func execution

        Raises:
            Exception: If all retries are exhausted
        """
        delay = self.initial_delay
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.max_retries + 1}: {func.__name__}")
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"✓ {func.__name__} succeeded after {attempt} retries")
                return result

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(f"✗ Attempt {attempt + 1} failed: {str(e)}")
                    logger.info(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    logger.error(f"✗ All {self.max_retries + 1} attempts failed: {str(e)}")

        raise last_exception


class RateLimiter:
    """
    Implements rate limiting to avoid API throttling
    """

    def __init__(self, calls_per_second: float = 1.0):
        """
        Args:
            calls_per_second: Maximum API calls per second
        """
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0

    def wait(self):
        """
        Wait if necessary to maintain rate limit
        """
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.3f}s")
            time.sleep(wait_time)
        self.last_call_time = time.time()


class DataValidator:
    """
    Validates collected data for quality and completeness
    """

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, indicator_name: str, min_completeness: float = 0.8) -> Dict[str, Any]:
        """
        Validate a dataframe containing economic indicator data

        Args:
            df: DataFrame to validate
            indicator_name: Name of the indicator for logging
            min_completeness: Minimum acceptable data completeness (0-1)

        Returns:
            Dictionary with validation results
        """
        results = {
            'indicator': indicator_name,
            'valid': True,
            'issues': [],
            'stats': {}
        }

        # Check if dataframe is empty
        if df.empty:
            results['valid'] = False
            results['issues'].append("DataFrame is empty")
            return results

        # Check for required columns
        if 'date' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            results['issues'].append("Missing date column or datetime index")

        # Check data completeness
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        results['stats']['completeness'] = completeness
        if completeness < min_completeness:
            results['valid'] = False
            results['issues'].append(f"Completeness {completeness:.1%} below minimum {min_completeness:.1%}")

        # Check for duplicates
        if not isinstance(df.index, pd.DatetimeIndex):
            duplicates = df.duplicated(subset=['date']).sum()
        else:
            duplicates = df.index.duplicated().sum()
        results['stats']['duplicate_rows'] = duplicates
        if duplicates > 0:
            results['issues'].append(f"Found {duplicates} duplicate rows")

        # Check for outliers (z-score > 5)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'value':
                continue
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > 5).sum()
            if outliers > 0:
                results['issues'].append(f"Found {outliers} potential outliers in {col}")

        # Log validation results
        if results['valid']:
            logger.info(f"✓ {indicator_name}: VALID (Completeness: {completeness:.1%})")
        else:
            logger.warning(f"✗ {indicator_name}: INVALID - {', '.join(results['issues'])}")

        return results

    @staticmethod
    def validate_multiple(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple dataframes and return aggregate results

        Args:
            dataframes: Dictionary of {indicator_name: dataframe}

        Returns:
            Dictionary with validation results for each indicator
        """
        results = {}
        for indicator, df in dataframes.items():
            results[indicator] = DataValidator.validate_dataframe(df, indicator)
        return results


class DataQualityReport:
    """
    Generates comprehensive data quality reports
    """

    @staticmethod
    def generate_report(validation_results: Dict[str, Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Generate a data quality report

        Args:
            validation_results: Results from DataValidator
            output_path: Path to save the report

        Returns:
            Report as string
        """
        report = "=" * 80 + "\n"
        report += "DATA QUALITY REPORT\n"
        report += f"Generated: {datetime.now().isoformat()}\n"
        report += "=" * 80 + "\n\n"

        # Summary statistics
        total_indicators = len(validation_results)
        valid_indicators = sum(1 for v in validation_results.values() if v['valid'])
        invalid_indicators = total_indicators - valid_indicators

        report += f"Summary:\n"
        report += f"  Total Indicators: {total_indicators}\n"
        report += f"  Valid: {valid_indicators} ({valid_indicators/total_indicators*100:.1f}%)\n"
        report += f"  Invalid: {invalid_indicators} ({invalid_indicators/total_indicators*100:.1f}%)\n\n"

        # Detailed results
        report += "Detailed Results:\n"
        report += "-" * 80 + "\n"
        for indicator, result in sorted(validation_results.items()):
            status = "✓ VALID" if result['valid'] else "✗ INVALID"
            report += f"{indicator:40s} {status}\n"
            if 'completeness' in result['stats']:
                report += f"  Completeness: {result['stats']['completeness']:.1%}\n"
            if result['issues']:
                for issue in result['issues']:
                    report += f"  ⚠ {issue}\n"
            report += "\n"

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Data quality report saved to {output_path}")

        return report


class MetadataManager:
    """
    Manages metadata for collected data (timestamps, sources, versions)
    """

    @staticmethod
    def save_metadata(metadata: Dict[str, Any], output_path: str):
        """
        Save metadata to JSON file

        Args:
            metadata: Dictionary of metadata
            output_path: Path to save metadata
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {output_path}")

    @staticmethod
    def load_metadata(metadata_path: str) -> Dict[str, Any]:
        """
        Load metadata from JSON file

        Args:
            metadata_path: Path to metadata file

        Returns:
            Dictionary of metadata
        """
        if not Path(metadata_path).exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {}

        with open(metadata_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def create_collection_metadata(source: str, indicators: list, date_range: tuple) -> Dict[str, Any]:
        """
        Create metadata for a data collection run

        Args:
            source: Data source (e.g., 'FRED', 'OECD')
            indicators: List of collected indicators
            date_range: Tuple of (start_date, end_date)

        Returns:
            Dictionary with collection metadata
        """
        return {
            'source': source,
            'collection_timestamp': datetime.now().isoformat(),
            'start_date': date_range[0],
            'end_date': date_range[1],
            'indicators_count': len(indicators),
            'indicators': indicators,
            'python_version': __import__('sys').version,
            'pandas_version': pd.__version__
        }


class FileOrganizer:
    """
    Organizes downloaded data files into standardized directory structure
    """

    @staticmethod
    def organize_fred_data(raw_data_dir: str):
        """
        Organize FRED data by frequency (daily, monthly, quarterly)

        Args:
            raw_data_dir: Root directory for FRED data
        """
        raw_path = Path(raw_data_dir)
        raw_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for freq in ['daily', 'monthly', 'quarterly', 'weekly']:
            (raw_path / freq).mkdir(exist_ok=True)
            logger.info(f"Created directory: {raw_path / freq}")

    @staticmethod
    def organize_oecd_data(raw_data_dir: str):
        """
        Organize OECD data by country and indicator

        Args:
            raw_data_dir: Root directory for OECD data
        """
        raw_path = Path(raw_data_dir)
        raw_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for indicator categories
        for category in ['gdp', 'employment', 'trade', 'production', 'consumption']:
            (raw_path / category).mkdir(exist_ok=True)
            logger.info(f"Created directory: {raw_path / category}")


# Utility functions
def setup_logging(log_file: str, verbose: bool = True):
    """Setup logging configuration"""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)
    logger.info("Logging initialized")


def format_date_range(start_date: str, end_date: str) -> tuple:
    """
    Format and validate date range

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        Tuple of (start_date, end_date) as strings
    """
    # Validate format
    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except Exception as e:
        logger.error(f"Invalid date format: {e}")
        raise

    return start_date, end_date
