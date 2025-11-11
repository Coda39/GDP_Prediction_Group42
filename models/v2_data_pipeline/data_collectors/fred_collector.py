"""
FRED Data Collector
Collects economic indicators from the Federal Reserve Economic Data (FRED) API
Saves data to Data_v2/raw/fred/ organized by frequency
"""

import os
import logging
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml

from .utils import (
    RetryHandler,
    RateLimiter,
    DataValidator,
    MetadataManager,
    FileOrganizer,
)

logger = logging.getLogger(__name__)


class FREDCollector:
    """
    Collects economic indicators from FRED API
    Reference: https://fred.stlouisfed.org/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: str = "config/data_sources.yaml",
    ):
        """
        Initialize FRED collector

        Args:
            api_key: FRED API key (or set FRED_API_KEY environment variable)
            config_path: Path to data sources configuration file
        """
        # Get API key
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key not provided. "
                "Set FRED_API_KEY environment variable or pass api_key parameter."
            )

        # Load configuration
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            self.config = self._get_default_config()
        else:
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)["fred"]

        # Initialize helpers
        self.retry_handler = RetryHandler(
            max_retries=self.config.get("max_retries", 3),
            initial_delay=self.config.get("retry_delay", 2),
            backoff_factor=self.config.get("backoff_factor", 1.5),
        )
        self.rate_limiter = RateLimiter(calls_per_second=1.0)

        # Setup directories
        self.raw_data_dir = Path(self.config.get("raw_fred_dir", "Data_v2/raw/fred"))
        FileOrganizer.organize_fred_data(str(self.raw_data_dir))

        # Collected data storage
        self.collected_data: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}

        logger.info(
            f"FREDCollector initialized with {len(self._get_all_indicators())} indicators"
        )

    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            "api_base": "https://api.stlouisfed.org/fred",
            "max_retries": 3,
            "retry_delay": 2,
            "backoff_factor": 1.5,
            "raw_fred_dir": "Data_v2/raw/fred",
        }

    def _get_all_indicators(self) -> Dict[str, Dict]:
        """
        Get all configured FRED indicators

        Returns:
            Dictionary of all indicators from config
        """
        all_indicators = {}
        for category in [
            "target_variables",
            "hard_data_production",
            "hard_data_labor",
            "hard_data_consumption",
            "hard_data_housing",
            "soft_data_pmi",
            "soft_data_sentiment",
            "financial_rates",
            "financial_spreads",
            "financial_equity",
            "monetary",
            "inflation",
            "alternative_uncertainty",
        ]:
            if category in self.config:
                all_indicators.update(self.config[category])
        return all_indicators

    def collect_indicator(
        self, series_id: str, indicator_info: Dict, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Collect a single indicator from FRED API

        Args:
            series_id: FRED series ID (e.g., 'GDPC1')
            indicator_info: Dictionary with indicator metadata
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with collected data, or None if failed
        """

        def _fetch():
            """Internal fetch function for retry logic"""
            self.rate_limiter.wait()
            url = f"{self.config['api_base']}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        try:
            logger.info(
                f"Fetching {series_id}: {indicator_info.get('name', 'Unknown')}"
            )
            data = self.retry_handler.execute(_fetch)

            # Parse response
            observations = data.get("observations", [])
            if not observations:
                logger.warning(f"No observations returned for {series_id}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            # Remove missing values
            df = df.dropna(subset=["value"])
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"✓ {series_id}: {len(df)} observations")

            # Validate data
            validation = DataValidator.validate_dataframe(df, series_id)
            if not validation["valid"]:
                logger.warning(
                    f"Validation issues for {series_id}: {validation['issues']}"
                )

            # Store metadata
            self.metadata[series_id] = {
                "name": indicator_info.get("name", series_id),
                "frequency": indicator_info.get("frequency", "d"),
                "category": indicator_info.get("category", "unknown"),
                "timing": indicator_info.get("timing", "unknown"),
                "observations_count": len(df),
                "date_range": [str(df["date"].min()), str(df["date"].max())],
                "validation": validation,
            }

            return df

        except Exception as e:
            logger.error(f"✗ Failed to fetch {series_id}: {str(e)}")
            return None

    def collect_all_indicators(
        self,
        start_date: str = "2000-01-01",
        end_date: str = "2025-12-31",
        categories: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all indicators from configuration

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            categories: List of categories to collect (None = all)

        Returns:
            Dictionary of {series_id: DataFrame}
        """
        all_indicators = self._get_all_indicators()

        logger.info(f"Starting collection of {len(all_indicators)} indicators")
        logger.info(f"Date range: {start_date} to {end_date}")

        successful = 0
        failed = 0

        for series_id, indicator_info in all_indicators.items():
            df = self.collect_indicator(series_id, indicator_info, start_date, end_date)
            if df is not None:
                self.collected_data[series_id] = df
                successful += 1
            else:
                failed += 1

        logger.info(f"Collection complete: {successful} successful, {failed} failed")
        return self.collected_data

    def save_collected_data(self, organize_by_frequency: bool = True):
        """
        Save collected data to disk

        Args:
            organize_by_frequency: If True, organize files by frequency
        """
        saved_files = 0

        for series_id, df in self.collected_data.items():
            metadata = self.metadata.get(series_id, {})
            frequency = metadata.get("frequency", "d")

            # Determine subdirectory
            if organize_by_frequency:
                subdir = self.raw_data_dir / frequency
            else:
                subdir = self.raw_data_dir

            subdir.mkdir(parents=True, exist_ok=True)

            # Save CSV
            output_path = subdir / f"{series_id}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved: {output_path}")
            saved_files += 1

        # Save metadata
        metadata_path = self.raw_data_dir / "metadata.json"
        MetadataManager.save_metadata(self.metadata, str(metadata_path))

        logger.info(f"✓ Saved {saved_files} files to {self.raw_data_dir}")

    def load_from_disk(self) -> Dict[str, pd.DataFrame]:
        """
        Load previously collected FRED data from disk

        Returns:
            Dictionary of {series_id: DataFrame}
        """
        data = {}
        for csv_file in self.raw_data_dir.glob("**/*.csv"):
            series_id = csv_file.stem
            if series_id == "metadata":
                continue
            try:
                df = pd.read_csv(csv_file)
                df["date"] = pd.to_datetime(df["date"])
                data[series_id] = df
                logger.info(f"Loaded: {series_id} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {str(e)}")

        # Load metadata
        metadata_path = self.raw_data_dir / "metadata.json"
        if metadata_path.exists():
            self.metadata = MetadataManager.load_metadata(str(metadata_path))

        return data

    def get_summary(self) -> Dict:
        """
        Get summary of collected data

        Returns:
            Dictionary with collection summary
        """
        return {
            "total_indicators": len(self.collected_data),
            "indicators": list(self.collected_data.keys()),
            "total_observations": sum(len(df) for df in self.collected_data.values()),
            "date_ranges": {
                sid: metadata.get("date_range")
                for sid, metadata in self.metadata.items()
            },
        }


def main():
    """Example usage of FREDCollector"""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Initialize collector
    collector = FREDCollector()

    # Collect all indicators
    data = collector.collect_all_indicators(
        start_date="2000-01-01", end_date="2025-12-31"
    )

    # Save to disk
    collector.save_collected_data()

    # Print summary
    summary = collector.get_summary()
    print("\n" + "=" * 80)
    print("FRED Data Collection Summary")
    print("=" * 80)
    print(f"Total Indicators: {summary['total_indicators']}")
    print(f"Total Observations: {summary['total_observations']}")
    print("\nSuccessfully collected indicators:")
    for indicator in sorted(summary["indicators"]):
        print(f"  - {indicator}")


if __name__ == "__main__":
    main()
