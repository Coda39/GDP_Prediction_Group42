"""
OECD Data Collector
Collects economic indicators for G7 countries from OECD databases
Saves data to Data_v2/raw/oecd/ organized by indicator category
"""

import os
import logging
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

from .utils import RetryHandler, RateLimiter, DataValidator, MetadataManager, FileOrganizer

logger = logging.getLogger(__name__)


class OECDCollector:
    """
    Collects economic indicators for G7 countries from OECD APIs
    Uses both direct API calls and SDMX interface
    """

    def __init__(self, raw_data_dir: str = 'Data_v2/raw/oecd', start_date: str = '2000',
                 end_date: str = '2025'):
        """
        Initialize OECD collector

        Args:
            raw_data_dir: Directory to save OECD data
            start_date: Start year (e.g., '2000')
            end_date: End year (e.g., '2025')
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.start_date = start_date
        self.end_date = end_date

        # G7 countries with OECD codes
        self.g7_countries = {
            'USA': 'United States',
            'CAN': 'Canada',
            'JPN': 'Japan',
            'GBR': 'United Kingdom',
            'FRA': 'France',
            'DEU': 'Germany',
            'ITA': 'Italy'
        }

        # Initialize helpers
        self.retry_handler = RetryHandler(max_retries=3, initial_delay=2, backoff_factor=1.5)
        self.rate_limiter = RateLimiter(calls_per_second=0.5)  # OECD has stricter limits

        # Setup directories
        FileOrganizer.organize_oecd_data(str(self.raw_data_dir))

        self.collected_data: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}

        logger.info(f"OECDCollector initialized for {len(self.g7_countries)} G7 countries")

    def collect_qna_data(self, indicator: str, country_code: str) -> Optional[pd.DataFrame]:
        """
        Collect quarterly national accounts data from OECD

        Args:
            indicator: OECD indicator code (e.g., 'GDP', 'HFCEXP', 'GFCF')
            country_code: OECD country code (e.g., 'USA', 'CAN')

        Returns:
            DataFrame with collected data, or None if failed
        """
        def _fetch():
            """Fetch data using OECD Stats API"""
            self.rate_limiter.wait()

            # OECD Stats API for QNA (Quarterly National Accounts)
            url = f"https://stats.oecd.org/sdmx-json/data/QNA/{country_code}.{indicator}.VOBARSA.Q"
            params = {
                'startTime': self.start_date,
                'endTime': self.end_date
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()

        try:
            logger.info(f"Fetching OECD QNA: {indicator} for {country_code}")
            data = self.retry_handler.execute(_fetch)

            # Parse SDMX-JSON response
            observations = self._parse_sdmx_json(data, country_code, indicator)

            if not observations:
                logger.warning(f"No observations returned for {indicator} ({country_code})")
                return None

            df = pd.DataFrame(observations)
            df = df.sort_values('date').reset_index(drop=True)

            logger.info(f"✓ {indicator} ({country_code}): {len(df)} observations")
            return df

        except Exception as e:
            logger.error(f"✗ Failed to fetch {indicator} ({country_code}): {str(e)}")
            return None

    def collect_employment_data(self, country_code: str) -> Optional[pd.DataFrame]:
        """
        Collect employment data from OECD ALFS (Annual Labour Force Statistics)

        Args:
            country_code: OECD country code

        Returns:
            DataFrame with employment data, or None if failed
        """
        def _fetch():
            """Fetch ALFS employment data"""
            self.rate_limiter.wait()
            url = f"https://stats.oecd.org/sdmx-json/data/ALFS_EMP/{country_code}.EMP.Q"
            params = {
                'startTime': self.start_date,
                'endTime': self.end_date
            }
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()

        try:
            logger.info(f"Fetching OECD Employment: {country_code}")
            data = self.retry_handler.execute(_fetch)
            observations = self._parse_sdmx_json(data, country_code, 'EMP')

            if observations:
                df = pd.DataFrame(observations)
                df = df.sort_values('date').reset_index(drop=True)
                logger.info(f"✓ Employment ({country_code}): {len(df)} observations")
                return df
            return None

        except Exception as e:
            logger.error(f"✗ Failed to fetch employment ({country_code}): {str(e)}")
            return None

    def collect_all_g7_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect all QNA indicators for all G7 countries

        Returns:
            Dictionary structure: {country: {indicator: DataFrame}}
        """
        # Core QNA indicators
        qna_indicators = {
            'GDP': 'Gross Domestic Product',
            'HFCEXP': 'Household Final Consumption Expenditure',
            'GFCF': 'Gross Fixed Capital Formation',
            'XGSV': 'Exports of Goods and Services',
            'MGSV': 'Imports of Goods and Services'
        }

        all_data = {}

        for country_code, country_name in self.g7_countries.items():
            logger.info(f"Collecting data for {country_name} ({country_code})")
            country_data = {}

            # Collect QNA data
            for indicator, indicator_name in qna_indicators.items():
                df = self.collect_qna_data(indicator, country_code)
                if df is not None:
                    country_data[indicator] = df
                    self.collected_data[f"{country_code}_{indicator}"] = df

                    # Save to disk
                    subdir = self.raw_data_dir / indicator.lower()
                    subdir.mkdir(parents=True, exist_ok=True)
                    output_path = subdir / f"{country_code}_{indicator}.csv"
                    df.to_csv(output_path, index=False)
                    logger.info(f"  Saved: {output_path}")

            # Collect employment data
            emp_df = self.collect_employment_data(country_code)
            if emp_df is not None:
                country_data['EMP'] = emp_df
                self.collected_data[f"{country_code}_EMP"] = emp_df

                subdir = self.raw_data_dir / 'employment'
                subdir.mkdir(parents=True, exist_ok=True)
                output_path = subdir / f"{country_code}_EMP.csv"
                emp_df.to_csv(output_path, index=False)
                logger.info(f"  Saved: {output_path}")

            all_data[country_code] = country_data

        return all_data

    @staticmethod
    def _parse_sdmx_json(response_data: dict, country_code: str, indicator: str) -> List[Dict]:
        """
        Parse SDMX-JSON response from OECD API

        Args:
            response_data: JSON response from OECD API
            country_code: Country code for filtering
            indicator: Indicator code

        Returns:
            List of dictionaries with {date, value}
        """
        try:
            dimensions = response_data.get('structure', {}).get('dimensions', {})
            observations = response_data.get('data', {}).get('observations', {})

            if not observations:
                return []

            # Find time dimension index (usually 0)
            time_dim_idx = 0

            results = []
            for obs_key, obs_values in observations.items():
                key_parts = obs_key.split(':')
                if len(key_parts) > time_dim_idx:
                    time_idx = int(key_parts[time_dim_idx])

                    # Get time value from dimensions
                    time_dim = dimensions.get('observation', [{}])[0]
                    if 'category' in time_dim:
                        time_values = list(time_dim['category'].keys())
                        if time_idx < len(time_values):
                            date_str = time_values[time_idx]

                            # Convert date (e.g., "2020-Q1" to datetime)
                            try:
                                if 'Q' in date_str:
                                    year, quarter = date_str.split('-Q')
                                    month = (int(quarter) - 1) * 3 + 1
                                    date = pd.to_datetime(f'{year}-{month:02d}-01')
                                else:
                                    date = pd.to_datetime(date_str)

                                # Get observation value
                                value = obs_values[0] if obs_values else None
                                if value is not None:
                                    results.append({
                                        'date': date,
                                        'value': float(value)
                                    })
                            except Exception as e:
                                logger.debug(f"Error parsing observation: {e}")
                                continue

            return results
        except Exception as e:
            logger.error(f"Error parsing SDMX-JSON response: {str(e)}")
            return []

    def save_metadata(self):
        """Save metadata about collected OECD data"""
        metadata_path = self.raw_data_dir / "metadata.json"
        metadata = {
            'source': 'OECD',
            'collection_timestamp': datetime.now().isoformat(),
            'countries': list(self.g7_countries.keys()),
            'date_range': [self.start_date, self.end_date],
            'total_series': len(self.collected_data)
        }
        MetadataManager.save_metadata(metadata, str(metadata_path))

    def get_summary(self) -> Dict:
        """Get summary of collected OECD data"""
        return {
            'total_series': len(self.collected_data),
            'countries': list(self.g7_countries.keys()),
            'total_observations': sum(len(df) for df in self.collected_data.values() if df is not None),
            'series_list': list(self.collected_data.keys())
        }


def main():
    """Example usage of OECDCollector"""
    import logging
    logging.basicConfig(level=logging.INFO)

    # Initialize collector
    collector = OECDCollector(start_date='2000', end_date='2025')

    # Collect all G7 data
    data = collector.collect_all_g7_data()

    # Save metadata
    collector.save_metadata()

    # Print summary
    summary = collector.get_summary()
    print("\n" + "=" * 80)
    print("OECD Data Collection Summary")
    print("=" * 80)
    print(f"Total Series: {summary['total_series']}")
    print(f"Total Observations: {summary['total_observations']}")
    print(f"Countries: {', '.join(summary['countries'])}")
    print("\nCollected series:")
    for series in sorted(summary['series_list']):
        print(f"  - {series}")


if __name__ == '__main__':
    main()
