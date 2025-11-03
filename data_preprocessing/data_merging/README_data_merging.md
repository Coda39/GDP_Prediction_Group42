Data Merging Scripts - Phase 3 Historical Data Expansion
This folder contains scripts to download and merge historical data (1980-2000) with current data (2001-2025).
Files in this folder:

download_historical_data.py - Downloads 1980-2000 data from FRED API
merge_historical_data.py - Merges historical with current data
test_phase3_data.py - Verifies the merge worked correctly

Setup (One-time):
powershell# Install required library
pip install fredapi
Usage (Run in order):
powershellcd data_preprocessing\data_merging

# Step 1: Download historical data (5-10 minutes)
python download_historical_data.py

# Step 2: Merge with current data (1 minute)
python merge_historical_data.py

# Step 3: Verify it worked (optional)
python test_phase3_data.py
What it does:
Before Phase 3:

Training data: 72 quarters (2001-2018)
Data location: Data/united_states_fred_data.csv, etc.

After Phase 3:

Training data: 152 quarters (1980-2018) - +111% more data!
Data location: Data/united_states_fred_data_extended.csv, etc.
Includes: 1980s stagflation, high inflation periods, more regime variety

Output files created:
In Data/ directory:

united_states_fred_data_extended.csv (1980-2025)
canada_data_extended.csv (1980-2025)
japan_data_extended.csv (1980-2025)
united_kingdom_data_extended.csv (1980-2025)

In Data/historical/ directory:

usa_historical_1980-2000.csv
canada_historical_1980-2000.csv
japan_historical_1980-2000.csv
uk_historical_1980-2000.csv

After running these scripts:
You need to update preprocessing_pipeline.py to use the *_extended.csv files instead of the original files.
See QUICK_START.md for detailed instructions.
FRED API Key:
Your API key is already configured in download_historical_data.py:
pythonFRED_API_KEY = '3477091a9ab79fb20b9ac8aca531d2dd'
Troubleshooting:
"File not found" errors:

Make sure you're running from data_preprocessing/data_merging/ directory
Scripts use relative paths: ../../Data/