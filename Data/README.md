## 📁 Folder Structure

### raw/
Original OECD data files (2000-2025, monthly frequency)
- Source: OECD API
- Includes metadata JSON files
- Countries: Canada, UK, Japan, Germany, France, Italy

### historical/
Historical data (1980-2000, quarterly)
- Source: Manually collected/FRED extended
- Used to extend training data back to 1980
- Countries: Canada, UK, Japan, USA
- European countries were extended along with bloomberg extension

### bloomberg/
Bloomberg Terminal exports (Excel files)
- Corporate bond indices (investment grade)
- Government bond yields (2Y, 5Y, 10Y)
- Monthly frequency, 1989-2025
- Shared: Europe Corporate (Germany/France/Italy)

### extended/
**FINAL DATASETS USED FOR MODELING** (1980-2025, quarterly)
- Combines: raw + historical + bloomberg/yahoo finance
- Created by: data_preprocessing/data_merging/{{Country}}/extend_{{country}}_data.py
- All financial indicators included (yield curves, credit spreads)
- **7 files - one per G7 country**

### archive/
Old versions, backups, and deprecated files
- Not used in current pipeline
- Kept for reference only

## 📊 Data Coverage by Country

| Country | Raw Data | Historical | Financial Indicators | Total Quarters |
|---------|----------|------------|----------------------|----------------|
| Canada  | 2000-2025 | 1980-2000 | Bloomberg 2002-2025 | 184 | 
| UK      | 2000-2025 | 1980-2000 | Bloomberg 1998-2025 | 184 |
| Japan   | 2000-2025 | 1980-2000 | Bloomberg 1987-2025 | 184 |
| Germany | 2000-2025 | 1991+ (FRED) | Bloomberg 1998-2025 | 183 |
| France  | 2000-2025 | 1991+ (FRED) | Bloomberg 1998-2025 | 139 |
| Italy   | 2000-2025 | 1991+ (FRED) | Bloomberg 1998-2025 | 139 |
| USA     | 2000-2025 | 1980-2000 | Yahoo Finance 1997-2025 | 184 |

## 🔄 Data Pipeline

1. **Download** → `raw/` (OECD API)
2. **Extend** → Merge with `historical/` + `bloomberg/`
3. **Output** → `extended/` (final datasets)
4. **Process** → `data_preprocessing/resampled_data/`
5. **Model** → `models/nowcasting_v7/`