## Day 7 Summary — Chronological Train/Test Split & ML Data Preparation

### Goal
Implement correct train/test splitting for time-series crop price data.
Built: src/price_predictor.py

### Key Concept: Why Random Split is Wrong for Time-Series

ML rows are normally independent — random splitting is fine.
Time-series rows are NOT independent — each row's lag features are
computed from previous rows. Random splitting allows test-set rows
to appear before training-set rows chronologically, meaning training
features may be computed from test-set prices.

This is **data leakage** — the model trains on future information it
should not have access to. Results look unrealistically good during
development but collapse in production.

### Solution: Chronological Splitting
Sort by date → split positionally at 80% → verify train ends before
test begins. Never shuffle. Never use sklearn's train_test_split.

### Data Splits Created

| Crop   | Total Rows | Train Rows | Test Rows | Train Period          | Test Period          |
|--------|------------|------------|-----------|----------------------|---------------------|
| Onion  | 21         | 16         | 5         | Sep-2023 → Dec-2024  | Feb-2025 → Jun-2025 |
| Potato | 22         | 17         | 5         | ~Sep-2023 → Jan-2025 | Feb-2025 → Jun-2025 |
| Wheat  | 6          | 4          | 2         | Sep-2023 → Nov-2023  | Dec-2023 → Jan-2024 |
| Tomato | 3          | 2          | 1         | Sep-2023 → Oct-2023  | Nov-2023            |
| Rice   | 0          | —          | —         | Excluded (0 usable rows after feature engineering) |

### Features Used (8 total)
- **Month**: Seasonal signal — Onion peaks Month=10/11
- **Year**: Long-term trend — MSP increases, inflation
- **Quarter**: Coarser seasonal grouping
- **lag_1**: Last month's price (strongest predictor)
- **lag_2**: 2 months ago
- **lag_3**: 3 months ago
- **rolling_mean_3**: 3-month average before current month (trend)
- **rolling_std_3**: 3-month volatility before current month (risk)

### Target
**Avg_Modal_Price** — monthly average modal price (₹/quintal)

### Files Created Today
- `src/__init__.py` — makes src/ importable as a Python package
- `src/price_predictor.py` — data loading + chronological splitting

### Key Functions
- `load_data(path)` — loads CSV, converts Date, sorts chronologically
- `split_crop_data(df, commodity, train_ratio)` — chronological split
  for one crop with NaN validation and temporal integrity assertion
- `prepare_all_crops(df, crops, train_ratio)` — splits all crops,
  returns dictionary of {crop: {X_train, X_test, y_train, y_test}}
- `get_feature_info()` — prints human-readable feature explanations

### Data Leakage Prevention
1. `shift(1)` before `rolling()` → rolling features don't use current month
2. Chronological split → test rows are always chronologically after train rows
3. `assert train['Date'].max() < test['Date'].min()` → automatic verification

### Limitations Documented
- Onion/Potato: reliable (21-22 rows, 2+ years of data)
- Wheat: marginal (6 rows — treat model as illustrative)
- Tomato: insufficient (3 rows — single year only)
- Rice: excluded (0 usable rows after feature engineering)

### Interview Answer: "Why chronological split?"
"In time-series, lag features create dependencies between rows —
each row's inputs are computed from previous rows' targets.
Random splitting allows future rows in training, which is data leakage.
Chronological splitting ensures the model only learns from the past
and is evaluated on the genuine future, matching real deployment."

### Next Step
Day 8: Train Linear Regression and Random Forest models using
`from src.price_predictor import load_data, prepare_all_crops`
Evaluate using MAE, RMSE, R².