# 🌾 Crop Advisor — Agricultural Market Intelligence System

Crop Advisor predicts mandi (market) prices for major crops in Uttar Pradesh
using historical price trends, and recommends which crops show favorable
predicted price movement. Built as a summer project to explore the full
lifecycle of a machine learning system — from raw data to a deployed,
interactive dashboard.

## Dataset

**Source:** Indian Agricultural Mandi Prices (2023–2025)
**Subset used:** Uttar Pradesh
**Crops covered:** Onion, Potato, Wheat, Tomato *(Rice was excluded — see Limitations)*

Raw fields: State, District, Market Name, Commodity, Variety, Grade,
Min/Max/Modal Price, Price Date.

After cleaning, aggregation, and feature engineering (lag features,
rolling averages), the model-ready dataset contains **52 rows across 4 crops**:

| Crop   | Rows | Notes |
|--------|------|-------|
| Potato | 22   | Sufficient for train/test evaluation |
| Onion  | 21   | Sufficient for train/test evaluation |
| Wheat  | 6    | Too few rows for reliable evaluation — flagged low-confidence |
| Tomato | 3    | Too few rows for reliable evaluation — flagged low-confidence |

## Project Structure

```
crop-advisor/
├── data/
│   ├── raw/
│   │   └── mandi_prices_raw.csv
│   └── processed/
│       ├── clean_mandi_prices.csv
│       ├── monthly_crop_prices.csv
│       ├── feature_engineered_prices.csv
│       ├── model_ready_prices.csv
│       └── model_evaluation_summary.csv
├── models/
│   ├── onion_linear_regression.pkl
│   ├── onion_random_forest.pkl
│   ├── potato_linear_regression.pkl
│   ├── potato_random_forest.pkl
│   ├── wheat_linear_regression.pkl
│   ├── wheat_random_forest.pkl
│   ├── tomato_linear_regression.pkl
│   └── tomato_random_forest.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_aggregation.ipynb
│   ├── 04_visualization.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_training.ipynb
│   └── 07_train_all_crops.ipynb
├── src/
│   └── predictor.py
├── plots/
├── app.py
├── requirements.txt
└── README.md
```

## Key Findings

- **Random Forest outperformed Linear Regression** on both crops with
  reliable data (Onion, Potato), based on MAE. Linear Regression tended
  to overextrapolate the 2024 upward price trend into 2025's actual
  decline, while Random Forest predictions stayed anchored closer to
  historical price ranges.

- **R² was unreliable on this dataset** due to small test set sizes
  (4–5 rows after an 80/20 split). Deeply negative R² values don't
  necessarily indicate a broken model — they reflect known instability
  of R² on very small samples. MAE was used as the primary evaluation
  metric instead.

- **Wheat and Tomato had insufficient data** (6 and 3 rows respectively)
  for a meaningful train/test split. Rather than reporting misleading
  metrics, these crops are flagged as "low confidence" throughout the
  system — in the evaluation summary, the recommendation engine, and
  the dashboard itself.

- **Rice was excluded entirely** — after feature engineering (lag
  features require prior months of data), zero usable rows remained.

## How to Run It

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd crop-advisor
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

   The app will open automatically at `http://localhost:8501`.

### Retraining models (optional)

If you want to regenerate the trained models from raw data, run the
notebooks in order (01 through 07) inside the `notebooks/` folder.
Models are saved automatically to `models/` and are already included
in this repo, so this step is optional unless you're modifying the
pipeline.

## Limitations

- **Small dataset.** Only 52 rows across 4 crops after feature
  engineering — Wheat and Tomato in particular have too few rows for
  statistically meaningful evaluation, and are flagged as low-confidence
  throughout the system rather than presented with false authority.

- **No cost or yield data.** Recommendations are based purely on
  predicted price trend relative to recent average — not full
  profitability, which would require cultivation cost and expected
  yield per crop.

- **Single-state, single-district-level aggregation.** The dataset
  covers Uttar Pradesh only; prices and patterns may not generalize
  to other states.

- **R² is not a reliable metric here** due to small test set sizes.
  MAE was used as the primary evaluation metric instead — see Key
  Findings above.

## Tech Stack

- **Language:** Python
- **Data processing:** pandas, numpy
- **Visualization:** matplotlib, Streamlit's built-in charting
- **Machine Learning:** scikit-learn (Linear Regression, Random Forest)
- **Model persistence:** joblib
- **Dashboard:** Streamlit
- **Development:** Jupyter Notebook
