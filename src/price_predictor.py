"""
price_predictor.py
==================
Handles data loading, chronological train/test splitting,
and feature/target preparation for crop price prediction.

Part of: Smart Crop Planning Advisor
Author: [Your Name]
Dataset: Indian Agricultural Mandi Prices (2023-2025), Uttar Pradesh
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Path to the model-ready dataset (relative to project root)
DATA_PATH = "data/processed/model_ready_prices.csv"

# Features the model will use as inputs
FEATURE_COLUMNS = [
    'Month',           # Captures seasonal patterns (Onion peaks in Nov)
    'Year',            # Captures long-term price trend
    'Quarter',         # Coarser seasonal signal
    'lag_1',           # Last month's price (strongest predictor)
    'lag_2',           # Price 2 months ago
    'lag_3',           # Price 3 months ago
    'rolling_mean_3',  # Average of last 3 months (smoothed trend)
    'rolling_std_3',   # Volatility of last 3 months (market stability)
]

# What the model is trying to predict
TARGET_COLUMN = 'Avg_Modal_Price'

# Crops with enough data for reliable modeling
# Rice (0 usable rows) and Tomato/Wheat (very few rows) are documented
RELIABLE_CROPS = ['Onion', 'Potato']

# All crops in the dataset (for reference and documentation)
ALL_CROPS = ['Onion', 'Potato', 'Wheat', 'Tomato']
# Note: Rice excluded — 0 usable rows after feature engineering


# ─────────────────────────────────────────────
# FUNCTION: load_data
# ─────────────────────────────────────────────
def load_data(path=DATA_PATH):
    """
    Load the model-ready feature-engineered dataset.
    Converts Date column to datetime and sorts chronologically.

    Parameters:
        path (str): Path to model_ready_prices.csv

    Returns:
        pd.DataFrame: Loaded and sorted DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Make sure you have run Day 6's feature engineering notebook "
            f"and that model_ready_prices.csv exists in data/processed/."
        )

    df = pd.read_csv(path)

    # Convert Date column back to datetime
    # CSV saves datetime as text — must re-convert on every load
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by Commodity first, then Date
    # This ensures lag features are in correct chronological order per crop
    df = df.sort_values(['Commodity', 'Date']).reset_index(drop=True)

    print(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m')} "
          f"to {df['Date'].max().strftime('%Y-%m')}")
    print(f"Crops available: {sorted(df['Commodity'].unique().tolist())}")
    print(f"Rows per crop:\n{df.groupby('Commodity').size().to_string()}\n")

    return df


# ─────────────────────────────────────────────
# FUNCTION: split_crop_data
# ─────────────────────────────────────────────
def split_crop_data(df, commodity, train_ratio=0.8):
    """
    Perform a chronological train/test split for a single crop.

    WHY chronological (not random):
    - Lag features create temporal dependencies between rows
    - Random splitting would leak future prices into training features
    - Chronological split ensures the model only learns from the past

    Parameters:
        df (pd.DataFrame): Full model-ready DataFrame
        commodity (str): Crop name (e.g., 'Onion', 'Potato')
        train_ratio (float): Proportion of data for training (default 0.8)

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame/Series)
    """
    # Step 1: Filter to this crop only
    crop_df = df[df['Commodity'] == commodity].copy()

    # Step 2: Sort chronologically — CRITICAL
    # Must be sorted before splitting so "first 80%" = earliest 80%
    crop_df = crop_df.sort_values('Date').reset_index(drop=True)

    # Step 3: Verify no NaN values remain in features or target
    missing_features = crop_df[FEATURE_COLUMNS].isnull().sum().sum()
    missing_target = crop_df[TARGET_COLUMN].isnull().sum()
    if missing_features > 0 or missing_target > 0:
        raise ValueError(
            f"NaN values found for {commodity}: "
            f"{missing_features} in features, {missing_target} in target. "
            f"Check that model_ready_prices.csv was generated correctly."
        )

    # Step 4: Calculate split index
    n_total = len(crop_df)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    # Step 5: Split — positionally, NOT randomly
    train_df = crop_df.iloc[:n_train]   # First n_train rows (earliest dates)
    test_df = crop_df.iloc[n_train:]    # Remaining rows (latest dates)

    # Step 6: Separate features (X) from target (y)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # Step 7: Print split summary
    print(f"\n{'='*50}")
    print(f"SPLIT SUMMARY: {commodity}")
    print(f"{'='*50}")
    print(f"Total rows:    {n_total}")
    print(f"Training rows: {n_train} "
          f"({train_df['Date'].min().strftime('%Y-%m')} → "
          f"{train_df['Date'].max().strftime('%Y-%m')})")
    print(f"Test rows:     {n_test} "
          f"({test_df['Date'].min().strftime('%Y-%m')} → "
          f"{test_df['Date'].max().strftime('%Y-%m')})")
    print(f"Features used: {FEATURE_COLUMNS}")
    print(f"Target:        {TARGET_COLUMN}")

    # Step 8: Verify temporal integrity — test must be after train
    assert train_df['Date'].max() < test_df['Date'].min(), (
        f"CRITICAL ERROR: Training data contains dates AFTER test data "
        f"for {commodity}. This indicates a sorting error."
    )
    print(f"Temporal integrity: ✓ (train ends before test begins)")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# FUNCTION: prepare_all_crops
# ─────────────────────────────────────────────
def prepare_all_crops(df, crops=None, train_ratio=0.8):
    """
    Run chronological split for multiple crops and return
    a dictionary of train/test splits.

    Parameters:
        df (pd.DataFrame): Full model-ready DataFrame
        crops (list): List of crop names. Defaults to RELIABLE_CROPS.
        train_ratio (float): Proportion for training (default 0.8)

    Returns:
        dict: {crop_name: {'X_train', 'X_test', 'y_train', 'y_test'}}
    """
    if crops is None:
        crops = RELIABLE_CROPS

    splits = {}

    for crop in crops:
        if crop not in df['Commodity'].unique():
            print(f"Warning: {crop} not found in dataset. Skipping.")
            continue

        X_train, X_test, y_train, y_test = split_crop_data(
            df, crop, train_ratio
        )

        splits[crop] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    print(f"\n{'='*50}")
    print(f"ALL SPLITS COMPLETE")
    print(f"{'='*50}")
    print(f"Crops split: {list(splits.keys())}")
    print(f"Ready for model training (Day 8).")

    return splits


# ─────────────────────────────────────────────
# FUNCTION: get_feature_info
# ─────────────────────────────────────────────
def get_feature_info():
    """
    Print a human-readable explanation of every feature.
    Useful for documentation and presentations.
    """
    feature_explanations = {
        'Month': (
            'Calendar month (1-12). Captures seasonal price patterns. '
            'Onion prices consistently peak at Month=10/11.'
        ),
        'Year': (
            'Calendar year (2023/2024/2025). Captures long-term price trend '
            'driven by inflation and government MSP increases.'
        ),
        'Quarter': (
            'Calendar quarter (1-4). Coarser seasonal signal. '
            'Q4 (Oct-Dec) is typically high-price period for Onion/Potato.'
        ),
        'lag_1': (
            'Previous month average Modal Price. Strongest single predictor '
            'due to price inertia — this month prices correlate with last '
            'month prices.'
        ),
        'lag_2': (
            'Price 2 months ago. Captures medium-term momentum. '
            'Rising lag_3 → lag_2 → lag_1 suggests continuing upward trend.'
        ),
        'lag_3': (
            'Price 3 months ago. Captures quarterly price cycle position. '
            'Helps model distinguish seasonal peaks from noise.'
        ),
        'rolling_mean_3': (
            '3-month rolling average of prices BEFORE current month. '
            'Smooths short-term noise. Gap between rolling_mean and lag_1 '
            'is a momentum signal (is current price above/below trend?).'
        ),
        'rolling_std_3': (
            '3-month rolling standard deviation BEFORE current month. '
            'Measures recent market volatility. High rolling_std signals '
            'turbulent market conditions.'
        ),
    }

    print("FEATURE EXPLANATIONS")
    print("=" * 60)
    for feature, explanation in feature_explanations.items():
        print(f"\n{feature}:")
        print(f"  {explanation}")
    print("=" * 60)


# ─────────────────────────────────────────────
# MAIN — runs when file is executed directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CROP PRICE PREDICTOR — DATA PREPARATION")
    print("Day 7: Chronological Train/Test Split")
    print("=" * 60)

    # Load data
    df = load_data()

    # Show feature explanations
    get_feature_info()

    # Split for reliable crops
    splits = prepare_all_crops(df, crops=RELIABLE_CROPS)

    # Also split Wheat and Tomato with documented caveats
    print("\n" + "="*50)
    print("SUPPLEMENTARY CROPS (limited data — treat with caution)")
    print("="*50)
    for crop in ['Wheat', 'Tomato']:
        if crop in df['Commodity'].unique():
            rows = len(df[df['Commodity'] == crop])
            print(f"\n{crop}: {rows} usable rows — "
                  f"{'marginal' if rows >= 5 else 'insufficient'} for modeling")
            if rows >= 3:
                X_train, X_test, y_train, y_test = split_crop_data(
                    df, crop
                )
                splits[crop] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }

    # Final summary
    print("\n" + "="*60)
    print("PREPARATION COMPLETE")
    print("="*60)
    for crop, split in splits.items():
        train_size = len(split['X_train'])
        test_size = len(split['X_test'])
        print(f"{crop:10s}: {train_size} train rows, {test_size} test rows")

    print("\nNext step: Day 8 — Train price prediction models")
    print("Import with: from src.price_predictor import load_data, "
          "prepare_all_crops")