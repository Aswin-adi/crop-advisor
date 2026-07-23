import os
import joblib
import pandas as pd

# Folder where all trained models are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Feature order must match the order used during model training
FEATURE_COLUMNS = [
    "Month",
    "Year",
    "Quarter",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3"
]


def load_model(crop_name: str,
               model_type: str = "random_forest"):
    """
    Load a saved machine learning model.

    Parameters:
        crop_name (str): Crop name (e.g. Onion)
        model_type (str): linear_regression or random_forest

    Returns:
        Trained sklearn model
    """

    filename = f"{crop_name}_{model_type}.pkl"
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    return joblib.load(model_path)


def predict_price(crop_name: str,
                  features: dict,
                  model_type: str = "random_forest") -> float:
    """
    Predict the crop price.

    Parameters:
        crop_name (str): Crop name
        features (dict): Dictionary containing feature values
        model_type (str): Model to use

    Returns:
        float: Predicted crop price
    """
    # Validate that all required features are present
    missing = [col for col in FEATURE_COLUMNS if col not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")


    # Load the trained model
    model = load_model(crop_name, model_type)

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([features])

    # Ensure feature order matches training
    input_df = input_df[FEATURE_COLUMNS]

    # Predict
    prediction = model.predict(input_df)[0]

    return round(float(prediction), 2)
def get_latest_features(crop_name: str, df: pd.DataFrame) -> dict:
    """Auto-build feature dict from a crop's most recent row."""
    crop_df = df[df['Commodity'].str.lower() == crop_name.lower()].sort_values('Year')
    latest = crop_df.iloc[-1]
    return {
        "Month": int(latest["Month"]),
        "Year": int(latest["Year"]),
        "Quarter": int(latest["Quarter"]),
        "lag_1": latest["lag_1"],
        "lag_2": latest["lag_2"],
        "lag_3": latest["lag_3"],
        "rolling_mean_3": latest["rolling_mean_3"],
        "rolling_std_3": latest["rolling_std_3"],
    }


def recommend_crop(df: pd.DataFrame, crops: list = None, min_rows: int = 15) -> pd.DataFrame:
    """Predict prices for all crops, rank by expected % growth, flag confidence."""
    if crops is None:
        crops = df['Commodity'].unique()
    rows = []
    for crop in crops:
        crop_row_count = len(df[df['Commodity'].str.lower() == crop.lower()])
        try:
            features = get_latest_features(crop, df)
            predicted = predict_price(crop.lower(), features)
            recent_avg = features["rolling_mean_3"]
            if recent_avg == 0:
                pct_change = None
            else:
                pct_change = round(((predicted - recent_avg) / recent_avg) * 100, 2)
            confidence = "High" if crop_row_count >= min_rows else "Low (limited data)"
            rows.append({"Crop": crop, "Recent_Avg_Price": round(recent_avg, 2),
                        "Predicted_Price": predicted, "Expected_Change_%": pct_change,
                        "Confidence": confidence})
        except FileNotFoundError:
            rows.append({"Crop": crop, "Recent_Avg_Price": None, "Predicted_Price": None,
                        "Expected_Change_%": None, "Confidence": "N/A"})
    return pd.DataFrame(rows).sort_values("Expected_Change_%", ascending=False).reset_index(drop=True)


def get_price_history(crop_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Return chronological price history for a crop.
    Useful for plotting price trends in the dashboard.
    """
    crop_df = (
        df[df["Commodity"].str.lower() == crop_name.lower()]
        .sort_values(["Year", "Month"])
    )
    return crop_df[["Year", "Month", "Avg_Modal_Price"]].reset_index(drop=True)


def compare_models(crop_name: str, features: dict) -> dict:
    """Return predictions from both models side-by-side for a crop."""
    result = {}
    for model_type in ["linear_regression", "random_forest"]:
        try:
            result[model_type] = predict_price(crop_name, features, model_type)
        except FileNotFoundError:
            result[model_type] = None
    return result