import streamlit as st
import pandas as pd
import sys
sys.path.append("src")
from predictor import predict_price, get_latest_features, recommend_crop, get_price_history, compare_models

st.set_page_config(page_title="Crop Advisor", page_icon="🌾", layout="centered")
with st.sidebar:
    st.header("About")
    st.write(
        "Crop Advisor predicts mandi (market) prices for major crops in "
        "Uttar Pradesh using historical price trends, and suggests which "
        "crops may be worth growing based on predicted price movement."
    )
    st.subheader("Data")
    st.write("Indian Agricultural Mandi Prices (2023–2025), Uttar Pradesh subset.")
    st.subheader("Limitations")
    st.write(
        "- Wheat and Tomato have limited historical data, so their predictions "
        "are marked low-confidence.\n"
        "- Recommendations reflect price trend only, not cultivation cost or yield."
    )

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/model_ready_prices.csv")

df = load_data()
crop_list = sorted(df['Commodity'].unique())

st.title("🌾 Crop Advisor")
st.caption("Agricultural Market Intelligence — Uttar Pradesh Mandi Prices (2023–2025)")

tab1, tab2, tab3 = st.tabs(["Price Prediction", "Model Comparison", "Crop Recommendation"])

# ---------------- TAB 1: Price Prediction ----------------
with tab1:
    st.subheader("Predict Next Price")
    selected_crop = st.selectbox("Select a crop", crop_list, key="predict_crop")

    if st.button("Predict Price", key="predict_btn"):
        try:
            features = get_latest_features(selected_crop, df)
            price = predict_price(selected_crop.lower(), features)
            recent_avg = round(features['rolling_mean_3'], 2)

            st.success(f"Predicted price for {selected_crop}: ₹{price}")
            st.caption(f"Recent 3-month average: ₹{recent_avg}")

            # Price history chart
            history = get_price_history(selected_crop, df)
            history['Period'] = history['Year'].astype(str) + "-" + history['Month'].astype(str).str.zfill(2)
            st.line_chart(history.set_index('Period')['Avg_Modal_Price'])
            st.caption("Historical monthly average price for this crop")

        except FileNotFoundError:
            st.error(f"No trained model found for {selected_crop}. It may have insufficient data.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------------- TAB 2: Model Comparison ----------------
with tab2:
    st.subheader("Linear Regression vs Random Forest")
    compare_crop = st.selectbox("Select a crop", crop_list, key="compare_crop")
    LOW_DATA_CROPS = ["wheat", "tomato"]  # crops with insufficient rows for reliable evaluation
    if compare_crop.lower() in LOW_DATA_CROPS:
        st.warning(f"{compare_crop} has limited historical data — treat this comparison as a rough estimate only.")

    if st.button("Compare Models", key="compare_btn"):
        try:
            features = get_latest_features(compare_crop, df)
            comparison = compare_models(compare_crop.lower(), features)

            col1, col2 = st.columns(2)
            with col1:
                lr_val = comparison["linear_regression"]
                st.metric("Linear Regression", f"₹{lr_val}" if lr_val else "N/A")
            with col2:
                rf_val = comparison["random_forest"]
                st.metric("Random Forest", f"₹{rf_val}" if rf_val else "N/A")

            st.caption("Random Forest is generally more reliable on this dataset (see evaluation summary), but both are shown for transparency.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------------- TAB 3: Crop Recommendation ----------------
with tab3:
    st.subheader("Which Crop Looks Favorable?")
    if st.button("Get Recommendation", key="recommend_btn"):
        rec_df = recommend_crop(df)
        st.dataframe(rec_df, use_container_width=True)
        # Bar chart comparing predicted price vs recent average
        chart_data = rec_df.dropna(subset=["Predicted_Price"]).set_index("Crop")
        st.bar_chart(chart_data[["Recent_Avg_Price", "Predicted_Price"]])
        st.caption(
            "Positive % = predicted price rising above recent average. "
            "'Low confidence' crops have limited historical data and should be treated as rough signals only."
        )