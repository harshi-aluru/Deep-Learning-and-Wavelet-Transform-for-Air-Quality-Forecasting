import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# ====== IMPORTS FROM EXISTING REPO ======
from datasets import AirQualityDataset
from BaselineArchitectures import Baseline1D
# from StackedResnet import StackedResnet   # optional

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Air Quality Forecasting",
    layout="wide"
)

st.title("Urban Air Quality Forecasting")
st.caption("Wavelet Transform + Deep Learning")

# ====== LOAD DATA ======
st.sidebar.header("Dataset")

DATA_PATH = "AirQuality.csv"   # already present in repo

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ====== MODEL OPTIONS ======
st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Model",
    ["Baseline1D"]  # keep simple for now
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (hours)",
    min_value=1,
    max_value=24,
    value=1
)

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    model = Baseline1D()
    model.eval()
    return model

model = load_model()

# ====== RUN FORECAST ======
if st.button("Run Forecast"):

    with st.spinner("Preparing dataset..."):
        dataset = AirQualityDataset(
            df=df,
            horizon=forecast_horizon
        )

        X_train, y_train, X_test, y_test = dataset.get_train_test_split()

    with st.spinner("Running inference..."):
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test).float()
            y_pred = model(X_test_tensor).numpy()

    # ====== VISUALIZATION ======
    st.subheader("Prediction vs Actual")

    results = pd.DataFrame({
        "Actual": y_test.flatten(),
        "Predicted": y_pred.flatten()
    })

    st.line_chart(results.head(200))

    # ====== METRICS ======
    mse = np.mean((results["Actual"] - results["Predicted"]) ** 2)
    mae = np.mean(np.abs(results["Actual"] - results["Predicted"]))

    col1, col2 = st.columns(2)
    col1.metric("MSE", round(mse, 4))
    col2.metric("MAE", round(mae, 4))

    st.success("Forecast completed successfully")
