import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="California House Price Predictor", layout="centered")

st.title("🏠 California House Price Prediction")
st.write("Predict the median house value in California based on housing features.")

# Load dataset (cached for performance)
@st.cache_data
def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['price'] = data.target
    return df

df = load_data()

# Train model (cached to avoid retraining on every interaction)
@st.cache_resource
def train_model(df):
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(df)

# Sidebar inputs for features
st.sidebar.header("Input Features")

MedInc = st.sidebar.slider("Median Income (10k USD)", 0.0, 20.0, 3.0, 0.1)
HouseAge = st.sidebar.slider("House Age (years)", 1, 52, 20)
AveRooms = st.sidebar.slider("Average Rooms", 0.5, 15.0, 5.0, 0.1)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)
Population = st.sidebar.slider("Population", 1, 40000, 1000)
AveOccup = st.sidebar.slider("Average Occupants", 0.5, 30.0, 3.0, 0.1)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0, 0.01)
Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -118.0, 0.01)

# Create input dataframe for prediction
input_df = pd.DataFrame({
    "MedInc": [MedInc],
    "HouseAge": [HouseAge],
    "AveRooms": [AveRooms],
    "AveBedrms": [AveBedrms],
    "Population": [Population],
    "AveOccup": [AveOccup],
    "Latitude": [Latitude],
    "Longitude": [Longitude],
})

# Predict button and display
if st.button("Predict House Price"):
    pred = model.predict(input_df)[0]
    st.success(f"🏡 Predicted Median House Price: **${pred * 100_000:,.2f}**")

# Optional: Show sample of raw data
with st.expander("🔍 Sample raw data"):
    st.dataframe(df.sample(10))

# Optional: Show model performance
with st.expander("📊 Model Performance (on test data)"):
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.3f}")
