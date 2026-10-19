
import os
import joblib
import pandas as pd
import streamlit as st

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "machine_failure_model.pkl"
)

# Page settings
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Machine Failure Prediction")
st.write(
    "Enter the sensor readings below to predict whether "
    "a machine may be at risk of failure."
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

# Input form
with st.form("sensor_form"):
    st.subheader("Machine Sensor Readings")

    temperature = st.number_input(
        "Temperature", min_value=0.0, max_value=200.0,
        value=70.0
    )
    pressure = st.number_input(
        "Pressure", min_value=0.0, max_value=100.0,
        value=32.0
    )
    vibration = st.number_input(
        "Vibration", min_value=0.0, max_value=5.0,
        value=0.3
    )
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0,
        value=50.0
    )
    power = st.number_input(
        "Power Consumption", min_value=0.0,
        max_value=500.0, value=130.0
    )

    submitted = st.form_submit_button("Predict Machine Status")

if submitted:
    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Pressure": pressure,
        "Vibration": vibration,
        "Humidity": humidity,
        "Power_Consumption": power
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ The model predicts a possible machine failure.")
    else:
        st.success("✅ The model predicts no machine failure.")

    st.caption(
        "Demonstration only: this model uses a very small synthetic "
        "dataset. Do not use its predictions for real equipment "
        "safety or maintenance decisions."
    )
