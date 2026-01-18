import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
# Using st.cache_resource to load them only once
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

def main():
    # Set page config
    st.set_page_config(
        page_title="Forest Fire Risk Predictor",
        page_icon="🔥",
        layout="centered"
    )

    # Title and description
    st.title("🌲 Forest Fire Risk Classification")
    st.markdown("""
    This app predicts the risk of forest fire based on environmental conditions.
    
    **Model:** Logistic Regression (trained on Algerian Forest Fires dataset).
    """)

    # Check if model loaded
    if model is None or scaler is None:
        st.error("Model artifacts not found. Please run `train_model.py` first.")
        return

    # Sidebar for inputs
    st.sidebar.header("Environmental Conditions")
    
    # Input fields
    # Ranges based on dataset typical values (Temperature: 20-45, RH: 20-100, Ws: 6-30, Rain: 0-20)
    temperature = st.sidebar.slider("Temperature (°C)", 20, 45, 30)
    rh = st.sidebar.slider("Relative Humidity (%)", 20, 100, 50)
    ws = st.sidebar.slider("Wind Speed (km/h)", 5, 30, 15)
    rain = st.sidebar.number_input("Rainfall (mm)", 0.0, 20.0, 0.0, step=0.1)

    # Feature descriptions
    with st.expander("ℹ️ Feature Explanations"):
        st.markdown("""
        - **Temperature**: Daily maximum temperature in Celsius.
        - **Relative Humidity (RH)**: Percentage of moisture in the air.
        - **Wind Speed (Ws)**: Speed of wind in km/h.
        - **Rainfall**: Amount of rain in mm.
        """)

    # Prediction button
    if st.button("Predict Risk"):
        # Prepare input data
        # Feature order must match training: Temperature, RH, Ws, Rain
        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'RH': [rh],
            'Ws': [ws],
            'Rain': [rain]
        })

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Display result
        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🔥 **High Fire Risk Detected!**")
            st.write("The model classifies this as a **FIRE** scenario.")
        else:
            st.success("✅ **Low Fire Risk**")
            st.write("The model classifies this as a **NOT FIRE** scenario.")

        st.info(f"Probability of Fire: **{probability:.2%}**")

        # Explainability note
        st.caption("Note: This is an educational prototype using Logistic Regression. Do not rely on it for real-world safety.")

if __name__ == "__main__":
    main()
