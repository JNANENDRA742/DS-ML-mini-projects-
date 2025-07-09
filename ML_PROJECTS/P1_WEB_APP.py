import pandas as pd
import pickle
import streamlit as st

# Load the model, scaler, encoder and feature names
le_encoder = pickle.load(open("house_encoder.pkl", "rb"))
scaler = pickle.load(open("house_scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))
le_model = pickle.load(open("house_model.pkl", "rb"))

st.title("🏠 House Price Prediction App")

col1, col2 = st.columns(2)
with col1:
    year_built = st.number_input("Year Built", min_value=1900, max_value=2050, value=2000)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=6, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=6, value=2)
with col2:
    garage_area = st.selectbox("GarageArea", ["Small", "Medium", "Large"])
    swimming_pool = st.selectbox("Swimming Pool", ["Yes", "No"])
    square_footage = st.number_input("Square Footage", min_value=0, max_value=10000, value=1500)

if st.button("Predict Price"):
    input_dict = {
        'Bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'GarageArea': garage_area,
        'swimming_pool': swimming_pool,
        'square_footage': square_footage
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical columns
    for col in ['GarageArea', 'swimming_pool']:
        input_df[col] = le_encoder[col].transform(input_df[col])

    # Ensure the same column order
    input_df = input_df[feature_names]

    # Scale
    input_df_scaled = scaler.transform(input_df)

    # Predict
    prediction = le_model.predict(input_df_scaled)
    st.success(f"🏷️ Predicted House Price: ₹ {prediction[0]:,.2f}")

    # Show the predicted price
    # st.write(f"🏷️ Predicted House Price: ₹ {prediction[0]:,.2f}")