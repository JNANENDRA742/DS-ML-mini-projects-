import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Load Data
df = pd.read_csv("house_price_prediction_500rows_categorical.csv")

# Fill missing values with most repeated value
df["GarageArea"] = df["GarageArea"].fillna(df["GarageArea"].mode()[0])

# Encode categorical columns
encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Scale numerical columns
numerical_columns = ['Bedrooms', 'bathrooms', 'GarageArea', 'swimming_pool', 'square_footage']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Train-test split
X = df.drop(["price", "year_built"], axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
le_model = LinearRegression()
le_model.fit(X_train, y_train)

# Show R² score
y_pred = le_model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))









# Building the Streamlit app
import pickle
import streamlit as st


st.title("🏠 House Price Prediction App")

col1, col2 = st.columns(2)
with col1:
    year_built = st.number_input("Year Built", min_value=1900, max_value=2050, value=2000)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=5, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
with col2:
    garage_area = st.selectbox("GarageArea", ["Small", "Medium", "Large"])
    swimming_pool = st.selectbox("Swimming Pool", ["Yes", "No"])
    square_footage = st.number_input("Square Footage", min_value=1000, max_value=10000, value=1500)

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
        input_df[col] = encoders[col].transform(input_df[col])

    # Ensure the same column order
    input_df = input_df[numerical_columns]

    # Scale
    input_df_scaled = scaler.transform(input_df)

    # Predict
    prediction = le_model.predict(input_df_scaled)
    st.success(f"🏷️ Predicted House Price: ₹ {prediction[0]:,.2f}")
