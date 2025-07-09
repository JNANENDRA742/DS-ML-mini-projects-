import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Load Data
df = pd.read_csv("house_price_prediction_500rows_categorical.csv")

# Fill missing values
df["GarageArea"] = df["GarageArea"].fillna(df["GarageArea"].mode()[0])

# Encode categorical columns
encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
pickle.dump(encoders, open("house_encoder.pkl", "wb"))

# Scale numerical columns
numerical_columns = ['Bedrooms', 'bathrooms', 'GarageArea', 'swimming_pool', 'square_footage']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save scaler and its feature names
pickle.dump(scaler, open("house_scaler.pkl", "wb"))
pickle.dump(numerical_columns, open("feature_names.pkl", "wb"))

# Train-test split
X = df.drop(["price", "year_built"], axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("house_model.pkl", "wb"))

# Show R² score
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
