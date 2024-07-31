import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.geocoders import Nominatim
import requests

# Load datasets
agricultural_dataset = pd.read_csv('data/agricultural_dataset.csv')
insect_dataset = pd.read_csv('data/period.csv')

# Check if dataset columns align with expected columns
expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
if not all(column in agricultural_dataset.columns for column in expected_columns):
    st.error(f"agricultural_dataset.csv should contain columns: {expected_columns}")
    st.stop()

# Prepare agricultural dataset
# Fill missing values and encode the soil type
soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky', 'Unknown']
soil_type_encoder = LabelEncoder()
soil_type_encoder.fit(soil_types)

# Ensure 'soil_type' exists in the dataset
if 'soil_type' not in agricultural_dataset.columns:
    agricultural_dataset['soil_type'] = 'Unknown'  # Assuming default if missing

agricultural_dataset['soil_type'] = agricultural_dataset['soil_type'].fillna('Unknown')
agricultural_dataset['soil_type_encoded'] = soil_type_encoder.transform(agricultural_dataset['soil_type'])

# Prepare features and target
X_agriculture = agricultural_dataset.drop(columns=['label', 'soil_type'])
y_agriculture = agricultural_dataset['label']

# Scale features
scaler_agriculture = StandardScaler()
X_agriculture_scaled = scaler_agriculture.fit_transform(X_agriculture)

# Train Random Forest Classifier
agricultural_model = RandomForestClassifier(n_estimators=100, random_state=42)
agricultural_model.fit(X_agriculture_scaled, y_agriculture)

# Preprocess insect dataset
severity_encoder = LabelEncoder()
insect_dataset['Severity_Encoded'] = severity_encoder.fit_transform(insect_dataset['Severity'])

crop_severity = insect_dataset.groupby('Crop Affected')['Severity_Encoded'].mean().sort_values()

def categorize_crop(severity):
    if severity < 0.4:
        return "Low Risk"
    elif severity < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

crop_risk = crop_severity.apply(categorize_crop)

def get_location():
    try:
        ip_address = requests.get('https://api64.ipify.org?format=json').json()['ip']
        response = requests.get(f'https://ipinfo.io/{ip_address}/json').json()
        loc = response['loc'].split(',')
        latitude, longitude = loc[0], loc[1]
        return latitude, longitude
    except Exception as e:
        return None, None

def get_location_details(latitude, longitude):
    try:
        geolocator = Nominatim(user_agent="crop_recommendation")
        location = geolocator.reverse(f"{latitude}, {longitude}")
        return location.address
    except Exception as e:
        return None

def recommend_crops(N, P, K, temperature, humidity, ph, rainfall, soil_type):
    latitude, longitude = get_location()
    location = get_location_details(latitude, longitude) if latitude and longitude else None
    
    soil_type_encoded = soil_type_encoder.transform([soil_type])[0]
    input_features_agriculture = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_type_encoded]])
    input_features_agriculture_scaled = scaler_agriculture.transform(input_features_agriculture)
    
    probabilities = agricultural_model.predict_proba(input_features_agriculture_scaled)[0]
    crop_probabilities = list(zip(agricultural_model.classes_, probabilities))
    crop_probabilities.sort(key=lambda x: x[1], reverse=True)
    
    top_two_crops = crop_probabilities[:2]
    
    recommendations = {}
    for crop, _ in top_two_crops:
        if crop in crop_risk.index:
            pests_info = insect_dataset[insect_dataset['Crop Affected'].str.contains(crop, na=False)]
            recommendations[crop] = pests_info
        else:
            recommendations[crop] = pd.DataFrame()
    
    return recommendations, crop_risk

# Streamlit app
st.title("Crop Recommendation System")

# Input fields for user input
N = st.number_input("Nitrogen Content (N)", min_value=0.0, max_value=300.0, step=1.0)
P = st.number_input("Phosphorus Content (P)", min_value=0.0, max_value=300.0, step=1.0)
K = st.number_input("Potassium Content (K)", min_value=0.0, max_value=300.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=1.0)
soil_type = st.selectbox("Soil Type", options=soil_types)

if st.button("Get Crop Recommendations"):
    # Get recommendations and crop risks
    recommendations, crop_risk = recommend_crops(N, P, K, temperature, humidity, ph, rainfall, soil_type)
    
    # Display recommendations
    st.subheader("Crop Recommendations")
    for crop, pests_info in recommendations.items():
        st.write(f"Crop: {crop}")
        if not pests_info.empty:
            st.write(pests_info)
        else:
            st.write("No pest information available.")

    # Display crop risk information
    st.subheader("Crop Risk Levels")
    st.table(crop_risk)

    # Calculate most affected crops and region severity
    most_affected_crops = crop_risk.tail(5)
    region_severity = insect_dataset.groupby('Region')['Severity_Encoded'].mean().sort_values(ascending=False)
    insect_counts = insect_dataset['Insect Name'].value_counts()

    st.subheader("Most Affected Crops")
    if not most_affected_crops.empty:
        st.table(most_affected_crops)
    else:
        st.write("No data available.")

    st.subheader("Region Severity")
    if not region_severity.empty:
        st.table(region_severity.head())
    else:
        st.write("No data available.")

    st.subheader("Insect Counts")
    if not insect_counts.empty:
        st.table(insect_counts.head())
    else:
        st.write("No data available.")
