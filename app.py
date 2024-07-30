import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.geocoders import Nominatim
import requests

app = Flask(__name__)

# Load datasets
agricultural_dataset = pd.read_csv('data/agricultural_dataset.csv')
insect_dataset = pd.read_csv('data/period.csv')

# Prepare agricultural dataset
agricultural_dataset['soil_type'] = np.nan
soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky', 'Unknown']
soil_type_encoder = LabelEncoder()
soil_type_encoder.fit(soil_types)
agricultural_dataset['soil_type'] = agricultural_dataset['soil_type'].fillna('Unknown')
agricultural_dataset['soil_type_encoded'] = soil_type_encoder.transform(agricultural_dataset['soil_type'])

X_agriculture = agricultural_dataset.drop(columns=['label', 'soil_type'])
y_agriculture = agricultural_dataset['label']

scaler_agriculture = StandardScaler()
X_agriculture_scaled = scaler_agriculture.fit_transform(X_agriculture)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        soil_type = request.form['soil_type']
        
        recommendations, crop_risk = recommend_crops(N, P, K, temperature, humidity, ph, rainfall, soil_type)
        
        most_affected_crops = crop_risk.tail(5)
        region_severity = insect_dataset.groupby('Region')['Severity_Encoded'].mean().sort_values(ascending=False)
        insect_counts = insect_dataset['Insect Name'].value_counts()
        
        return render_template(
            'index.html',
            recommendations=recommendations,
            most_affected_crops=most_affected_crops,
            region_severity=region_severity.head(),
            insect_counts=insect_counts.head()
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
