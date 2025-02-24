import numpy as np
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.geocoders import Nominatim
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import streamlit as st

# Database configuration
db_config = {
    'host': st.secrets["db_host"],
    'user': st.secrets["db_username"],
    'password': st.secrets["db_password"],
    'database': st.secrets["db_name"]
}

# Load datasets
agricultural_dataset = pd.read_csv('data/agricultural_dataset.csv')
insect_dataset = pd.read_csv('data/period.csv')

# Prepare agricultural dataset
expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
if not all(column in agricultural_dataset.columns for column in expected_columns):
    raise ValueError(f"agricultural_dataset.csv should contain columns: {expected_columns}")

# Prepare soil types
soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky', 'Unknown']
soil_type_encoder = LabelEncoder()
soil_type_encoder.fit(soil_types)

if 'soil_type' not in agricultural_dataset.columns:
    agricultural_dataset['soil_type'] = 'Unknown'

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
    if severity < 0.4: return "Low Risk"
    elif severity < 0.7: return "Moderate Risk"
    else: return "High Risk"

crop_risk = crop_severity.apply(categorize_crop)

def get_location():
    try:
        ip_address = requests.get('https://api64.ipify.org?format=json').json()['ip']
        response = requests.get(f'https://ipinfo.io/{ip_address}/json').json()
        loc = response['loc'].split(',')
        return loc[0], loc[1]
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
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_type_encoded]])
    input_features_scaled = scaler_agriculture.transform(input_features)
    
    probabilities = agricultural_model.predict_proba(input_features_scaled)[0]
    crop_probabilities = list(zip(agricultural_model.classes_, probabilities))
    crop_probabilities.sort(key=lambda x: x[1], reverse=True)
    
    top_two_crops = crop_probabilities[:2]
    recommendations = {}
    for crop, prob in top_two_crops:
        if crop in crop_risk.index:
            pests_info = insect_dataset[insect_dataset['Crop Affected'].str.contains(crop, na=False)]
            recommendations[crop] = {'probability': prob, 'pests': pests_info}
        else:
            recommendations[crop] = {'probability': prob, 'pests': pd.DataFrame()}
    
    return recommendations, crop_risk, location

def show_prediction_page():
    st.title("AI Crop Recommendation System")
    
    # Get unique regions from the dataset
    regions = sorted(insect_dataset['Region'].unique())
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            N = st.slider("Nitrogen (kg/ha)", 
                min_value=0, max_value=150, value=20)
            P = st.slider("Phosphorus (kg/ha)", 
                min_value=0, max_value=150, value=20)
            K = st.slider("Potassium (kg/ha)", 
                min_value=0, max_value=200, value=40)
            temperature = st.slider("Temperature (°C)", 
                min_value=0, max_value=50, value=25)
            region = st.selectbox("Select Region", options=regions)

        with col2:
            humidity = st.slider("Humidity (%)", 
                min_value=0, max_value=100, value=60)
            ph = st.slider("pH", 
                min_value=0.0, max_value=14.0, value=6.0, step=0.1)
            rainfall = st.slider("Rainfall (mm)", 
                min_value=10, max_value=300, value=100)
            soil_type = st.selectbox("Soil Type", 
                options=['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky'])
        
        submitted = st.form_submit_button("Recommend Crops")
        
        if submitted:
            try:
                recommendations, crop_risk, location = recommend_crops(
                    N, P, K, temperature, humidity, ph, rainfall, soil_type
                )
                
                st.success("Prediction Complete!")
                
                # Display selected region information
                st.subheader(f"Selected Region: {region}")
                region_data = insect_dataset[insect_dataset['Region'] == region]
                if not region_data.empty:
                    avg_severity = region_data['Severity_Encoded'].mean()
                    st.write(f"Region Average Severity: {avg_severity:.2f}")
                    
                    # Common pests in the region
                    st.write("Common Pests in this Region:")
                    region_pests = region_data['Insect Name'].value_counts().head(3)
                    for pest, count in region_pests.items():
                        st.write(f"- {pest} ({count} occurrences)")
                
                # Simple display of crops and their fit percentages
                st.subheader("Recommended Crops:")
                for crop, data in recommendations.items():
                    fit_percentage = data['probability'] * 100
                    st.markdown(f"**{crop}**: {fit_percentage:.1f}%")

                # Additional Information Section
                st.header("Additional Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Most Affected Crops Table
                    st.subheader("Most Affected Crops")
                    most_affected = pd.DataFrame({
                        'Crop': crop_risk.index,
                        'Risk Category': crop_risk.values
                    })
                    st.table(most_affected.head())
                    
                    # Region Severity Table
                    st.subheader("Region Severity")
                    region_severity = insect_dataset.groupby('Region')['Severity_Encoded'].mean()
                    region_df = pd.DataFrame({
                        'Region': region_severity.index,
                        'Average Severity': region_severity.values.round(2)
                    })
                    st.table(region_df)
                
                with col2:
                    # Insect Counts Table
                    st.subheader("Insect Counts")
                    insect_counts = insect_dataset['Insect Name'].value_counts()
                    insect_df = pd.DataFrame({
                        'Insect Name': insect_counts.index,
                        'Count': insect_counts.values
                    })
                    st.table(insect_df.head())
                    
                    # Region-specific pest information
                    st.subheader(f"Pests in {region}")
                    region_pests_full = region_data[['Insect Name', 'Severity', 'Control Measure']].drop_duplicates()
                    if not region_pests_full.empty:
                        st.table(region_pests_full)

            except Exception as e:
                st.error("Unable to make prediction. Please try again.")

def init_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as e:
        st.error(f"Database connection failed. Please check your database settings.")
        return None

def create_user(username, password):
    conn = init_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (%s, %s)', 
                      (username, password_hash))
        conn.commit()
        return True
    except Exception as e:
        st.error("Registration failed. Username might already exist.")
        return False
    finally:
        cursor.close()
        conn.close()

def login_user(username, password):
    conn = init_connection()
    if not conn:
        return False
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            return True
        return False
    except Exception as e:
        st.error("Login failed. Please try again.")
        return False
    finally:
        cursor.close()
        conn.close()

def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None

    # Show appropriate page based on login status
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.header("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if login_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            st.header("Register")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            
            if st.button("Register"):
                if create_user(reg_username, reg_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Registration failed")
    else:
        # Add logout button in sidebar
        with st.sidebar:
            st.write(f"Welcome, {st.session_state.username}!")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.rerun()
        
        # Show the prediction page
        show_prediction_page()

if __name__ == "__main__":
    main()

