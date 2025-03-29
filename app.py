import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import geopy.geocoders
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import streamlit as st
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64
from mpesa_integration import show_payment_form

# Configure page settings
st.set_page_config(
    page_title="AgriSmart - Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2e7d32; text-align: center; margin-bottom: 2rem; }
    .dashboard-card { background-color: #f5f5f5; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .metric-container { background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1b5e20; }
    .metric-label { font-size: 0.9rem; color: #555; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f8f0; border-radius: 4px 4px 0px 0px; padding: 10px 16px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #2e7d32 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Database Configuration using Streamlit secrets
db_config = {
    'host': st.secrets["db_host"],
    'user': st.secrets["db_username"],
    'password': st.secrets["db_password"],
    'database': st.secrets["db_name"]
}

# Add this function to verify database configuration
def verify_db_config():
    required_keys = ["db_host", "db_username", "db_password", "db_name"]
    missing = []
    for key in required_keys:
        if key not in st.secrets:
            missing.append(key)
    
    if missing:
        st.error(f"Missing database configuration in secrets.toml for: {', '.join(missing)}")
        return False
    return True

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.now()
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = {
        'total_predictions': 0,
        'recent_crops': [],
        'favorite_crops': {}
    }
if 'subscription_status' not in st.session_state:
    st.session_state.subscription_status = 'free'  # Can be 'free' or 'premium'
if 'predictions_today' not in st.session_state:
    st.session_state.predictions_today = 0
if 'last_prediction_date' not in st.session_state:
    st.session_state.last_prediction_date = None

# Session management
SESSION_TIMEOUT = 3600
def update_session_activity():
    st.session_state.last_activity = datetime.now()

def is_session_valid():
    if not st.session_state.logged_in:
        return False
    now = datetime.now()
    time_diff = (now - st.session_state.last_activity).total_seconds()
    if time_diff > SESSION_TIMEOUT:
        return False
    update_session_activity()
    return True

# Load datasets with absolute path
@st.cache_data
def load_datasets():
    base_path = r"C:\Users\cossi\OneDrive\Desktop\limo\crop-model\data"
    try:
        agricultural_dataset = pd.read_csv(os.path.join(base_path, 'agricultural_dataset.csv'))
        period_dataset = pd.read_csv(os.path.join(base_path, 'period.csv'))
        pest_disease_dataset = pd.read_csv(os.path.join(base_path, 'pest_disease_dataset.csv'))
        return agricultural_dataset, period_dataset, pest_disease_dataset
    except FileNotFoundError as e:
        st.error(f"Error loading datasets: {str(e)}. Please ensure all files are in {base_path}")
        return pd.DataFrame(), pd.DataFrame(columns=['Severity', 'Crop Affected', 'Region', 'Insect Name']), pd.DataFrame(columns=['Crop', 'Pest', 'Disease'])
    except Exception as e:
        st.error(f"Unexpected error loading datasets: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(columns=['Severity', 'Crop Affected', 'Region', 'Insect Name']), pd.DataFrame(columns=['Crop', 'Pest', 'Disease'])

# Train and save the model
def train_and_save_model(agricultural_dataset, model_path=r"C:\Users\cossi\OneDrive\Desktop\limo\crop-model\models\agricultural_model.joblib", 
                        scaler_path=r"C:\Users\cossi\OneDrive\Desktop\limo\crop-model\models\scaler.joblib", 
                        encoder_path=r"C:\Users\cossi\OneDrive\Desktop\limo\crop-model\models\soil_type_encoder.joblib"):
    expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    if agricultural_dataset.empty or not all(column in agricultural_dataset.columns for column in expected_columns):
        raise ValueError(f"agricultural_dataset.csv should contain columns: {expected_columns}")

    soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky', 'Unknown']
    soil_type_encoder = LabelEncoder()
    soil_type_encoder.fit(soil_types)

    if 'soil_type' not in agricultural_dataset.columns:
        agricultural_dataset['soil_type'] = 'Unknown'
    agricultural_dataset['soil_type'] = agricultural_dataset['soil_type'].fillna('Unknown')
    agricultural_dataset['soil_type_encoded'] = soil_type_encoder.transform(agricultural_dataset['soil_type'])

    X = agricultural_dataset.drop(columns=['label', 'soil_type'])
    y = agricultural_dataset['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(soil_type_encoder, encoder_path)

    print(f"Model trained with accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model, scaler, soil_type_encoder, accuracy, class_report, conf_matrix

# Load or train model
def load_or_train_model(agricultural_dataset):
    model_path = r"C:\Users\cossi\OneDrive\Desktop\limo\crop-model\models\agricultural_model.joblib"
    scaler_path = r"C:\Users\cossi\OneDrive\Desktop\limo\crop-model\models\scaler.joblib"
    encoder_path = r"C:\Users\cossi\OneDrive\Desktop\limo\crop-model\models\soil_type_encoder.joblib"
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        accuracy, class_report, conf_matrix = None, None, None
    else:
        model, scaler, encoder, accuracy, class_report, conf_matrix = train_and_save_model(agricultural_dataset)
    return model, scaler, encoder, accuracy, class_report, conf_matrix

# Process additional datasets
@st.cache_data
def process_additional_data(period_dataset, pest_disease_dataset):
    try:
        # Period Dataset
        if not all(col in period_dataset.columns for col in ['Severity', 'Crop Affected', 'Region', 'Insect Name']):
            period_dataset = pd.DataFrame(columns=['Severity', 'Crop Affected', 'Region', 'Insect Name'])
        if 'Severity' not in period_dataset.columns:
            period_dataset['Severity'] = 'Low'
        severity_encoder = LabelEncoder()
        severity_values = ['Low', 'Medium', 'High']
        severity_encoder.fit(severity_values)
        period_dataset['Severity_Encoded'] = severity_encoder.transform(period_dataset['Severity'].fillna('Low'))
        crop_severity = period_dataset.groupby('Crop Affected')['Severity_Encoded'].mean().sort_values()
        crop_risk = crop_severity.apply(lambda x: "Low Risk" if x < 0.4 else "Moderate Risk" if x < 0.7 else "High Risk")

        # Pest Disease Dataset
        if not all(col in pest_disease_dataset.columns for col in ['Crop', 'Pest', 'Disease']):
            pest_disease_dataset = pd.DataFrame(columns=['Crop', 'Pest', 'Disease'])

        return severity_encoder, crop_risk, period_dataset, pest_disease_dataset
    except Exception as e:
        st.error(f"Error processing additional data: {str(e)}")
        return LabelEncoder(), pd.Series(), pd.DataFrame(), pd.DataFrame()

# Load and process data
agricultural_dataset, period_dataset, pest_disease_dataset = load_datasets()
try:
    agricultural_model, scaler_agriculture, soil_type_encoder, model_accuracy, class_report, conf_matrix = load_or_train_model(agricultural_dataset)
    severity_encoder, crop_risk, period_dataset, pest_disease_dataset = process_additional_data(period_dataset, pest_disease_dataset)
except Exception as e:
    st.error(f"Error initializing: {str(e)}")
    agricultural_model, scaler_agriculture, soil_type_encoder = None, None, None
    model_accuracy, class_report, conf_matrix = None, None, None
    severity_encoder, crop_risk, period_dataset, pest_disease_dataset = LabelEncoder(), pd.Series(), pd.DataFrame(), pd.DataFrame()

def reset_daily_predictions():
    """Reset prediction count if it's a new day"""
    current_date = datetime.now().date()
    if (st.session_state.last_prediction_date is None or 
        st.session_state.last_prediction_date != current_date):
        st.session_state.predictions_today = 0
        st.session_state.last_prediction_date = current_date

def increment_prediction_count():
    """Increment the prediction count for the day"""
    st.session_state.predictions_today += 1
    st.session_state.last_prediction_date = datetime.now().date()

def can_make_prediction():
    """Check if user can make a prediction"""
    if st.session_state.subscription_status == 'premium':
        return True
    reset_daily_predictions()
    return st.session_state.predictions_today < 3

def recommend_crops(N, P, K, temperature, humidity, ph, rainfall, soil_type):
    try:
        latitude, longitude = get_location()
        location = get_location_details(latitude, longitude) if latitude and longitude else None
        
        soil_type_encoded = soil_type_encoder.transform([soil_type])[0]
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_type_encoded]])
        input_features_scaled = scaler_agriculture.transform(input_features)
        
        probabilities = agricultural_model.predict_proba(input_features_scaled)[0]
        crop_probabilities = list(zip(agricultural_model.classes_, probabilities))
        crop_probabilities.sort(key=lambda x: x[1], reverse=True)
        
        top_crops = crop_probabilities[:5]
        recommendations = {}
        for crop, prob in top_crops:
            pest_info = period_dataset[period_dataset['Crop Affected'].str.contains(crop, case=False, na=False)]
            pest_disease_info = pest_disease_dataset[pest_disease_dataset['Crop'].str.contains(crop, case=False, na=False)]
            recommendations[crop] = {
                'probability': prob,
                'pests': pest_info,
                'pest_diseases': pest_disease_info
            }
        
        st.session_state.dashboard_data['total_predictions'] += 1
        top_crop = top_crops[0][0]
        if len(st.session_state.dashboard_data['recent_crops']) >= 5:
            st.session_state.dashboard_data['recent_crops'].pop(0)
        st.session_state.dashboard_data['recent_crops'].append(top_crop)
        st.session_state.dashboard_data['favorite_crops'][top_crop] = st.session_state.dashboard_data['favorite_crops'].get(top_crop, 0) + 1
        
        return recommendations, crop_risk, location
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return {}, pd.Series(), None

def get_location():
    try:
        ip_address = requests.get('https://api64.ipify.org?format=json').json()['ip']
        response = requests.get(f'https://ipinfo.io/{ip_address}/json').json()
        loc = response['loc'].split(',')
        return loc[0], loc[1]
    except Exception:
        return None, None

def get_location_details(latitude, longitude):
    try:
        geolocator = geopy.geocoders.Nominatim(user_agent="crop_recommendation")
        location = geolocator.reverse(f"{latitude}, {longitude}")
        return location.address
    except Exception:
        return None

def create_severity_heatmap(period_dataset):
    pivot_data = period_dataset.pivot_table(values='Severity_Encoded', index='Region', columns='Crop Affected', aggfunc='mean').fillna(0)
    fig = go.Figure(data=go.Heatmap(z=pivot_data.values, x=pivot_data.columns, y=pivot_data.index, colorscale='RdYlBu_r', colorbar=dict(title='Severity')))
    fig.update_layout(title='Pest Severity by Region', xaxis_title='Crop', yaxis_title='Region', height=500, template="plotly_white")
    return fig

def create_crop_distribution_chart(agricultural_dataset):
    crop_counts = agricultural_dataset['label'].value_counts()
    fig = px.pie(values=crop_counts.values, names=crop_counts.index, title='Crop Distribution', color_discrete_sequence=px.colors.sequential.Greens)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white")
    return fig

def create_nutrient_boxplot(agricultural_dataset):
    nutrients_df = agricultural_dataset[['N', 'P', 'K', 'label']].melt(id_vars=['label'], var_name='Nutrient', value_name='Value')
    fig = px.box(nutrients_df, x='Nutrient', y='Value', color='Nutrient', title='N-P-K Distribution', color_discrete_sequence=['#2e7d32', '#388e3c', '#4caf50'])
    fig.update_layout(template="plotly_white")
    return fig

def create_model_performance_viz(class_report):
    if not class_report:
        return None
    df = pd.DataFrame(class_report).transpose().reset_index()
    df = df[df['index'].isin(agricultural_model.classes_)]
    fig = px.bar(df, x='index', y=['precision', 'recall', 'f1-score'], barmode='group', title='Performance per Crop', labels={'index': 'Crop', 'value': 'Score'})
    fig.update_layout(template="plotly_white")
    return fig

def show_prediction_page():
    st.markdown('<h1 class="main-header">AgriSmart - Crop Recommendation</h1>', unsafe_allow_html=True)
    
    # Add subscription status and features display
    is_premium = st.session_state.subscription_status == 'premium'
    subscription_color = "#4CAF50" if is_premium else "#FFA726"
    
    st.markdown(f"""
        <div style="
            background-color: {subscription_color};
            padding: 10px;
            border-radius: 5px;
            color: white;
            text-align: center;
            margin-bottom: 20px;">
            Current Plan: {st.session_state.subscription_status.upper()}
        </div>
    """, unsafe_allow_html=True)

    # Display feature comparison if user is on free plan
    if not is_premium:
        st.info("🌟 Upgrade to Premium to unlock:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Premium Features:**
            - ✅ Detailed pest analysis
            - ✅ Advanced soil recommendations
            - ✅ Export predictions to PDF/CSV
            - ✅ Historical data analysis
            - ✅ Priority support
            """)
        with col2:
            st.markdown("""
            **Free Features:**
            - ✅ Basic crop recommendations
            - ✅ Simple weather data
            - ✅ Basic soil analysis
            - ❌ Limited to 3 predictions/day
            - ❌ Basic support
            """)

    tabs = st.tabs(["📊 Dashboard", "🌱 Recommendation", "📈 Analytics", "🔍 Research"])
    
    with tabs[0]:
        st.subheader("Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.dashboard_data["total_predictions"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Predictions</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            month = datetime.now().month
            season = "Spring" if 3 <= month <= 5 else "Summer" if 6 <= month <= 8 else "Autumn" if 9 <= month <= 11 else "Winter"
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{season}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Season</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            unique_crops = len(set(st.session_state.dashboard_data['recent_crops']))
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{unique_crops}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Unique Crops</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            favorite_crop = max(st.session_state.dashboard_data['favorite_crops'], key=st.session_state.dashboard_data['favorite_crops'].get, default="None")
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{favorite_crop}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Top Crop</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Recent Activity")
        if st.session_state.dashboard_data['recent_crops']:
            recent_df = pd.DataFrame({"Crop": st.session_state.dashboard_data['recent_crops'], "Prediction": range(1, len(st.session_state.dashboard_data['recent_crops'])+1)})
            fig = px.line(recent_df, x="Prediction", y="Crop", markers=True, title="Recent Predictions", color_discrete_sequence=['#2e7d32'])
            st.plotly_chart(fig)
        else:
            st.info("No predictions yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add premium features to dashboard
        if is_premium:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Premium Analytics")
            # Add historical data analysis
            if st.session_state.dashboard_data['recent_crops']:
                st.line_chart(pd.DataFrame({"predictions": st.session_state.dashboard_data['recent_crops']}))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        regions = sorted(period_dataset['Region'].unique()) if 'Region' in period_dataset.columns else ["Unknown"]
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Input Parameters")

        # Show prediction count for free users
        if st.session_state.subscription_status == 'free':
            reset_daily_predictions()
            predictions_left = 3 - st.session_state.predictions_today
            st.warning(f"Free Plan: {predictions_left} predictions left today. Upgrade to Premium for unlimited predictions!")

        # Create a container for recommendations
        recommendations_container = st.container()

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                N = st.slider("Nitrogen (kg/ha)", 0, 150, 20)
                P = st.slider("Phosphorus (kg/ha)", 0, 150, 20)
                K = st.slider("Potassium (kg/ha)", 0, 200, 40)
                temperature = st.slider("Temperature (°C)", 0, 50, 25)
                region = st.selectbox("Region", options=regions)
            with col2:
                humidity = st.slider("Humidity (%)", 0, 100, 60)
                ph = st.slider("pH", 0.0, 14.0, 6.0, step=0.1)
                rainfall = st.slider("Rainfall (mm)", 10, 300, 100)
                soil_type = st.selectbox("Soil Type", options=['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky'])
            submitted = st.form_submit_button("Recommend Crops")

            if submitted:
                if can_make_prediction():
                    with st.spinner("Analyzing..."):
                        recommendations, crop_risk, location = recommend_crops(N, P, K, temperature, humidity, ph, rainfall, soil_type)
                        increment_prediction_count()
                        
                        # Store recommendations in session state for display outside the form
                        st.session_state.current_recommendations = recommendations
                        st.session_state.current_crop_risk = crop_risk
                        st.session_state.current_location = location
                        st.session_state.show_recommendations = True
                else:
                    st.error("You've reached your daily prediction limit. Upgrade to Premium for unlimited predictions!")
                    if st.button("Upgrade to Premium"):
                        st.session_state.show_payment = True

        # Display recommendations outside the form
        if st.session_state.get('show_recommendations', False):
            with recommendations_container:
                st.subheader("Recommended Crops")
                for crop, details in st.session_state.current_recommendations.items():
                    with st.expander(f"{crop} (Probability: {details['probability']:.2%})"):
                        st.write(f"Risk Level: {st.session_state.current_crop_risk.get(crop, 'Unknown')}")
                        if st.session_state.current_location:
                            st.write(f"Location: {st.session_state.current_location}")
                        
                        # Show pest information if available
                        if not details['pests'].empty:
                            st.write("Pest Information:")
                            st.dataframe(details['pests'])
                        
                        # Show disease information if available
                        if not details['pest_diseases'].empty:
                            st.write("Disease Information:")
                            st.dataframe(details['pest_diseases'])
                
                # Premium features section
                if st.session_state.subscription_status == 'premium':
                    st.markdown("---")
                    st.subheader("Premium Features")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Report (PDF)",
                            "Premium Report Content",  # Replace with actual PDF generation
                            "crop_report.pdf"
                        )
                    with col2:
                        st.download_button(
                            "Export Data (CSV)",
                            "CSV Data",  # Replace with actual CSV data
                            "crop_data.csv"
                        )
                else:
                    # Free users get limited results
                    st.info("⭐ Upgrade to Premium for detailed reports and data export!")
                    # Limit the number of recommendations shown
                    recommendations = dict(list(st.session_state.current_recommendations.items())[:3])
                    
                    # Show remaining predictions
                    predictions_left = 3 - st.session_state.predictions_today
                    if predictions_left > 0:
                        st.warning(f"You have {predictions_left} predictions left today.")
                    else:
                        st.warning("You've used all your predictions for today. Upgrade to Premium for unlimited predictions!")

        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        if is_premium:
            # Premium analytics features
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Advanced Analytics")
            # Add more detailed analytics for premium users
            st.plotly_chart(create_nutrient_boxplot(agricultural_dataset))
            st.plotly_chart(create_crop_distribution_chart(agricultural_dataset))
            st.plotly_chart(create_severity_heatmap(period_dataset))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("🔒 Analytics features are available for Premium members only")
            st.button("Upgrade to Access Analytics")
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Model Performance")
        if model_accuracy:
            st.metric("Accuracy", f"{model_accuracy * 100:.2f}%")
            fig = create_model_performance_viz(class_report)
            if fig:
                st.plotly_chart(fig)
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            st.subheader("Classification Report")
            report_df = pd.DataFrame(class_report).transpose()
            st.dataframe(report_df.style.format("{:.2f}", subset=['precision', 'recall', 'f1-score']))
        else:
            st.warning("Performance data unavailable for pre-trained model.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        if is_premium:
            # Premium research features
            st.subheader("Advanced Research Tools")
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Soil Optimization")
            target_crop = st.selectbox("Target Crop", sorted(agricultural_dataset['label'].unique()))
            if st.button("Optimize Soil"):
                crop_data = agricultural_dataset[agricultural_dataset['label'] == target_crop]
                st.success(f"Optimal Parameters for {target_crop}:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nitrogen", f"{crop_data['N'].mean():.1f} kg/ha", f"Range: {crop_data['N'].mean() - crop_data['N'].std():.1f}-{crop_data['N'].mean() + crop_data['N'].std():.1f}")
                    st.metric("Phosphorus", f"{crop_data['P'].mean():.1f} kg/ha", f"Range: {crop_data['P'].mean() - crop_data['P'].std():.1f}-{crop_data['P'].mean() + crop_data['P'].std():.1f}")
                with col2:
                    st.metric("Potassium", f"{crop_data['K'].mean():.1f} kg/ha", f"Range: {crop_data['K'].mean() - crop_data['K'].std():.1f}-{crop_data['K'].mean() + crop_data['K'].std():.1f}")
                    st.metric("pH", f"{crop_data['ph'].mean():.1f}", f"Range: {crop_data['ph'].mean() - crop_data['ph'].std():.1f}-{crop_data['ph'].mean() + crop_data['ph'].std():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("🔒 Research tools are available for Premium members only")
            st.button("Upgrade to Access Research Tools")

def create_user(username, password):
    try:
        if not all(db_config.values()):
            st.error("Database config incomplete.")
            return False
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            st.error("Username exists.")
            return False
        password_hash = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, password_hash))
        connection.commit()
        st.success("User created!")
        return True
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False
    finally:
        if 'connection' in locals():
            cursor.close()
            connection.close()

def check_subscription(username):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT subscription_status, subscription_end FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        
        if result:
            status, end_date = result
            if end_date and end_date > datetime.now():
                return status
        return 'free'
    except Exception as e:
        st.error(f"Error checking subscription: {str(e)}")
        return 'free'
    finally:
        if 'connection' in locals():
            cursor.close()
            connection.close()

def upgrade_subscription(username):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # Set subscription end date to 30 days from now
        end_date = datetime.now() + datetime.timedelta(days=30)
        
        # Update the users table instead of subscriptions table
        cursor.execute("""
            UPDATE users 
            SET subscription_status = 'premium',
                subscription_end = %s
            WHERE username = %s
        """, (end_date, username))
        
        connection.commit()
        st.session_state.subscription_status = 'premium'
        return True
    except Exception as e:
        st.error(f"Error upgrading subscription: {str(e)}")
        return False
    finally:
        if 'connection' in locals():
            cursor.close()
            connection.close()

def login_user(username, password):
    try:
        if not all(db_config.values()):
            st.error("Database config incomplete.")
            return False
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT username, password_hash FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user[1], password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.subscription_status = check_subscription(username)
            return True
        st.error("Invalid credentials.")
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False
    finally:
        if 'connection' in locals():
            cursor.close()
            connection.close()

def show_login_page():
    st.title("AgriSmart Login")
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if login_user(username, password):
                    st.rerun()
    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords don't match.")
                elif create_user(new_username, new_password):
                    st.success("Registered! Please login.")

def main():
    try:
        if not all(db_config.values()):
            st.warning("Limited mode.")
        else:
            connection = mysql.connector.connect(**db_config)
            connection.close()
    except Exception as e:
        st.error(f"Database error: {str(e)}")

    if not is_session_valid():
        st.session_state.logged_in = False
        st.session_state.username = None

    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_prediction_page()
        with st.sidebar:
            st.write(f"Welcome, {st.session_state.username}!")
            
            # Add subscription management
            st.markdown("---")
            st.subheader("Subscription Status")
            if st.session_state.subscription_status == 'free':
                st.warning("Free Plan Limitations:")
                st.markdown("""
                - 3 predictions per day
                - Basic analysis only
                - No data export
                - Limited support
                """)
                
                if st.button("🌟 Upgrade to Premium"):
                    st.session_state.show_payment = True
                
                # Show payment form when upgrade button is clicked
                if 'show_payment' in st.session_state and st.session_state.show_payment:
                    show_payment_form(upgrade_subscription)
                    
            else:
                st.success("Premium Member Benefits:")
                st.markdown("""
                - Unlimited predictions
                - Advanced analysis
                - Data export (PDF/CSV)
                - Priority support
                - Historical data access
                """)
            
            st.markdown("---")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.subscription_status = 'free'
                if 'show_payment' in st.session_state:
                    del st.session_state.show_payment
                st.rerun()

if __name__ == "__main__":
    model_path = r"C:\Users\cossi\OneDrive\Desktop\limo\crop-model\models\agricultural_model.joblib"
    if not os.path.exists(model_path):
        agricultural_dataset, _, _ = load_datasets()
        if not agricultural_dataset.empty:
            train_and_save_model(agricultural_dataset)
        else:
            st.error("Cannot train model: agricultural_dataset.csv is missing or invalid.")
    main()