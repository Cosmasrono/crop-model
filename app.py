import numpy as np
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.geocoders import Nominatim
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import streamlit as st
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import datetime
import json
import time

# Configure page settings
st.set_page_config(
    page_title="AgriSmart - Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dashboard-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1b5e20;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    footer {
        text-align: center;
        padding: 20px;
        font-size: 0.8rem;
        color: #666;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f8f0;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e7d32 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables from .env file
load_dotenv()

# Access the variables
db_config = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USERNAME"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME")
}

# Initialize session state for persistent login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.datetime.now()
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = {
        'total_predictions': 0,
        'recent_crops': [],
        'favorite_crops': {}
    }

# Session management - keep users logged in
def update_session_activity():
    st.session_state.last_activity = datetime.datetime.now()

# Check if session is active and valid
SESSION_TIMEOUT = 3600  # 1 hour in seconds
def is_session_valid():
    if not st.session_state.logged_in:
        return False
    now = datetime.datetime.now()
    time_diff = (now - st.session_state.last_activity).total_seconds()
    if time_diff > SESSION_TIMEOUT:
        return False
    update_session_activity()
    return True

# Load datasets
@st.cache_data
def load_datasets():
    agricultural_dataset = pd.read_csv('data/agricultural_dataset.csv')
    insect_dataset = pd.read_csv('data/period.csv')
    return agricultural_dataset, insect_dataset

try:
    agricultural_dataset, insect_dataset = load_datasets()
except Exception as e:
    st.error(f"Error loading datasets: {str(e)}")
    agricultural_dataset = pd.DataFrame()
    insect_dataset = pd.DataFrame()

# Prepare agricultural dataset
@st.cache_data
def prepare_agricultural_data(agricultural_dataset):
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
    
    return agricultural_model, scaler_agriculture, soil_type_encoder

# Preprocess insect dataset
@st.cache_data
def preprocess_insect_data(insect_dataset):
    try:
        # Check if required columns exist
        required_columns = ['Severity', 'Crop Affected', 'Region', 'Insect Name']
        if not all(col in insect_dataset.columns for col in required_columns):
            # If columns don't exist, create a default dataset
            insect_dataset = pd.DataFrame(columns=required_columns)
            
        # Add Severity if it doesn't exist
        if 'Severity' not in insect_dataset.columns:
            insect_dataset['Severity'] = 'Low'
            
        severity_encoder = LabelEncoder()
        # Ensure there's at least one severity value
        if len(insect_dataset) == 0:
            severity_values = ['Low', 'Medium', 'High']
            severity_encoder.fit(severity_values)
            insect_dataset['Severity_Encoded'] = 0
        else:
            insect_dataset['Severity_Encoded'] = severity_encoder.fit_transform(insect_dataset['Severity'])
        
        # Calculate crop severity
        if len(insect_dataset) > 0:
            crop_severity = insect_dataset.groupby('Crop Affected')['Severity_Encoded'].mean().sort_values()
        else:
            crop_severity = pd.Series(dtype=float)
        
        def categorize_crop(severity):
            if severity < 0.4: return "Low Risk"
            elif severity < 0.7: return "Moderate Risk"
            else: return "High Risk"
        
        crop_risk = crop_severity.apply(categorize_crop)
        return severity_encoder, crop_risk, insect_dataset
    except Exception as e:
        st.error(f"Error preprocessing insect data: {str(e)}")
        # Return default values
        return LabelEncoder(), pd.Series(), pd.DataFrame()

# Prepare data
try:
    agricultural_model, scaler_agriculture, soil_type_encoder = prepare_agricultural_data(agricultural_dataset)
    severity_encoder, crop_risk, insect_dataset = preprocess_insect_data(insect_dataset)
except Exception as e:
    st.error(f"Error processing data: {str(e)}")

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
    try:
        latitude, longitude = get_location()
        location = get_location_details(latitude, longitude) if latitude and longitude else None
        
        soil_type_encoded = soil_type_encoder.transform([soil_type])[0]
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_type_encoded]])
        input_features_scaled = scaler_agriculture.transform(input_features)
        
        probabilities = agricultural_model.predict_proba(input_features_scaled)[0]
        crop_probabilities = list(zip(agricultural_model.classes_, probabilities))
        crop_probabilities.sort(key=lambda x: x[1], reverse=True)
        
        top_crops = crop_probabilities[:5]  # Increased to top 5 for dashboard
        recommendations = {}
        for crop, prob in top_crops:
            try:
                if crop in crop_risk.index:
                    pests_info = insect_dataset[insect_dataset['Crop Affected'].str.contains(crop, na=False)]
                    recommendations[crop] = {'probability': prob, 'pests': pests_info}
                else:
                    recommendations[crop] = {'probability': prob, 'pests': pd.DataFrame()}
            except Exception:
                recommendations[crop] = {'probability': prob, 'pests': pd.DataFrame()}
        
        # Update dashboard data
        st.session_state.dashboard_data['total_predictions'] += 1
        
        # Update recent crops list (keep last 5)
        top_crop = top_crops[0][0]
        if len(st.session_state.dashboard_data['recent_crops']) >= 5:
            st.session_state.dashboard_data['recent_crops'].pop(0)
        st.session_state.dashboard_data['recent_crops'].append(top_crop)
        
        # Update favorite crops counter
        if top_crop in st.session_state.dashboard_data['favorite_crops']:
            st.session_state.dashboard_data['favorite_crops'][top_crop] += 1
        else:
            st.session_state.dashboard_data['favorite_crops'][top_crop] = 1
        
        return recommendations, crop_risk, location
    except Exception as e:
        st.error(f"Error in recommend_crops: {str(e)}")
        return {}, pd.Series(), None

def create_severity_heatmap(insect_dataset):
    # Create a pivot table of severity by region and crop
    pivot_data = insect_dataset.pivot_table(
        values='Severity_Encoded',
        index='Region',
        columns='Crop Affected',
        aggfunc='mean'
    ).fillna(0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlBu_r',
        colorbar=dict(title='Severity')
    ))
    
    fig.update_layout(
        title='Crop Severity by Region',
        xaxis_title='Crop',
        yaxis_title='Region',
        height=500,
        template="plotly_white"
    )
    return fig

def create_crop_distribution_chart(agricultural_dataset):
    crop_counts = agricultural_dataset['label'].value_counts()
    fig = px.pie(
        values=crop_counts.values,
        names=crop_counts.index,
        title='Crop Distribution in Dataset',
        color_discrete_sequence=px.colors.sequential.Greens
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white")
    return fig

def create_nutrient_boxplot(agricultural_dataset):
    # Melt the dataset for nutrients
    nutrients_df = agricultural_dataset[['N', 'P', 'K', 'label']].melt(
        id_vars=['label'],
        var_name='Nutrient',
        value_name='Value'
    )
    
    fig = px.box(
        nutrients_df,
        x='Nutrient',
        y='Value',
        color='Nutrient',
        title='Distribution of N-P-K Values',
        color_discrete_sequence=['#2e7d32', '#388e3c', '#4caf50']
    )
    fig.update_layout(template="plotly_white")
    return fig

def create_dashboard_metrics():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{st.session_state.dashboard_data["total_predictions"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Predictions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        # Get current season
        month = datetime.datetime.now().month
        if 3 <= month <= 5:
            season = "Spring"
        elif 6 <= month <= 8:
            season = "Summer"
        elif 9 <= month <= 11:
            season = "Autumn"
        else:
            season = "Winter"
            
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{season}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Current Season</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Count unique crops predicted
        unique_crops = len(set(st.session_state.dashboard_data['recent_crops']))
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{unique_crops}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Unique Crops Explored</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        # Most recommended crop
        if st.session_state.dashboard_data['favorite_crops']:
            favorite_crop = max(st.session_state.dashboard_data['favorite_crops'], 
                               key=st.session_state.dashboard_data['favorite_crops'].get)
        else:
            favorite_crop = "None"
            
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{favorite_crop}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Most Recommended Crop</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_recent_activity_chart():
    if not st.session_state.dashboard_data['recent_crops']:
        st.info("No predictions made yet. Your recent activity will appear here.")
        return
        
    # Create recent activity chart
    recent_df = pd.DataFrame({"Crop": st.session_state.dashboard_data['recent_crops'],
                             "Prediction": range(1, len(st.session_state.dashboard_data['recent_crops'])+1)})
    
    fig = px.line(recent_df, x="Prediction", y="Crop", markers=True,
                 title="Recent Crop Predictions",
                 color_discrete_sequence=['#2e7d32'])
    
    fig.update_layout(
        xaxis_title="Prediction Number",
        yaxis_title="Recommended Crop",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_page():
    st.markdown('<h1 class="main-header">AgriSmart - AI Crop Recommendation System</h1>', unsafe_allow_html=True)
    
    # Dashboard tabs
    tabs = st.tabs(["📊 Dashboard", "🌱 Crop Recommendation", "📈 Analytics", "🔍 Research"])
    
    with tabs[0]:
        st.subheader("Dashboard Overview")
        
        # Dashboard metrics
        create_dashboard_metrics()
        
        # Recent activity and insights
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Recent Activity")
        create_recent_activity_chart()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Weather information
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Today's Weather")
            st.info("Weather data integration pending. Connect your local weather station or API to see real-time conditions.")
            
        with col2:
            st.subheader("Seasonal Advisory")
            current_month = datetime.datetime.now().month
            if 3 <= current_month <= 5:  # Spring
                st.success("Spring planting season is here! Consider soil preparation and early crops.")
            elif 6 <= current_month <= 8:  # Summer
                st.success("Summer growing season. Focus on irrigation and pest management.")
            elif 9 <= current_month <= 11:  # Fall
                st.success("Fall harvest approaching. Plan your storage and prepare for winter crops.")
            else:  # Winter
                st.success("Winter planning season. Research crop varieties and prepare for spring planting.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        # Get unique regions from the dataset
        regions = sorted(insect_dataset['Region'].unique())
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Input Your Field Parameters")
        
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
                    with st.spinner("Analyzing soil data and generating recommendations..."):
                        # Add a small delay to show the spinner
                        time.sleep(1)
                        recommendations, crop_risk, location = recommend_crops(
                            N, P, K, temperature, humidity, ph, rainfall, soil_type
                        )
                    
                    st.success("Analysis Complete! Here are your crop recommendations:")
                    
                    # Results section
                    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                    
                    # Display location if available
                    if location:
                        st.info(f"📍 Your detected location: {location}")
                    
                    # Display selected region information
                    st.subheader(f"Selected Region: {region}")
                    region_data = insect_dataset[insect_dataset['Region'] == region]
                    if not region_data.empty:
                        avg_severity = region_data['Severity_Encoded'].mean()
                        
                        # Create severity gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = avg_severity,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Region Pest Severity"},
                            gauge = {
                                'axis': {'range': [0, 1], 'tickwidth': 1},
                                'bar': {'color': "rgba(0,0,0,0)"},
                                'steps': [
                                    {'range': [0, 0.4], 'color': "#c8e6c9"},
                                    {'range': [0.4, 0.7], 'color': "#fff9c4"},
                                    {'range': [0.7, 1], 'color': "#ffcdd2"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': avg_severity
                                }
                            }
                        ))
                        
                        fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Common pests in the region
                        st.subheader("Common Pests in this Region:")
                        region_pests = region_data['Insect Name'].value_counts().head(5)
                        fig = px.bar(
                            x=region_pests.index,
                            y=region_pests.values,
                            title=f'Pest Distribution in {region}',
                            labels={'x': 'Pest', 'y': 'Count'},
                            color_discrete_sequence=['#ff7043']
                        )
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Crop recommendations card
                    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                    st.subheader("Top Crop Recommendations:")
                    
                    # Create bar chart for crop recommendations
                    crop_probs = {crop: data['probability'] * 100 for crop, data in recommendations.items()}
                    fig_recommendations = px.bar(
                        y=list(crop_probs.keys()),
                        x=list(crop_probs.values()),
                        orientation='h',
                        title='Crop Suitability Analysis',
                        labels={'y': 'Crop', 'x': 'Confidence (%)'},
                        color=list(crop_probs.values()),
                        color_continuous_scale='Greens',
                    )
                    fig_recommendations.update_layout(template="plotly_white")
                    st.plotly_chart(fig_recommendations, use_container_width=True)
                    
                    # Detailed crop information
                    top_crop = list(recommendations.keys())[0]
                    st.subheader(f"Detailed Information: {top_crop}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Risk assessment
                        if top_crop in crop_risk.index:
                            risk_level = crop_risk[top_crop]
                            risk_color = {
                                "Low Risk": "#c8e6c9",
                                "Moderate Risk": "#fff9c4",
                                "High Risk": "#ffcdd2"
                            }
                            st.markdown(f'<div style="background-color:{risk_color[risk_level]};padding:10px;border-radius:5px;">'
                                       f'<strong>Pest Risk Level:</strong> {risk_level}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Confidence level
                        confidence = crop_probs[top_crop]
                        st.metric("Confidence Score", f"{confidence:.1f}%")
                    
                    # Show pests that affect this crop
                    crop_pests = recommendations[top_crop]['pests']
                    if not crop_pests.empty:
                        st.subheader(f"Common Pests for {top_crop}")
                        pest_counts = crop_pests['Insect Name'].value_counts().head(5)
                        st.bar_chart(pest_counts)
                        
                        # Pest management recommendations
                        st.subheader("Pest Management Recommendations")
                        st.info("Based on your region and crop selection, we recommend the following pest management strategies:")
                        strategies = [
                            "Regular crop monitoring for early pest detection",
                            "Implement crop rotation to break pest cycles",
                            "Consider resistant varieties for future planting",
                            "Use integrated pest management (IPM) approaches",
                            "Consult with local agricultural extension for specific treatments"
                        ]
                        for strategy in strategies:
                            st.write(f"• {strategy}")
                    else:
                        st.info(f"No specific pest data available for {top_crop} in our database.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Unable to make prediction. Please try again. Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.subheader("Crop Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Soil nutrient analysis
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Soil Nutrient Analysis")
            st.plotly_chart(create_nutrient_boxplot(agricultural_dataset), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Crop distribution
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Crop Distribution")
            st.plotly_chart(create_crop_distribution_chart(agricultural_dataset), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a full-width chart
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Regional Pest Severity Analysis")
        st.plotly_chart(create_severity_heatmap(insect_dataset), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance
        if hasattr(agricultural_model, 'feature_importances_'):
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Feature Importance")
            
            # Get feature names - only include columns that exist
            feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            feature_names = [col for col in feature_columns if col in agricultural_dataset.columns]
            
            # Get feature importances and ensure they match feature names
            importances = agricultural_model.feature_importances_[:len(feature_names)]  # Limit to actual features
            indices = np.argsort(importances)
            
            # Create dataframe for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                labels={'Importance': 'Feature Importance', 'Feature': 'Parameter'},
                title='Feature Importance for Crop Prediction',
                color='Importance',
                color_continuous_scale='Greens'
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        st.subheader("Agricultural Research")
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Experimental Features")
        
        experiment_tabs = st.tabs(["Soil Optimization", "Climate Impact", "Yield Prediction"])
        
        with experiment_tabs[0]:
            st.info("Soil optimization feature is in development. This will help you optimize soil nutrients based on your target crop.")
            
            st.subheader("Target Crop")
            target_crop = st.selectbox("Select your target crop", options=sorted(agricultural_dataset['label'].unique()))
            
            if st.button("Generate Soil Recommendations"):
                st.spinner("Analyzing optimal soil conditions...")
                # Filter the dataset for the selected crop
                crop_data = agricultural_dataset[agricultural_dataset['label'] == target_crop]
                
                # Calculate optimal ranges
                n_range = (crop_data['N'].mean() - crop_data['N'].std(), crop_data['N'].mean() + crop_data['N'].std())
                p_range = (crop_data['P'].mean() - crop_data['P'].std(), crop_data['P'].mean() + crop_data['P'].std())
                k_range = (crop_data['K'].mean() - crop_data['K'].std(), crop_data['K'].mean() + crop_data['K'].std())
                ph_range = (crop_data['ph'].mean() - crop_data['ph'].std(), crop_data['ph'].mean() + crop_data['ph'].std())
                
                st.success(f"Recommended soil parameters for {target_crop}:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nitrogen (N)", f"{crop_data['N'].mean():.1f} kg/ha", f"Range: {n_range[0]:.1f}-{n_range[1]:.1f}")
                    st.metric("Phosphorus (P)", f"{crop_data['P'].mean():.1f} kg/ha", f"Range: {p_range[0]:.1f}-{p_range[1]:.1f}")
                with col2:
                    st.metric("Potassium (K)", f"{crop_data['K'].mean():.1f} kg/ha", f"Range: {k_range[0]:.1f}-{k_range[1]:.1f}")
                    st.metric("pH Level", f"{crop_data['ph'].mean():.1f}", f"Range: {ph_range[0]:.1f}-{ph_range[1]:.1f}")
        
        with experiment_tabs[1]:
            st.info("Climate impact analysis feature is coming soon. This will help you understand how climate change might affect your crops.")
            
            st.image("https://via.placeholder.com/800x400?text=Climate+Impact+Visualization", use_column_width=True)
            
            st.warning("This feature requires climate projection data which will be available in the next update.")
        
        with experiment_tabs[2]:
            st.info("Yield prediction feature is under development. This will provide estimates of potential crop yields based on your input parameters.")
            
            st.subheader("Coming Soon - Yield Predictor")
            st.write("In the next update, you'll be able to predict yields for each recommended crop based on:")
            st.write("• Historical yield data")
            st.write("• Current soil conditions")
            st.write("• Weather forecasts")
            st.write("• Management practices")
            
            st.success("Sign up for our newsletter to be notifie")
            
def create_user(username, password):
    """Create a new user and store credentials in the database"""
    try:
        if not all(db_config.values()):
            st.error("Database configuration is incomplete. Please contact the administrator.")
            return False
            
        if not username or not password:
            st.error("Username and password cannot be empty.")
            return False

        # Connect to the MySQL database using the config
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Check if user already exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            st.error("Username already exists")
            return False

        # Insert new user data
        password_hash = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", 
                      (username, password_hash))
        connection.commit()

        st.success("User created successfully!")
        return True
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def login_user(username, password):
    """Authenticate user from database"""
    try:
        if not all(db_config.values()):
            st.error("Database configuration is incomplete. Please contact the administrator.")
            return False

        # Connect to the MySQL database using the config
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Get user from database
        cursor.execute("SELECT username, password_hash FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user[1], password):
            st.session_state.logged_in = True
            st.session_state.username = username
            return True
        else:
            st.error("Invalid username or password")
            return False
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def show_login_page():
    """Display login/registration page"""
    st.title("Welcome to AgriSmart")
    
    # Create tabs for login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if login_user(username, password):
                    st.success("Login successful!")
                    st.rerun()
    
    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    if create_user(new_username, new_password):
                        st.success("Registration successful! Please login.")

def main():
    # Check database connectivity
    try:
        if not all(db_config.values()):
            st.warning("⚠️ Application is running in limited mode. Some features may not be available.")
        else:
            connection = mysql.connector.connect(**db_config)
            connection.close()
    except Exception as e:
        st.error(f"⚠️ Database connection error: {str(e)}")

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.datetime.now()

    # Check if session is valid
    if not is_session_valid():
        st.session_state.logged_in = False
        st.session_state.username = None

    # Show appropriate page based on login status
    if not st.session_state.logged_in:
        show_login_page()
    else:
        # Show the main application
        show_prediction_page()
        
        # Add logout button in sidebar
        with st.sidebar:
            st.write(f"Welcome, {st.session_state.username}!")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.rerun()

if __name__ == "__main__":
    main()
            