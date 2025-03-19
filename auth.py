import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import csv
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_user(username, password):
    """Create a new user and store credentials locally"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Define the users file path
        users_file = 'data/users.csv'
        
        # Check if user already exists
        if os.path.exists(users_file):
            df = pd.read_csv(users_file)
            if username in df['username'].values:
                st.error("Username already exists")
                return False
        
        # Create new user data
        user_data = {
            'username': username,
            'password_hash': generate_password_hash(password),
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Write to CSV
        with open(users_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=user_data.keys())
            if not os.path.exists(users_file):
                writer.writeheader()
            writer.writerow(user_data)
        
        return True
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False

def login_user(username, password):
    """Authenticate user from local CSV file"""
    try:
        users_file = 'data/users.csv'
        
        # Check if users file exists
        if not os.path.exists(users_file):
            st.error("No users found")
            return False
        
        # Read users data
        df = pd.read_csv(users_file)
        
        # Find user
        user = df[df['username'] == username]
        
        if user.empty:
            return False
        
        # Check password
        return check_password_hash(user.iloc[0]['password_hash'], password)
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def show_login_page():
    """Display login/register page"""
    st.markdown('<h1 class="main-header">Welcome to AgriSmart</h1>', unsafe_allow_html=True)
    
    # Create two columns for login/register
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Login")
            
            if login_submit:
                if login_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Register")
        with st.form("register_form"):
            reg_username = st.text_input("Choose Username", key="reg_username")
            reg_password = st.text_input("Choose Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            register_submit = st.form_submit_button("Register")
            
            if register_submit:
                if not reg_username or not reg_password:
                    st.error("Please fill in all fields")
                elif reg_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    if create_user(reg_username, reg_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add some information about the application
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("About AgriSmart")
    st.write("""
    AgriSmart is an intelligent crop recommendation system that helps farmers make informed decisions 
    about crop selection based on soil conditions, climate, and pest risks. Our AI-powered system 
    analyzes multiple factors to suggest the most suitable crops for your field.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🌱 Smart Recommendations")
        st.write("Get personalized crop suggestions based on your field conditions")
    with col2:
        st.markdown("### 📊 Data Analytics")
        st.write("Access detailed analytics about soil health and pest risks")
    with col3:
        st.markdown("### 🔍 Research Tools")
        st.write("Explore experimental features and agricultural research")
    st.markdown('</div>', unsafe_allow_html=True) 