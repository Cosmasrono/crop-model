import streamlit as st
import requests
import base64
from datetime import datetime, timedelta
import time
import json
import logging
import mysql.connector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# M-Pesa Sandbox Configuration
# NOTE: Replace these with your actual credentials from the Safaricom Developer Portal
MPESA_SANDBOX_URL = "https://sandbox.safaricom.co.ke"
CONSUMER_KEY = "fi5E6lESRdN0KWZ08aBAuOTkZ2dH3C7PyTPaLnzTMMyvdV8r"
CONSUMER_SECRET = "W3YAbHIaIGV5Mkofkl5npWHcsNn8hGmux1OYZANbGvuPzxN4RBww841zEGEIoObk"
BUSINESS_SHORT_CODE = "174379"  # Default sandbox shortcode
PASSKEY = "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919"  # Default sandbox passkey
CALLBACK_URL = "https://your-callback-url.com/callback"  # Must be a public HTTPS URL

# Database Configuration
db_config = {
    'host': st.secrets["db_host"],
    'user': st.secrets["db_username"],
    'password': st.secrets["db_password"],
    'database': st.secrets["db_name"]
}

# Initialize session state
if 'payment_initiated' not in st.session_state:
    st.session_state.payment_initiated = False
if 'account_status' not in st.session_state:
    st.session_state.account_status = "free"
if 'payment_processed' not in st.session_state:
    st.session_state.payment_processed = False

def get_database_connection():
    """Get a database connection"""
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def generate_access_token():
    """Generate M-Pesa API access token"""
    try:
        # Create the authorization string
        auth_string = f"{CONSUMER_KEY}:{CONSUMER_SECRET}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode('utf-8')

        # Set up headers
        headers = {
            'Authorization': f'Basic {encoded_auth}'
        }

        # Make the request
        url = f"{MPESA_SANDBOX_URL}/oauth/v1/generate?grant_type=client_credentials"
        response = requests.get(url, headers=headers)
        
        # Log the response for debugging (not visible to users)
        logger.info(f"Token Response Status: {response.status_code}")
        logger.info(f"Token Response: {response.text}")

        if response.status_code == 200:
            result = response.json()
            token = result.get('access_token')
            logger.info("Access token generated successfully")
            return token
        else:
            logger.error(f"Failed to generate access token: {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error generating access token: {str(e)}")
        return None

def generate_password():
    """Generate M-Pesa password for the transaction"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    data_to_encode = f"{BUSINESS_SHORT_CODE}{PASSKEY}{timestamp}"
    encoded = base64.b64encode(data_to_encode.encode())
    return encoded.decode('utf-8'), timestamp

def initiate_stk_push(phone_number, amount, account_reference, transaction_desc):
    """
    Initiate M-Pesa STK Push to customer's phone
    Returns (success_status, message, checkout_request_id)
    """
    try:
        # Format phone number (ensure it starts with 254)
        if phone_number.startswith('0'):
            phone_number = '254' + phone_number[1:]
        elif not phone_number.startswith('254'):
            phone_number = '254' + phone_number
        
        logger.info(f"Initiating STK push to {phone_number} for amount {amount}")
        
        # Generate access token
        access_token = generate_access_token()
        if not access_token:
            return False, "Could not authenticate with M-Pesa. Please try again.", None
        
        # Generate password and timestamp
        password, timestamp = generate_password()
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Prepare payload
        payload = {
            "BusinessShortCode": BUSINESS_SHORT_CODE,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": int(amount),  # Amount must be integer
            "PartyA": phone_number,
            "PartyB": BUSINESS_SHORT_CODE,
            "PhoneNumber": phone_number,
            "CallBackURL": CALLBACK_URL,
            "AccountReference": account_reference,
            "TransactionDesc": transaction_desc
        }
        
        # Log the request payload (for debugging, not visible to users)
        logger.info(f"STK Push Payload: {json.dumps(payload)}")
        
        # Send the request
        stk_push_url = f"{MPESA_SANDBOX_URL}/mpesa/stkpush/v1/processrequest"
        response = requests.post(stk_push_url, json=payload, headers=headers)
        
        # Log the response (for debugging, not visible to users)
        logger.info(f"STK Push Response Status: {response.status_code}")
        logger.info(f"STK Push Response: {response.text}")
        
        # Process the response
        if response.status_code == 200:
            result = response.json()
            response_code = result.get('ResponseCode')
            
            if response_code == "0":
                checkout_request_id = result.get('CheckoutRequestID')
                logger.info(f"STK Push successful with CheckoutRequestID: {checkout_request_id}")
                return True, "Please check your phone to complete the payment", checkout_request_id
            else:
                error_msg = result.get('errorMessage', 'Unknown error occurred')
                logger.error(f"STK Push failed: {error_msg}")
                return False, f"M-Pesa error: {error_msg}", None
        else:
            logger.error(f"STK Push HTTP Error: {response.status_code}, {response.text}")
            return False, f"Communication error with M-Pesa. Please try again.", None
            
    except Exception as e:
        logger.exception("Exception during STK push")
        return False, f"An unexpected error occurred: {str(e)}", None

def check_transaction_status(checkout_request_id):
    """
    Check the status of an STK push transaction
    Returns (success_status, message)
    """
    try:
        # Generate access token
        access_token = generate_access_token()
        if not access_token:
            return False, "Could not authenticate with M-Pesa"
        
        # Generate password and timestamp
        password, timestamp = generate_password()
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Prepare payload
        payload = {
            "BusinessShortCode": BUSINESS_SHORT_CODE,
            "Password": password,
            "Timestamp": timestamp,
            "CheckoutRequestID": checkout_request_id
        }
        
        # Send request
        query_url = f"{MPESA_SANDBOX_URL}/mpesa/stkpushquery/v1/query"
        response = requests.post(query_url, json=payload, headers=headers)
        
        # Log response (for debugging)
        logger.info(f"Status Query Response: {response.status_code}, {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            result_code = result.get('ResultCode')
            
            if result_code == "0":
                return True, "Payment completed successfully"
            else:
                result_desc = result.get('ResultDesc', 'Unknown status')
                return False, f"Payment status: {result_desc}"
        else:
            return False, "Could not determine payment status"
            
    except Exception as e:
        logger.exception("Exception checking transaction status")
        return False, f"Error checking payment status: {str(e)}"

def upgrade_to_premium(phone_number, username=None):
    """Upgrade user account from free to premium"""
    try:
        # Update session state
        st.session_state.account_status = "premium"
        st.session_state.payment_processed = True
        st.session_state.subscription_status = 'premium'  # Update app.py's subscription status
        
        # Update the database
        if username:
            try:
                # Connect to database
                conn = get_database_connection()
                cursor = conn.cursor()
                
                # First check if user exists and is currently free
                check_query = "SELECT subscription_status FROM users WHERE username = %s"
                cursor.execute(check_query, (username,))
                result = cursor.fetchone()
                
                if not result:
                    logger.error(f"User {username} not found in database")
                    return False
                
                current_status = result[0]
                if current_status == 'premium':
                    logger.info(f"User {username} is already premium")
                    return True
                
                # Set subscription end date to 30 days from now
                end_date = datetime.now() + datetime.timedelta(days=30)
                
                # Update the subscription status in the users table
                update_query = """
                    UPDATE users 
                    SET subscription_status = 'premium',
                        subscription_end = %s,
                        updated_at = NOW()
                    WHERE username = %s AND subscription_status = 'free'
                """
                cursor.execute(update_query, (end_date, username))
                
                # Check if update was successful
                if cursor.rowcount == 0:
                    logger.error(f"Failed to update subscription status for user {username}")
                    return False
                
                # Commit the changes and close the connection
                conn.commit()
                cursor.close()
                conn.close()
                
                # Log the successful upgrade
                logger.info(f"User {username} with phone {phone_number} successfully upgraded to premium in database")
                return True
                
            except Exception as db_error:
                logger.error(f"Database update error: {str(db_error)}")
                # Try to close connections if they exist
                if 'cursor' in locals():
                    cursor.close()
                if 'conn' in locals():
                    conn.close()
                return False
        else:
            logger.warning(f"Username not provided, database not updated for phone {phone_number}")
            return False
            
    except Exception as e:
        logger.error(f"Error upgrading account: {str(e)}")
        return False

def process_callback(callback_data):
    """Process the callback data from M-Pesa"""
    try:
        # Parse the callback data
        result_code = callback_data.get("Body", {}).get("stkCallback", {}).get("ResultCode")
        phone_number = callback_data.get("Body", {}).get("stkCallback", {}).get("Metadata", {}).get("PhoneNumber")
        username = callback_data.get("Body", {}).get("stkCallback", {}).get("Metadata", {}).get("Username")
        
        if result_code == 0:  # Successful payment
            try:
                # Connect to database
                conn = get_database_connection()
                cursor = conn.cursor()
                
                # Set subscription end date to 30 days from now
                end_date = datetime.now() + timedelta(days=30)
                
                # Update the subscription status in the users table
                update_query = """
                    UPDATE users 
                    SET subscription_status = 'premium',
                        subscription_end = %s
                    WHERE username = %s AND subscription_status = 'free'
                """
                cursor.execute(update_query, (end_date, username))
                
                # Check if update was successful
                if cursor.rowcount == 0:
                    logger.error(f"Failed to update subscription status for user {username}")
                    return {"status": "failed", "message": "Failed to update subscription status"}
                
                # Commit the changes
                conn.commit()
                
                # Update session state
                st.session_state.subscription_status = 'premium'
                st.session_state.account_status = "premium"
                st.session_state.payment_processed = True
                
                # Log the successful upgrade
                logger.info(f"User {username} with phone {phone_number} successfully upgraded to premium")
                
                return {"status": "success", "message": "Payment successful, account upgraded to premium"}
                
            except Exception as db_error:
                logger.error(f"Database update error: {str(db_error)}")
                return {"status": "failed", "message": "Database error occurred"}
            finally:
                if 'cursor' in locals():
                    cursor.close()
                if 'conn' in locals():
                    conn.close()
        else:
            return {"status": "failed", "message": "Payment failed or was cancelled"}
    except Exception as e:
        logger.error(f"Error processing callback: {str(e)}")
        return {"status": "error", "message": str(e)}

def show_payment_form(upgrade_subscription_callback):
    """Display the M-Pesa payment form for upgrading to premium"""
    st.subheader("Upgrade to Premium")
    
    with st.form("mpesa_payment_form"):
        st.subheader("Enter Payment Details")
        
        # Get customer's phone number
        phone_number = st.text_input(
            "Phone Number", 
            placeholder="07XXXXXXXX", 
            max_chars=10
        )
        
        # Payment amount
        amount = st.number_input(
            "Amount (KSH)", 
            min_value=1, 
            max_value=150000,  # M-Pesa transaction limit
            value=1  # Default payment amount
        )
        
        # Payment reference
        account_reference = st.text_input(
            "Payment For", 
            value="Premium Subscription",
            help="Description of what the payment is for"
        )
        
        # Submit button
        submitted = st.form_submit_button("Pay Now")
        
        if submitted:
            if not phone_number or len(phone_number) != 10 or not phone_number.startswith('07'):
                st.error("Invalid phone number. Use format 07XXXXXXXX")
                return
                
            # Show processing message
            with st.spinner("Processing payment request..."):
                # Initiate STK push
                success, message, checkout_request_id = initiate_stk_push(
                    phone_number=phone_number,
                    amount=amount,
                    account_reference=account_reference,
                    transaction_desc=f"Payment for {account_reference}"
                )
                
                if success:
                    # Show success message
                    st.success(message)
                    
                    # Store checkout request ID in session state
                    st.session_state.checkout_request_id = checkout_request_id
                    st.session_state.payment_initiated = True
                    
                    # Inform user to check their phone
                    st.info("Please enter your M-Pesa PIN on your phone to complete the transaction")
                    
                    # Wait and check payment status
                    max_retries = 5
                    for i in range(max_retries):
                        time.sleep(5)  # Wait 5 seconds between checks
                        
                        # Check the transaction status with M-Pesa
                        success, message = check_transaction_status(checkout_request_id)
                        
                        if success:
                            # Process successful payment
                            mock_callback = {
                                "Body": {
                                    "stkCallback": {
                                        "ResultCode": 0,
                                        "Metadata": {
                                            "PhoneNumber": phone_number,
                                            "Username": st.session_state.username  # Include username in callback
                                        }
                                    }
                                }
                            }
                            result = process_callback(mock_callback)
                            
                            if result["status"] == "success":
                                st.success(result["message"])
                                st.balloons()
                                
                                # Call the upgrade callback function
                                if upgrade_subscription_callback:
                                    upgrade_subscription_callback(st.session_state.username)
                                
                                time.sleep(2)
                                st.rerun()
                                break
                        else:
                            # If we're on the last attempt and payment is still not confirmed
                            if i == max_retries - 1:
                                st.error("Payment not confirmed. Your account was not upgraded.")
                            else:
                                st.info(f"Waiting for payment confirmation... ({i+1}/{max_retries})")
                else:
                    st.error(message)

def show_dashboard():
    """Display the premium dashboard"""
    st.title("Premium Dashboard")
    st.success("Welcome to your Premium Account!")
    
    # Add your dashboard content here
    st.write("Your premium features:")
    st.write("• Unlimited crop predictions")
    st.write("• Advanced analytics")
    st.write("• Detailed pest analysis")
    st.write("• Data export capabilities")
    
    if st.button("Logout"):
        # Reset session state
        st.session_state.account_status = "free"
        st.session_state.payment_processed = False
        st.session_state.subscription_status = 'free'
        st.experimental_rerun()

def main():
    st.title("M-Pesa Integration")
    
    # Simple page routing
    if st.session_state.account_status == "premium" and st.session_state.payment_processed:
        show_dashboard()
    else:
        show_payment_form(None)

if __name__ == "__main__":
    main()