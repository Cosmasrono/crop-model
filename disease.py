import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cv2
import streamlit as st
from PIL import Image
import joblib

# Add crop type verification
CROP_TYPES = {
    "Tomato": ["Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Late_Blight", "Tomato_Healthy"],
    "Corn": ["Corn_Cercospora_Leaf_Spot", "Corn_Common_Rust", "Corn_Northern_Leaf_Blight", "Corn_Healthy"],
    "Pepper": ["Pepper_Bacterial_Spot", "Pepper_Early_Blight", "Pepper_Healthy"]
}

# Update disease descriptions
DISEASE_DESCRIPTIONS = {
    "Corn_Cercospora_Leaf_Spot": {
        "description": "A fungal disease that causes small, circular to oval spots on corn leaves.",
        "symptoms": [
            "Gray to tan circular spots with reddish-brown borders",
            "Lesions may merge to form larger affected areas",
            "Spots visible on both upper and lower leaf surfaces"
        ],
        "causes": "Caused by the fungus Cercospora zeae-maydis, favored by warm, humid conditions.",
        "crop_type": "Corn"
    },
    "Corn_Common_Rust": {
        "description": "A common fungal disease affecting corn plants worldwide.",
        "symptoms": [
            "Small, circular to elongated brown pustules on leaves",
            "Pustules appear on both leaf surfaces",
            "Rust-colored spores that can be rubbed off"
        ],
        "causes": "Caused by the fungus Puccinia sorghi, spreads rapidly in cool, moist conditions."
    },
    "Corn_Northern_Leaf_Blight": {
        "description": "A serious fungal disease that can cause significant yield loss in corn.",
        "symptoms": [
            "Long, cigar-shaped gray-green to tan lesions",
            "Lesions begin on lower leaves and move upward",
            "Dark areas of fungal sporulation in mature lesions"
        ],
        "causes": "Caused by Exserohilum turcicum, favored by moderate temperatures and humid conditions."
    },
    "Pepper_Bacterial_Spot": {
        "description": "A bacterial disease affecting pepper plants, causing spots on leaves and fruits.",
        "symptoms": [
            "Small, dark, raised spots on leaves and fruits",
            "Spots may have yellow halos",
            "Leaves may become yellow and fall off"
        ],
        "causes": "Caused by Xanthomonas bacteria, spreads through water splash and humid conditions."
    },
    "Tomato_Early_Blight": {
        "description": "A fungal disease that affects tomato plants early in the growing season.",
        "symptoms": [
            "Dark brown spots with concentric rings",
            "Yellow areas around the spots",
            "Lower leaves are affected first"
        ],
        "causes": "Caused by Alternaria solani, common in warm, humid conditions."
    },
    "Healthy": {
        "description": "Plant showing no signs of disease.",
        "symptoms": [
            "Vibrant green color",
            "No spots or lesions",
            "Normal growth pattern"
        ],
        "causes": "Good agricultural practices and proper plant care."
    }
}

def extract_features_from_image(image):
    """Extract comprehensive image features for disease detection"""
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize image to a standard size
    img_resized = cv2.resize(img_cv, (224, 224))
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    
    features = []
    
    # 1. Color features from multiple color spaces
    for color_space in [img_resized, hsv, lab]:
        for channel in cv2.split(color_space):
            features.extend([
                np.mean(channel),      # Average color
                np.std(channel),       # Color variation
                np.percentile(channel, 25),  # First quartile
                np.percentile(channel, 75)   # Third quartile
            ])
    
    # 2. Texture features
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # GLCM (Gray-Level Co-Occurrence Matrix) features
    def compute_glcm_features(img):
        glcm = cv2.calcHist([img], [0], None, [8], [0, 256])
        stats = [np.mean(glcm), np.std(glcm), 
                np.max(glcm), np.min(glcm)]
        return stats
    
    features.extend(compute_glcm_features(gray))
    
    # 3. Edge features using different methods
    edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    for edge_img in [edges_sobel_x, edges_sobel_y, edges_laplacian]:
        features.extend([
            np.mean(np.abs(edge_img)),  # Average edge strength
            np.std(edge_img),           # Edge variation
            np.percentile(np.abs(edge_img), 90)  # Strong edges
        ])
    
    # 4. Disease-specific color ratios
    b, g, r = cv2.split(img_resized)
    features.extend([
        np.mean(g) / (np.mean(r) + 1e-6),  # Green/Red ratio (vegetation health)
        np.mean(b) / (np.mean(g) + 1e-6),  # Blue/Green ratio
        np.std(g) / (np.std(r) + 1e-6)     # Color variation ratio
    ])
    
    return np.array(features)

def train_model():
    """Train an improved model using comprehensive image features"""
    try:
        # Load the dataset
        data = pd.read_csv("data/crop-disease.csv")
        
        # Load and process all training images
        X = []
        y = []
        
        # Group by Image_ID to get unique images
        unique_images = data.groupby('Image_ID').first().reset_index()
        
        for _, row in unique_images.iterrows():
            try:
                # Load image (you'll need to modify this path according to your dataset structure)
                image_path = f"data/images/{row['Image_ID']}"
                image = Image.open(image_path)
                
                # Extract features
                features = extract_features_from_image(image)
                X.append(features)
                y.append(row['class'])
                
            except Exception as e:
                st.warning(f"Skipping image {row['Image_ID']}: {str(e)}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        # Create and train the model with more trees and balanced class weights
        model = RandomForestClassifier(
            n_estimators=200,          # More trees
            max_depth=20,              # Control overfitting
            min_samples_split=5,       # Minimum samples to split a node
            min_samples_leaf=2,        # Minimum samples in a leaf
            class_weight='balanced',   # Handle class imbalance
            random_state=42
        )
        model.fit(X, y)
        
        # Save the model
        joblib.dump(model, 'crop_disease_model.pkl')
        
        # Save class names
        class_names = list(set(y))
        joblib.dump(class_names, 'class_names.pkl')
        
        return model, class_names
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return None, None

def predict_disease(model, image, class_names):
    """Make prediction using comprehensive image features"""
    # Extract features from the image
    features = extract_features_from_image(image)
    
    # Reshape features for prediction
    features_reshaped = features.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_reshaped)
    probabilities = model.predict_proba(features_reshaped)
    confidence = np.max(probabilities[0])
    
    return prediction[0], confidence

def get_crop_type(disease_name):
    """Get the crop type from the disease name"""
    for crop, diseases in CROP_TYPES.items():
        if any(disease_name.startswith(disease.split('_')[0]) for disease in diseases):
            return crop
    return None

def verify_prediction(prediction, confidence):
    """Verify if the prediction makes sense based on crop type and confidence"""
    crop_type = get_crop_type(prediction)
    
    if confidence < 0.20:  # If confidence is less than 20%
        return False, f"Low confidence ({confidence:.1%}) in prediction. Please provide a clearer image."
    
    if crop_type is None:
        return False, "Unable to determine crop type from the image."
        
    return True, f"Detected {crop_type} plant with {prediction} disease"

def display_disease_info(disease_name, confidence):
    """Enhanced disease information display with verification"""
    is_valid, message = verify_prediction(disease_name, confidence)
    
    if not is_valid:
        st.warning(message)
        st.info("Please ensure:")
        st.write("• The image is clear and well-lit")
        st.write("• The affected area is clearly visible")
        st.write("• The image shows the plant's leaves or affected parts")
        return
    
    if disease_name in DISEASE_DESCRIPTIONS:
        disease_info = DISEASE_DESCRIPTIONS[disease_name]
        
        st.subheader("Disease Information:")
        st.write(f"**Crop Type:** {get_crop_type(disease_name)}")
        st.write(f"**Description:** {disease_info['description']}")
        
        st.write("**Symptoms:**")
        for symptom in disease_info['symptoms']:
            st.write(f"• {symptom}")
            
        st.write(f"**Cause:** {disease_info['causes']}")
        
        if confidence < 0.50:
            st.warning("⚠️ Note: Prediction confidence is low. Consider taking another image or consulting an expert.")
    else:
        st.write("Detailed information not available for this specific disease.")

def main():
    st.title("Crop Disease Detection System")
    
    tab1, tab2 = st.tabs(["Train Model", "Predict Disease"])
    
    with tab1:
        st.header("Train Model")
        if st.button("Train New Model"):
            with st.spinner("Training model..."):
                model, class_names = train_model()
                st.success("Model trained successfully!")
    
    with tab2:
        st.header("Predict Disease")
        
        try:
            model = joblib.load('crop_disease_model.pkl')
            class_names = joblib.load('class_names.pkl')
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error("Model files not found! Please train the model first using the 'Train Model' tab.")
            return
        
        uploaded_file = st.file_uploader("Choose a crop image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            try:
                # Make prediction
                prediction, confidence = predict_disease(model, image, class_names)
                
                # Display results
                st.success("Prediction Results:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Disease", prediction)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Display disease information
                display_disease_info(prediction, confidence)
                
                # Display treatment recommendations if confidence is high enough
                if confidence >= 0.20:
                    st.subheader("Recommended Treatment:")
                    if "healthy" in prediction.lower():
                        st.success("Your crop appears to be healthy! Continue with regular maintenance.")
                    else:
                        st.warning("Disease detected! Here are some treatment recommendations:")
                        
                        treatments = {
                            "Bacterial_Spot": [
                                "Apply copper-based fungicides",
                                "Ensure proper plant spacing",
                                "Avoid overhead irrigation"
                            ],
                            "Early_Blight": [
                                "Remove infected leaves",
                                "Apply appropriate fungicide",
                                "Maintain proper plant nutrition"
                            ],
                            "Late_Blight": [
                                "Apply fungicide immediately",
                                "Remove infected plants",
                                "Improve air circulation"
                            ],
                            "Leaf_Curl": [
                                "Use disease-resistant varieties",
                                "Apply appropriate insecticides",
                                "Remove infected leaves"
                            ],
                            "Mosaic_Virus": [
                                "Remove infected plants",
                                "Control insect vectors",
                                "Use virus-resistant varieties"
                            ]
                        }
                        
                        # Display relevant treatments
                        for disease, treatment_list in treatments.items():
                            if disease.lower() in prediction.lower():
                                for treatment in treatment_list:
                                    st.write(f"• {treatment}")
                        
                        # Additional care instructions
                        st.info("General Care Instructions:")
                        st.write("• Monitor the affected plants regularly")
                        st.write("• Ensure proper watering and drainage")
                        st.write("• Maintain good garden hygiene")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("Please make sure the image is clear and in the correct format.")

if __name__ == "__main__":
    main()