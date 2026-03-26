import streamlit as st
import numpy as np
import pickle
import shap  # Import SHAP for explanation
from sklearn.preprocessing import LabelEncoder

# Define the feature columns (this should be defined before use)
feature_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                   'Torque [Nm]', 'Tool wear [min]', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11']

# Load the model and scaler before the app starts
try:
    model = pickle.load(open(r"c:\\Users\\91995\\OneDrive\\Desktop\\p\\model.pkl", "rb"))
    scaler = pickle.load(open(r"c:\\Users\\91995\\OneDrive\\Desktop\\p\\scaler.pkl", "rb"))
    st.write("Model and scaler loaded successfully!")
except Exception as e:
    st.write(f"Error loading model or scaler: {e}")
    model = None  # If model loading fails, set model to None
    scaler = None

# Streamlit app layout
st.title("Smart Predictive Maintenance Model")
st.write("This tool predicts machine failure based on input features")

# Input fields
type_ = st.selectbox('Select Machine Type', [1, 2, 3]) 
air_temp = st.slider('Air Temperature [K]', 290.0, 320.0, step=0.1)
process_temp = st.slider('Process Temperature [K]', 290.0, 320.0, step=0.1)
rotational_speed = st.number_input('Rotational Speed [rpm]', min_value=1000, max_value=3000, step=100)
torque = st.number_input('Torque [Nm]', min_value=0.0, max_value=100.0, step=1.0)
tool_wear = st.number_input('Tool Wear [min]', min_value=0, max_value=300, step=10)

# Additional features
feature7 = st.number_input('Feature 7', min_value=0.0, max_value=100.0, step=1.0)
feature8 = st.number_input('Feature 8', min_value=0.0, max_value=100.0, step=1.0)
feature9 = st.number_input('Feature 9', min_value=0.0, max_value=100.0, step=1.0)
feature10 = st.number_input('Feature 10', min_value=0.0, max_value=100.0, step=1.0)
feature11 = st.number_input('Feature 11', min_value=0.0, max_value=100.0, step=1.0)

# Label encode 'Type' feature (matching backend preprocessing)
label_encoder = LabelEncoder()
type_encoded = label_encoder.fit_transform([type_])[0]  # Assuming type_ is 1, 2, or 3

# Create input data as a numpy array (ensure it has 11 features)
input_data = np.array([[type_encoded, air_temp, process_temp, rotational_speed, torque, tool_wear, 
                        feature7, feature8, feature9, feature10, feature11]])

# Apply the same preprocessing (standardization) to input data
if scaler:
    input_data_scaled = scaler.transform(input_data)

# Display input data for debugging
st.write("Input data passed to the model:", input_data)

# Check if the model expects the same number of features
st.write("Expected features for the model:", feature_columns)

# Prediction on button click
if st.button('Predict Machine Failure'):
    # Show loading spinner
    with st.spinner('Making prediction...'):
        # Ensure model is loaded before making prediction
        if model and scaler:
            try:
                # Make prediction
                prediction = model.predict(input_data_scaled)
                st.write("Prediction result:", prediction)

                if prediction[0] == 1:
                    st.error('The machine is likely to fail.')
                else:
                    st.success('The machine is not likely to fail.')

                # Dynamically create the SHAP explainer based on model type
                explainer = shap.TreeExplainer(model)  # Assuming the model is a tree-based model
                shap_values = explainer.shap_values(input_data_scaled)
                
                # SHAP expects 2D input data, so we need to pass the first element
                shap_value = shap_values[0][0]  # Get SHAP values for the first prediction
                
                # Check if SHAP values are returned as expected
                if shap_values is not None:
                    st.write("SHAP values:", shap_value)
                    shap.summary_plot(shap_values, input_data_scaled)  # Plot SHAP values for the input data
                else:
                    st.write("Error: SHAP explanation does not contain expected values.")
                
            except Exception as e:
                st.write(f"Error during prediction: {e}")
        else:
            st.error("Model or scaler is not loaded. Please check the model file.")