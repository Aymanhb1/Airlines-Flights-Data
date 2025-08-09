
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os

# Define the Google Drive file ID and destination filename
google_drive_file_id = '1cfO1LkDZNc14qpZqI4kXLtwYSrBklVE3' # Your Google Drive File ID
model_filename = 'best_random_forest_model.joblib'
download_url = f'https://drive.google.com/uc?export=download&id={google_drive_file_id}'

# Function to download the model file from Google Drive
@st.cache_resource # Cache the model loading to avoid re-downloading on every interaction
def load_model_from_drive(file_id, destination):
    if not os.path.exists(destination):
        st.info(f"Downloading model from Google Drive (ID: {file_id})...")
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(download_url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded successfully.")
    else:
        st.info("Model file already exists locally.")
    return joblib.load(destination)

# Load the trained model from Google Drive
try:
    model = load_model_from_drive(google_drive_file_id, model_filename)
except Exception as e:
    st.error(f"Failed to load the model from Google Drive: {e}")
    st.stop()


# Assuming you have the list of columns the model was trained on
# Replace with the actual columns from your X_train_cleaned after one-hot encoding
# You can get this list from X_train_cleaned.columns
model_features = [
    'duration', 'days_left', 'airline_Air_India', 'airline_GO_FIRST',
    'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara',
    'departure_time_Early_Morning', 'departure_time_Evening',
    'departure_time_Late_Night', 'departure_time_Morning',
    'departure_time_Night', 'stops_two_or_more', 'stops_zero',
    'arrival_time_Early_Morning', 'arrival_time_Evening',
    'arrival_time_Late_Night', 'arrival_time_Morning', 'arrival_time_Night',
    'destination_city_Chennai', 'destination_city_Hyderabad',
    'destination_city_Kolkata', 'destination_city_Mumbai'
]

# Define the original categorical column names and their possible values
# This is needed for one-hot encoding user input
categorical_info = {
    'airline': ['AirAsia', 'Air_India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara'], # Example values, replace with actual unique values
    'source_city': ['Delhi'], # Based on your data, only Delhi
    'departure_time': ['Afternoon', 'Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night'], # Example values
    'stops': ['one', 'two_or_more', 'zero'], # Example values
    'arrival_time': ['Afternoon', 'Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night'], # Example values
    'destination_city': ['Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 'Mumbai'], # Example values
    'class': ['Business', 'Economy'] # Example values
}


st.title("Flight Price Prediction")

st.sidebar.header("Input Features")

# Create input fields for the features
duration = st.sidebar.number_input("Duration", min_value=0.0, value=5.0)
days_left = st.sidebar.number_input("Days Left", min_value=0, value=10)

# Dropdowns for categorical features - ensure these match your training data categories
airline = st.sidebar.selectbox("Airline", categorical_info['airline'])
source_city = st.sidebar.selectbox("Source City", categorical_info['source_city'])
departure_time = st.sidebar.selectbox("Departure Time", categorical_info['departure_time'])
stops = st.sidebar.selectbox("Stops", categorical_info['stops'])
arrival_time = st.sidebar.selectbox("Arrival Time", categorical_info['arrival_time'])
destination_city = st.sidebar.selectbox("Destination City", categorical_info['destination_city'])
flight_class = st.sidebar.selectbox("Class", categorical_info['class']) # Renamed to avoid conflict with class keyword

# Create a dictionary from the input values
input_data = {
    'duration': duration,
    'days_left': days_left,
    'airline': airline,
    'source_city': source_city,
    'departure_time': departure_time,
    'stops': stops,
    'arrival_time': arrival_time,
    'destination_city': destination_city,
    'class': flight_class
}

# Convert input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# --- Preprocessing for new input data ---
# Apply one-hot encoding to the input data, aligning with training data columns
input_encoded = pd.get_dummies(input_df, columns=categorical_info.keys(), drop_first=True)

# Ensure all columns from training data are present in the input data, fill missing with 0
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match the training data - crucial for consistent predictions
input_processed = input_encoded[model_features]


# Make prediction when the button is clicked
if st.sidebar.button("Predict Price"):
    try:
        prediction = model.predict(input_processed)
        st.subheader("Predicted Flight Price:")
        st.write(f"The predicted price is: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


import os
import joblib

try:
    # Check if file exists and its size
    if os.path.exists('best_random_forest_model.joblib'):
        file_size = os.path.getsize('best_random_forest_model.joblib')
        st.write(f"File exists, size: {file_size} bytes")
        
        # Try to load with more detailed error info
        model = joblib.load('best_random_forest_model.joblib')
        st.success("Model loaded successfully!")
    else:
        st.error("Model file not found after download")
        
except Exception as e:
    st.error(f"Detailed error: {str(e)}")
    st.error(f"Error type: {type(e).__name__}")
