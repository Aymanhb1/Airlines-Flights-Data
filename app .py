import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os
import tempfile

# Define the Google Drive file ID and destination filename
# Make sure this ID is exactly correct from your Google Drive share link
google_drive_file_id = '1cfO1LkDZNc14qpZqI4kXLtwYSrBklVE3'  # Correct file ID from Google Drive link
model_filename = 'best_random_forest_model.joblib'

# Function to download the model file from Google Drive using gdown (more reliable)
@st.cache_resource
def load_model_from_drive(file_id, destination):
    try:
        if not os.path.exists(destination):
            st.info(f"Downloading model from Google Drive (ID: {file_id})...")
            
            # Use gdown which handles Google Drive downloads better
            try:
                import gdown
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, destination, quiet=False)
                
                # Verify the download
                actual_size = os.path.getsize(destination)
                st.success(f"Model downloaded successfully. Size: {actual_size:,} bytes")
                
                # Check if file is too small (likely HTML error page)
                if actual_size < 10000:  # Model should be much larger
                    with open(destination, 'r', errors='ignore') as f:
                        content_preview = f.read(200)
                        if '<html>' in content_preview.lower() or 'doctype html' in content_preview.lower():
                            st.error("Downloaded HTML error page instead of model file.")
                            st.error("This usually means the file is too large for direct download.")
                            st.error("Try using a smaller model or a different hosting service.")
                            os.remove(destination)  # Remove the bad file
                            return None
                            
            except ImportError:
                st.error("gdown library not found. Falling back to requests method...")
                # Fallback to requests method with better handling
                session = requests.Session()
                
                # First request to get the confirmation token
                response = session.get(f'https://drive.google.com/uc?export=download&id={file_id}', stream=True)
                
                # Look for confirmation token in the response
                token = None
                for line in response.iter_lines():
                    if b'confirm=' in line:
                        # Extract token from the line
                        line_str = line.decode('utf-8', errors='ignore')
                        if 'confirm=' in line_str:
                            token = line_str.split('confirm=')[1].split('&')[0].split('"')[0]
                            break
                
                if token:
                    st.info("Large file detected, using confirmation token...")
                    # Download with confirmation token
                    response = session.get(f'https://drive.google.com/uc?export=download&id={file_id}&confirm={token}', stream=True)
                else:
                    response = session.get(f'https://drive.google.com/uc?export=download&id={file_id}', stream=True)
                
                response.raise_for_status()
                
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                actual_size = os.path.getsize(destination)
                st.success(f"Model downloaded successfully. Size: {actual_size:,} bytes")
                        
        else:
            # Delete existing file if it's too small (likely corrupted)
            if os.path.getsize(destination) < 10000:
                st.warning("Existing file is too small, re-downloading...")
                os.remove(destination)
                return load_model_from_drive(file_id, destination)
            else:
                st.info("Model file already exists locally.")
        
        # Try to load the model
        st.info("Loading model...")
        model = joblib.load(destination)
        st.success("Model loaded successfully!")
        return model
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error downloading model: {e}")
        return None
    except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
        st.error(f"Model loading error (possibly memory issue): {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {type(e).__name__}: {e}")
        # Try to show file info if it exists
        if os.path.exists(destination):
            file_size = os.path.getsize(destination)
            st.info(f"File exists with size: {file_size:,} bytes")
            
            # Show first few bytes to debug
            try:
                with open(destination, 'rb') as f:
                    first_bytes = f.read(50)
                st.info(f"First bytes: {first_bytes}")
            except:
                pass
        return None

# Load the trained model from Google Drive
st.title("Flight Price Prediction")

# Show loading status
with st.spinner("Loading prediction model..."):
    model = load_model_from_drive(google_drive_file_id, model_filename)

if model is None:
    st.error("Failed to load the model. Please check:")
    st.error("1. The Google Drive file ID is correct")
    st.error("2. The file sharing is set to 'Anyone with the link can view'")
    st.error("3. Your internet connection is stable")
    st.stop()

# Assuming you have the list of columns the model was trained on
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
categorical_info = {
    'airline': ['AirAsia', 'Air_India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara'],
    'source_city': ['Delhi'],
    'departure_time': ['Afternoon', 'Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night'],
    'stops': ['one', 'two_or_more', 'zero'],
    'arrival_time': ['Afternoon', 'Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night'],
    'destination_city': ['Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 'Mumbai'],
    'class': ['Business', 'Economy']
}

st.sidebar.header("Input Features")

# Create input fields for the features
duration = st.sidebar.number_input("Duration (hours)", min_value=0.0, value=5.0, step=0.5)
days_left = st.sidebar.number_input("Days Left", min_value=0, value=10, step=1)

# Dropdowns for categorical features
airline = st.sidebar.selectbox("Airline", categorical_info['airline'])
source_city = st.sidebar.selectbox("Source City", categorical_info['source_city'])
departure_time = st.sidebar.selectbox("Departure Time", categorical_info['departure_time'])
stops = st.sidebar.selectbox("Stops", categorical_info['stops'])
arrival_time = st.sidebar.selectbox("Arrival Time", categorical_info['arrival_time'])
destination_city = st.sidebar.selectbox("Destination City", categorical_info['destination_city'])
flight_class = st.sidebar.selectbox("Class", categorical_info['class'])

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

# Apply one-hot encoding to the input data
input_encoded = pd.get_dummies(input_df, columns=categorical_info.keys(), drop_first=True)

# Ensure all columns from training data are present in the input data
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match the training data
input_processed = input_encoded[model_features]

# Display current input
st.subheader("Current Input:")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Duration:** {duration} hours")
    st.write(f"**Days Left:** {days_left}")
    st.write(f"**Airline:** {airline}")
    st.write(f"**Source:** {source_city}")
    
with col2:
    st.write(f"**Departure:** {departure_time}")
    st.write(f"**Stops:** {stops}")
    st.write(f"**Arrival:** {arrival_time}")
    st.write(f"**Destination:** {destination_city}")
    st.write(f"**Class:** {flight_class}")

# Make prediction when the button is clicked
if st.button("🔮 Predict Flight Price", use_container_width=True):
    try:
        with st.spinner("Predicting price..."):
            prediction = model.predict(input_processed)
            
        st.success("Prediction completed!")
        st.subheader("💰 Predicted Flight Price:")
        
        # Display prediction with nice formatting
        price = prediction[0]
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #1f77b4; margin: 0;">₹{price:,.0f}</h2>
            <p style="margin: 5px 0 0 0; color: #666;">Estimated flight price</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check that all input values are valid.")
