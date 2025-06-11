# Import necessary libraries for the Flask API
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS # To handle Cross-Origin Resource Sharing for frontend
import joblib # For loading trained models and preprocessing objects
import pandas as pd
import numpy as np
import os # For managing file paths

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable CORS for all routes, allowing your frontend (even on different port/domain) to access this API
CORS(app)

# --- Configuration and File Paths for Saved Models ---
MODEL_DIR = 'trained_models' # Directory where trained models and scalers are saved
# IMPORTANT: Ensure these paths correctly point to the .pkl files saved by
# your 'wine_feature_engineering_and_training.py' script after tuning.
# If you used '_fast.pkl' suffix in the training script, update these paths accordingly.
MODEL_PATH = os.path.join(MODEL_DIR, 'decision_tree_tuned_model.pkl') # Check if this matches your saved model filename!
SCALER_PATH = os.path.join(MODEL_DIR, 'standard_scaler.pkl')
PCA_PATH = os.path.join(MODEL_DIR, 'pca_transformer.pkl')
KMEANS_PATH = os.path.join(MODEL_DIR, 'kmeans_clusterer.pkl')

# Global variables to hold the loaded model and preprocessing objects
loaded_model = None
loaded_scaler = None
loaded_pca = None
loaded_kmeans = None

# Define the expected input features (order matters!)
# These are the original features from your dataset, before any transformations.
EXPECTED_ORIGINAL_FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# --- Function to Load Models (Run once when app starts) ---
def load_all_artifacts():
    global loaded_model, loaded_scaler, loaded_pca, loaded_kmeans
    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        loaded_pca = joblib.load(PCA_PATH)
        loaded_kmeans = joblib.load(KMEANS_PATH)
        print("All machine learning artifacts loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e.filename} not found.")
        print("Please ensure you have run 'wine_feature_engineering_and_training.py' to save the models.")
        # Set all to None to indicate failure and prevent server from trying to use them
        loaded_model, loaded_scaler, loaded_pca, loaded_kmeans = None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading artifacts: {e}")
        loaded_model, loaded_scaler, loaded_pca, loaded_kmeans = None, None, None, None

# Load models when the Flask application starts
# This context ensures models are loaded only once at app startup
with app.app_context():
    load_all_artifacts()

# --- API Routes ---

@app.route('/')
def home():
    """Simple home route to confirm API is running."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the quality of wine based on physicochemical properties.
    Expects a JSON payload with 11 features.
    """
    # Check if models are loaded. If not, return an error.
    if loaded_model is None or loaded_scaler is None or loaded_pca is None or loaded_kmeans is None:
        return jsonify({"error": "ML models not loaded. Server issue. Please check server logs for loading errors."}), 500

    try:
        # Get JSON data from the request
        data = request.get_json(force=True) # force=True allows parsing even if content-type isn't strictly application/json

        # Validate if all expected features are present in the input JSON
        if not all(feature in data for feature in EXPECTED_ORIGINAL_FEATURES):
            missing_features = [f for f in EXPECTED_ORIGINAL_FEATURES if f not in data]
            return jsonify({
                "error": "Missing input features.",
                "details": f"Please provide values for all required wine attributes: {', '.join(missing_features)}."
            }), 400

        # Convert input dictionary to a Pandas DataFrame
        # IMPORTANT: Ensure column order matches the training data.
        # Creating a DataFrame from a list of dicts handles order if keys match.
        input_df = pd.DataFrame([data], columns=EXPECTED_ORIGINAL_FEATURES)

        # --- Apply Preprocessing Pipeline (MUST match training order) ---
        # 1. Scale the input data using the loaded StandardScaler
        input_scaled = loaded_scaler.transform(input_df)

        # 2. Apply PCA transformation using the loaded PCA transformer
        input_pca = loaded_pca.transform(input_scaled)

        # 3. Get K-Means cluster assignment using the loaded KMeans clusterer
        # Predict on the PCA-transformed input
        cluster_assignment = loaded_kmeans.predict(input_pca)[0] # Get the single cluster ID for the input sample

        # 4. Combine PCA components with the cluster assignment for the final model input
        # Create a DataFrame for PCA components, as the Decision Tree expects these as features
        pca_feature_names = [f'PC_{i+1}' for i in range(loaded_pca.n_components_)]
        processed_input_df = pd.DataFrame(input_pca, columns=pca_feature_names)
        processed_input_df['cluster'] = cluster_assignment # Add cluster as the last feature, matching training structure

        # --- Make Prediction ---
        # Use the loaded Decision Tree model to make the prediction
        prediction_numeric = loaded_model.predict(processed_input_df)[0]

        # Convert numeric prediction (0 or 1) to human-readable label
        quality_label = "Bad" if prediction_numeric == 0 else "Good"

        # Return the prediction and label as JSON
        return jsonify({
    "predicted_quality_numeric": int(prediction_numeric),
    "predicted_quality_label": quality_label,
    "message": "Prediction successful!",
 # <-- ADD THIS LINE
}), 200

    except Exception as e:
        # Catch any unexpected errors during the prediction process and return a generic error
        print(f"Error during prediction: {e}") # Log the error to your server's console
        return jsonify({"error": f"An internal server error occurred during prediction: {str(e)}"}), 500

# --- Main entry point to run the Flask app ---
if __name__ == '__main__':
    # Run the Flask app in debug mode (only for development!)
    # debug=True provides helpful error messages in development.
    # host='0.0.0.0' makes the server accessible from other devices on your local network
    # port=5000 is a common port for Flask apps
    app.run(debug=True, host='0.0.0.0', port=5001)
