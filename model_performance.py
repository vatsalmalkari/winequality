# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For loading models and preprocessing objects
import os # For managing file paths

# --- Configuration and File Paths ---
DATA_FILE = 'winequality-red.csv'
MODEL_DIR = 'trained_models' # Directory where trained models and scalers are saved
MODEL_PATH = os.path.join(MODEL_DIR, 'decision_tree_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'standard_scaler.pkl')

# --- 1. Load the Dataset ---
print(f"Loading dataset from {DATA_FILE}...")
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully!")
else:
    print(f"Error: The data file '{DATA_FILE}' was not found. Please ensure it's in the project root.")
    exit() # Exit if the data file is not found


# --- 2. Data Preprocessing (Identical to Training Script to prepare X_test, y_test) ---

# 2.1 Feature and Target Separation
X = df.drop('quality', axis=1) # Features
y = df['quality']             # Target

# 2.2 Transform Target Variable (Quality) into Binary Classification
# 'Good' wine (quality >= 7) -> 1
# 'Bad' wine (quality < 7)  -> 0
y_binary = (y >= 7).astype(int)

# 2.3 Split Data into Training and Testing Sets
# IMPORTANT: Use the exact same random_state and stratify settings as in your training script
# This ensures that X_test and y_test are IDENTICAL to what was used for initial evaluation during training.
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

print(f"\nRe-created X_test shape: {X_test.shape}")
print(f"Re-created y_test shape: {y_test.shape}")


# --- 3. Load the Trained Model and Scaler ---
print("\n" + "="*50 + "\n")
print(f"Loading trained model from {MODEL_PATH} and scaler from {SCALER_PATH}...")
try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model or Scaler file not found. Ensure '{MODEL_DIR}' directory and files exist.")
    exit()
except Exception as e:
    print(f"An error occurred while loading files: {e}")
    exit()


# --- 4. Prepare Test Data for Prediction ---
# Scale X_test using the loaded scaler
# IMPORTANT: Only transform X_test; do NOT fit the scaler again.
X_test_scaled = loaded_scaler.transform(X_test)

# Convert scaled array back to DataFrame if the model expects it, or if using column names
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\nTest data scaled using the loaded scaler.")
print("First 5 rows of scaled X_test_df:")
print(X_test_scaled_df.head())


# --- 5. Make Predictions using the Loaded Model ---
print("\n" + "="*50 + "\n")
print("Making predictions on the test set using the loaded model...")
y_pred_loaded = loaded_model.predict(X_test_scaled_df)
print("Predictions made.")


# --- 6. Evaluate Performance ---
print("\n" + "="*50 + "\n")
print("Evaluating the Loaded Model's Performance:")

accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
class_report_loaded = classification_report(y_test, y_pred_loaded)
conf_matrix_loaded = confusion_matrix(y_test, y_pred_loaded)

print(f"Accuracy (Loaded Model): {accuracy_loaded:.4f}")
print("\nClassification Report (Loaded Model):")
print(class_report_loaded)
print("\nConfusion Matrix (Loaded Model):")
print(conf_matrix_loaded)

print("\nModel evaluation script finished. Performance should match training script results.")
