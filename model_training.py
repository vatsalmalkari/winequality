import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving/loading models and preprocessing objects
import os # For managing file paths
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = 'winequality-red.csv'
MODEL_DIR = 'trained_models' # Directory to save trained models and scalers


# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)

X = df.drop('quality', axis=1) # Features
y = df['quality']             # Target

print(f"\nOriginal Features (X) shape: {X.shape}")
print(f"Original Target (y) shape: {y.shape}")

y_binary = (y >= 5).astype(int)

print("\nOriginal 'quality' distribution:")
print(y.value_counts().sort_index())
print("\nBinary 'quality' distribution (0: Bad, 1: Good):")
print(y_binary.value_counts())

# 2.3 Split Data into Training and Testing Sets
# Stratify ensures class balance in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

print(f"\nTraining set shape (X_train): {X_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}")
print(f"Training set binary quality distribution:\n{y_train.value_counts()}")
print(f"Testing set binary quality distribution:\n{y_test.value_counts()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames to preserve column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\nFeatures scaled successfully!")
print("First 5 rows of X_train_scaled_df:")
print(X_train_scaled_df.head())

print("\n" + "="*50 + "\n")
print("Starting Decision Tree Classifier training...")

# Initialize the Decision Tree Classifier
# random_state for reproducibility
# max_depth is a hyperparameter to prevent overfitting. A shallow tree (e.g., max_depth=5)
# is often a good starting point to improve generalization.
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=7)

# Train the model using the scaled training data
dt_classifier.fit(X_train_scaled_df, y_train)

print("Decision Tree Classifier training complete!")


# --- 4. Model Evaluation ---
print("\n" + "="*50 + "\n")
print("Evaluating the Decision Tree Classifier...")

y_pred = dt_classifier.predict(X_test_scaled_df)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)
print("\nConfusion Matrix:")
print(conf_matrix)

model_save_path = os.path.join(MODEL_DIR, 'decision_tree_model.pkl')
scaler_save_path = os.path.join(MODEL_DIR, 'standard_scaler.pkl')

joblib.dump(dt_classifier, model_save_path)
joblib.dump(scaler, scaler_save_path) # Save the fitted scaler

print(f"\nModel saved to: {model_save_path}")
print(f"Scaler saved to: {scaler_save_path}")