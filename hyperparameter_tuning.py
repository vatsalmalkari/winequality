# Import necessary libraries (ensure GridSearchCV is imported)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV # <-- Ensure GridSearchCV is imported
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier # Removed GridSearchCV as it's a bottleneck for speed
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving/loading models and preprocessing objects
import os # For managing file paths
from yellowbrick.cluster import KElbowVisualizer # Commented out for speed, as it involves multiple KMeans runs

# --- Configuration and File Paths ---
DATA_FILE = 'winequality-red.csv'
MODEL_DIR = 'trained_models' # Directory to save trained models and preprocessing objects

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


# --- 1. Load the Dataset ---
print(f"Loading dataset from {DATA_FILE}...")
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully! (First 5 rows displayed below)")
    print(df.head())
else:
    print(f"Error: The data file '{DATA_FILE}' was not found. Please ensure it's in the project root.")
    exit() # Exit if the data file is not found

print("\n" + "="*70 + "\n") # Separator


# --- 2. Initial Data Preprocessing ---

# 2.1 Handle Nulls/Missing Values (Confirm no missing values)
print("Checking for missing values:")
print(df.isnull().sum())
print("\nNo missing values found, proceeding with preprocessing.")


# 2.2 Feature and Target Separation
X = df.drop('quality', axis=1) # Features (all physicochemical properties)
y = df['quality']             # Target (wine quality score)

print(f"\nOriginal Features (X) shape: {X.shape}")
print(f"Original Target (y) shape: {y.shape}")

# 2.3 Transform Target Variable (Quality) into Binary Classification
# 'Good' wine (quality >= 7) -> 1
# 'Bad' wine (quality < 7)  -> 0
y_binary = (y >= 7).astype(int)

print("\nOriginal 'quality' distribution:")
print(y.value_counts().sort_index())
print("\nBinary 'quality' distribution (0: Bad, 1: Good):")
print(y_binary.value_counts())


# 2.4 Split Data into Training and Testing Sets
# test_size=0.2 means 20% for testing, 80% for training.
# random_state ensures reproducibility of the split.
# stratify=y_binary is crucial due to class imbalance in 'quality',
# ensuring proportional representation of 'Good' and 'Bad' wines in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

print(f"\nTraining set shape (X_train): {X_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}")
print(f"Training set binary quality distribution:\n{y_train.value_counts()}")
print(f"Testing set binary quality distribution:\n{y_test.value_counts()}")


# --- 3. Feature Scaling (StandardScaler) ---
# Scale features to have zero mean and unit variance.
# ESSENTIAL for PCA and K-Means, which are distance-based algorithms.
# Fit scaler ONLY on X_train to prevent data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames to keep column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\nFeatures scaled successfully!")
print("First 5 rows of X_train_scaled_df:")
print(X_train_scaled_df.head())

print("\n" + "="*70 + "\n") # Separator


# --- 4. Feature Engineering ---
print("Starting Feature Engineering with PCA and K-Means...")

# --- 4.1 Principal Component Analysis (PCA) ---
# PCA reduces dimensionality while retaining most of the variance.
# It can help simplify the feature space and sometimes improve model performance
# by reducing noise or multicollinearity.

# Determine optimal number of components for PCA by plotting explained variance
# This plot takes time, but it's important for initial understanding.
pca_full = PCA().fit(X_train_scaled) # Fit PCA on scaled training data
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance Ratio by Number of Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle=':', label='95% Variance Explained')
plt.legend()
plt.show()

# Apply PCA with a chosen number of components (e.g., to explain 95% of variance)
pca = PCA(n_components=0.95, random_state=42) # Still aiming for 95% variance
X_train_pca = pca.fit_transform(X_train_scaled) # Fit PCA on scaled training data
X_test_pca = pca.transform(X_test_scaled)     # Transform test data using the fitted PCA

print(f"\nOriginal number of features: {X_train.shape[1]}")
print(f"Number of PCA components selected (to explain 95% variance): {pca.n_components_}")
print(f"Cumulative variance explained by PCA: {pca.explained_variance_ratio_.sum():.4f}")

# Convert PCA results to DataFrame for easier handling and column naming
pca_feature_names = [f'PC_{i+1}' for i in range(pca.n_components_)]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_feature_names, index=X_train.index)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_feature_names, index=X_test.index)

print("\nFirst 5 rows of PCA-transformed training data:")
print(X_train_pca_df.head())


# --- 4.2 K-Means Clustering ---
# K-Means can find natural groupings in your data. The cluster assignment can then
# be used as a new categorical feature for your Decision Tree.
# We'll apply K-Means on the PCA-transformed data to work in a reduced, uncorrelated space.

# Determine optimal number of clusters (k) using the Elbow Method (re-enabled for guidance)
# Use PCA-transformed data for K-Means
model_kmeans = KMeans(random_state=42, n_init=10) # Using default n_init for better results here
visualizer = KElbowVisualizer(model_kmeans, k=(2, 10), metric='distortion', timings=False) # Distortion is WCSS
visualizer.fit(X_train_pca) # Fit K-Means visualizer on PCA-transformed training data
visualizer.show() # Display the elbow plot

# Based on the Elbow plot, choose a suitable 'k'. Let's pick k=3 as a common choice from past runs.
k_chosen = 3 # Adjust based on your Elbow Method analysis

kmeans = KMeans(n_clusters=k_chosen, random_state=42, n_init=10) # Using default n_init for better results here
kmeans.fit(X_train_pca) # Fit K-Means on PCA-transformed training data

# Get cluster assignments for both training and testing sets
train_clusters = kmeans.predict(X_train_pca)
test_clusters = kmeans.predict(X_test_pca)

# Add cluster assignments as a new feature to the PCA-transformed DataFrames
X_train_processed = X_train_pca_df.copy() # Start with PCA features
X_train_processed['cluster'] = train_clusters

X_test_processed = X_test_pca_df.copy() # Start with PCA features
X_test_processed['cluster'] = test_clusters

print(f"\nK-Means clustering complete with {k_chosen} clusters.")
print("Cluster assignments added as a new feature.")
print("First 5 rows of final processed training data (PCA + Cluster):")
print(X_train_processed.head())

print("\n" + "="*70 + "\n") # Separator


# --- 5. Model Training: Decision Tree Classifier with Hyperparameter Tuning (GridSearchCV) ---
print("Starting Decision Tree Classifier training with hyperparameter tuning (GridSearchCV)...")

# Initialize the Decision Tree Classifier
# Add class_weight='balanced' to handle the class imbalance in the target variable.
# This gives more importance to the minority class (Bad wine) during training.
dt_classifier = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Define a grid of hyperparameters to search
# You can expand or shrink this grid based on computational resources and desired search thoroughness.
param_grid = {
    'max_depth': [3, 5, 7, 10, None], # None means no limit on depth
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy'] # Gini impurity or information gain
}

# Use GridSearchCV for exhaustive hyperparameter search
# cv=5 means 5-fold cross-validation. This might take some time.
# scoring='f1_weighted' is crucial for imbalanced datasets, as it provides a balanced
# measure of precision and recall for both classes, weighted by their support.
# n_jobs=-1 uses all available CPU cores for faster computation.
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
grid_search.fit(X_train_processed, y_train)

best_dt_classifier = grid_search.best_estimator_

print("\nDecision Tree Classifier training with hyperparameter tuning complete!")
print(f"Best hyperparameters found: {grid_search.best_params_}")


# --- 6. Model Evaluation ---
print("\n" + "="*70 + "\n")
print("Evaluating the Best Decision Tree Classifier with engineered features...")

# Make predictions on the processed test set using the best model from GridSearchCV
y_pred_engineered = best_dt_classifier.predict(X_test_processed)

# Calculate and print evaluation metrics
accuracy_engineered = accuracy_score(y_test, y_pred_engineered)
class_report_engineered = classification_report(y_test, y_pred_engineered)
conf_matrix_engineered = confusion_matrix(y_test, y_pred_engineered)

print(f"Accuracy (Engineered Features Model): {accuracy_engineered:.4f}")
print("\nClassification Report (Engineered Features Model):")
print(class_report_engineered)
print("\nConfusion Matrix (Engineered Features Model):")
print(conf_matrix_engineered)

# Interpretation of Confusion Matrix (re-iterated):
# [[TN, FP],
#  [FN, TP]]
# TN (True Negatives): Correctly predicted 'Bad' wine (0)
# FP (False Positives): Incorrectly predicted 'Good' wine (1) when it was 'Bad' (Type I error)
# FN (False Negatives): Incorrectly predicted 'Bad' wine (0) when it was 'Good' (Type II error)
# TP (True Positives): Correctly predicted 'Good' wine (1)


# --- 7. Save All Trained Artifacts ---
# Save the final trained Decision Tree model, StandardScaler, PCA, and KMeans objects.
# Use distinct filenames to differentiate from the "fast" version.
model_save_path = os.path.join(MODEL_DIR, 'decision_tree_tuned_model.pkl')
scaler_save_path = os.path.join(MODEL_DIR, 'standard_scaler.pkl') # This remains the same
pca_save_path = os.path.join(MODEL_DIR, 'pca_transformer.pkl')     # This remains the same
kmeans_save_path = os.path.join(MODEL_DIR, 'kmeans_clusterer.pkl') # This remains the same

joblib.dump(best_dt_classifier, model_save_path)
joblib.dump(scaler, scaler_save_path)
joblib.dump(pca, pca_save_path)
joblib.dump(kmeans, kmeans_save_path)

print(f"\nAll trained artifacts saved to '{MODEL_DIR}':")
print(f"  - Model: {model_save_path}")
print(f"  - Scaler: {scaler_save_path}")
print(f"  - PCA Transformer: {pca_save_path}")
print(f"  - KMeans Clusterer: {kmeans_save_path}")

print("\n" + "="*70 + "\n")
print("Feature engineering and hyperparameter-tuned model training script finished.")
print("The saved model should be more robust and balanced.")