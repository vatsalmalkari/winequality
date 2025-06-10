import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving/loading models and preprocessing objects
import os # For managing file paths
from yellowbrick.cluster import KElbowVisualizer # For Elbow Method visualization (install with: pip install yellowbrick)
import time
DATA_FILE = 'winequality-red.csv'
MODEL_DIR = 'trained_models' # Directory to save trained models and preprocessing objects

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
print(df.head())

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

print("Starting Feature Engineering with PCA and K-Means...")


start = time.time()

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

pca = PCA(n_components=0.95, random_state=42) # Still aiming for 95% variance
X_train_pca = pca.fit_transform(X_train_scaled) # Fit PCA on scaled training data
X_test_pca = pca.transform(X_test_scaled)     # Transform test data using the fitted PCA

print(f"\nOriginal number of features: {X_train.shape[1]}")
print(f"Number of PCA components selected (to explain 95% variance): {pca.n_components_}")
print(f"Cumulative variance explained by PCA: {pca.explained_variance_ratio_.sum():.4f}")
pca_feature_names = [f'PC_{i+1}' for i in range(pca.n_components_)]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_feature_names, index=X_train.index)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_feature_names, index=X_test.index)

print("\nFirst 5 rows of PCA-transformed training data:")
print(X_train_pca_df.head())

k_chosen = 3 # Choose your desired number of clusters directly

kmeans = KMeans(n_clusters=k_chosen, random_state=42, n_init=1) # Minimal n_init for max speed
kmeans.fit(X_train_pca) # Fit K-Means on PCA-transformed training data

# Get cluster assignments for both training and testing sets
train_clusters = kmeans.predict(X_train_pca)
test_clusters = kmeans.predict(X_test_pca)

# Add cluster assignments as a new feature to the PCA-transformed DataFrames
X_train_processed = X_train_pca_df.copy() # Start with PCA features
X_train_processed['cluster'] = train_clusters

X_test_processed = X_test_pca_df.copy() # Start with PCA features
X_test_processed['cluster'] = test_clusters

print(f"\nK-Means clustering complete with {k_chosen} clusters (n_init=1 for speed).")
print("Cluster assignments added as a new feature.")
print("First 5 rows of final processed training data (PCA + Cluster):")
print(X_train_processed.head())

print("\n" + "="*70 + "\n") # Separator

print("Starting Decision Tree Classifier training with engineered features (optimized for speed)...")

# Initialize the Decision Tree Classifier with predefined hyperparameters (no GridSearchCV)
# This is the fastest way to train a Decision Tree, but might not yield the optimal model.
best_dt_classifier = DecisionTreeClassifier(
    random_state=42,
    max_depth=7,  # A reasonable depth to prevent severe overfitting but allow complexity
    min_samples_leaf=5 # Ensures each leaf has at least 5 samples, preventing extreme overfitting
)

best_dt_classifier.fit(X_train_processed, y_train)

print(f"Best hyperparameters used: {best_dt_classifier.get_params()}")

y_pred_engineered = best_dt_classifier.predict(X_test_processed)

# Calculate and print evaluation metrics
accuracy_engineered = accuracy_score(y_test, y_pred_engineered)
class_report_engineered = classification_report(y_test, y_pred_engineered)
conf_matrix_engineered = confusion_matrix(y_test, y_pred_engineered)

print(f"Accuracy (Engineered Features Model - Optimized for Speed): {accuracy_engineered:.4f}")
print("\nClassification Report (Engineered Features Model - Optimized for Speed):")
print(class_report_engineered)
print("\nConfusion Matrix (Engineered Features Model - Optimized for Speed):")
print(conf_matrix_engineered)

model_save_path = os.path.join(MODEL_DIR, 'decision_tree_engineered_model_fast.pkl') # Changed filename
scaler_save_path = os.path.join(MODEL_DIR, 'standard_scaler_fast.pkl') # Changed filename
pca_save_path = os.path.join(MODEL_DIR, 'pca_transformer_fast.pkl') # Changed filename
kmeans_save_path = os.path.join(MODEL_DIR, 'kmeans_clusterer_fast.pkl') # Changed filename

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