# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree # For plotting the decision tree
import joblib # For loading trained models and preprocessing objects
import os # For managing file paths

# --- Configuration and File Paths ---
MODEL_DIR = 'trained_models' # Directory where trained models and scalers are saved
# Ensure these paths match the filenames you used when saving in the previous training script
# (e.g., _fast.pkl if you ran the optimized version)
MODEL_PATH = os.path.join(MODEL_DIR, 'decision_tree_engineered_model_fast.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'standard_scaler_fast.pkl')
PCA_PATH = os.path.join(MODEL_DIR, 'pca_transformer_fast.pkl')
KMEANS_PATH = os.path.join(MODEL_DIR, 'kmeans_clusterer_fast.pkl')

# --- Data File (needed to get original feature names for PCA/KMeans interpretation) ---
DATA_FILE = 'winequality-red.csv'


# --- 1. Load All Trained Artifacts ---
print(f"Loading trained model and preprocessing objects from '{MODEL_DIR}'...")
try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_pca = joblib.load(PCA_PATH)
    loaded_kmeans = joblib.load(KMEANS_PATH)
    print("All artifacts loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: One or more files not found. Ensure '{MODEL_DIR}' exists and contains all saved .pkl files.")
    print(f"Missing file: {e.filename}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    exit()

print("\n" + "="*70 + "\n") # Separator


# --- 2. Prepare Feature Names for Visualization and Interpretation ---

# Load a small dummy DataFrame to get original feature names for interpretation
# This ensures original_feature_names are available even if not running sequentially
if os.path.exists(DATA_FILE):
    temp_df_original = pd.read_csv(DATA_FILE).drop('quality', axis=1)
    original_feature_names = temp_df_original.columns.tolist()
    print(f"Original Feature Names (for interpretation): {original_feature_names}")
else:
    print(f"Warning: Original data file '{DATA_FILE}' not found. Cannot interpret PCA/KMeans in original feature space.")
    original_feature_names = [f'feature_{i}' for i in range(loaded_pca.n_features_in_)] # Fallback
    print("Using generic feature names for interpretation.")


# Get PCA feature names (PC_1, PC_2, etc.)
pca_feature_names = [f'PC_{i+1}' for i in range(loaded_pca.n_components_)]

# Combine PCA feature names with the 'cluster' feature for the Decision Tree
# The order must match how they were passed to the Decision Tree during training.
# In the training script, 'cluster' was added as the last column.
feature_names_for_tree = pca_feature_names + ['cluster']

# Define class names for the target variable (binary quality)
class_names = ['Bad (0)', 'Good (1)']

print(f"Feature names used by Decision Tree: {feature_names_for_tree}")
print(f"Class names: {class_names}")

print("\n" + "="*70 + "\n") # Separator


# --- 3. Plot the Decision Tree ---
print("Generating Decision Tree visualization...")
# Adjust figure size for better readability, especially if the tree is deep.
# max_depth=4 limits the plot depth for clarity; remove for full tree
plt.figure(figsize=(25, 15))
plot_tree(loaded_model,
          feature_names=feature_names_for_tree,
          class_names=class_names,
          filled=True,      # Fill nodes with colors to indicate class
          rounded=True,     # Round node corners
          proportion=False, # Show counts instead of proportions in nodes
          fontsize=10,      # Adjust font size for readability
          max_depth=4       # Limit depth for clarity in plot; remove for full tree
         )
plt.title('Decision Tree for Wine Quality Prediction (Top 4 Levels)', fontsize=18)
plt.show()

print("\nDecision Tree plot displayed.")
print("Note: The plot might be truncated to 4 levels for better readability. Remove 'max_depth' in plot_tree to see full tree.")

print("\n" + "="*70 + "\n") # Separator


# --- 4. Interpret PCA Components (Feature Loadings) ---
print("Interpreting PCA Components (Loadings on Original Features):")

# Create a DataFrame to show component loadings
pca_components_df = pd.DataFrame(loaded_pca.components_,
                                 columns=original_feature_names,
                                 index=[f'PC_{i+1}' for i in range(loaded_pca.n_components_)])

# Display components, focusing on the features with highest absolute loadings for each PC
# Sort by absolute value within each PC to see the most influential original features
print("PCA Components (feature loadings - sorted by absolute influence for each PC):")
# Use .T (transpose) for easier reading if you have many PCs
# Use a diverging colormap like 'coolwarm' or 'RdBu' for loadings (positive/negative impact)
print(pca_components_df.T.style.background_gradient(cmap='RdBu', axis=None).format(precision=3))

print("\nTop 3 original features contributing to each Principal Component:")
for i, pc_name in enumerate(pca_components_df.index):
    # Sort by absolute value to find most influential features
    top_features = pca_components_df.loc[pc_name].abs().sort_values(ascending=False).head(3)
    print(f"\n  - {pc_name}:")
    for feature, loading_abs in top_features.items():
        original_loading = pca_components_df.loc[pc_name, feature]
        print(f"    - {feature}: {original_loading:.3f}") # Show original loading
print("\nInterpretation: Positive loadings mean the original feature contributes positively to the PC; negative means negatively.")

print("\n" + "="*70 + "\n") # Separator


# --- 5. Interpret K-Means Clusters (Centroids) ---
print("Interpreting K-Means Clusters (Centroids in Scaled Original Feature Space):")

# Get cluster centroids (these will be in the PCA-transformed space)
cluster_centers_pca = loaded_kmeans.cluster_centers_

# Inverse transform PCA components back to the scaled original feature space for interpretability
cluster_centers_scaled = loaded_pca.inverse_transform(cluster_centers_pca)

# Create a DataFrame for scaled centroids
cluster_centers_df_scaled = pd.DataFrame(cluster_centers_scaled,
                                         columns=original_feature_names, # Use original feature names
                                         index=[f'Cluster {i}' for i in range(loaded_kmeans.n_clusters)])

print("Cluster Centroids (in Scaled Original Feature Space):")
# Use a diverging colormap to easily see which features are high/low for each cluster
print(cluster_centers_df_scaled.T.style.background_gradient(cmap='RdBu', axis=None).format(precision=3))

print("\nInterpretation: Values represent how many standard deviations away from the mean each feature is for that cluster's centroid.")
print("Positive values are higher than average, negative are lower than average.")

print("\n" + "="*70 + "\n") # Separator


# --- 6. Plot Feature Importances ---
print("Calculating and plotting Feature Importances...")

# Get feature importances from the loaded model
feature_importances = loaded_model.feature_importances_

# Create a Pandas Series for easy sorting and plotting
importance_df = pd.Series(feature_importances, index=feature_names_for_tree)
importance_df = importance_df.sort_values(ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x=importance_df.values, y=importance_df.index, palette='viridis')
plt.title('Feature Importances for Wine Quality Prediction (Engineered Features)', fontsize=16)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(axis='x', alpha=0.75)
plt.show()

print("\nFeature importances plot displayed.")
print("\n" + "="*70 + "\n")
print("Decision Tree visualization and feature importance analysis complete.")
