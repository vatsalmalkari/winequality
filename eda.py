import pandas as pd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
# Define the path to your CSV file
file_path = 'winequality-red.csv'

# Check if the file exists before attempting to load
if os.path.exists(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
print("\n" + "="*50 + "\n") # Separator for readability

# --- 2. Handle Nulls/Missing Values ---
print("Checking for missing values:")
print(df.isnull().sum())
# As expected for this dataset, you should see 0 missing values for all columns.
print("\nNo missing values found, proceeding with preprocessing.")
print("\n" + "="*50 + "\n")

# --- 3. Feature and Target Separation ---
# X contains all physicochemical properties (features)
# y is the 'quality' column (target)
X = df.drop('quality', axis=1) # Drop the 'quality' column from features
y = df['quality']             # Select the 'quality' column as the target

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("\n" + "="*50 + "\n") # Separator

# --- 4. Transform Target Variable (Quality) into Binary Classification ---
# The original 'quality' scores range from 3 to 8.
# We'll convert this into a binary classification problem:
# 'Good' wine (quality >= 7) will be labeled 1
# 'Bad' wine (quality < 7) will be labeled 0
# This threshold (7) is common for this dataset, feel free to adjust if needed.
y_binary = (y >= 5).astype(int)

print("Original 'quality' distribution:")
print(y.value_counts().sort_index())
print("\nBinary 'quality' distribution (0: Bad, 1: Good):")
print(y_binary.value_counts())
print("\n" + "="*50 + "\n") # Separator


X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("\n" + "="*50 + "\n") # Separator

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for easier readability and consistency
# This is especially helpful if you plan to add cluster features later.
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("Features scaled successfully!")
print(f"X_train_scaled (first 5 rows):\n{X_train_scaled_df.head()}")
print(f"X_test_scaled (first 5 rows):\n{X_test_scaled_df.head()}")
print("\n" + "="*50 + "\n") # Separator

print("Starting Exploratory Data Analysis and Visualizations...")
# --- 7.1 Distribution of Each Physicochemical Property (Features) ---
# Histograms help visualize the distribution, skewness, and potential outliers of each feature.
plt.figure(figsize=(18, 15))
for i, column in enumerate(X.columns):
    plt.subplot(3, 4, i + 1) # Arrange plots in a 3x4 grid
    sns.histplot(df[column], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.suptitle('Distributions of Wine Physicochemical Properties', y=1.02, fontsize=18) # Add a main title
plt.show()

print("\n" + "="*50 + "\n") # Separator

# --- 7.2 Relationship between Features and Binary Quality ---
# Box plots are good for visualizing the distribution of a numerical variable across categories.
# Here, we see how each feature's values differ between 'Bad' (0) and 'Good' (1) wines.
# We'll use the original (unscaled) features for better interpretability of values.
df_eda = df.copy() # Create a copy for EDA with binary quality
df_eda['quality_binary'] = y_binary

plt.figure(figsize=(18, 15))
for i, column in enumerate(X.columns):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(x='quality_binary', y=column, data=df_eda, palette='viridis')
    plt.title(f'{column} vs. Binary Quality')
    plt.xlabel('Quality (0: Bad, 1: Good)')
    plt.ylabel(column)
    plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.suptitle('Feature Distributions Across Binary Wine Quality', y=1.02, fontsize=18)
plt.show()

print("\n" + "="*50 + "\n") # Separator

# --- 7.3 Correlation Matrix Heatmap ---
# A heatmap shows the correlation coefficients between all pairs of variables.
# This helps identify which features are strongly correlated with 'quality'
# and also if there's multicollinearity among features.
plt.figure(figsize=(12, 10))
# Include the binary quality column in the correlation matrix for insight
correlation_matrix = df_eda.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Wine Attributes and Binary Quality', fontsize=16)
plt.show()

print("\nEDA and Preprocessing complete. The data is now ready for model building!")
print("X_train_scaled_df and y_train are ready for training.")
print("X_test_scaled_df and y_test are ready for evaluation.")