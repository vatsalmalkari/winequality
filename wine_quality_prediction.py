import pandas as pd
import os

# Define the path to your CSV file
file_path = 'winequality-red.csv'

# Check if the file exists before attempting to load
if os.path.exists(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # --- Initial Data Inspection ---
    print("Dataset loaded successfully! Here are the first 5 rows:")
    print(df.head())

    print("\n--- Basic Info about the DataFrame (columns, non-nulls, dtypes) ---")
    df.info()

    print("\n--- Descriptive Statistics (mean, std, min, max, etc.) ---")
    print(df.describe())

    print("\n--- Value Counts for the 'quality' column ---")
    print(df['quality'].value_counts().sort_index())

else:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure 'winequality-red.csv' is in the root of your 'wine_quality_predictor' directory.")