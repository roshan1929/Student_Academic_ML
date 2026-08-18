import pandas as pd

# Load the dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# Display number of rows and columns
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Display information about columns and data types
print("\n--- DATASET INFORMATION ---")
print(df.info())

# Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Check duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())