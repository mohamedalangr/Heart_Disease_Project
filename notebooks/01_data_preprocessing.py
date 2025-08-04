# Step 1: Data Preprocessing & Cleaning
# 1.1 Load the Dataset
import pandas as pd

df = pd.read_csv('data/heart_disease.csv')

print(df.head())

# 1.2 Handle Missing Values
# . Replace '?' with NaN and Convert to Numeric
import numpy as np

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Show missing values count
print("Missing values per column:\n", df.isnull().sum())


# . Impute Missing Values
# median for numerical columns and mode for categorical ones
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype in ['float64', 'int64']:
            # Use median for numeric
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
        else:
            # Use mode for categorical
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)


print("Remaining missing values:", df.isnull().sum().sum())  

# 1.3 Encode Categorical Features
# Convert target to binary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# One-hot encoding for categorical features
categorical_cols = ['cp', 'restecg', 'slope', 'thal', 'ca']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 1.4 Scale Numerical Features
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
X_scaled = scaler.fit_transform(X)

# convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Check shape
print(X_scaled_df.shape)

# 1.5 Final Cleaned Dataset
# Combine X and y into a final DataFrame for exporting if needed
final_df = X_scaled_df.copy()
final_df['target'] = y

# Save to CSV
final_df.to_csv('heart_disease_cleaned.csv', index=False)