# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Initialize Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load the dataset directly from Hugging Face data space
DATASET_PATH = "hf://datasets/samuelrego/mlops_dataset/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")

# --- Data Cleaning ---

# Drop unnecessary columns: unnamed index column and CustomerID (unique identifier)
df.drop(columns=[df.columns[0], 'CustomerID'], inplace=True)

# Fix inconsistent Gender values ('Fe Male' -> 'Female')
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Encode categorical columns using LabelEncoder
label_encoder = LabelEncoder()
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'Designation']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("Data cleaning completed.")

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split (80% train, 20% test)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set size: {Xtrain.shape[0]}, Test set size: {Xtest.shape[0]}")

# Save the split datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train and test datasets saved locally.")

# Upload the split datasets back to Hugging Face data space
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="samuelrego/mlops_dataset",
        repo_type="dataset",
    )

print("Train and test datasets uploaded to Hugging Face Hub.")
