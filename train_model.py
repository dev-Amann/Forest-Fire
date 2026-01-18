import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Set up paths
DATA_PATH = 'data/Algerian_forest_fires_dataset.csv'
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

def load_and_clean_data(filepath):
    """
    Load the dataset and clean it by handling the two concatenated regions.
    """
    # Read the file
    # The dataset has two regions concatenated. We'll read it all, then handle the split.
    # Skip header=1 because the first line is "Bejaia Region Dataset"
    df = pd.read_csv(filepath, header=1)
    
    # Drop the row that separates the two regions (around row 122-127 in raw file, but let's find it dynamically)
    # The separator row usually has 'day','month', etc. as values or 'Sidi-Bel Abbes Region Dataset'
    
    # Identify the row where 'day' column repeats (indicating the second header)
    # Be careful with column name strings, they might have spaces
    df.columns = [col.strip() for col in df.columns]
    
    # Locate the second header
    if 'day' in df.columns:
        # Check where 'day' shows up as a value, effectively filtering out the mid-file header
        df = df[df['day'] != 'day']
        
    # Drop rows where 'Sidi-Bel Abbes Region Dataset' might be in a column (e.g. 'day')
    df = df[df['day'] != 'Sidi-Bel Abbes Region Dataset']
    
    # Drop fully empty rows just in case
    df.dropna(how='all', inplace=True)
    
    # Add a Region column? The user instructions didn't strictly ask for it, 
    # but it helps to know we merged them. 
    # For this simple model, we might just treat them as one dataset.
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    return df

def preprocess_columns(df):
    """
    Convert columns to appropriate types and encode the target.
    """
    # Columns to convert to numeric
    cols_to_numeric = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
    
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows with NaN after conversion
    df.dropna(inplace=True)
    
    # Clean 'Classes' column
    # It might have spaces like 'not fire   ', 'fire   '
    df['Classes'] = df['Classes'].astype(str).str.strip()
    
    # Encode Target: 1 for 'fire', 0 for 'not fire'
    df['target'] = df['Classes'].apply(lambda x: 1 if 'not fire' not in x else 0)
    
    return df

def train_model():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = load_and_clean_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    
    df = preprocess_columns(df)
    print(f"Data preprocessed. Shape after cleaning: {df.shape}")
    
    # Select features and target
    # User requested: Temperature, humidity (RH), wind (Ws), rainfall (Rain)
    X = df[['Temperature', 'RH', 'Ws', 'Rain']]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model and scaler
    print(f"Saving model to {MODEL_PATH} and scaler to {SCALER_PATH}...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Done.")

if __name__ == "__main__":
    train_model()
