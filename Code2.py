import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib

# Step 1: Setup Logging
logging.basicConfig(
    filename="fraud_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Step 2: Generate Synthetic Data (Replace with Database Fetch in Production)
def generate_synthetic_data(num_samples=10000):
    np.random.seed(42)
    random.seed(42)
    
    data = {
        "transaction_id": range(1, num_samples + 1),
        "amount": np.random.exponential(scale=50, size=num_samples).round(2),
        "timestamp": pd.date_range(start="2023-01-01", periods=num_samples, freq="T"),
        "user_id": np.random.choice(["U001", "U002", "U003", "U004"], size=num_samples),
        "location": np.random.choice(["New York", "London", "Sydney", "Mumbai"], size=num_samples),
        "is_fraud": np.random.choice([0, 1], size=num_samples, p=[0.98, 0.02])  # 2% fraud
    }
    logging.info(f"Generated {num_samples} synthetic transaction records.")
    return pd.DataFrame(data)

# Step 3: Perform Exploratory Data Analysis (EDA)
def exploratory_data_analysis(data):
    print("\n--- Exploratory Data Analysis ---")
    print("\nSummary statistics:")
    print(data.describe())
    
    print("\nFraud distribution:")
    print(data['is_fraud'].value_counts(normalize=True) * 100)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data['is_fraud'])
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data['amount'])
    plt.title("Transaction Amount Distribution")
    plt.show()
    
    logging.info("Completed EDA.")

# Step 4: Preprocess Data
def preprocess_data(data):
    # Feature engineering: Extract day, hour, and month
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month

    # Encode categorical variables
    data = pd.get_dummies(data, columns=['user_id', 'location'], drop_first=True)
    
    # Drop irrelevant columns
    data.drop(['transaction_id', 'timestamp'], axis=1, inplace=True)
    
    # Standardize 'amount'
    scaler = StandardScaler()
    data['amount'] = scaler.fit_transform(data[['amount']])
    
    logging.info("Data preprocessing completed.")
    return data

# Step 5: Split Data
def split_data(data):
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    logging.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

# Step 6: Train Model with Hyperparameter Tuning
def train_model(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Model trained with best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Step 7: Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    roc_auc = roc_auc_score(y_test, probs)
    print(f"ROC AUC Score: {roc_auc:.2f}")
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()
    
    logging.info("Model evaluation completed.")

# Step 8: Save Model
def save_model(model, filename="fraud_detection_model.pkl"):
    joblib.dump(model, filename)
    logging.info(f"Model saved to {filename}.")

# Step 9: Load Model
def load_model(filename="fraud_detection_model.pkl"):
    model = joblib.load(filename)
    logging.info(f"Model loaded from {filename}.")
    return model

# Step 10: Main Execution
if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic transaction data...")
    data = generate_synthetic_data(20000)
    
    # Perform EDA
    exploratory_data_analysis(data)
    
    # Preprocess data
    print("\nPreprocessing data...")
    data = preprocess_data(data)
    
    # Split data into training and testing sets
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Save model
    print("\nSaving model...")
    save_model(model)
    
    # Load and reuse model
    print("\nLoading saved model...")
    loaded_model = load_model()
    print("\nEvaluating loaded model...")
    evaluate_model(loaded_model, X_test, y_test)
