import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_spambase_dataset
from src.data.preprocess import preprocess_data

def train_logistic_regression():
    # Step 1: Load and preprocess data
    df = load_spambase_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Step 2: Train Logistic Regression
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)

    # Step 3: Save model
    joblib.dump(model, 'models/logistic_regression/model.pkl')
    print("✅ Logistic Regression model trained and saved successfully.")

if __name__ == "__main__":
    train_logistic_regression()
