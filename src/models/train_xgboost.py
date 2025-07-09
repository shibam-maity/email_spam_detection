import pandas as pd
import joblib
from xgboost import XGBClassifier

from src.data.load_data import load_spambase_dataset
from src.data.preprocess import preprocess_data

def train_xgboost():
    # Step 1: Load and preprocess data
    df = load_spambase_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Step 2: Train XGBoost model with best tuned hyperparameters
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=7,
        learning_rate=0.01,
        subsample=1.0,
        colsample_bytree=0.8,
        gamma=0,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Step 3: Save model
    joblib.dump(model, 'models/xgboost/model.pkl')
    print("✅ XGBoost model trained and saved successfully.")

if __name__ == "__main__":
    train_xgboost()
