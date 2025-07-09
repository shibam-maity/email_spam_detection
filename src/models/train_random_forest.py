import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_spambase_dataset
from src.data.preprocess import preprocess_data

def train_random_forest():
    # Step 1: Load and preprocess data
    df = load_spambase_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Step 2: Train Random Forest with best params
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features='log2',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Step 3: Save model and scaler
    joblib.dump(model, 'models/random_forest/model.pkl')
    joblib.dump(scaler, 'models/random_forest/scaler.pkl')

    print("✅ Random Forest model trained and saved successfully.")

if __name__ == "__main__":
    train_random_forest()
