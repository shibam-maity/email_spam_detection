import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier

from src.data.load_data import load_spambase_dataset
from src.data.preprocess import preprocess_data

def train_neural_network():
    # Step 1: Load and preprocess data
    df = load_spambase_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Step 2: Train Neural Network
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Step 3: Save model
    model.save("models/neural_network/model.keras")
    print("✅ Neural Network model trained and saved successfully.")

if __name__ == "__main__":
    train_neural_network()
