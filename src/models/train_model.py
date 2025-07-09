from src.models.train_logistic_regression import train_logistic_regression
from src.models.train_xgboost import train_xgboost
from src.models.train_neural_network import train_neural_network

def run_all_trainings():
    print("🚀 Training all models (excluding Random Forest)...")

    print("\n🔹 Training Logistic Regression...")
    train_logistic_regression()

    print("\n🔹 Training XGBoost...")
    train_xgboost()


    print("\n✅ All specified models trained and saved successfully!")

if __name__ == "__main__":
    run_all_trainings()
