# src/evaluation/evaluate_models.py
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.models import load_model

# Load preprocessed dataset
data = pd.read_csv('data/processed/cleaned_dataset.csv')
X = data.drop(columns=['is_spam'])
y = data['is_spam']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models directory
models_dir = "models"
models_info = {
    "Logistic Regression": os.path.join(models_dir, "logistic_regression", "model.pkl"),
    "XGBoost": os.path.join(models_dir, "xgboost", "model.pkl"),
    "Random Forest": os.path.join(models_dir, "random_forest", "model.pkl")
}

# Evaluation results
evaluation_results = []

print("\n🔍 Model Evaluation Results:\n")

for name, path in models_info.items():
    if not os.path.exists(path):
        print(f"❌ Model file not found: {path}")
        continue

    print(f"Evaluating {name}...")

    if name == "Neural Network":
        model = load_model(path)
        y_pred = model.predict(X_test_scaled)
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    else:
        model = joblib.load(path)
        y_pred_classes = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)

    evaluation_results.append((name, accuracy, precision, recall, f1))

# Sort and display
sorted_results = sorted(evaluation_results, key=lambda x: x[3], reverse=True)

print("\n📊 Comparison of All Models:")
print("Model\t\tAccuracy\tPrecision\tRecall\tF1 Score")
for name, acc, prec, rec, f1 in sorted_results:
    print(f"{name:<16} {acc*100:.2f}%\t{prec*100:.2f}%\t{rec*100:.2f}%\t{f1*100:.2f}%")

best_model = sorted_results[0]
print(f"\n🥇 Best Model: {best_model[0]} with F1 Score = {best_model[3]*100:.2f}%")
