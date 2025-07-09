
import streamlit as st
import pandas as pd
import joblib
import os
import sys

# ✅ Ensure src folder is available for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load model
model_path = "models/random_forest/model.pkl"
model = joblib.load(model_path)

# App Title
st.title("📊 Email Spam Classifier (with 57 Features CSV)")
st.markdown("Upload a `.csv` file **with 57 features** as per Spambase format, and the app will classify each email.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your `.csv` file (no label column)", type=["csv"])

if uploaded_file:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)

        # Show preview
        st.subheader("📄 Uploaded Data Preview:")
        st.dataframe(df.head())

        # Validate shape
        if df.shape[1] != 57:
            st.error("❌ Your CSV must have exactly 57 columns (Spambase features).")
            st.stop()

        # Predict
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]  # Probability of spam

        # Prepare results
        df['Prediction'] = predictions
        df['Spam Probability'] = probabilities

        # Display results
        st.subheader("🔍 Prediction Results:")
        st.dataframe(df[['Prediction', 'Spam Probability']])

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name='spam_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"❌ Error while processing the CSV: {e}")
