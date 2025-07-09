<h1 align="center">📧 End-to-End Email Spam Detection Pipeline</h1>
<p align="center">
  👨‍💻 Built from scratch with love by <b>Shibam Maity</b><br>
  🚀 A full-fledged Machine Learning + NLP project combining model training, evaluation, and live inference via Streamlit.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20XGBoost-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-blueviolet?style=flat-square"/>
  <img src="https://img.shields.io/badge/Project%20Status-Completed-success?style=flat-square"/>
</p>

---

## 🌟 About the Project

As a **machine learning enthusiast and aspiring data scientist**, I challenged myself to build an end-to-end **Spam Email Classifier** from the ground up — not just a notebook, but a complete **production-ready system**.

This project includes:

- 🧠 Training and evaluating multiple models (Random Forest, XGBoost, Logistic Regression)
- 🧼 Manually preprocessing the **UCI Spambase dataset**
- 📤 Extracting 57 handcrafted NLP-style features from `.txt` and `.pdf` email files
- 🧪 Model evaluation using accuracy, precision, recall, and F1
- 🖥️ Developing an **interactive Streamlit app** for real-time prediction
- 🧠 Deploying my trained model for user-friendly interaction

---

## 🚀 Live Streamlit App

Try it here 👉 [your-live-app-link](https://your-streamlit-link.com)

> Upload any `.txt` or `.pdf` email file and see the magic of AI detect spam!

---

## 💻 What I Did — My Contributions

| Component                   | Description |
|----------------------------|-------------|
| **🧹 Data Cleaning**        | Wrote Python script to preprocess raw `.data` file into a clean `.csv` |
| **🧠 Feature Engineering**  | Manually implemented 57 Spambase-style features with custom logic |
| **📈 Model Training**       | Trained & saved Logistic Regression, XGBoost, and Random Forest |
| **⚖️ Model Evaluation**     | Built `evaluate_models.py` to compare models on key metrics |
| **🧪 Feature Extractor**    | Developed feature parser for both `.txt` and `.pdf` files |
| **🖼 Streamlit UI**         | Created `app.py` for live interaction with email predictions |
| **🗂 Project Structure**    | Followed modular, scalable ML pipeline layout |
| **🛠️ Deployment Ready**     | Made sure the app is lightweight, fast, and easily deployable |

---

## 🧠 Tech Stack & Tools

| Category              | Stack                                        |
|----------------------|----------------------------------------------|
| Programming Language | Python 3                                     |
| ML Libraries         | Scikit-learn, XGBoost, Joblib                |
| Visualization        | Streamlit                                    |
| Data Handling        | Pandas, NumPy                                |
| File Parsing         | PyMuPDF (for `.pdf` support)                 |
| UI Framework         | Streamlit                                    |

---

## 🏆 Model Evaluation Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| 🥇 **Random Forest** | **98.81%** | 99.22% | 97.95% | **98.58%** |
| Logistic Regression| 94.57%   | 96.70%    | 90.26% | 93.37%   |
| XGBoost            | 92.51%   | 94.01%    | 92.83% | 93.61%   |

✅ Final model used in app: **Random Forest**

---

## 🧪 Sample Predictions

<p align="center">
  <img src="https://your-csv-prediction-screenshot.png" width="600"/>
</p>

---

## 🗂 Project Structure

email_spam_detection/
├── data/
│ ├── raw/ ← Original UCI data
│ └── processed/ ← Cleaned CSV dataset
├── models/ ← Trained models
├── src/
│ ├── data/ ← Preprocessing scripts
│ ├── models/ ← Training scripts
│ ├── evaluation/ ← Model comparison logic
│ └── utils/ ← Feature extractor for email files
├── streamlit_app/
│ ├── app.py ← Streamlit UI
│ └── requirements.txt

yaml
Copy
Edit

---

## 🔧 Run This Locally

### 1. Clone the repo

```bash
git clone https://github.com/shibammaity69/email_spam_detection.git
cd email_spam_detection
2. Install required packages
bash
Copy
Edit
pip install -r streamlit_app/requirements.txt
3. Preprocess data and train models
bash
Copy
Edit
python src/data/preprocess_data.py
python train_model.py
4. Launch the app
bash
Copy
Edit
streamlit run streamlit_app/app.py
📤 Input Formats Supported


.csv – Optional, for precomputed feature testing

👨‍💻 Author
Shibam Maity
Machine Learning Intern | AI for Healthcare Enthusiast | Open Source Contributor

📫 Email: shibammaitymaity@gmail.com
🔗 LinkedIn : https://www.linkedin.com/in/shibam-maity/