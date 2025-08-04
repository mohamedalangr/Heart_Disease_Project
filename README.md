# 💓 Heart Disease Risk Prediction – Full Machine Learning Pipeline

This project implements a complete end-to-end machine learning pipeline to predict the risk of heart disease using the UCI Heart Disease dataset. It includes data preprocessing, feature selection, dimensionality reduction, classification models, unsupervised learning, model tuning, deployment with Streamlit, and public sharing using Ngrok.

---

## 📊 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Used File**: `processed.cleveland.data`
- **Features**: Age, sex, chest pain type, cholesterol, ECG results, etc.
- **Target**: Presence of heart disease (binary: 0 = no disease, 1 = disease)

---

## 📁 Project Structure

Heart_Disease_Project/
├── data/ # Dataset files
├── models/ # Trained models (.pkl)
├── notebooks/ # Project step notebooks
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ └── 06_hyperparameter_tuning.ipynb
├── ui/ # Streamlit app
│ └── app.py
├── deployment/ # Ngrok
│ └── ngrok_setup.txt
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # This file


## 🧠 Machine Learning Steps

- ✅ Data cleaning & intelligent imputation
- ✅ Feature encoding & scaling
- ✅ PCA for dimensionality reduction
- ✅ Feature selection: Random Forest, RFE, Chi-Square
- ✅ Classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- ✅ Model evaluation: accuracy, precision, recall, F1-score, ROC-AUC
- ✅ Hyperparameter tuning: GridSearchCV, RandomizedSearchCV
- ✅ Unsupervised learning: KMeans, Hierarchical Clustering
- ✅ Streamlit UI for real-time predictions
- ✅ Ngrok for public sharing

---

## 🚀 How to Run the App Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
Run the Streamlit app:

streamlit run ui/app.py
🌐 How to Deploy Publicly (Optional)
Using Ngrok:

# Start Streamlit app
streamlit run ui/app.py

# In a new terminal
ngrok http 8501
This gives you a public URL to share your app.

📦 Requirements
All Python dependencies are listed in requirements.txt. To generate it:

pip freeze > requirements.txt
🧪 Evaluation Metrics
Model performance is tracked and saved in the results/ folder, including ROC curves, confusion matrices, and F1-scores for comparison.

📬 Contact
Project by Mohamed Fathy
For questions or contributions, feel free to open issues or pull requests.
