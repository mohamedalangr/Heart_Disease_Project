# 💓 Heart Disease Risk Prediction – Full ML Pipeline

This project builds an end-to-end ML system to predict heart disease risk using the UCI dataset. It includes data cleaning, feature selection, PCA, classification, clustering, tuning, and a Streamlit app for real-time prediction.

## 📦 How to Run

```bash
pip install -r requirements.txt
streamlit run ui/app.py

📁 Folder Structure
data/ → Dataset files

models/ → Saved models (.pkl)

notebooks/ → Jupyter steps

ui/ → Streamlit app

deployment/ → Ngrok/hosting instructions

results/ → Metrics/charts

📊 Dataset
Source: UCI Heart Disease Dataset

File used: processed.cleveland.data

🧠 Models
Logistic Regression

Decision Tree

Random Forest

SVM

KMeans & Hierarchical Clustering

🔧 Tools Used
Python, Pandas, Sklearn, Seaborn, Streamlit

---

#### ✅ Create `.gitignore`

Paste this:

```txt
__pycache__/
.ipynb_checkpoints/
*.pkl
*.pyc
.env

.DS_Store
