# Credit Card Default Prediction System

A professional machine learning web application that predicts credit card default risk using customer financial data. The system supports both batch processing and individual client assessment with explainable AI capabilities.

---

## 📁 Project Files

- `app.py` – Flask web application for prediction and routing
- `templates/index.html` – Frontend UI for file upload and manual input
- `final.ipynb` – Jupyter notebook for EDA, training, and evaluation
- `best_model.pkl` – Trained machine learning model
- `scaler.pkl` – Feature scaling model
- `feature_names.pkl` – Required input feature names
- `default_of_credit_card_clients.csv` – Dataset used for training
- `uploads/` – Temporary folder for uploaded files

---

## 🚀 Quick Start

### Install Dependencies

```powershell
pip install flask pandas numpy joblib openpyxl shap matplotlib
```

### Run Application

```powershell
cd "c:\Users\goyur\OneDrive\Desktop\ml pbl"
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## ✨ Key Features

### 📂 File Upload (Batch Prediction)

* Upload CSV / Excel files
* Predict multiple clients at once
* Download results as CSV

### 🧍 Manual Input Prediction

* Enter customer details manually
* Predict Default / Not Default
* Probability score
* Risk level classification

### 📊 Explainable AI (SHAP)

* Visual explanation for manual predictions
* Shows which features increased or decreased risk
* Improves model transparency

---

## 🎯 Prediction Output

* **Prediction:** Default / Not Default
* **Default Probability:** 0% – 100%
* **Risk Level:**

  * Low Risk (<30%)
  * Medium Risk (30%–70%)
  * High Risk (>70%)

---

## 🧠 Machine Learning Model

* Predicts next-month credit card default
* Trained using historical payment behavior
* Uses financial + demographic features
* Best model selected after comparison

---

## 📈 Result Dashboard

* Interactive results display
* Batch prediction table
* Downloadable reports
* SHAP explainability chart

---

## 🛠️ Technology Stack

* **Backend:** Flask (Python)
* **Frontend:** HTML5, CSS3, JavaScript
* **Machine Learning:** scikit-learn
* **Libraries:** pandas, numpy, SHAP
* **File Support:** CSV, Excel

## 📌 Future Enhancements

* Deploy online (Render / AWS)
* Add database integration
* Batch SHAP explainability
* Advanced analytics dashboard
* Improved UI/UX
