# Credit Card Default Prediction Using Explainable Machine Learning

## Overview

Credit Card Default Prediction Using Explainable Machine Learning is a machine learning project that predicts whether a customer is likely to default on their credit card payment in the following month.

The system uses customer financial and demographic information to perform classification and provides probability-based risk levels. SHAP is used to explain individual model predictions and identify the features that contributed to the prediction.

---

## Features

### Batch Prediction

- Upload CSV or Excel files containing customer information.
- Predict default risk for multiple customers simultaneously.
- Display default probability and risk level.
- Download prediction results as a CSV file.

### Manual Prediction

- Enter customer details through the web interface.
- Predict whether the customer is likely to default.
- Display default probability and risk classification.
- Generate SHAP-based feature contribution visualization.

### Explainable AI

- Uses SHAP to explain individual model predictions.
- Visualizes the contribution of input features using SHAP values.
- Helps interpret which features influenced the prediction.

### Risk Classification

- Low Risk: < 30%
- Medium Risk: 30% to < 70%
- High Risk: ≥ 70%

### Web Application

- Flask-based interactive web application.
- Supports manual and batch predictions.
- Provides prediction probabilities and risk levels.
- Provides SHAP visualization for manual predictions.

---

## Project Structure

```text
Credit-Card-Default-Prediction/
│
├── app.py
├── final.ipynb
├── scaler.pkl
├── feature_names.pkl
├── default_of_credit_card_clients.csv
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── shap_plot.png

## Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Logistic Regression
- Decision Tree
- Random Forest
- SMOTE

### Data Processing

- Pandas
- NumPy
- StandardScaler

### Explainable AI

- SHAP

### Web Application

- Flask
- HTML
- CSS
- JavaScript

### Tools and Libraries

- Matplotlib
- Joblib
- OpenPyXL
- Jupyter Notebook

---

## Dataset

The project uses the **Default of Credit Card Clients Dataset**.

The dataset contains customer demographic information, credit limits, repayment history, bill statements, and payment records.

### Important Features

- LIMIT_BAL
- SEX
- EDUCATION
- MARRIAGE
- AGE
- PAY_0 to PAY_6
- BILL_AMT1 to BILL_AMT6
- PAY_AMT1 to PAY_AMT6

### Target Variable

- `1` = Default
- `0` = Not Default

---

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Scaling using StandardScaler
4. Class Balancing using SMOTE
5. Model Training
6. Model Evaluation
7. Model Selection based on F1-Score
8. SHAP-based Explainability
9. Flask Web Application

---

## Model Comparison

The following machine learning algorithms were evaluated using Accuracy, Precision, Recall, and F1-Score:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 68.90% | 37.05% | 58.10% | 45.25% |
| Decision Tree | 68.62% | 36.79% | 58.33% | 45.12% |
| Random Forest | 78.40% | 51.25% | 47.85% | 49.49% |

The model with the highest F1-Score was selected as the final model. In this evaluation, Random Forest achieved the highest F1-Score among the evaluated models.

---

## Installation

### Install Dependencies

```bash
pip install flask pandas numpy scikit-learn joblib shap matplotlib openpyxl imbalanced-learn
```

### Generate the Model

The trained model file is not included in the repository because of its file size.

First, open and run:

```text
final.ipynb
```

The notebook generates:

```text
best_model.pkl
scaler.pkl
feature_names.pkl
```

### Run the Flask Application

After generating `best_model.pkl`, run:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

---

## Explainable AI with SHAP

SHAP is used to explain individual predictions made by the selected machine learning model.

The application generates visual explanations showing which features contributed to the prediction and their relative impact.

---

## Future Enhancements

- Cloud deployment
- Database integration
- Advanced analytics dashboard
- Batch SHAP explainability
- Enhanced user interface
- Real-time risk monitoring
- Mobile application support

---

## Author

**Rohini Goguri**

B.Tech – Artificial Intelligence & Machine Learning

Project: **Credit Card Default Prediction Using Explainable Machine Learning**
