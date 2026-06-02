from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import numpy as np
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model files
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Risk grouping
def risk_group(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    risk = None
    batch_results = None
    error_message = None

    if request.method == 'POST':

        # ================= FILE UPLOAD =================
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                try:
                    # Read file
                    if filename.lower().endswith('.csv'):
                        df = pd.read_csv(filepath)
                    else:
                        df = pd.read_excel(filepath)

                    # Check required columns
                    missing_cols = [col for col in feature_names if col not in df.columns]
                    if missing_cols:
                        error_message = f"Missing required columns: {', '.join(missing_cols)}"
                    else:
                        # Ensure correct order
                        df = df[feature_names]

                        input_scaled = scaler.transform(df)

                        predictions = model.predict(input_scaled)
                        probabilities = model.predict_proba(input_scaled)[:, 1]

                        results_df = df.copy()
                        results_df['Prediction'] = ['Default' if p == 1 else 'Not Default' for p in predictions]
                        results_df['Default_Probability'] = probabilities.round(4)
                        results_df['Risk_Level'] = [risk_group(p) for p in probabilities]

                        batch_results = results_df.to_dict('records')

                        # Save for download
                        results_df.to_csv("last_results.csv", index=False)

                except Exception as e:
                    error_message = f"Error processing file: {str(e)}"

                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)

        # ================= MANUAL INPUT =================
        else:
            try:
                input_data = []

                for feature in feature_names:
                    value = request.form.get(feature)

                    if value is None or value.strip() == "":
                        raise ValueError(f"{feature} is missing")

                    input_data.append(float(value))

                input_df = pd.DataFrame([input_data], columns=feature_names)
                input_scaled = scaler.transform(input_df)

                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1]

                prediction = "Default" if pred == 1 else "Not Default"
                probability = round(prob, 4)
                risk = risk_group(prob)

                # SHAP Explanation
                explainer = shap.Explainer(model)
                shap_values = explainer(input_scaled)
                shap_array = np.asarray(shap_values.values)

                if shap_array.ndim == 3 and shap_array.shape[2] == 2:
                    plot_values = shap_array[0][:, 1]
                else:
                    plot_values = shap_array[0]

                # Create bar plot manually
                plt.figure(figsize=(10, 6))
                plt.barh(range(len(feature_names)), plot_values)
                plt.yticks(range(len(feature_names)), feature_names)
                plt.xlabel('SHAP Value')
                plt.title('Feature Importance (SHAP)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("static/shap_plot.png", bbox_inches="tight")
                plt.close()

            except ValueError as e:
                error_message = str(e)

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction=prediction,
        probability=probability,
        risk=risk,
        batch_results=batch_results,
        error_message=error_message,
        shap_plot=prediction is not None
    )


# ================= DOWNLOAD ROUTE =================
@app.route('/download_results')
def download_results():
    try:
        return send_file(
            "last_results.csv",
            as_attachment=True,
            download_name="batch_predictions.csv",
            mimetype='text/csv'
        )
    except:
        return "No results available to download"


if __name__ == "__main__":
    app.run(debug=True)