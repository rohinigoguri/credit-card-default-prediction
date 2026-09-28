from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import numpy as np
import os
from werkzeug.utils import secure_filename
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
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


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

            if not allowed_file(file.filename):
                error_message = "Invalid file type. Please upload CSV or Excel files."

            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    filename
                )

                file.save(filepath)

                try:
                    # Read uploaded file
                    if filename.lower().endswith('.csv'):
                        df = pd.read_csv(filepath)
                    else:
                        df = pd.read_excel(filepath)

                    # Check required columns
                    missing_cols = [
                        col for col in feature_names
                        if col not in df.columns
                    ]

                    if missing_cols:
                        error_message = (
                            f"Missing required columns: "
                            f"{', '.join(missing_cols)}"
                        )

                    else:
                        # Ensure correct feature order
                        df = df[feature_names]

                        # Feature scaling
                        input_scaled = scaler.transform(df)

                        # Prediction
                        predictions = model.predict(input_scaled)
                        probabilities = model.predict_proba(
                            input_scaled
                        )[:, 1]

                        # Create results
                        results_df = df.copy()

                        results_df['Prediction'] = [
                            'Default' if p == 1 else 'Not Default'
                            for p in predictions
                        ]

                        results_df['Default_Probability'] = (
                            probabilities.round(4)
                        )

                        results_df['Risk_Level'] = [
                            risk_group(p)
                            for p in probabilities
                        ]

                        batch_results = results_df.to_dict('records')

                        # Save results for download
                        results_df.to_csv(
                            "last_results.csv",
                            index=False
                        )

                except Exception as e:
                    error_message = (
                        f"Error processing file: {str(e)}"
                    )

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
                        raise ValueError(
                            f"{feature} is missing"
                        )

                    input_data.append(float(value))

                input_df = pd.DataFrame(
                    [input_data],
                    columns=feature_names
                )

                # Feature scaling
                input_scaled = scaler.transform(input_df)

                # Prediction
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1]

                prediction = (
                    "Default" if pred == 1 else "Not Default"
                )

                probability = round(prob, 4)
                risk = risk_group(prob)

                # ================= SHAP EXPLANATION =================
                explainer = shap.Explainer(model)

                shap_values = explainer(
                    input_scaled,
                    check_additivity=False
                )

                shap_array = np.asarray(
                    shap_values.values
                )

                # Handle binary classification output
                if (
                    shap_array.ndim == 3
                    and shap_array.shape[2] == 2
                ):
                    plot_values = shap_array[0][:, 1]
                else:
                    plot_values = shap_array[0]

                # Create SHAP bar plot
                plt.figure(figsize=(10, 6))

                plt.barh(
                    range(len(feature_names)),
                    plot_values
                )

                plt.yticks(
                    range(len(feature_names)),
                    feature_names
                )

                plt.xlabel('SHAP Value')
                plt.title('Feature Importance (SHAP)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plt.savefig(
                    "static/shap_plot.png",
                    bbox_inches="tight"
                )

                plt.close()

            except Exception as e:
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
    except Exception:
        return "No results available to download"


if __name__ == "__main__":
    app.run(debug=True)
