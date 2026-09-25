"""
Student Dropout Risk Prediction API
------------------------------------
Reproduces the exact pipeline from Phase 3 (cleaning/encoding/scaling) and
Phase 5 (Logistic Regression training) of the notebook project, then serves
the trained model behind a REST API for the React frontend to call.

Run:
    pip install -r requirements.txt
    python app.py

The model is trained once at startup (takes ~5-10 seconds) and then reused
for every request. Server runs at http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

app = Flask(__name__)
CORS(app)  # allow the React dev server to call this API from a different port

# ---------------------------------------------------------------------------
# 1. Load, clean, and encode the dataset (same as Phase 3)
# ---------------------------------------------------------------------------
print("Loading dataset...")
dataset = fetch_ucirepo(id=697)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
df = df.drop_duplicates().reset_index(drop=True)

# Binary target: Dropout = 1, Not Dropout (Graduate/Enrolled) = 0
df["Dropout_Binary"] = (df["Target"] == "Dropout").astype(int)

CATEGORICAL_COLS = [
    "Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"
]
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]

NUMERIC_COLS = [
    "Application order", "Previous qualification (grade)", "Admission grade",
    "Age at enrollment", "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP"
]
NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df.columns]

features = df.drop(columns=["Target", "Dropout_Binary"])
features_encoded = pd.get_dummies(features, columns=CATEGORICAL_COLS, drop_first=True)

scaler = StandardScaler()
features_encoded[NUMERIC_COLS] = scaler.fit_transform(features_encoded[NUMERIC_COLS])

X = features_encoded
y = df["Dropout_Binary"]

TRAINING_COLUMNS = X.columns.tolist()  # remember exact column order/shape for inference

# ---------------------------------------------------------------------------
# 2. Train/test split + train the Logistic Regression model (same as Phase 5)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

METRICS = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "training_rows": int(X_train.shape[0]),
    "test_rows": int(X_test.shape[0]),
    "feature_count": int(X.shape[1]),
}
print("Model trained. Test metrics:", METRICS)

# ---------------------------------------------------------------------------
# 3. Helper: build a single-row feature vector matching TRAINING_COLUMNS
#    from the small set of fields the frontend form actually collects.
#    Any column not explicitly set falls back to the dataset's median/mode,
#    so the model always receives a full, correctly-shaped input row.
# ---------------------------------------------------------------------------
DEFAULT_ROW = features_encoded.median(numeric_only=True)


def build_feature_row(payload):
    row = DEFAULT_ROW.copy()

    numeric_map = {
        "admission_grade": "Admission grade",
        "sem1_grade": "Curricular units 1st sem (grade)",
        "sem2_grade": "Curricular units 2nd sem (grade)",
        "sem2_approved": "Curricular units 2nd sem (approved)",
        "age": "Age at enrollment",
    }
    for key, col in numeric_map.items():
        if key in payload and col in row.index:
            row[col] = float(payload[key])

    # Scale the numeric fields the user actually supplied, using the *same*
    # fitted scaler as training, so they're on the correct standardized scale.
    numeric_input = pd.DataFrame([row[NUMERIC_COLS]])
    numeric_input[NUMERIC_COLS] = scaler.transform(numeric_input[NUMERIC_COLS])
    row[NUMERIC_COLS] = numeric_input.iloc[0]

    binary_map = {
        "scholarship": "Scholarship holder_1",
        "tuition_up_to_date": "Tuition fees up to date_1",
        "debtor": "Debtor_1",
    }
    for key, col in binary_map.items():
        if key in payload and col in row.index:
            row[col] = float(payload[key])

    ordered = pd.DataFrame([row])[TRAINING_COLUMNS]
    return ordered


def risk_label(p):
    if p < 0.30:
        return "Low risk"
    if p < 0.60:
        return "Medium risk"
    return "High risk"


# ---------------------------------------------------------------------------
# 4. Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "metrics": METRICS})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    row = build_feature_row(payload)

    proba = float(model.predict_proba(row)[0][1])
    prediction = int(model.predict(row)[0])

    return jsonify({
        "probability": round(proba, 4),
        "probability_pct": round(proba * 100, 1),
        "prediction": "Dropout" if prediction == 1 else "Not Dropout",
        "risk_level": risk_label(proba),
        "input_echo": payload,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
