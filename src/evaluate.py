# src/evaluate.py

import pandas as pd
import joblib
import yaml
import json
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load processed data
df = pd.read_csv("data/processed/processed.csv")
X = df.drop(params["target_column"], axis=1)
y = df[params["target_column"]]

# Split again (same split as train.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=42
)

# Load the best model
model = joblib.load("models/model.pkl")

# Predict and evaluate
y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

# Log metrics to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("employee-attrition")

with mlflow.start_run(run_name="evaluation"):
    for key, val in metrics.items():
        if key == "confusion_matrix":
            mlflow.log_text(json.dumps({"confusion_matrix": val}), "confusion_matrix.json")
        else:
            mlflow.log_metric(key, val)

# Save to local metrics file
with open("metrics/eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation complete. Metrics saved and logged to MLflow.")

