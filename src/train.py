# src/train.py

import pandas as pd
import os
import joblib
import mlflow
import yaml

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_path = "data/processed/processed.csv"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)
X = df.drop(params["target_column"], axis=1)
y = df[params["target_column"]]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=42
)

# Set MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("employee-attrition")

models = {
    "logistic_regression": LogisticRegression(max_iter=params["logistic"]["max_iter"]),
    "random_forest": RandomForestClassifier(n_estimators=params["rf"]["n_estimators"]),
    "svm": SVC(kernel=params["svm"]["kernel"], C=params["svm"]["C"])
}

best_model = None
best_score = 0

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log params and metrics
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)

        # Save each model
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"✅ Trained {name} with accuracy: {acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_model_name = name

# Save best model as model.pkl
joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))
print(f"🏆 Best model: {best_model_name} with accuracy: {best_score:.4f}")

