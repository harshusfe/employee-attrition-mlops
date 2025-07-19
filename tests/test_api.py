import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)

def test_model_load():
    """Ensure the model loads without error"""
    response = client.get("/")
    assert response.status_code == 200

def test_prediction_endpoint():
    """Basic smoke test for prediction route"""
    response = client.post("/predict", data={
        "Age": 35,
        "MonthlyIncome": 5000,
        "Gender": "Male",
        "Education": 3,
        "YearsAtCompany": 4,
        "PerformanceRating": 3,
        "JobLevel": 2,
        "JobSatisfaction": 4,
        "OverTime": "No",
        "MaritalStatus": "Single"
    })
    assert response.status_code == 200
    assert "Attrition" in response.text
