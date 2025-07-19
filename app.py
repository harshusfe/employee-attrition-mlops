from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your model
model = joblib.load("models/model.pkl")  # Ensure correct path inside Docker

# Helper functions for encoding
def encode_inputs(form_data):
    categorical_options = {
        "BusinessTravel": ["Travel_Frequently", "Travel_Rarely", "Non-Travel"],
        "Department": ["Research & Development", "Sales", "Human Resources"],
        "EducationField": ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree", "Human Resources"],
        "Gender": ["Male", "Female"],
        "JobRole": ["Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", 
                    "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"],
        "MaritalStatus": ["Married", "Single", "Divorced"],
        "OverTime": ["Yes", "No"]
    }

    encoded = []

    # Numerical features
    num_fields = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
                  "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked",
                  "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                  "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
                  "YearsWithCurrManager"]
    for field in num_fields:
        encoded.append(int(form_data[field]))

    # One-hot encoding for categorical features
    for cat in ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]:
        options = categorical_options[cat]
        for option in options:
            encoded.append(1 if form_data[cat] == option else 0)

    return np.array([encoded])


@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict_attrition(request: Request,
    Age: int = Form(...), DailyRate: int = Form(...), DistanceFromHome: int = Form(...),
    Education: int = Form(...), EnvironmentSatisfaction: int = Form(...),
    JobInvolvement: int = Form(...), JobLevel: int = Form(...), JobSatisfaction: int = Form(...),
    MonthlyIncome: int = Form(...), NumCompaniesWorked: int = Form(...), PerformanceRating: int = Form(...),
    RelationshipSatisfaction: int = Form(...), StockOptionLevel: int = Form(...),
    TotalWorkingYears: int = Form(...), TrainingTimesLastYear: int = Form(...),
    WorkLifeBalance: int = Form(...), YearsAtCompany: int = Form(...),
    YearsInCurrentRole: int = Form(...), YearsWithCurrManager: int = Form(...),
    BusinessTravel: str = Form(...), Department: str = Form(...),
    EducationField: str = Form(...), Gender: str = Form(...), JobRole: str = Form(...),
    MaritalStatus: str = Form(...), OverTime: str = Form(...)
):
    form_data = locals()
    form_data.pop("request")
    input_data = encode_inputs(form_data)
    prediction = model.predict(input_data)[0]

    result_text = "Attrition: YES" if prediction == 1 else "Attrition: NO"
    return templates.TemplateResponse("form.html", {"request": request, "result": result_text})
