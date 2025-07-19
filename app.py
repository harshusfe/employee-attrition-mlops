from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("models/model.pkl")

# Predefined values for unused attributes
PREDEFINED = {
    "DailyRate": 800,
    "DistanceFromHome": 5,
    "EnvironmentSatisfaction": 3,
    "JobInvolvement": 3,
    "NumCompaniesWorked": 2,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 6,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 2,
    "YearsInCurrentRole": 3,
    "YearsWithCurrManager": 3,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "EducationField": "Life Sciences",
    "JobRole": "Research Scientist"
}

#One-hot encoding helper
def encode_inputs(form_data):
    categorical_options = {
        "BusinessTravel": ["Travel_Frequently", "Travel_Rarely"],
        "Department": ["Research & Development", "Sales"],
        "EducationField": ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
        "Gender": ["Male"],
        "JobRole": ["Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", 
                    "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"],
        "MaritalStatus": ["Married", "Single"],
        "OverTime": ["Yes"]
    }

    input_vals = {
        **form_data,
        **PREDEFINED
    }

    num_fields = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
                  "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked",
                  "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                  "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
                  "YearsWithCurrManager"]

    encoded = [int(input_vals[f]) for f in num_fields]

    for cat in categorical_options:
        options = categorical_options[cat]
        encoded += [1 if input_vals[cat] == opt else 0 for opt in options]

    return np.array([encoded])


# def encode_inputs(form_data):
#     categorical_options = {
#         "BusinessTravel": ["Travel_Frequently", "Travel_Rarely"],
#         "Department": ["Research & Development", "Sales"],
#         "EducationField": ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
#         "Gender": ["Male"],
#         "JobRole": ["Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", 
#                     "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"],
#         "MaritalStatus": ["Married", "Single"],
#         "OverTime": ["Yes"]
#     }

#     input_vals = {
#         **form_data,
#         **PREDEFINED
#     }

#     num_fields = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
#                   "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked",
#                   "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
#                   "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
#                   "YearsWithCurrManager"]

#     feature_names = []
#     encoded_values = []

#     for f in num_fields:
#         feature_names.append(f)
#         encoded_values.append(int(input_vals[f]))

#     for cat in categorical_options:
#         for opt in categorical_options[cat]:
#             col_name = f"{cat}_{opt}"
#             feature_names.append(col_name)
#             encoded_values.append(1 if input_vals[cat] == opt else 0)

#     return np.array([encoded_values]), feature_names


@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict_attrition(request: Request,
    Age: int = Form(...),
    MonthlyIncome: int = Form(...),
    Gender: str = Form(...),
    Education: int = Form(...),
    YearsAtCompany: int = Form(...),
    PerformanceRating: int = Form(...),
    JobLevel: int = Form(...),
    JobSatisfaction: int = Form(...),
    OverTime: str = Form(...),
    MaritalStatus: str = Form(...)
):
    form_data = {
        "Age": Age,
        "MonthlyIncome": MonthlyIncome,
        "Gender": Gender,
        "Education": Education,
        "YearsAtCompany": YearsAtCompany,
        "PerformanceRating": PerformanceRating,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "OverTime": OverTime,
        "MaritalStatus": MaritalStatus
    }

    input_data = encode_inputs(form_data)


    prediction = model.predict(input_data)[0]
    result_text = "Attrition: YES" if prediction == 1 else "Attrition: NO"
    return templates.TemplateResponse("form.html", {"request": request, "result": result_text})
