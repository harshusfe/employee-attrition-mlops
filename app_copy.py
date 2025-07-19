from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Employee Attrition Predictor")

# Load your best model (change this as needed)
import joblib
model = joblib.load("models/model.pkl")

#model = pickle.load(open("models/model.pkl", "rb"))

# Define the expected input structure
class EmployeeData(BaseModel):
    Age: int
    DailyRate: int
    DistanceFromHome: int
    Education: int
    EnvironmentSatisfaction: int
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    MonthlyIncome: int
    NumCompaniesWorked: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsWithCurrManager: int
    BusinessTravel_Travel_Frequently: bool
    BusinessTravel_Travel_Rarely: bool
    Department_Research_Development: bool
    Department_Sales: bool
    EducationField_Life_Sciences: bool
    EducationField_Marketing: bool
    EducationField_Medical: bool
    EducationField_Other: bool
    EducationField_Technical_Degree: bool
    Gender_Male: bool
    JobRole_Human_Resources: bool
    JobRole_Laboratory_Technician: bool
    JobRole_Manager: bool
    JobRole_Manufacturing_Director: bool
    JobRole_Research_Director: bool
    JobRole_Research_Scientist: bool
    JobRole_Sales_Executive: bool
    JobRole_Sales_Representative: bool
    MaritalStatus_Married: bool
    MaritalStatus_Single: bool
    OverTime_Yes: bool
    # Add all one-hot encoded fields used during training

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.post("/predict")
def predict(data: EmployeeData):
    input_data = np.array([[ 
        data.Age,
        data.DailyRate,
        data.DistanceFromHome,
        data.Education,
        data.EnvironmentSatisfaction,
        data.JobInvolvement,
        data.JobLevel,
        data.JobSatisfaction,
        data.MonthlyIncome,
        data.NumCompaniesWorked,
        data.PerformanceRating,
        data.RelationshipSatisfaction,
        data.StockOptionLevel,
        data.TotalWorkingYears,
        data.TrainingTimesLastYear,
        data.WorkLifeBalance,
        data.YearsAtCompany,
        data.YearsInCurrentRole,
        data.YearsWithCurrManager,
        int(data.BusinessTravel_Travel_Frequently),
        int(data.BusinessTravel_Travel_Rarely),
        int(data.Department_Research_Development),
        int(data.Department_Sales),
        int(data.EducationField_Life_Sciences),
        int(data.EducationField_Marketing),
        int(data.EducationField_Medical),
        int(data.EducationField_Other),
        int(data.EducationField_Technical_Degree),
        int(data.Gender_Male),
        int(data.JobRole_Human_Resources),
        int(data.JobRole_Laboratory_Technician),
        int(data.JobRole_Manager),
        int(data.JobRole_Manufacturing_Director),
        int(data.JobRole_Research_Director),
        int(data.JobRole_Research_Scientist),
        int(data.JobRole_Sales_Executive),
        int(data.JobRole_Sales_Representative),
        int(data.MaritalStatus_Married),
        int(data.MaritalStatus_Single),
        int(data.OverTime_Yes)
    ]])

    prediction = model.predict(input_data)[0]
    return {"Attrition": int(prediction)}

