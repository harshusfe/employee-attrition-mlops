import pandas as pd
import os

def preprocess(input_path, output_path):
    # Load Excel data
    df = pd.read_csv(input_path, encoding='latin1')
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values (if any)
    df.dropna(inplace=True)
    
    # Convert target column
    df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})

    df = df.drop(['Over18',"EmployeeNumber","HourlyRate","MonthlyRate","PercentSalaryHike","StandardHours","YearsSinceLastPromotion"], axis=1)
    
    # Encode categorical variables using one-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")

if __name__ == "__main__":
    input_path = "data/raw/employee.csv"
    output_path = "data/processed/processed.csv"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preprocess(input_path, output_path)


