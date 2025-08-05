import joblib
import pandas as pd
import numpy as np

# ✅ Load model and preprocessing objects
model = joblib.load("models/fake_job_detector_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
scaler = joblib.load("models/scaler.pkl")  # Remove if you didn’t use one

# ✅ Define a predict function
def predict_job(input_dict):
    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])

    # Encode categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Scale numerical features (only if you used a scaler)
    df_scaled = scaler.transform(df)

    # Make prediction
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    # Return result
    result = "Fake Job Posting" if pred == 1 else "Genuine Job Posting"
    return {"prediction": result, "probability": round(prob, 2)}

# ✅ Example usage
if __name__ == "__main__":
    # Example input based on your dataset columns
    sample_input = {
        "title": "Software Engineer",
        "location": "New York",
        "department": "Engineering",
        "salary_range": "70000-90000",
        "company_profile": "We are a software company...",
        "description": "We are hiring developers...",
        "requirements": "Python, JS",
        "benefits": "Health, Bonus",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
        "industry": "Tech",
        "function": "Engineering"
    }

    result = predict_job(sample_input)
    print(result)
