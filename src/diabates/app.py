import joblib
import pandas as pd

MODEL_PATH = "models/diabetes_model.pkl"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Load model
model = joblib.load(MODEL_PATH)

def predict_diabetes(user_input: dict):
    """
    user_input example:
    {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 94,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    """
    df = pd.DataFrame([user_input], columns=FEATURES)

    probability = model.predict_proba(df)[0][1]  # probability of diabetes
    percentage = round(probability * 100, 2)

    return {
        "diabetes_risk": f"{percentage}%",
        "risk_level": (
            "Low" if percentage < 30 else
            "Moderate" if percentage < 60 else
            "High"
        )
    }


# 🔹 Test run
if __name__ == "__main__":
    sample_input = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 94,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }

    result = predict_diabetes(sample_input)
    print("🩺 Diabetes Risk Assessment:")
    print(f"Chance of Diabetes: {result['diabetes_risk']}")
    print(f"Risk Level: {result['risk_level']}")
